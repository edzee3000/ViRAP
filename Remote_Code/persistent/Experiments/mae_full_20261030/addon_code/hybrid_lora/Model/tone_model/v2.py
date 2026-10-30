"""V2.1 architecture: shared DINOv2 + reference-conditioned spatial weights.

Uses the upstream DINOv2 implementation through a pinned local checkout. No
weights are silently downloaded. Structure-only execution requires explicit opt-in.
This implementation is a design baseline, not a trained or validated method.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .model import (rgb_to_normalized_lab, identity_lut, sample_lut,
                    color_basis, project_residual)


@dataclass
class V2Config:
    visual_size: int = 280
    photo_size: int = 256
    readout_dim: int = 256
    num_queries: int = 16
    readout_layers: int = 2
    readout_heads: int = 8
    readout_ffn: int = 1024
    global_dim: int = 256
    visual_layers: tuple = (3, 7, 11)
    pair_attention: bool = True
    context_size: int = 64
    lut_size: int = 17
    color_width: int = 128
    local_rank: int = 2
    global_limit: float = 1.0
    local_limit: float = .15
    local_output_limit: float = .15
    projection: str = 'none'
    projection_rcond: float = 1e-4
    render_chunk_rows: int = 128
    flat_field_scale: float = .05
    spatial_conditioning: str = 'pair'
    spatial_film_limit: float = .25
    visual_dim: int = 384
    visual_patch_size: int = 14

    def __post_init__(self):
        self.visual_layers=tuple(self.visual_layers)
        if self.visual_patch_size<1 or self.visual_size%self.visual_patch_size or self.visual_size<2*self.visual_patch_size:
            raise ValueError('visual_size must be a multiple of visual_patch_size and at least two patches')
        if self.visual_dim<1: raise ValueError('visual_dim must be positive')
        if self.readout_dim%self.readout_heads: raise ValueError('readout dimensions/head mismatch')
        if self.visual_layers!=(3,7,11): raise ValueError('v0.2 freezes the feature protocol at layers 4/8/12')
        if min(self.readout_layers,self.num_queries,self.local_rank,self.render_chunk_rows)<1: raise ValueError('invalid dimensions')
        if self.projection not in ('none','mean','color'): raise ValueError('invalid projection')
        if self.spatial_conditioning not in ('none','pair'): raise ValueError('invalid spatial conditioning')
        if not 0 < self.spatial_film_limit <= 1: raise ValueError('spatial FiLM limit must be in (0,1]')


def load_official_backbone(repo, weights=None, allow_uninitialized=False):
    repo=Path(repo).resolve()
    if not (repo/'dinov2/hub/backbones.py').is_file(): raise FileNotFoundError('Expected official DINOv2 checkout')
    if weights is None and not allow_uninitialized:
        raise ValueError('Official pretrained weights required. Use allow_uninitialized only for structure/smoke checks.')
    # Upstream imports remain in their original namespace/license; not rewritten.
    sys.path.insert(0,str(repo))
    from dinov2.hub.backbones import dinov2_vits14
    backbone=dinov2_vits14(pretrained=False)  # keep upstream img_size=518
    if weights is not None:
        state=torch.load(weights,map_location='cpu',weights_only=True)
        backbone.load_state_dict(state,strict=True)
    return backbone


class PhotometricBranch(nn.Module):
    """RGB-pretrained trunk is separate from this unnormalized Lab pathway."""
    def __init__(self):
        super().__init__()
        self.stages=nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1,padding_mode='replicate'),nn.GELU())
                                  for a,b in [(3,32),(32,64),(64,128),(128,192)]])
        self.register_buffer('light_centers',torch.linspace(0,1,32))
        self.register_buffer('chroma_centers',torch.linspace(0,1,16))
        self.register_buffer('zone_centers',torch.linspace(0,1,6))

    @staticmethod
    def hats(values,centers):
        return (1-(values[...,None]-centers).abs()*(len(centers)-1)).clamp_min(0)

    def forward(self,lab):
        outputs=[];feature=lab
        for stage in self.stages:
            feature=stage(feature);outputs.append(feature)
        return outputs,self.statistics(lab)

    def statistics(self,lab):
        pixels=lab.flatten(2).transpose(1,2);light,a,b=pixels.unbind(-1)
        histogram_l=self.hats(light,self.light_centers).mean(1)
        wa=self.hats(a,self.chroma_centers);wb=self.hats(b,self.chroma_centers)
        joint=torch.einsum('bni,bnj->bij',wa,wb).flatten(1)/pixels.shape[1]
        zones=self.hats(light,self.zone_centers);mass=zones.sum(1)
        ab=pixels[:,:,1:];denom=mass[...,None].clamp_min(1e-6)
        means=torch.einsum('bnk,bnc->bkc',zones,ab)/denom
        second=torch.einsum('bnk,bnc->bkc',zones,ab.square())/denom
        stds=(second-means.square()).clamp_min(1e-8).sqrt()
        # 6 bins x (a,b means + a,b stds + occupancy) = 30.
        conditional=torch.cat([means,stds,(mass/pixels.shape[1])[...,None]],-1).flatten(1)
        luma=lab[:,:1];contrast=luma-F.avg_pool2d(F.pad(luma,(1,1,1,1),mode='replicate'),3,stride=1)
        moments=torch.cat([pixels.mean(1),pixels.std(1,correction=0),
                           contrast.flatten(1).mean(1,keepdim=True),contrast.flatten(1).std(1,correction=0,keepdim=True)],1)
        return [histogram_l,joint,conditional,moments]


class SharedPhotographicEncoder(nn.Module):
    def __init__(self,backbone,cfg):
        super().__init__();self.backbone=backbone;self.cfg=cfg;d=cfg.readout_dim
        self.photometric=PhotometricBranch()
        self.visual_projections=nn.ModuleList([nn.Linear(cfg.visual_dim,d) for _ in cfg.visual_layers])
        self.photo_projections=nn.ModuleList([nn.Linear(c,d) for c in [64,128,192]])
        self.stat_projections=nn.ModuleList([nn.Linear(c,d) for c in [32,256,30,8]])
        self.source_types=nn.Parameter(torch.empty(10,d))
        self.queries=nn.Parameter(torch.empty(1,cfg.num_queries,d))
        nn.init.normal_(self.source_types,std=.02);nn.init.normal_(self.queries,std=.02)
        self.readout=nn.ModuleList([nn.TransformerDecoderLayer(d,cfg.readout_heads,cfg.readout_ffn,
                                  dropout=0.,activation='gelu',batch_first=True,norm_first=True)
                                   for _ in range(cfg.readout_layers)])
        self.token_norm=nn.LayerNorm(d)
        self.global_head=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,256),nn.GELU(),nn.Linear(256,cfg.global_dim))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,image):
        cfg=self.cfg;visual=F.interpolate(image,(cfg.visual_size,cfg.visual_size),mode='bilinear',align_corners=False,antialias=True)
        visual=(visual-self.rgb_mean)/self.rgb_std
        context=nullcontext() if any(p.requires_grad for p in self.backbone.parameters()) else torch.no_grad()
        with context:
            stages=self.backbone.get_intermediate_layers(visual,n=cfg.visual_layers,reshape=False,return_class_token=True,norm=True)
        # Photometric statistics retain FP32 under optional CUDA mixed precision.
        with torch.autocast(device_type=image.device.type,enabled=False):
            lab=rgb_to_normalized_lab(F.interpolate(image.float(),(cfg.photo_size,cfg.photo_size),mode='bilinear',align_corners=False,antialias=True))
            photo,statistics=self.photometric(lab)
        memory=[]
        for i,((patches,cls),project) in enumerate(zip(stages,self.visual_projections)):
            memory.append(project(torch.cat([cls[:,None],patches],1))+self.source_types[i])
        for i,(features,project) in enumerate(zip(photo[1:],self.photo_projections)):
            tokens=F.adaptive_avg_pool2d(features,(8,8)).flatten(2).transpose(1,2)
            memory.append(project(tokens)+self.source_types[3+i])
        for i,(values,project) in enumerate(zip(statistics,self.stat_projections)):
            memory.append(project(values)[:,None]+self.source_types[6+i])
        memory=torch.cat(memory,1);queries=self.queries.expand(image.shape[0],-1,-1)
        for layer in self.readout: queries=layer(queries,memory)
        tokens=self.token_norm(queries)
        global_vector=self.global_head(torch.cat([tokens.mean(1),tokens.std(1,correction=0)],1))
        return {'global':global_vector,'tokens':tokens,'dense':stages[-1][0],
                'photo_dense':photo[1],'memory_tokens':memory.shape[1]}


class PairConditioner(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg.readout_dim;self.cfg=cfg
        self.global_pair=nn.Sequential(nn.Linear(4*cfg.global_dim,512),nn.GELU(),nn.Linear(512,d),nn.GELU())
        if cfg.pair_attention:
            self.query_norm=nn.LayerNorm(d);self.memory_norm=nn.LayerNorm(d);self.ffn_norm=nn.LayerNorm(d)
            self.cross_attention=nn.MultiheadAttention(d,cfg.readout_heads,dropout=0.,batch_first=True)
            self.ffn=nn.Sequential(nn.Linear(d,cfg.readout_ffn),nn.GELU(),nn.Linear(cfg.readout_ffn,d))
            self.delta_head=nn.Linear(d,d)
            self.fusion=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d))

    def forward(self,content,reference):
        zc,zr=content['global'],reference['global']
        pair=self.global_pair(torch.cat([zc,zr,zr-zc,zc*zr],1))
        if not self.cfg.pair_attention:return pair
        q=self.query_norm(content['tokens']);memory=self.memory_norm(reference['tokens'])
        delta=self.cross_attention(q,memory,memory,need_weights=False)[0]
        fused=content['tokens']+delta;fused=fused+self.ffn(self.ffn_norm(fused))
        return self.fusion(torch.cat([pair,self.delta_head((fused-content['tokens']).mean(1))],1))


class SourceContentContext(nn.Module):
    """Input-coordinate maps, optionally modulated by the input/reference pair.

    No reference feature raster is copied or warped into the input. This is not
    a proof of semantic correspondence or freedom from color/halo artifacts.
    spatial_conditioning='none' retains the original parameter/key structure.
    """
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.visual_adapter=nn.Conv2d(cfg.visual_dim,128,1)
        self.spatial=nn.Sequential(nn.Conv2d(198,96,3,padding=1,padding_mode='replicate'),nn.GroupNorm(12,96),nn.SiLU(),
                                   nn.Conv2d(96,64,3,padding=1,padding_mode='replicate'),nn.GroupNorm(8,64),nn.SiLU(),
                                   nn.Conv2d(64,cfg.local_rank,1),nn.Tanh())
        if cfg.spatial_conditioning=='pair':
            self.pair_film=nn.Linear(cfg.readout_dim,2*(96+64))
            # Small nonzero weights expose conditioning from initialization;
            # identity initialization is still guaranteed by the zero LUT head.
            nn.init.normal_(self.pair_film.weight,std=.001)
            nn.init.zeros_(self.pair_film.bias)

    def forward(self,image,encoded,condition=None):
        side=self.cfg.visual_size//self.cfg.visual_patch_size;size=(self.cfg.context_size,)*2
        dense=encoded['dense'].transpose(1,2).reshape(image.shape[0],self.cfg.visual_dim,side,side)
        dense=F.interpolate(self.visual_adapter(dense),size,mode='bilinear',align_corners=False)
        photo=F.interpolate(encoded['photo_dense'],size,mode='bilinear',align_corners=False)
        rgb=F.interpolate(image,size,mode='bilinear',align_corners=False)
        luma=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
        local=luma-F.avg_pool2d(F.pad(luma,(1,1,1,1),mode='replicate'),3,stride=1)
        dx=F.pad(luma[:,:,:,1:]-luma[:,:,:,:-1],(0,1,0,0));dy=F.pad(luma[:,:,1:]-luma[:,:,:-1],(0,0,0,1))
        features=torch.cat([dense,photo,rgb,luma,local,dx.abs()+dy.abs()],1)
        if self.cfg.spatial_conditioning=='pair':
            if condition is None or condition.shape!=(image.shape[0],self.cfg.readout_dim):
                raise ValueError('Pair-conditioned spatial maps require a [B, readout_dim] condition')
            parameters=self.pair_film(condition).tanh()*self.cfg.spatial_film_limit
            gamma1,beta1,gamma2,beta2=parameters.split([96,96,64,64],dim=1)
            features=self.spatial[1](self.spatial[0](features))
            features=self.spatial[2](features*(1+gamma1[:,:,None,None])+beta1[:,:,None,None])
            features=self.spatial[4](self.spatial[3](features))
            features=self.spatial[5](features*(1+gamma2[:,:,None,None])+beta2[:,:,None,None])
            maps=self.spatial[7](self.spatial[6](features))
        else:
            maps=self.spatial(features)
        # DINO contains positional information. An input-wide contrast gate
        # prevents its position prior from painting structure on a constant field.
        # This heuristic is not a guarantee for ordinary photos or near-flat noise.
        if self.cfg.flat_field_scale>0:
            support=(image.float().std((2,3),correction=0).mean(1)/self.cfg.flat_field_scale).clamp(0,1)
            maps=maps*support[:,None,None,None]
        return maps


class ConditionalColorMLP(nn.Module):
    """Generate all LUTs on a color grid once; no per-pixel deep image decoder."""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;h=cfg.color_width
        self.color_layers=nn.ModuleList([nn.Linear(3,h),nn.Linear(h,h)])
        self.film=nn.Linear(cfg.readout_dim,4*h)
        self.output=nn.Linear(h,3*(1+cfg.local_rank))
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('identity',identity_lut(cfg.lut_size))

    def forward(self,condition):
        cfg=self.cfg;batch=condition.shape[0];h=cfg.color_width
        feature=(self.identity.permute(0,2,3,4,1).reshape(1,-1,3)*2-1).expand(batch,-1,-1)
        parameters=self.film(condition).reshape(batch,2,2,h)
        for i,layer in enumerate(self.color_layers):
            gamma,beta=parameters[:,i].unbind(1)
            feature=F.silu(layer(feature)*(1+gamma[:,None])+beta[:,None])
        values=self.output(feature).reshape(batch,cfg.lut_size,cfg.lut_size,cfg.lut_size,1+cfg.local_rank,3)
        values=values.permute(0,4,5,1,2,3).tanh()
        return self.identity+cfg.global_limit*values[:,0],cfg.local_limit*values[:,1:]


class ReferenceToneModelV2(nn.Module):
    def __init__(self,backbone,cfg=None,*,pretrained_loaded=False,allow_uninitialized=False):
        super().__init__();self.config=cfg or V2Config();self.pretrained_loaded=pretrained_loaded
        if not pretrained_loaded and not allow_uninitialized:
            raise ValueError('Uninitialized backbone allowed for explicit structure/smoke inspection only')
        self.allow_uninitialized=allow_uninitialized
        self.encoder=SharedPhotographicEncoder(backbone,self.config)
        self.conditioner=PairConditioner(self.config)
        self.content_context=SourceContentContext(self.config)
        self.color_generator=ConditionalColorMLP(self.config)
        self.gates=nn.Linear(self.config.readout_dim,self.config.local_rank)
        nn.init.zeros_(self.gates.weight);nn.init.constant_(self.gates.bias,-2.)
        self.set_stage('heads')

    @classmethod
    def from_official(cls,repo,weights=None,cfg=None,allow_uninitialized=False):
        backbone=load_official_backbone(repo,weights,allow_uninitialized)
        return cls(backbone,cfg,pretrained_loaded=weights is not None,allow_uninitialized=allow_uninitialized)

    def set_stage(self,stage):
        if stage not in ('encoder','heads','joint','full'):raise ValueError(stage)
        for parameter in self.parameters():parameter.requires_grad_(stage!='encoder')
        if stage=='encoder':
            for parameter in self.encoder.parameters():parameter.requires_grad_(True)
        if stage!='full':
            for parameter in self.encoder.backbone.parameters():parameter.requires_grad_(False)
        if stage=='joint':
            for module in [*self.encoder.backbone.blocks[-4:],self.encoder.backbone.norm]:
                for parameter in module.parameters():parameter.requires_grad_(True)
        self.stage=stage;self.encoder.backbone.eval()

    def train(self,mode=True):
        super().train(mode);self.encoder.backbone.eval()
        return self

    def forward(self,content,reference,return_aux=False):
        if content.ndim!=4 or reference.ndim!=4 or content.shape[1]!=3 or reference.shape[1]!=3:raise ValueError('RGB BCHW required')
        c=self.config;encoded=self.encoder(content);style=self.encoder(reference)
        condition=self.conditioner(encoded,style)
        global_lut,local_luts=self.color_generator(condition)
        maps=self.content_context(content,encoded,condition);gates=self.gates(condition).sigmoid()
        proxy=F.interpolate(content,(c.context_size,)*2,mode='bilinear',align_corners=False)
        raw=sum((sample_lut(local_luts[:,k],proxy)*maps[:,k:k+1]*gates[:,k,None,None,None]
                 for k in range(c.local_rank)),torch.zeros_like(proxy))
        projected,coefficients,basis=project_residual(raw,proxy,c.projection,c.projection_rcond)
        bound=c.local_limit*gates.sum(1)+coefficients.abs().sum(1).amax(1)
        scale=(c.local_output_limit/bound.clamp_min(1e-8)).clamp(max=1)
        height,width=content.shape[-2:];blocks=[]
        for start in range(0,height,c.render_chunk_rows):
            end=min(height,start+c.render_chunk_rows);pixels=content[:,:,start:end]
            yy=(torch.arange(start,end,device=content.device,dtype=content.dtype)+.5)*2/height-1
            xx=(torch.arange(width,device=content.device,dtype=content.dtype)+.5)*2/width-1
            gy,gx=torch.meshgrid(yy,xx,indexing='ij');grid=torch.stack([gx,gy],-1)[None].expand(content.shape[0],-1,-1,-1)
            spatial=F.grid_sample(maps,grid,align_corners=False,padding_mode='border')
            correction=sum((sample_lut(local_luts[:,k],pixels)*spatial[:,k:k+1]*gates[:,k,None,None,None]
                            for k in range(c.local_rank)),torch.zeros_like(pixels))
            correction=correction-torch.einsum('bhwm,bmc->bchw',color_basis(pixels),coefficients)
            block=sample_lut(global_lut,pixels)+scale[:,None,None,None]*correction
            blocks.append(block)
        preclamp=torch.cat(blocks,2);result=preclamp.clamp(0,1)
        if not return_aux:return result
        return result,{'content_global':encoded['global'],'reference_global':style['global'],
                       'content_tokens':encoded['tokens'],'reference_tokens':style['tokens'],
                       'memory_tokens':encoded['memory_tokens'],'maps':maps,'gates':gates,
                       'condition':condition,'global_lut':global_lut,'local_luts':local_luts,
                       'global_delta':global_lut-self.color_generator.identity,'preclamp_output':preclamp,
                       'proxy_local':projected*scale[:,None,None,None],'proxy_basis':basis,
                       'pretrained_loaded':self.pretrained_loaded}
