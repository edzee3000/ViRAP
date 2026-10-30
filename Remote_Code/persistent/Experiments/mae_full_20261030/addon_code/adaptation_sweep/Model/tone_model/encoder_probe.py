"""Controlled frozen-encoder probe, NOT the complete spatial V2 architecture.

All groups use the same directional conditioner and global-only color renderer.
Visual features are cached from public pretrained encoders; no PETAL features.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .v2 import V2Config, PhotometricBranch, PairConditioner
from .model import rgb_to_normalized_lab, identity_lut, sample_lut


GROUPS = {
    'A': {'backbone': None, 'photo': 'statistics'},
    'B': {'backbone': 'dinov2_s', 'photo': 'full'},
    'C': {'backbone': 'dinov3_s', 'photo': 'full'},
    'D': {'backbone': 'mae_b', 'photo': 'full'},
    'E': {'backbone': 'dinov3_s', 'photo': 'none'},
    'F': {'backbone': 'dinov2_b', 'photo': 'full'},
}


class StatisticsOnly(PhotometricBranch):
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('light_centers',torch.linspace(0,1,32))
        self.register_buffer('chroma_centers',torch.linspace(0,1,16))
        self.register_buffer('zone_centers',torch.linspace(0,1,6))

    def forward(self,lab):
        return [],self.statistics(lab)


class ProbeEncoder(nn.Module):
    def __init__(self,group,feature_dim=384):
        super().__init__();self.spec=GROUPS[group];d=256
        self.visual_projections=nn.ModuleList([nn.Linear(feature_dim,d) for _ in range(3)]) if self.spec['backbone'] else nn.ModuleList()
        self.photo=PhotometricBranch() if self.spec['photo']=='full' else StatisticsOnly() if self.spec['photo']=='statistics' else None
        self.photo_projections=nn.ModuleList([nn.Linear(c,d) for c in [64,128,192]]) if self.spec['photo']=='full' else nn.ModuleList()
        self.stat_projections=nn.ModuleList([nn.Linear(c,d) for c in [32,256,30,8]]) if self.photo is not None else nn.ModuleList()
        # Fixed slots preserve the same source-type protocol across groups.
        self.source_types=nn.Parameter(torch.empty(10,d));self.queries=nn.Parameter(torch.empty(1,16,d))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+230)
            nn.init.normal_(self.source_types,std=.02);nn.init.normal_(self.queries,std=.02)
            self.readout=nn.ModuleList([nn.TransformerDecoderLayer(d,8,1024,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
            self.norm=nn.LayerNorm(d)
            self.global_head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Linear(256,256))

    def forward(self,image,visual=None):
        memory=[]
        if self.spec['backbone']:
            if visual is None or visual.shape[1:3]!=(3,197):raise ValueError('Expected cached layers 4/8/12, each CLS + 14x14 patches')
            for i,project in enumerate(self.visual_projections):memory.append(project(visual[:,i].float())+self.source_types[i])
        if self.photo is not None:
            photo,statistics=self.photo(rgb_to_normalized_lab(image))
            for i,(features,project) in enumerate(zip(photo[1:],self.photo_projections)):
                tokens=F.adaptive_avg_pool2d(features,(8,8)).flatten(2).transpose(1,2)
                memory.append(project(tokens)+self.source_types[3+i])
            for i,(values,project) in enumerate(zip(statistics,self.stat_projections)):
                memory.append(project(values)[:,None]+self.source_types[6+i])
        memory=torch.cat(memory,1);q=self.queries.expand(image.shape[0],-1,-1)
        for layer in self.readout:q=layer(q,memory)
        tokens=self.norm(q)
        z=self.global_head(torch.cat([tokens.mean(1),tokens.std(1,correction=0)],1))
        return {'tokens':tokens,'global':z}


class GlobalRenderer(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(3,128),nn.Linear(128,128)])
        self.film=nn.Linear(256,512);self.output=nn.Linear(128,3)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('identity',identity_lut(17))

    def forward(self,image,condition):
        b=image.shape[0];features=(self.identity.permute(0,2,3,4,1).reshape(1,-1,3)*2-1).expand(b,-1,-1)
        film=self.film(condition).reshape(b,2,2,128)
        for i,layer in enumerate(self.layers):
            gamma,beta=film[:,i].unbind(1);features=F.silu(layer(features)*(1+gamma[:,None])+beta[:,None])
        delta=self.output(features).tanh().reshape(b,17,17,17,3).permute(0,4,1,2,3)
        raw=sample_lut(self.identity+delta,image)
        return raw.clamp(0,1),{'delta':delta,'raw':raw}


class EncoderProbe(nn.Module):
    def __init__(self,group,feature_dim=384):
        super().__init__();self.group=group
        self.encoder=ProbeEncoder(group,feature_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+720)
            self.conditioner=PairConditioner(V2Config())
            self.renderer=GlobalRenderer()

    def forward(self,content,reference,render_content,content_features=None,reference_features=None):
        c=self.encoder(content,content_features);r=self.encoder(reference,reference_features)
        return self.renderer(render_content,self.conditioner(c,r))


def probe_loss(prediction,target,aux):
    l1=F.l1_loss(prediction,target)
    gradient=(F.l1_loss(prediction.diff(dim=-1),target.diff(dim=-1))+F.l1_loss(prediction.diff(dim=-2),target.diff(dim=-2)))/2
    smooth=sum(aux['delta'].diff(dim=d).square().mean() for d in (-1,-2,-3))/3
    overflow=(F.relu(-aux['raw'])+F.relu(aux['raw']-1)).mean()
    return l1+.05*gradient+.001*smooth+.01*overflow


@torch.no_grad()
def image_metrics(prediction,target):
    """Per-image values. PSNR is computed per image, not from pooled MSE."""
    p=prediction.float();t=target.float();dims=(1,2,3)
    mse=(p-t).square().mean(dims)
    labp=rgb_to_normalized_lab(p);labt=rgb_to_normalized_lab(t)
    scale=p.new_tensor([100.,255.,255.])[None,:,None,None]
    de76=((labp-labt)*scale).square().sum(1).sqrt().mean((1,2))
    # Standard 11x11 Gaussian-window SSIM, valid interior, RGB channel mean.
    x=torch.arange(11,device=p.device,dtype=p.dtype)-5
    kernel=torch.exp(-x.square()/(2*1.5**2));kernel=kernel/kernel.sum()
    window=(kernel[:,None]*kernel[None,:]).expand(3,1,11,11).contiguous()
    meanp=F.conv2d(p,window,groups=3);meant=F.conv2d(t,window,groups=3)
    varp=(F.conv2d(p*p,window,groups=3)-meanp.square()).clamp_min(0)
    vart=(F.conv2d(t*t,window,groups=3)-meant.square()).clamp_min(0)
    cov=F.conv2d(p*t,window,groups=3)-meanp*meant
    ssim=((2*meanp*meant+.01**2)*(2*cov+.03**2)/((meanp.square()+meant.square()+.01**2)*(varp+vart+.03**2))).mean(dims)
    return {'l1':(p-t).abs().mean(dims),'psnr':-10*mse.clamp_min(1e-12).log10(),'ssim':ssim,'delta_e76':de76}
