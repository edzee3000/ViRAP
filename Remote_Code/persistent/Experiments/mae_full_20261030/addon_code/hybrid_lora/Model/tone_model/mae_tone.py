"""MAE-tone design: strict public MAE encoder, shared photographic readout.

M0 is the backbone-substitution control (K=2); M1 uses K=4. No random
masking or MAE reconstruction decoder is used. Structure-only opt-in is not
pretraining and does not establish quality. Keeps official sin/cos positions fixed.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from .v2 import V2Config,ReferenceToneModelV2


@dataclass
class MAEToneConfig(V2Config):
    visual_size: int = 224
    photo_size: int = 224
    visual_dim: int = 768
    visual_patch_size: int = 16
    local_rank: int = 4

    def __post_init__(self):
        super().__post_init__()
        if (self.visual_size,self.visual_dim,self.visual_patch_size)!=(224,768,16):
            raise ValueError('This revision fixes the official MAE-B/16 224 feature protocol')


class MAEBackbone(nn.Module):
    def __init__(self,trunk):
        super().__init__();self.trunk=trunk
        if trunk.embed_dim!=768 or len(trunk.blocks)!=12:raise ValueError('Expected ViT-B encoder')
        if tuple(trunk.pos_embed.shape)!=(1,197,768):raise ValueError('Unexpected MAE position shape')
        self.trunk.pos_embed.requires_grad_(False)

    @property
    def blocks(self):return self.trunk.blocks

    @property
    def norm(self):return self.trunk.norm

    @property
    def pos_embed(self):return self.trunk.pos_embed

    def get_intermediate_layers(self,image,n=(3,7,11),reshape=False,return_class_token=True,norm=True):
        if reshape or not return_class_token:raise ValueError('Expected token/CLS extraction protocol')
        patches=self.trunk.patch_embed(image)+self.pos_embed[:,1:]
        cls=(self.trunk.cls_token+self.pos_embed[:,:1]).expand(image.shape[0],-1,-1)
        features=torch.cat([cls,patches],1);outputs=[]
        for i,block in enumerate(self.blocks):
            features=block(features)
            if i in n:
                values=self.norm(features) if norm else features
                outputs.append((values[:,1:],values[:,0]))
        return tuple(outputs)


def load_mae_encoder(weights=None,*,allow_uninitialized=False):
    if weights is None and not allow_uninitialized:
        raise ValueError('Provide official MAE pretraining checkpoint, or explicitly opt into structure-only inspection')
    import timm
    trunk=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    # Match the fixed 2D sin/cos position convention even in structure-only
    # runs. Real checkpoint loading below overwrites with its exact tensor.
    axis=torch.arange(14,dtype=torch.float32)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    frequencies=1/(10000**(torch.arange(192,dtype=torch.float32)/192))
    xphase=xx.flatten()[:,None]*frequencies[None]
    yphase=yy.flatten()[:,None]*frequencies[None]
    positions=torch.cat([xphase.sin(),xphase.cos(),yphase.sin(),yphase.cos()],dim=1)
    with torch.no_grad():trunk.pos_embed.copy_(torch.cat([torch.zeros(1,768),positions],0)[None])
    if weights is not None:
        state=torch.load(Path(weights),map_location='cpu',weights_only=True)
        if 'model' not in state:raise ValueError('Expected official MAE pretraining checkpoint with model key')
        encoder={key:value for key,value in state['model'].items() if not key.startswith('decoder') and key!='mask_token'}
        trunk.load_state_dict(encoder,strict=True)
    return MAEBackbone(trunk)


class MAEToneModel(ReferenceToneModelV2):
    def __init__(self,backbone,cfg=None,*,pretrained_loaded=False,allow_uninitialized=False):
        super().__init__(backbone,cfg or MAEToneConfig(),pretrained_loaded=pretrained_loaded,allow_uninitialized=allow_uninitialized)

    @classmethod
    def from_pretrained(cls,weights=None,cfg=None,*,allow_uninitialized=False):
        backbone=load_mae_encoder(weights,allow_uninitialized=allow_uninitialized)
        return cls(backbone,cfg,pretrained_loaded=weights is not None,allow_uninitialized=allow_uninitialized)

    @classmethod
    def from_official(cls,*args,**kwargs):
        raise ValueError('Use from_pretrained with the official MAE checkpoint, not the DINOv2 factory')

    def set_stage(self,stage):
        if stage=='encoder':
            raise ValueError('Encoder-only stage requires a separately specified representation objective; use heads then joint')
        super().set_stage(stage)
        # Official MAE sin/cos embeddings remain fixed even in full fine-tuning.
        self.encoder.backbone.pos_embed.requires_grad_(False)
