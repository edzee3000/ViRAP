"""MAE hybrid adaptation: full last four blocks + norm, LoRA on frozen matrices.

Prefix LayerNorm/bias/CLS/position vectors stay frozen; LoRA is applied to
matrix-valued linear/patch weights, not to one-dimensional normalization vectors.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .mae_tone import MAEToneConfig,MAEToneModel,load_mae_encoder


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16.):
        super().__init__()
        if not isinstance(base,nn.Linear) or rank<1:raise ValueError('Expected linear weight and positive rank')
        self.base=base;self.base.requires_grad_(False);self.rank=rank;self.scaling=alpha/rank
        self.in_features=base.in_features;self.out_features=base.out_features
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scaling


class LoRAPatchProjection(nn.Module):
    def __init__(self,base,rank=8,alpha=16.):
        super().__init__()
        if not isinstance(base,nn.Conv2d) or base.groups!=1 or base.padding_mode!='zeros':raise ValueError('Expected standard patch projection')
        self.base=base;self.base.requires_grad_(False);self.scaling=alpha/rank
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_channels,*base.kernel_size))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_channels,rank,1,1))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,x):
        low=F.conv2d(x,self.lora_A,None,self.base.stride,self.base.padding,self.base.dilation)
        return self.base(x)+F.conv2d(low,self.lora_B)*self.scaling


def inject_prefix_lora(backbone,rank=8,alpha=16.):
    targets=[]
    for index,block in enumerate(backbone.blocks[:8]):
        for parent_name,leaf in [('attn','qkv'),('attn','proj'),('mlp','fc1'),('mlp','fc2')]:
            parent=getattr(block,parent_name)
            setattr(parent,leaf,LoRALinear(getattr(parent,leaf),rank,alpha));targets.append(f'blocks.{index}.{parent_name}.{leaf}')
    patch=backbone.trunk.patch_embed
    patch.proj=LoRAPatchProjection(patch.proj,rank,alpha);targets.append('patch_embed.proj')
    return targets


class MAEHybridLoRAModel(MAEToneModel):
    def __init__(self,backbone,cfg=None,*,lora_rank=8,lora_alpha=16.,pretrained_loaded=False,allow_uninitialized=False):
        super().__init__(backbone,cfg or MAEToneConfig(),pretrained_loaded=pretrained_loaded,allow_uninitialized=allow_uninitialized)
        self.lora_rank=lora_rank;self.lora_alpha=lora_alpha
        self.lora_targets=inject_prefix_lora(self.encoder.backbone,lora_rank,lora_alpha)
        self.set_stage('hybrid')
    @classmethod
    def from_pretrained(cls,weights=None,cfg=None,*,lora_rank=8,lora_alpha=16.,allow_uninitialized=False):
        return cls(load_mae_encoder(weights,allow_uninitialized=allow_uninitialized),cfg,lora_rank=lora_rank,lora_alpha=lora_alpha,
                   pretrained_loaded=weights is not None,allow_uninitialized=allow_uninitialized)
    def set_stage(self,stage):
        if stage=='heads':return super().set_stage(stage)
        if stage!='hybrid':raise ValueError('Hybrid experiment uses heads only during construction, then hybrid adaptation')
        super().set_stage('joint')
        for module in self.encoder.backbone.modules():
            if isinstance(module,(LoRALinear,LoRAPatchProjection)):
                module.lora_A.requires_grad_(True);module.lora_B.requires_grad_(True)
        self.stage='hybrid'


def adaptation_counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'lora':sum(p.numel() for name,p in model.named_parameters() if name.endswith(('lora_A','lora_B'))),
            'targets':model.lora_targets}
