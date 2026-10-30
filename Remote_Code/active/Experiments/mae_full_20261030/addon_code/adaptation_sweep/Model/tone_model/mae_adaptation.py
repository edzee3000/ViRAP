"""Controlled MAE adaptation variants; identical photographic heads."""
from .mae_tone import MAEToneConfig,MAEToneModel,load_mae_encoder
from .mae_hybrid_lora import LoRALinear,LoRAPatchProjection,MAEHybridLoRAModel


class MAEAllLoRAModel(MAEToneModel):
    def __init__(self,backbone,cfg=None,*,lora_rank=8,lora_alpha=16.,pretrained_loaded=False,allow_uninitialized=False):
        super().__init__(backbone,cfg or MAEToneConfig(),pretrained_loaded=pretrained_loaded,allow_uninitialized=allow_uninitialized)
        self.lora_targets=[]
        for index,block in enumerate(self.encoder.backbone.blocks):
            for parent_name,leaf in [('attn','qkv'),('attn','proj'),('mlp','fc1'),('mlp','fc2')]:
                parent=getattr(block,parent_name)
                setattr(parent,leaf,LoRALinear(getattr(parent,leaf),lora_rank,lora_alpha))
                self.lora_targets.append(f'blocks.{index}.{parent_name}.{leaf}')
        patch=self.encoder.backbone.trunk.patch_embed
        patch.proj=LoRAPatchProjection(patch.proj,lora_rank,lora_alpha)
        self.lora_targets.append('patch_embed.proj');self.set_stage('all_lora')

    @classmethod
    def from_pretrained(cls,weights=None,cfg=None,*,lora_rank=8,lora_alpha=16.,allow_uninitialized=False):
        return cls(load_mae_encoder(weights,allow_uninitialized=allow_uninitialized),cfg,lora_rank=lora_rank,lora_alpha=lora_alpha,
                   pretrained_loaded=weights is not None,allow_uninitialized=allow_uninitialized)

    def set_stage(self,stage):
        if stage=='heads':return super().set_stage(stage)
        if stage!='all_lora':raise ValueError(stage)
        super().set_stage('heads')
        for name,p in self.encoder.backbone.named_parameters():
            p.requires_grad_(name.endswith(('lora_A','lora_B')))
        self.stage=stage


def build_adaptation(weights,cfg,adaptation,lora_rank=8,lora_alpha=16.,allow_uninitialized=False):
    kwargs=dict(lora_rank=lora_rank,lora_alpha=lora_alpha,allow_uninitialized=allow_uninitialized)
    if adaptation=='all_lora':return MAEAllLoRAModel.from_pretrained(weights,cfg,**kwargs)
    if adaptation=='hybrid':return MAEHybridLoRAModel.from_pretrained(weights,cfg,**kwargs)
    if adaptation=='full':
        model=MAEToneModel.from_pretrained(weights,cfg,allow_uninitialized=allow_uninitialized)
        # Literal full adaptation includes CLS and positional embeddings.
        model.requires_grad_(True);model.stage='full';model.lora_targets=[]
        return model
    raise ValueError(adaptation)
