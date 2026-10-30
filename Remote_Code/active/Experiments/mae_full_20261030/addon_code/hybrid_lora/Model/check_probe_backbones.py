"""Strict public-checkpoint loading and GPU forward preflight, no optimization."""
import gc
import json
import os
from pathlib import Path
import torch
from prepare_encoder_experiment import load_backbone,save_json
from train import guard_experiment_paths

root=Path('/tmp/shihangz/photo-tone-transfer');cache=root/'Experiments/encoder_probe_20261030/cache'
os.chdir(root)
guard_experiment_paths(root,cache);cache.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
report={}
for name in ('dinov2_s','dinov3_s','mae_b','dinov2_b'):
    model,meta=load_backbone(name,root,cache)
    model=model.cuda()
    with torch.inference_mode():
        image=torch.rand(1,3,224,224,device='cuda')
        output=model.get_intermediate_layers(image,n=(3,7,11),return_class_token=True) if name.startswith('dinov2') else model.forward_features(image)
        meta['finite']=bool(torch.isfinite(output[0][0] if name.startswith('dinov2') else output).all())
    report[name]=meta;save_json(cache/'backbone_preflight.json',report)
    print(name,json.dumps(meta),flush=True)
    del model,output;gc.collect();torch.cuda.empty_cache()
