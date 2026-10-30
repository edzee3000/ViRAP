"""Independent checkpoint preview inference. Never updates models or training.

Runs on an otherwise idle compute GPU, uses fixed train/validation samples,
and publishes checkpoint-labelled assets. Existing training processes untouched.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--model-code',type=Path,required=True);ap.add_argument('--count',type=int,default=48);ap.add_argument('--poll',type=int,default=30)
    args=ap.parse_args();sys.path.insert(0,str(args.model_code))
    import numpy as np
    from PIL import Image
    import torch
    from train import guard_experiment_paths
    from train_mae_full import atomic_json,ValidationMemoryCache,to_device
    from tone_model.data import load_triplets,TripletDataset
    from tone_model.mae_tone import MAEToneConfig,MAEToneModel
    from tone_model.mae_hybrid_lora import MAEHybridLoRAModel
    from tone_model.mae_adaptation import build_adaptation
    from mae_gallery import fixed_selection,safe_run

    root=args.root.resolve();base=args.experiment.resolve();guard_experiment_paths(root,base)
    if hasattr(os,'nice'):os.nice(10)
    torch.set_num_threads(1);device=torch.device('cuda')
    records=load_triplets(root/'Dataset/Photo_Triplets_v1/triplets.json')
    guard_experiment_paths(*{row['_'+role+'_path'].parent for row in records for role in ('content','reference','target')})
    selection=fixed_selection(records,args.count)
    selection_hash=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    atomic_json(base/'gallery_selection.json',{'seed':20261030,'count_per_split':args.count,'selection':selection,
        'selection_sha256':selection_hash,'policy':'source-stratified, unique content per split; fixed across all runs; no quality cherry-picking'})
    data={split:ValidationMemoryCache(TripletDataset(records,split,384,384)) for split in selection}
    indices={split:{row['pair_id']:index for index,row in enumerate(data[split].dataset.records)} for split in selection}
    sources={row['pair_id']:row for row in records};assets=base/'gallery_assets';assets.mkdir(exist_ok=True)
    weights=root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'
    models={};seen={};shared_ready=set()
    def save_image(tensor,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        value=tensor.detach().float().clamp(0,1)
        if value.ndim==4:value=value[0]
        pixels=(value.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        temp=path.with_name(path.stem+'.tmp.png');Image.fromarray(pixels).save(temp,compress_level=3);temp.replace(path)
    def url(path):return '/gallery/'+path.relative_to(assets).as_posix()
    def emit(**values):atomic_json(base/'gallery_worker.json',{'pid':os.getpid(),'heartbeat_unix':time.time(),**values})
    @torch.inference_mode()
    def generate(folder,checkpoint,signature):
        run=folder.name;config=json.loads((folder/'config.json').read_text())
        if config.get('long_edge',384)!=384:raise ValueError('Gallery data protocol must match training long edge')
        expected=config.get('data_fingerprints',{})
        if any(expected.get(split)!=data[split].fingerprint() for split in data):raise ValueError('Gallery dataset differs from run')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);step=state['step']
        hybrid='lora_rank' in config
        adaptation=config.get('adaptation','hybrid' if hybrid else 'baseline')
        key=(adaptation,config['model']['local_rank'],config.get('lora_rank',0),config.get('lora_alpha',0))
        if key not in models:
            cfg=MAEToneConfig(**config['model'])
            if 'adaptation' in config:models[key]=build_adaptation(weights,cfg,adaptation,lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'])
            elif hybrid:models[key]=MAEHybridLoRAModel.from_pretrained(weights,cfg,lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'])
            else:models[key]=MAEToneModel.from_pretrained(weights,cfg)
            models[key].requires_grad_(False).eval()
        model=models[key].to(device);model.load_state_dict(state['model'],strict=True);del state
        version=f'step{step:07d}_{signature[0]}';version_root=assets/run/version
        previous={}
        try:previous=json.loads((folder/'gallery.json').read_text())
        except (OSError,ValueError):pass
        publish_partial=not previous.get('complete',False)
        manifest={'run':run,'version':version,'checkpoint_step':step,'checkpoint_kind':checkpoint.stem,
            'selection_sha256':selection_hash,'requested':{split:len(ids) for split,ids in selection.items()},
            'validation':[],'train':[],'complete':False,'updated_unix':time.time()}
        for position in range(max(map(len,selection.values()))):
            for split,ids in selection.items():
                if position>=len(ids):continue
                pair_id=ids[position];sample=data[split][indices[split][pair_id]];row=sources[pair_id]
                slug=hashlib.sha256(pair_id.encode()).hexdigest()[:24]
                common=assets/'shared'/selection_hash[:12]/split
                common_paths={role:common/(slug+'_'+role+'.png') for role in ('content','reference','target')}
                if (split,pair_id) not in shared_ready:
                    for role,path in common_paths.items():
                        if not path.exists():save_image(sample[role],path)
                    shared_ready.add((split,pair_id))
                item=to_device(sample,device)
                with torch.autocast('cuda',dtype=torch.float16,enabled=config.get('amp',False)):
                    output=model(item['content'],item['reference'])
                output_path=version_root/split/(slug+'_output.png');save_image(output,output_path)
                manifest[split].append({'pair_id':pair_id,'content_source':row['content_key'].split('/')[0],
                    'reference_source':row['reference_key'].split('/')[0],
                    **{role+'_url':url(path) for role,path in common_paths.items()},'output_url':url(output_path)})
            if (position+1)%6==0:
                manifest['updated_unix']=time.time()
                if publish_partial:atomic_json(folder/'gallery.json',manifest)
                emit(state='generating',run=run,checkpoint_step=step,ready={s:len(manifest[s]) for s in selection})
        manifest.update(complete=True,updated_unix=time.time());atomic_json(folder/'gallery.json',manifest)
        model.cpu();torch.cuda.empty_cache();gc.collect()
        # Keep the current and previous complete version for images open in a
        # browser. Only generated assets under this run's gallery root are removed.
        versions=sorted((assets/run).iterdir(),key=lambda path:path.stat().st_mtime,reverse=True)
        for old in versions[2:]:
            if old.is_dir() and old.resolve().is_relative_to((assets/run).resolve()) and old.name.startswith('step'):
                shutil.rmtree(old)
        seen[run]=signature
        print('gallery_complete',run,step,{split:len(manifest[split]) for split in selection},flush=True)
    while True:
        folders=sorted((base/'runs').glob('*'),key=lambda p:(0 if p.name=='M0_s17' else 1 if 'LoRA' in p.name else 2 if p.name=='M1_s17' else 3,p.name))
        for folder in folders:
            if not (folder/'config.json').exists():continue
            try:safe_run(base,folder.name)
            except ValueError:continue
            checkpoint=folder/'latest.pt'
            if not checkpoint.exists():checkpoint=folder/'best.pt'
            if not checkpoint.exists():continue
            stat=checkpoint.stat();signature=(stat.st_mtime_ns,stat.st_size,checkpoint.stem)
            if seen.get(folder.name)==signature:continue
            try:generate(folder,checkpoint,signature)
            except Exception as error:
                emit(state='error',run=folder.name,error=type(error).__name__+': '+str(error));print('gallery_error',folder.name,repr(error),flush=True)
                for model in models.values():model.cpu()
                torch.cuda.empty_cache();gc.collect()
        emit(state='waiting',completed_runs=len(seen));time.sleep(args.poll)


if __name__=='__main__':main()
