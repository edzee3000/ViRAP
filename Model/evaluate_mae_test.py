"""Evaluate a precommitted validation-selected checkpoint; never selects on test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--run',required=True)
    args=ap.parse_args();base=args.experiment;out=base/'test_eval_v1';folder=base/'runs'/args.run
    sys.path.insert(0,str(base/'addon_code/hybrid_lora/Model'))
    import torch
    from tone_model.data import load_triplets,TripletDataset
    from tone_model.mae_tone import MAEToneConfig,MAEToneModel
    from tone_model.mae_hybrid_lora import MAEHybridLoRAModel
    from train import guard_experiment_paths
    from train_mae_full import atomic_json,to_device
    from test_metrics import image_metrics
    guard_experiment_paths(args.root,base,out);torch.set_num_threads(1)
    torch.manual_seed(20261030);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((out/'protocol.json').read_text());entry=protocol['checkpoints'][args.run]
    config=json.loads((folder/'config.json').read_text());checkpoint=folder/'best.pt'
    if sha256(checkpoint)!=entry['sha256']:raise ValueError('Precommitted checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['step']!=entry['step']:raise ValueError('Checkpoint step mismatch')
    cfg=MAEToneConfig(**config['model']);weights=Path(config['weights'])
    if 'lora_rank' in config:
        model=MAEHybridLoRAModel.from_pretrained(weights,cfg,lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'])
    else:model=MAEToneModel.from_pretrained(weights,cfg)
    model.load_state_dict(state['model'],strict=True);del state
    model.requires_grad_(False).eval().cuda()
    records=load_triplets(args.root/'Dataset/Photo_Triplets_v1/triplets.json')
    for split,expected in config['data_fingerprints'].items():
        if TripletDataset(records,split,384,384,verify_files=False).fingerprint()!=expected:
            raise ValueError('Frozen data mapping changed')
    dataset=TripletDataset(records,'test',384,384)
    if len(dataset)!=400:raise ValueError('Expected exactly 400 frozen test pairs')
    rows=[];started=time.time()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample=dataset[index];item=to_device(sample,torch.device('cuda'))
            # Match validation's FP16 forward, compute all metrics in FP32.
            with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                prediction=model(item['content'],item['reference'])
            row=dataset.records[index]
            rows.append({'pair_id':sample['pair_id'],'content_key':row['content_key'],'reference_key':row['reference_key'],
                         'model':image_metrics(prediction,item['target']),
                         'identity':image_metrics(item['content'],item['target'])})
            if (index+1)%50==0:
                atomic_json(out/(args.run+'_status.json'),{'state':'running','count':index+1,'total':len(dataset)})
                print(args.run,index+1,flush=True)
    means={kind:{metric:sum(row[kind][metric] for row in rows)/len(rows) for metric in rows[0][kind]}
           for kind in ('model','identity')}
    result={'run':args.run,'checkpoint':entry,'test_fingerprint':dataset.fingerprint(),'count':len(rows),
            'mean_per_image':means,'elapsed_seconds':time.time()-started,'rows':rows,
            'note':'Generated targets; these metrics measure target agreement, not human aesthetic preference.'}
    atomic_json(out/(args.run+'.json'),result)
    atomic_json(out/(args.run+'_status.json'),{'state':'completed','count':len(rows),'total':len(rows)})
    print('completed',args.run,means,flush=True)


if __name__=='__main__':main()
