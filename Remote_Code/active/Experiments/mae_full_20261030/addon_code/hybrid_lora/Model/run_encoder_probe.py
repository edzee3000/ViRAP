"""Train/evaluate frozen-encoder probes on a shared, deterministic pair schedule."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from train import guard_experiment_paths,seed_everything
from prepare_encoder_experiment import save_json
from tone_model.encoder_probe import EncoderProbe,GROUPS,probe_loss,image_metrics


class Banks:
    def __init__(self,cache,group):
        self.meta=json.loads((cache/'images.json').read_text());self.rows=self.meta['rows']
        self.photos=np.load(cache/'photos.npy',mmap_mode='r');self.render=np.load(cache/'render.npy',mmap_mode='r')
        self.targets=np.load(cache/'targets.npy',mmap_mode='r');name=GROUPS[group]['backbone']
        self.features=None;self.feature_meta=None
        if name:
            self.feature_meta=json.loads((cache/(name+'.json')).read_text())
            if self.feature_meta['image_fingerprint']!=self.meta['fingerprint']:raise ValueError('Feature fingerprint mismatch')
            self.features=np.load(cache/(name+'.npy'),mmap_mode='r')
        self.dim=self.feature_meta['feature_dim'] if name else 384
        self.splits={split:np.array([i for i,r in enumerate(self.rows) if r['split']==split],dtype=np.int64)
                     for split in ('train','validation','test')}

    def batch(self,indices,device,reference_indices=None):
        rows=[self.rows[int(i)] for i in indices]
        ci=np.array([r['content_index'] for r in rows]);ri=np.array([r['reference_index'] for r in rows])
        if reference_indices is not None:ri=np.array([self.rows[int(i)]['reference_index'] for i in reference_indices])
        def tensor(value,image=True):
            result=torch.from_numpy(np.array(value)).to(device)
            return result.float()/255 if image else result.float()
        batch={'content':tensor(self.photos[ci,0]),'reference':tensor(self.photos[ri,1]),
               'render_content':tensor(self.render[ci]),'content_features':None,'reference_features':None}
        if self.features is not None:
            batch['content_features']=tensor(self.features[ci,0],False);batch['reference_features']=tensor(self.features[ri,1],False)
        return batch,tensor(self.targets[indices])


def schedule(indices,epochs,seed,batch_size):
    rng=np.random.default_rng(seed+1000)
    for epoch in range(epochs):
        order=rng.permutation(indices)
        for start in range(0,len(order),batch_size):yield epoch,order[start:start+batch_size]


@torch.no_grad()
def evaluate(model,banks,split,device,batch_size=8,limit=0,export=None,shuffle_reference=False):
    model.eval();indices=banks.splits[split]
    if limit:indices=indices[:limit]
    rows=[]
    for start in range(0,len(indices),batch_size):
        group=indices[start:start+batch_size]
        references=np.roll(indices,1)[start:start+batch_size] if shuffle_reference else None
        batch,target=banks.batch(group,device,references)
        prediction,_=model(**batch);metrics=image_metrics(prediction,target)
        baseline=image_metrics(batch['render_content'],target)
        for j,index in enumerate(group):
            row=banks.rows[int(index)]
            rows.append({'pair_id':row['pair_id'],'content_key':row['content_key'],
                         **{k:float(v[j]) for k,v in metrics.items()},
                         **{'identity_'+k:float(v[j]) for k,v in baseline.items()}})
        if export is not None and start<24:
            export.mkdir(parents=True,exist_ok=True)
            for j,index in enumerate(group[:max(0,24-start)]):
                reference=torch.nn.functional.interpolate(batch['reference'][j:j+1],(128,128),mode='bilinear',align_corners=False)[0]
                strip=torch.cat([batch['render_content'][j],reference,target[j],prediction[j]],dim=2)
                pixels=(strip.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                Image.fromarray(pixels).save(export/(banks.rows[int(index)]['pair_id']+'.png'))
    summary={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ('pair_id','content_key')}
    summary['count']=len(rows);model.train();return summary,rows


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--group',choices=list(GROUPS),required=True);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--device',default='cuda')
    ap.add_argument('--max-steps',type=int,default=0);ap.add_argument('--validation-every',type=int,default=2000)
    ap.add_argument('--validation-limit',type=int,default=0);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--resume',action='store_true');args=ap.parse_args(argv)
    guard_experiment_paths(args.cache,args.output);torch.set_num_threads(4);seed_everything(args.seed)
    if min(args.epochs,args.batch_size,args.validation_every)<=0:raise ValueError('Positive training parameters required')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():raise FileExistsError('Run already completed')
    if (args.output/'config.json').exists() and not args.resume:raise FileExistsError('Use new output or --resume')
    banks=Banks(args.cache,args.group);model=EncoderProbe(args.group,banks.dim).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    total_steps=math.ceil(len(banks.splits['train'])/args.batch_size)*args.epochs
    if args.max_steps:total_steps=min(total_steps,args.max_steps)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(total_steps=total_steps,train_count=len(banks.splits['train']),validation_count=len(banks.splits['validation']),
                  fingerprint=banks.meta['fingerprint'],head_parameters=sum(p.numel() for p in model.parameters()),
                  backbone_parameters=banks.feature_meta['parameters'] if banks.feature_meta else 0,
                  feature_provenance=banks.feature_meta,group_spec=GROUPS[args.group],
                  stage='frozen representation probe; identical global-only renderer, no spatial DINO bypass',
                  evaluation='128x128 full-image square warp; not final native-resolution quality',
                  checkpoint_selection='lowest validation L1; test only once after selection; pilot never evaluates test')
    model_root=Path(__file__).resolve().parent
    config['code_sha256']={name:hashlib.sha256((model_root/name).read_bytes()).hexdigest()
                           for name in ('run_encoder_probe.py','tone_model/encoder_probe.py','tone_model/v2.py','tone_model/model.py')}
    config['packages']={name:importlib.metadata.version(name) for name in ('torch','numpy','pillow')}
    config['config_hash']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    step=0;best=float('inf');best_step=0;last_validation=0
    if args.resume:
        state=torch.load(args.output/'latest.pt',map_location='cpu',weights_only=False)
        old=json.loads((args.output/'config.json').read_text())
        mutable={'resume','config_hash'}
        if any(config[k]!=old.get(k) for k in config if k not in mutable):raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        step=state['step'];best=state['best'];best_step=state['best_step'];last_validation=state['last_validation']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    else:save_json(args.output/'config.json',config)
    def checkpoint():
        temp=args.output/'latest.pt.tmp'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'best':best,'best_step':best_step,
                    'last_validation':last_validation,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temp)
        temp.replace(args.output/'latest.pt')
    model.train();started=time.monotonic();schedule_hash=hashlib.sha256()
    with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as log:
        for position,(epoch,indices) in enumerate(schedule(banks.splits['train'],args.epochs,args.seed,args.batch_size),1):
            if position>total_steps:break
            schedule_hash.update(indices.tobytes())
            if position<=step:continue
            lr=args.lr*(.1+.9*.5*(1+math.cos(math.pi*(position-1)/max(1,total_steps))))
            for group in optimizer.param_groups:group['lr']=lr
            batch,target=banks.batch(indices,args.device);optimizer.zero_grad(set_to_none=True)
            prediction,aux=model(**batch);loss=probe_loss(prediction,target,aux)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step=position
            event={'step':step,'epoch':epoch+1,'loss':float(loss.detach()),'lr':lr,'elapsed_seconds':round(time.monotonic()-started,2)}
            if step%args.validation_every==0 or step==total_steps:
                summary,_=evaluate(model,banks,'validation',args.device,args.batch_size,args.validation_limit)
                event['validation']=summary;last_validation=step
                if summary['l1']<best:
                    best=summary['l1'];best_step=step;torch.save(model.state_dict(),args.output/'best.pt')
                checkpoint()
            if step==1 or step%50==0 or 'validation' in event:
                line=json.dumps(event);print(line,flush=True);log.write(line+'\n');log.flush()
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=args.device,weights_only=True))
    val,val_rows=evaluate(model,banks,'validation',args.device,args.batch_size,args.validation_limit,args.output/'validation_examples')
    shuffled,_=evaluate(model,banks,'validation',args.device,args.batch_size,args.validation_limit,shuffle_reference=True)
    save_json(args.output/'validation_per_pair.json',val_rows)
    result={'group':args.group,'seed':args.seed,'pilot':args.pilot,'best_step':best_step,'steps':step,
            'train_schedule_sha256':schedule_hash.hexdigest(),'validation':val,'validation_shuffled_reference':shuffled,
            'elapsed_seconds':time.monotonic()-started,'head_parameters':config['head_parameters'],
            'backbone_parameters':config['backbone_parameters'],'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None}
    if not args.pilot:
        test,test_rows=evaluate(model,banks,'test',args.device,args.batch_size,export=args.output/'test_examples')
        result['test']=test;save_json(args.output/'test_per_pair.json',test_rows)
    save_json(args.output/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
