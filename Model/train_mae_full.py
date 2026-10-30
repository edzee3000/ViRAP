"""Full MAE-tone training on aspect-preserved triplets; resumable two-stage run.

The held-out test split is never evaluated here. All live IO must be compute-local.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Sampler

from train import guard_experiment_paths,seed_everything,rng_state,restore_rng
from tone_model.data import load_triplets,TripletDataset,read_rgb,shrink,image_tensor
from tone_model.mae_tone import MAEToneConfig,MAEToneModel
from tone_model.losses import LossConfig,ToneLoss


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


class EpochOrder(Sampler):
    def __init__(self,size,epochs,seed,start=0):
        self.order=np.concatenate([np.random.default_rng(seed+1000+epoch).permutation(size) for epoch in range(epochs)])
        self.start=start
    def __iter__(self):return iter(self.order[self.start:].tolist())
    def __len__(self):return len(self.order)-self.start


class ValidationMemoryCache:
    """Cache the fixed validation set in CPU RAM; values/geometry do not change.

    Only 400 validation triplets are retained, never the full training corpus.
    The node has >300 GiB available RAM at deployment; eight caches fit easily.
    """
    def __init__(self,dataset):self.dataset=dataset;self.items={}
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        if index not in self.items:self.items[index]=self.dataset[index]
        return self.items[index]
    def fingerprint(self):return self.dataset.fingerprint()


class SourceMemoryTriplets(TripletDataset):
    """Cache deterministic source thumbnails per worker, never generated targets.

    Source photographs repeat across triplets. RGB PIL thumbnails consume one
    byte/channel; return values remain exactly the original loader's FP32 tensors.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.content_memory={};self.reference_memory={}
    def __getitem__(self,index):
        row=self.records[index];edge=row.get('generation',{}).get('max_image_edge')
        if not edge:return super().__getitem__(index)
        target=read_rgb(row['_target_path']);native_size=target.size
        target=shrink(target,self.long_edge)
        content_key=(str(row['_content_path']),edge,native_size,target.size)
        if content_key not in self.content_memory:
            content=read_rgb(row['_content_path']);content.thumbnail((edge,edge),Image.Resampling.LANCZOS)
            if content.size!=native_size:raise ValueError('Generated target raster mismatch '+row['pair_id'])
            self.content_memory[content_key]=content.resize(target.size,Image.Resampling.LANCZOS)
        reference_key=(str(row['_reference_path']),self.reference_edge)
        if reference_key not in self.reference_memory:
            self.reference_memory[reference_key]=shrink(read_rgb(row['_reference_path']),self.reference_edge)
        return {'content':image_tensor(self.content_memory[content_key]),'reference':image_tensor(self.reference_memory[reference_key]),
                'target':image_tensor(target),'pair_id':row['pair_id']}


def singleton(items):return items[0]


def init_worker(worker_id):torch.set_num_threads(1)


def to_device(sample,device):
    return {role:sample[role].unsqueeze(0).to(device,non_blocking=True) for role in ('content','reference','target')}


def optimizer_for(model,stage):
    backbone=[];heads=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(backbone if name.startswith('encoder.backbone.') else heads).append(p)
    groups=[{'params':heads,'lr':2e-4 if stage=='heads' else 5e-5,'initial_lr':2e-4 if stage=='heads' else 5e-5}]
    if backbone:groups.append({'params':backbone,'lr':1e-5,'initial_lr':1e-5})
    return torch.optim.AdamW(groups,weight_decay=.01)


def learning_rate(optimizer,stage_step,stage_steps):
    warm=min(100,max(1,stage_steps//10));factor=min(1.,(stage_step+1)/warm)
    progress=max(0.,(stage_step-warm)/max(1,stage_steps-warm))
    factor*=.1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor


@torch.no_grad()
def validation(model,criterion,dataset,device,amp,limit=0,preview=None):
    model.eval();totals={};count=min(limit or len(dataset),len(dataset));images=[]
    for index in range(count):
        item=to_device(dataset[index],device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            prediction,aux=model(item['content'],item['reference'],True)
            losses=criterion(prediction,item['target'],item['content'],aux)
        for key,value in losses.items():totals[key]=totals.get(key,0.)+float(value)
        mse=(prediction.float()-item['target']).square().mean().clamp_min(1e-12)
        totals['psnr']=totals.get('psnr',0.)+float(-10*mse.log10())
        if preview is not None and index<3:
            def tile(tensor):
                rgb=(tensor[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                image=Image.fromarray(rgb);image.thumbnail((256,192),Image.Resampling.LANCZOS)
                box=Image.new('RGB',(256,192),(24,29,38));box.paste(image,((256-image.width)//2,(192-image.height)//2));return box
            strip=Image.new('RGB',(1024,192))
            for j,tensor in enumerate([item['content'],item['reference'],item['target'],prediction]):strip.paste(tile(tensor),(j*256,0))
            images.append(strip)
    if preview is not None:
        canvas=Image.new('RGB',(1024,len(images)*192))
        for i,image in enumerate(images):canvas.paste(image,(0,i*192))
        preview.parent.mkdir(parents=True,exist_ok=True);temp=preview.with_suffix('.tmp.png');canvas.save(temp);temp.replace(preview)
    model.train();return {key:value/count for key,value in totals.items()}


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--local-rank',type=int,choices=[2,4],default=4)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--heads-epochs',type=int,default=2);ap.add_argument('--joint-epochs',type=int,default=2)
    ap.add_argument('--long-edge',type=int,default=384);ap.add_argument('--accumulate',type=int,default=8)
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--validate-every',type=int,default=500)
    ap.add_argument('--save-every',type=int,default=500);ap.add_argument('--log-every',type=int,default=20)
    ap.add_argument('--validation-limit',type=int,default=0);ap.add_argument('--device',default='cuda')
    ap.add_argument('--amp',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-updates',type=int,default=0);ap.add_argument('--heads-updates',type=int,default=0)
    ap.add_argument('--stop-after-updates',type=int,default=0,help='Engineering resume check only; writes a checkpoint and exits before completion')
    ap.add_argument('--resume-allow-trainer-update',action='store_true',help='Record a controlled trainer-only runtime fix; model/data/config must remain identical')
    args=ap.parse_args(argv);out=args.output.resolve();root=args.root.resolve()
    guard_experiment_paths(root,out,args.weights);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Run completed')
    if (out/'config.json').exists() and not args.resume:raise FileExistsError('Use new output or --resume')
    torch.set_num_threads(2);seed_everything(args.seed);device=torch.device(args.device)
    if args.amp and device.type!='cuda':raise ValueError('FP16 AMP is only enabled on CUDA')
    if min(args.accumulate,args.heads_epochs,args.joint_epochs,args.long_edge,args.log_every,args.save_every,args.validate_every)<1:raise ValueError('Positive settings required')
    records=load_triplets(root/'Dataset/Photo_Triplets_v1/triplets.json')
    guard_experiment_paths(*{r['_'+role+'_path'].parent for r in records for role in ('content','reference','target')})
    train=SourceMemoryTriplets(records,'train',args.long_edge,384)
    valid=ValidationMemoryCache(TripletDataset(records,'validation',args.long_edge,384))
    fingerprints={'train':train.fingerprint(),'validation':valid.fingerprint()}
    n=len(train);stage_micro=n*args.heads_epochs
    heads_steps=math.ceil(stage_micro/args.accumulate)
    joint_steps=math.ceil(n*args.joint_epochs/args.accumulate);total=heads_steps+joint_steps
    if args.max_updates:
        total=args.max_updates;heads_steps=args.heads_updates or max(1,total//2);joint_steps=total-heads_steps
        if joint_steps<1:raise ValueError('Smoke check must exercise both stages')
    cfg=MAEToneConfig(local_rank=args.local_rank)
    loss_cfg=LossConfig(style_invariance=0.,style_sensitivity=0.)
    code_dir=Path(__file__).resolve().parent
    code_hash={str(p.relative_to(code_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((code_dir/'tone_model').glob('*.py'))]}
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(model=asdict(cfg),loss=asdict(loss_cfg),data_fingerprints=fingerprints,code_sha256=code_hash,
                  train_count=n,validation_count=len(valid),heads_steps=heads_steps,total_steps=total,
                  effective_batch=args.accumulate,geometry='Aspect-preserved content/target at max edge; separate reference. No square output warping.',
                  note='Validation only; test never evaluated. Generated targets are not manually curated.')
    # JSON stores the feature-layer tuple as a list; compare canonical values
    # on resume rather than rejecting the identical architecture on tuple/list.
    config=json.loads(json.dumps(config))
    model=MAEToneModel.from_pretrained(args.weights,cfg).to(device)
    criterion=ToneLoss(loss_cfg).to(device);sampler_rng=torch.Generator().manual_seed(args.seed+3000)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp,init_scale=1024.)
    step=0;consumed=0;elapsed_previous=0.;best=float('inf');best_step=0;stage='heads';saved_validation=None;saved_validation_step=None
    model.set_stage(stage);optimizer=optimizer_for(model,stage)
    if args.resume:
        saved=json.loads((out/'config.json').read_text());mutable={'resume','stop_after_updates','resume_allow_trainer_update'}
        if args.resume_allow_trainer_update:
            old_hash=saved['code_sha256'];new_hash=config['code_sha256']
            changed={key for key in set(old_hash)|set(new_hash) if old_hash.get(key)!=new_hash.get(key)}
            if not changed.issubset({'train_mae_full.py'}):raise ValueError('Only a trainer-only runtime fix is allowed; model code changed')
            mutable.add('code_sha256')
            mutable.add('workers')
        if any(config[k]!=saved.get(k) for k in config if k not in mutable):raise ValueError('Resume configuration/code/data changed')
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True);stage=state['stage'];model.set_stage(stage)
        optimizer=optimizer_for(model,stage);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step=state['step'];consumed=state['consumed'];best=state['best'];best_step=state['best_step'];elapsed_previous=state['elapsed_seconds']
        saved_validation=state.get('last_validation');saved_validation_step=state.get('last_validation_step')
        restore_rng(state['rng'],sampler_rng)
        if args.resume_allow_trainer_update and saved['code_sha256']!=config['code_sha256']:
            history_path=out/'trainer_update_history.json'
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            history.append({'step':step,'time_unix':time.time(),'old_code_sha256':saved['code_sha256'],'new_code_sha256':config['code_sha256'],
                            'old_workers':saved.get('workers'),'new_workers':config.get('workers'),
                            'reason':'Runtime recovery and lossless RAM thumbnail/validation caches; finite-gradient math/model/data unchanged'})
            atomic_json(history_path,history);atomic_json(out/'config.json',config)
    else:atomic_json(out/'config.json',config)
    status={'run':out.name,'pid':os.getpid(),'node':socket.gethostname(),'state':'running','stage':stage,'step':step,'total_steps':total,
            'heads_steps':heads_steps,'seed':args.seed,'local_rank':args.local_rank,'consumed':consumed,'train_count':n,'best_valid':None,'best_step':0}
    if saved_validation is not None:status.update(validation=saved_validation,validation_step=saved_validation_step,preview='preview/latest.png')
    started=time.monotonic()
    def heartbeat(state='running',**extra):
        status.update(state=state,stage=stage,step=step,consumed=consumed,elapsed_seconds=elapsed_previous+time.monotonic()-started,
                      best_valid=best if math.isfinite(best) else None,best_step=best_step,heartbeat_unix=time.time(),**extra)
        atomic_json(out/'status.json',status)
    def checkpoint():
        guard_experiment_paths(out)
        payload={'format':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                 'step':step,'stage':stage,'consumed':consumed,'best':best,'best_step':best_step,
                 'last_validation':status.get('validation'),'last_validation_step':status.get('validation_step'),
                 'elapsed_seconds':elapsed_previous+time.monotonic()-started,'rng':rng_state(sampler_rng),'config':config}
        temp=out/'latest.pt.tmp';torch.save(payload,temp);os.replace(temp,out/'latest.pt')
    order=EpochOrder(n,args.heads_epochs+args.joint_epochs+1,args.seed,consumed)
    loader=DataLoader(train,batch_size=1,sampler=order,num_workers=args.workers,collate_fn=singleton,
                      worker_init_fn=init_worker,pin_memory=device.type=='cuda',persistent_workers=args.workers>0,generator=sampler_rng)
    iterator=iter(loader);model.train();log=(out/'metrics.jsonl').open('a',encoding='utf-8')
    def event(value):
        value.update(step=step,stage=stage,elapsed_seconds=elapsed_previous+time.monotonic()-started,time_unix=time.time())
        line=json.dumps(value);log.write(line+'\n');log.flush();print(line,flush=True)
    def check_validation():
        nonlocal best,best_step
        heartbeat('validating')
        values=validation(model,criterion,valid,device,args.amp,args.validation_limit,out/'preview/latest.png')
        if not all(math.isfinite(x) for x in values.values()):raise FloatingPointError('Nonfinite validation')
        if values['reconstruction']<best:
            best=values['reconstruction'];best_step=step
            temp=out/'best.pt.tmp';torch.save({'model':model.state_dict(),'model_config':asdict(cfg),'step':step,'stage':stage,'validation':values},temp);temp.replace(out/'best.pt')
        event({'validation':values});heartbeat('running',validation=values,validation_step=step,preview='preview/latest.png')
        return values
    heartbeat()
    if args.resume:event({'resume':True,'from_checkpoint':step})
    try:
        if step==0:check_validation()
        running={};running_count=0
        while step<total:
            desired='heads' if step<heads_steps else 'joint'
            if desired!=stage:
                stage=desired;model.set_stage(stage);optimizer=optimizer_for(model,stage)
                event({'stage_transition':True});heartbeat()
            stage_start=0 if stage=='heads' else heads_steps
            learning_rate(optimizer,step-stage_start,heads_steps if stage=='heads' else joint_steps)
            # Preserve epoch coverage at a stage boundary even if the count is
            # not divisible by effective batch. Production 2 epochs is divisible.
            micro=args.accumulate
            if not args.max_updates and step==heads_steps-1:micro=stage_micro-consumed
            if not args.max_updates and step==total-1:micro=n*(args.heads_epochs+args.joint_epochs)-consumed
            optimizer.zero_grad(set_to_none=True);aggregate={}
            for _ in range(micro):
                item=to_device(next(iterator),device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp):
                    prediction,aux=model(item['content'],item['reference'],True)
                    losses=criterion(prediction,item['target'],item['content'],aux)
                if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite train loss')
                scaler.scale(losses['total']/micro).backward()
                for key,value in losses.items():aggregate[key]=aggregate.get(key,0.)+float(value.detach())/micro
                consumed+=1
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=not args.amp)
            old_scale=scaler.get_scale()
            # GradScaler must be allowed to skip an overflowing update and
            # reduce its scale; aborting at clip_grad_norm defeats that recovery.
            scaler.step(optimizer);scaler.update();step+=1
            skipped=scaler.get_scale()<old_scale
            if skipped:event({'amp_skipped_update':True,'amp_scale':scaler.get_scale()})
            for key,value in aggregate.items():running[key]=running.get(key,0.)+value
            running_count+=1
            if step==1 or step%args.log_every==0 or step in (heads_steps,total):
                event({'train':{k:v/running_count for k,v in running.items()},'lr_head':optimizer.param_groups[0]['lr'],
                       'gradient_norm':float(norm) if torch.isfinite(norm) else None,'consumed':consumed,'epoch':consumed/n,
                       'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None})
                running={};running_count=0;heartbeat()
            validate=step%args.validate_every==0 or step in (min(50,heads_steps),heads_steps,total)
            if validate:check_validation()
            if validate or step%args.save_every==0:checkpoint()
            if args.stop_after_updates and step>=args.stop_after_updates:
                checkpoint();heartbeat('resume_check_stopped');return
        if status.get('validation_step')!=step:check_validation();checkpoint()
        final=status['validation'];heartbeat('completed')
        atomic_json(out/'complete.json',{'step':step,'best_step':best_step,'best_valid_l1':best,'last_validation':final,
                                       'elapsed_seconds':elapsed_previous+time.monotonic()-started,'test_evaluated':False})
    except BaseException as error:
        heartbeat('failed',error=type(error).__name__+': '+str(error));raise
    finally:log.close()


if __name__=='__main__':main()
