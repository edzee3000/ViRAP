"""Independent native-batch MAE hybrid LoRA experiment; baseline runs untouched."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
from torch.utils.data import DataLoader
from train import guard_experiment_paths,seed_everything,rng_state,restore_rng
from train_mae_full import atomic_json,SourceMemoryTriplets,ValidationMemoryCache,init_worker,learning_rate,validation
from tone_model.data import load_triplets,TripletDataset
from tone_model.mae_tone import MAEToneConfig
from tone_model.mae_hybrid_lora import adaptation_counts
from tone_model.mae_adaptation import build_adaptation
from tone_model.losses import ToneLoss,LossConfig
from hybrid_batching import ExactShapeBatches,collate_native,gpu_batch,target_shape


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--microbatch-size',type=int,default=16)
    ap.add_argument('--adaptation',choices=['all_lora','full','hybrid'],required=True)
    ap.add_argument('--lora-rank',type=int,default=8);ap.add_argument('--lora-alpha',type=float,default=16.)
    ap.add_argument('--long-edge',type=int,default=384);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--validate-every',type=int,default=500);ap.add_argument('--log-every',type=int,default=10)
    ap.add_argument('--amp',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args(argv);root=args.root.resolve();out=args.output.resolve();guard_experiment_paths(root,out,args.weights)
    if min(args.batch_size,args.microbatch_size,args.epochs,args.lora_rank,args.validate_every,args.log_every)<1:raise ValueError('Positive dimensions required')
    if args.microbatch_size>args.batch_size:raise ValueError('Microbatch cannot exceed nominal batch')
    if (out/'complete.json').exists():raise FileExistsError('Completed run')
    if (out/'config.json').exists() and not args.resume:raise FileExistsError('Use new output or --resume')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);seed_everything(args.seed);device=torch.device('cuda')
    records=load_triplets(root/'Dataset/Photo_Triplets_v1/triplets.json')
    guard_experiment_paths(*{r['_'+role+'_path'].parent for r in records for role in ('content','reference','target')})
    train=SourceMemoryTriplets(records,'train',args.long_edge,384)
    valid=ValidationMemoryCache(TripletDataset(records,'validation',args.long_edge,384))
    sampler=ExactShapeBatches(train.records,args.long_edge,args.batch_size,args.epochs,args.seed)
    if args.preflight:
        full=[batch for batch in sampler.batches if len(batch)==args.batch_size]
        if not full:raise ValueError('No full-sized native batch for memory check')
        sampler.batches=sorted(full,key=lambda batch:math.prod(target_shape(train.records[batch[0]],args.long_edge)),reverse=True)[:3]
    plan_hash=hashlib.sha256(json.dumps(sampler.batches,separators=(',',':')).encode()).hexdigest();total=len(sampler.batches)
    model_cfg=MAEToneConfig(local_rank=4);loss_cfg=LossConfig(style_invariance=0.,style_sensitivity=0.)
    model=build_adaptation(args.weights,model_cfg,args.adaptation,lora_rank=args.lora_rank,lora_alpha=args.lora_alpha).to(device)
    groups={'heads':[],'lora':[],'full_backbone':[]}
    for name,p in model.named_parameters():
        if p.requires_grad:groups['lora' if name.endswith(('lora_A','lora_B')) else 'full_backbone' if name.startswith('encoder.backbone.') else 'heads'].append(p)
    optimizer=torch.optim.AdamW([{'params':groups[name],'lr':lr,'initial_lr':lr,'name':name}
                                for name,lr in [('heads',2e-4),('lora',1e-4),('full_backbone',1e-5)]],weight_decay=.01)
    criterion=ToneLoss(loss_cfg).to(device);scaler=torch.amp.GradScaler('cuda',enabled=args.amp,init_scale=1024.)
    generator=torch.Generator().manual_seed(args.seed+3000)
    config={key:str(value.resolve()) if isinstance(value,Path) else value for key,value in vars(args).items()}
    code=Path(__file__).resolve().parent
    files=[Path(__file__),code/'train_mae_full.py',code/'hybrid_batching.py',*sorted((code/'tone_model').glob('*.py'))]
    config.update(model=asdict(model_cfg),loss=asdict(loss_cfg),counts=adaptation_counts(model),local_rank=4,
        total_steps=total,heads_steps=0,stage=args.adaptation,effective_batch=args.batch_size,physical_batch_size=args.microbatch_size,
        accumulate=math.ceil(args.batch_size/args.microbatch_size),
        train_count=len(train),validation_count=len(valid),mean_actual_batch=sum(map(len,sampler.batches))/total,
        full_batch_fraction=sum(len(batch)==args.batch_size for batch in sampler.batches)/total,
        batch_plan_sha256=plan_hash,data_fingerprints={'train':train.fingerprint(),'validation':valid.fingerprint()},
        code_sha256={str(path.relative_to(code)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        note='Exact native content-size buckets, reference-only 224 preprocessing. Bucket tails retained; all four epochs use the configured adaptation, no heads-only warmup. Test never evaluated.')
    config['validation_schedule']='every 8000 consumed training pairs, step 0/25 and final'
    config=json.loads(json.dumps(config));step=0;consumed=0;elapsed_previous=0.;best=float('inf');best_step=0;saved_valid=None
    if args.resume:
        saved=json.loads((out/'config.json').read_text())
        if any(config[key]!=saved.get(key) for key in config if key!='resume'):raise ValueError('Hybrid resume config/model/data changed')
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step=state['step'];consumed=state['consumed'];elapsed_previous=state['elapsed_seconds'];best=state['best'];best_step=state['best_step']
        saved_valid=state.get('last_validation');restore_rng(state['rng'],generator)
    else:atomic_json(out/'config.json',config)
    status={'run':out.name,'pid':os.getpid(),'node':socket.gethostname(),'stage':args.adaptation,'heads_steps':0,'total_steps':total,
            'local_rank':4,'lora_rank':args.lora_rank if args.adaptation!='full' else 0,'seed':args.seed,'train_count':len(train),
            'effective_batch':args.batch_size,'physical_batch_size':args.microbatch_size}
    started=time.monotonic();log=(out/'metrics.jsonl').open('a',encoding='utf-8')
    def heartbeat(state='running',**extra):
        status.update(state=state,step=step,consumed=consumed,elapsed_seconds=elapsed_previous+time.monotonic()-started,
            best_valid=best if math.isfinite(best) else None,best_step=best_step,heartbeat_unix=time.time(),**extra);atomic_json(out/'status.json',status)
    def event(value):
        value.update(step=step,stage=args.adaptation,elapsed_seconds=elapsed_previous+time.monotonic()-started,time_unix=time.time())
        line=json.dumps(value);print(line,flush=True);log.write(line+'\n');log.flush()
    def checkpoint():
        guard_experiment_paths(out)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,
            'consumed':consumed,'elapsed_seconds':elapsed_previous+time.monotonic()-started,'best':best,'best_step':best_step,
            'rng':rng_state(generator),'config':config,'last_validation':status.get('validation')}
        temp=out/'latest.pt.tmp';torch.save(payload,temp);temp.replace(out/'latest.pt')
    def validate():
        nonlocal best,best_step
        heartbeat('validating')
        values=validation(model,criterion,valid,device,args.amp,8 if args.preflight else 0,out/'preview/latest.png')
        if not all(math.isfinite(value) for value in values.values()):raise FloatingPointError('Nonfinite validation')
        if values['reconstruction']<best:
            best=values['reconstruction'];best_step=step
            temp=out/'best.pt.tmp';torch.save({'model':model.state_dict(),'config':config,'step':step,'validation':values},temp);temp.replace(out/'best.pt')
        event({'validation':values});heartbeat('running',validation=values,preview='preview/latest.png');return values
    sampler.start_batch=step
    loader=DataLoader(train,batch_sampler=sampler,num_workers=args.workers,collate_fn=collate_native,worker_init_fn=init_worker,
                      pin_memory=True,persistent_workers=args.workers>0,generator=generator)
    heartbeat();model.train()
    try:
        if step==0:validate()
        elif saved_valid:heartbeat(validation=saved_valid,preview='preview/latest.png');event({'resume':True,'from_checkpoint':step})
        averages={};images_in_window=0
        for batch in loader:
            if step>=total:break
            batch_count=len(batch['references']);learning_rate(optimizer,step,total);optimizer.zero_grad(set_to_none=True);losses_batch={}
            for start in range(0,batch_count,args.microbatch_size):
                end=min(batch_count,start+args.microbatch_size);item=gpu_batch(batch,device,start,end);weight=(end-start)/batch_count
                with torch.autocast('cuda',dtype=torch.float16,enabled=args.amp):
                    prediction,aux=model(item['content'],item['reference'],True)
                    losses=criterion(prediction,item['target'],item['content'],aux)
                if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite train loss')
                scaler.scale(losses['total']*weight).backward()
                for key,value in losses.items():losses_batch[key]=losses_batch.get(key,0.)+float(value.detach())*weight
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=not args.amp)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();step+=1;consumed+=batch_count
            if scaler.get_scale()<old_scale:event({'amp_skipped_update':True,'amp_scale':scaler.get_scale()})
            for key,value in losses_batch.items():averages[key]=averages.get(key,0.)+value*batch_count
            images_in_window+=batch_count
            if step==1 or step%args.log_every==0 or step==total:
                event({'train':{key:value/images_in_window for key,value in averages.items()},'consumed':consumed,'epoch':consumed/len(train),
                       'actual_batch_size':batch_count,'physical_microbatch':min(args.microbatch_size,batch_count),
                       'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'gradient_norm':float(norm) if torch.isfinite(norm) else None})
                averages={};images_in_window=0;heartbeat()
            if consumed//8000 > (consumed-batch_count)//8000 or step in (min(25,total),total):validate();checkpoint()
        if 'validation' not in status:validate()
        checkpoint();heartbeat('completed')
        atomic_json(out/'complete.json',{'step':step,'consumed':consumed,'best_valid_l1':best,'best_step':best_step,'counts':config['counts'],
            'batch_size':args.batch_size,'physical_batch_size':args.microbatch_size,'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,
            'elapsed_seconds':elapsed_previous+time.monotonic()-started,'test_evaluated':False})
    except BaseException as error:heartbeat('failed',error=type(error).__name__+': '+str(error));raise
    finally:log.close()


if __name__=='__main__':main()
