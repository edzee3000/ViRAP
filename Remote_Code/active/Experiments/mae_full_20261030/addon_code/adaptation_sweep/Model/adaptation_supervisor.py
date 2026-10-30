"""Eight controlled adaptation runs; preflight, recovery and archival on compute."""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from train_mae_full import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True)
    args=ap.parse_args();root=args.root;base=args.experiment;code=Path(__file__).resolve().parent
    assert socket.gethostname().split('.')[0]=='trinity-2-23'
    weights=root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'
    destination=Path('/data3/shihangz/photo-tone-transfer/Experiments')/base.name
    marker=base/'additional_supervisors/adaptation_sweep.json'
    jobs=[]
    for adaptation,batch,label in [('all_lora',16,'AllLoRA'),('full',16,'Full'),('all_lora',8,'AllLoRA'),('hybrid',8,'HybridLoRA')]:
        for seed in (17,29):
            jobs.append({'name':f'M1_{label}_b{batch}_s{seed}','adaptation':adaptation,'batch':batch,
                         'seed':seed,'gpu':len(jobs),'state':'preflight','attempts':0,'rank':4})
    def status():atomic_json(marker,{'pid':os.getpid(),'jobs':jobs,'heartbeat_unix':time.time()})
    def environment(gpu):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    def command(job,out,preflight=False):
        values=[sys.executable,'-u',str(code/'train_mae_adaptation.py'),'--root',str(root),'--output',str(out),'--weights',str(weights),
                '--adaptation',job['adaptation'],'--seed',str(job['seed']),'--batch-size',str(job['batch']),
                '--microbatch-size',str(job['batch']),'--amp','--workers','2']
        if preflight:values.extend(['--preflight','--log-every','1'])
        elif (out/'latest.pt').exists():values.append('--resume')
        return values
    status()
    # Largest (square 384px) native batches, three real forward/backward updates.
    checks=[]
    for gpu,(adaptation,batch) in enumerate([('all_lora',16),('full',16),('hybrid',8)]):
        usage=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if usage>100:raise RuntimeError(f'GPU{gpu} already occupied')
        job={'adaptation':adaptation,'batch':batch,'seed':17};out=base/f'preflight_{adaptation}_b{batch}'
        log=(base/(out.name+'.log')).open('w')
        process=subprocess.Popen(command(job,out,True),cwd=root,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
        checks.append((process,log,out))
    failures=[]
    for process,log,out in checks:
        result=process.wait();log.close()
        if result:failures.append(out.name)
        # Preflight states are disposable; configs/metrics/best weights preserved.
        checkpoint=out/'latest.pt'
        if result==0 and checkpoint.exists():checkpoint.unlink()
    if failures:
        for job in jobs:job.update(state='failed',error='Memory/pretraining preflight failed: '+','.join(failures))
        status();raise RuntimeError(str(failures))
    print('all_native_batch_preflights_passed',flush=True)
    handles={}
    def launch(job):
        gpu=job['gpu'];usage=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if usage>100:raise RuntimeError(f'GPU{gpu} occupied before launch')
        run=base/'runs'/job['name'];log=(base/'logs'/(job['name']+'.log')).open('a')
        p=subprocess.Popen(command(job,run),cwd=root,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
        job.update(state='running',pid=p.pid,attempts=job['attempts']+1);handles[job['name']]=(p,log)
    for job in jobs:launch(job)
    status();last_backup=time.monotonic()
    while any(job['state']=='running' for job in jobs):
        for job in jobs:
            if job['state']!='running':continue
            p,log=handles[job['name']];result=p.poll()
            if result is None:continue
            log.close();run=base/'runs'/job['name']
            if result==0 and (run/'complete.json').exists():
                job.update(state='completed',exit_code=0)
                dest=destination/'runs'/job['name'];dest.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-a','--exclude=*.tmp',str(run)+'/',str(dest)+'/'],check=True)
            elif job['attempts']<3 and (run/'latest.pt').exists():
                time.sleep(3);launch(job)
            else:job.update(state='failed',exit_code=result,error='Inspect '+str(base/'logs'/(job['name']+'.log')))
        if time.monotonic()-last_backup>7200:
            for job in jobs:
                run=base/'runs'/job['name'];dest=destination/'recovery'/job['name'];dest.mkdir(parents=True,exist_ok=True)
                for name in ('config.json','status.json','best.pt','latest.pt'):
                    source=run/name
                    if source.exists():
                        temp=dest/(name+'.tmp');shutil.copy2(source,temp);temp.replace(dest/name)
            last_backup=time.monotonic()
        status();time.sleep(15)
    status();(destination/'additional_supervisors').mkdir(exist_ok=True)
    shutil.copy2(marker,destination/'additional_supervisors'/marker.name)
    print('sweep_finished',jobs,flush=True)

if __name__=='__main__':main()
