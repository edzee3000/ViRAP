"""Eight independent GPU experiments, detached supervision and bounded recovery."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from train_mae_full import atomic_json
from train import guard_experiment_paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--amp',action='store_true');ap.add_argument('--approve-trainer-update',action='store_true')
    args=ap.parse_args();root=args.root.resolve();base=args.experiment.resolve()
    guard_experiment_paths(root,base);assert socket.gethostname().split('.')[0]=='trinity-2-23'
    code=Path(__file__).resolve().parent;weights=root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'
    (base/'runs').mkdir(parents=True,exist_ok=True);(base/'logs').mkdir(exist_ok=True)
    jobs=[{'name':f'M{0 if rank==2 else 1}_s{seed}','rank':rank,'seed':seed,'gpu':gpu,'attempts':0,'process':None}
          for gpu,(rank,seed) in enumerate((rank,seed) for seed in (17,29,43,71) for rank in (2,4))]
    for job in jobs:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>700:raise RuntimeError('GPU occupied: '+str(job['gpu']))
    archive=Path('/data3/shihangz/photo-tone-transfer/Experiments')/base.name
    def serializable():
        return {'pid':os.getpid(),'node':socket.gethostname(),'heartbeat_unix':time.time(),
                'jobs':[{k:v for k,v in job.items() if k not in ('process','log')} for job in jobs]}
    def start(job):
        output=base/'runs'/job['name'];complete=output/'complete.json'
        if complete.exists():job['state']='completed';return
        cmd=[sys.executable,'-u',str(code/'train_mae_full.py'),'--root',str(root),'--output',str(output),
             '--weights',str(weights),'--local-rank',str(job['rank']),'--seed',str(job['seed']),'--workers','4']
        if args.amp:cmd.append('--amp')
        if (output/'latest.pt').exists():
            cmd.append('--resume')
            if args.approve_trainer_update:cmd.append('--resume-allow-trainer-update')
        elif (output/'config.json').exists():
            job['state']='failed';job['error']='No recoverable checkpoint after initial failure';return
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        job['log']=(base/'logs'/(job['name']+'.log')).open('a')
        job['process']=subprocess.Popen(cmd,cwd=root,env=env,stdout=job['log'],stderr=subprocess.STDOUT)
        job['attempts']+=1;job['state']='running';job['pid']=job['process'].pid
        print('started',job['name'],'gpu',job['gpu'],'pid',job['pid'],flush=True)
    def backup(final=False):
        archive.mkdir(parents=True,exist_ok=True)
        if final:
            subprocess.run(['rsync','-a','--exclude=*.tmp',str(base)+'/',str(archive)+'/'],check=True)
        else:
            # Deliberate two-hour checkpoint recovery snapshots, not live NFS logging.
            for job in jobs:
                dest=archive/'recovery'/job['name'];dest.mkdir(parents=True,exist_ok=True)
                for name in ('latest.pt','best.pt','config.json','status.json'):
                    path=base/'runs'/job['name']/name
                    if path.exists():
                        temp=dest/(name+'.tmp');shutil.copy2(path,temp);temp.replace(dest/name)
        print('archived', 'final' if final else 'recovery',flush=True)
    for job in jobs:start(job)
    last_backup=time.monotonic()
    while any(job['state']=='running' for job in jobs):
        for job in jobs:
            process=job['process']
            if job['state']!='running' or process.poll() is None:continue
            job['log'].close();job['exit_code']=process.returncode
            if process.returncode==0 and (base/'runs'/job['name']/'complete.json').exists():
                job['state']='completed';print('completed',job['name'],flush=True)
            elif job['attempts']<3 and (base/'runs'/job['name']/'latest.pt').exists():
                print('recovering',job['name'],flush=True);start(job)
            else:job['state']='failed';print('failed',job['name'],flush=True)
        atomic_json(base/'supervisor.json',serializable())
        if time.monotonic()-last_backup>7200:
            try:backup()
            except Exception as error:print('recovery_archive_error',repr(error),flush=True)
            last_backup=time.monotonic()
        time.sleep(15)
    atomic_json(base/'supervisor.json',serializable());backup(final=True)
    atomic_json(base/'archive_complete.json',{'destination':str(archive),'time_unix':time.time()})
    print('all_tasks_finished',flush=True)


if __name__=='__main__':main()
