"""Single additional experiment on GPU8; existing eight jobs stay untouched."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from train_mae_full import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--microbatch',type=int,default=16);args=ap.parse_args();root=args.root;main_root=args.experiment
    name='M1_LoRA_r8_b16_s17';run=main_root/'runs'/name;code=Path(__file__).resolve().parent
    marker=main_root/'additional_supervisors/hybrid_lora.json';marker.parent.mkdir(exist_ok=True)
    destination=Path('/data3/shihangz/photo-tone-transfer/Experiments')/main_root.name
    metadata={'name':name,'rank':4,'seed':17,'gpu':8,'lora_rank':8,'attempts':0,'state':'starting'}
    def status():atomic_json(marker,{'pid':os.getpid(),'jobs':[metadata],'heartbeat_unix':time.time()})
    def backup(final=False):
        dest=destination/('runs' if final else 'recovery')/name;dest.mkdir(parents=True,exist_ok=True)
        if final:subprocess.run(['rsync','-a','--exclude=*.tmp',str(run)+'/',str(dest)+'/'],check=True)
        else:
            for filename in ('latest.pt','best.pt','config.json','status.json'):
                source=run/filename
                if source.exists():
                    temp=dest/(filename+'.tmp');shutil.copy2(source,temp);temp.replace(dest/filename)
        extra=destination/'additional_supervisors';extra.mkdir(exist_ok=True);shutil.copy2(marker,extra/marker.name)
    while metadata['attempts']<3:
        used=int(subprocess.check_output(['nvidia-smi','-i','8','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>700:raise RuntimeError('GPU8 occupied; refuse to overlap')
        command=[sys.executable,'-u',str(code/'train_mae_hybrid.py'),'--root',str(root),'--output',str(run),
                 '--weights',str(root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'),'--batch-size','16','--microbatch-size',str(args.microbatch),'--amp']
        if (run/'latest.pt').exists():command.append('--resume')
        elif (run/'config.json').exists():metadata.update(state='failed',error='No checkpoint for initial failure');status();return
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (main_root/'logs'/f'{name}.log').open('a') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            metadata.update(state='running',pid=process.pid,attempts=metadata['attempts']+1);status();last_backup=time.monotonic()
            while process.poll() is None:
                status()
                if time.monotonic()-last_backup>7200:
                    try:backup()
                    except Exception as error:print('backup_error',repr(error),flush=True)
                    last_backup=time.monotonic()
                time.sleep(15)
        if process.returncode==0:
            metadata.update(state='completed');status();backup(True);return
        if not (run/'latest.pt').exists():break
    metadata.update(state='failed');status();backup(False)


if __name__=='__main__':main()
