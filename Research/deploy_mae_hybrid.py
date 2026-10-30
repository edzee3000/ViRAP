"""Isolated addon deployment and GPU8 preflight; original eight models untouched."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','preflight','launch']);ap.add_argument('--microbatch',type=int,default=16);args=ap.parse_args()
    assert socket.gethostname().split('.')[0]=='trinity-2-23'
    root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030';addon=base/'addon_code/hybrid_lora';python=root/'Runtime/venv/bin/python'
    if args.action=='extract':
        addon.mkdir(parents=True,exist_ok=True)
        with tarfile.open(root/'Research/mae_hybrid_code.tar') as archive:
            for member in archive.getmembers():
                target=(addon/member.name).resolve()
                if not target.is_relative_to(addon.resolve()) or not member.isfile():raise ValueError('Unexpected addon archive member')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
        print('isolated_addon_ready');return
    used=int(subprocess.check_output(['nvidia-smi','-i','8','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>700:raise RuntimeError('GPU8 occupied')
    if args.action=='preflight':
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        command=[str(python),'-u',str(addon/'Model/train_mae_hybrid.py'),'--root',str(root),
                 '--output',str(base/f'preflight_hybrid_micro{args.microbatch}'),'--weights',str(root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'),
                 '--batch-size','16','--microbatch-size',str(args.microbatch),'--amp','--preflight','--workers','1','--log-every','1']
        with (base/f'preflight_hybrid_micro{args.microbatch}.log').open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('native_batch16_preflight_passed','microbatch',args.microbatch);return
    preflight=base/f'preflight_hybrid_micro{args.microbatch}/complete.json'
    if not preflight.exists():raise RuntimeError('Pass the memory/pretraining preflight first')
    command=f'cd {root} && {python} -u {addon}/Model/hybrid_supervisor.py --root {root} --experiment {base} --microbatch {args.microbatch} > {base}/hybrid_supervisor.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','mae-hybrid-lora',command],check=True)
    # Only the read-only monitor assets are refreshed. No baseline trainer or
    # model modules are overwritten, no training sessions are stopped.
    for name in ('mae_training_dashboard.py','mae_training_dashboard.html'):
        shutil.copy2(addon/'Model'/name,base/'code/Model'/name)
    subprocess.run(['tmux','kill-session','-t','mae-full-monitor'],check=True)
    monitor=f'cd {root} && {python} -u {base}/code/Model/mae_training_dashboard.py --root {base} --port 55318 >> {base}/dashboard.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','mae-full-monitor',monitor],check=True)
    print('hybrid_launched_gpu8_and_added_to_existing_dashboard')


if __name__=='__main__':main()
