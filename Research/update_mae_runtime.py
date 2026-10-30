"""Controlled trainer-only update: preserve checkpoint state/model/data, log code history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tarfile
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['install','check','restart']);args=ap.parse_args()
    assert socket.gethostname().split('.')[0]=='trinity-2-23'
    root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030';code=base/'code'
    python=root/'Runtime/venv/bin/python'
    if args.action=='install':
        history=base/'code_history/pre_amp_cache_fix';history.mkdir(parents=True,exist_ok=True)
        allowed={'Model/train_mae_full.py','Model/mae_full_supervisor.py','Model/mae_training_dashboard.py'}
        with tarfile.open(root/'Research/mae_runtime_update.tar') as archive:
            for member in archive.getmembers():
                if member.name not in allowed or not member.isfile():raise ValueError('Unexpected runtime-update member')
                target=(code/member.name).resolve();assert target.is_relative_to(code.resolve())
                old=history/target.name
                if not old.exists():shutil.copy2(target,old)
                versioned=base/'code_history'/hashlib.sha256(target.read_bytes()).hexdigest()[:12]
                versioned.mkdir(parents=True,exist_ok=True);shutil.copy2(target,versioned/target.name)
                target.write_bytes(archive.extractfile(member).read())
        print('trainer_only_update_installed');return
    if args.action=='check':
        used=int(subprocess.check_output(['nvidia-smi','-i','8','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>700:raise RuntimeError('Preflight GPU8 occupied')
        folder=base/'pilot_resume';shutil.copy2(Path('/data3/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/preflight/pilot_resume/latest.pt'),folder/'latest.pt')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        command=[str(python),'-u',str(code/'Model/train_mae_full.py'),'--root',str(root),'--output',str(folder),
                 '--weights',str(root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'),'--max-updates','6','--heads-updates','3',
                 '--accumulate','2','--validation-limit','8','--validate-every','3','--log-every','1','--workers','1','--amp','--resume','--resume-allow-trainer-update']
        with (base/'runtime_update_check.log').open('w') as log:subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('controlled_resume_check_passed');return
    supervisor=json.loads((base/'supervisor.json').read_text())
    for job in supervisor['jobs']:
        if not (base/'runs'/job['name']/'latest.pt').exists():raise RuntimeError('Wait for first atomic checkpoint: '+job['name'])
    pid=supervisor['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'mae_full_supervisor.py' not in cmd or str(base).encode() not in cmd:raise ValueError('Supervisor ownership mismatch')
    os.kill(pid,signal.SIGTERM)
    for job in supervisor['jobs']:
        pid=job['pid'];cmd_path=Path(f'/proc/{pid}/cmdline')
        if cmd_path.exists():
            cmd=cmd_path.read_bytes()
            if b'train_mae_full.py' not in cmd or str(base/'runs'/job['name']).encode() not in cmd:raise ValueError('Worker ownership mismatch')
            os.kill(pid,signal.SIGTERM)
    # Bounded exit wait, only the explicitly owned worker PIDs.
    for _ in range(20):
        if all(not Path(f'/proc/{job["pid"]}').exists() for job in supervisor['jobs']):break
        time.sleep(1)
    if any(Path(f'/proc/{job["pid"]}').exists() for job in supervisor['jobs']):raise RuntimeError('Owned workers did not exit')
    if subprocess.run(['tmux','has-session','-t','mae-full-train'],stderr=subprocess.DEVNULL).returncode==0:
        subprocess.run(['tmux','kill-session','-t','mae-full-train'],check=True)
    if subprocess.run(['tmux','has-session','-t','mae-full-monitor'],stderr=subprocess.DEVNULL).returncode==0:
        subprocess.run(['tmux','kill-session','-t','mae-full-monitor'],check=True)
    monitor=f'cd {root} && {python} -u {code}/Model/mae_training_dashboard.py --root {base} --port 55318 >> {base}/dashboard.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','mae-full-monitor',monitor],check=True)
    command=f'cd {root} && {python} -u {code}/Model/mae_full_supervisor.py --root {root} --experiment {base} --amp --approve-trainer-update >> {base}/supervisor.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','mae-full-train',command],check=True)
    print('eight_workers_restarted_from_checkpoints_with_logged_trainer_fix')


if __name__=='__main__':main()
