"""Compute-node bootstrap for immutable code deployment, dashboard and training."""
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import tarfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','pilot','resume-check','monitor','launch']);ap.add_argument('--amp',action='store_true');args=ap.parse_args()
    assert socket.gethostname().split('.')[0]=='trinity-2-23'
    root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030';base.mkdir(parents=True,exist_ok=True)
    code=base/'code';python=root/'Runtime/venv/bin/python'
    if args.action=='extract':
        code.mkdir(exist_ok=True)
        with tarfile.open(root/'Research/mae_full_code.tar') as archive:
            for member in archive.getmembers():
                if not (code/member.name).resolve().is_relative_to(code.resolve()) or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member')
            import inspect
            if 'filter' in inspect.signature(archive.extractall).parameters:archive.extractall(code,filter='data')
            else:archive.extractall(code)
        print('code_ready',code);return
    if args.action in ('pilot','resume-check'):
        used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>700:raise RuntimeError('GPU0 is occupied; do not run preflight over another job')
        name='pilot_resume_v2' if args.action=='resume-check' else 'pilot_amp_v2' if args.amp else 'pilot_fp32'
        command=[str(python),'-u',str(code/'Model/train_mae_full.py'),'--root',str(root),'--output',str(base/name),
                 '--weights',str(root/'Experiments/encoder_probe_20261030/cache/mae_b.pth'),'--max-updates','6','--heads-updates','3',
                 '--accumulate','2','--validation-limit','8','--validate-every','3','--log-every','1','--workers','1']
        if args.amp:command.append('--amp')
        import os
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (base/(name+'.log')).open('w') as log:
            subprocess.run(command+(['--stop-after-updates','3'] if args.action=='resume-check' else []),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if args.action=='resume-check':subprocess.run(command+['--resume'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('pilot_complete');return
    def session(name,command,log):
        if subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            if name=='mae-full-monitor':return
            raise RuntimeError('Session already exists: '+name)
        shell=f'cd {root} && '+ ' '.join(map(str,command))+f' > {log} 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',name,shell],check=True)
    session('mae-full-monitor',[python,'-u',code/'Model/mae_training_dashboard.py','--root',base,'--port','55318'],base/'dashboard.log')
    if args.action=='monitor':print('monitor_started',55318);return
    command=[python,'-u',code/'Model/mae_full_supervisor.py','--root',root,'--experiment',base]
    if args.amp:command.append('--amp')
    session('mae-full-train',command,base/'supervisor.log')
    print('started',base,'dashboard_port',55318)


if __name__=='__main__':main()
