"""Commit protocol before evaluating and run nine independent jobs in tmux."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030'
out=base/'test_eval_v1';python=root/'Runtime/venv/bin/python'
assert socket.gethostname().split('.')[0]=='trinity-2-23'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    if '--worker' not in sys.argv:
        assert not out.exists(), 'Existing test evaluation must not be overwritten'
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(line.split(',')[1])<100 for line in usage.splitlines() if int(line.split(',')[0])<9)
        out.mkdir();code=out/'code';code.mkdir()
        for name in ('evaluate_mae_test.py','test_metrics.py'):shutil.copy2(root/'Research'/name,code/name)
        runs=sorted(path for path in (base/'runs').iterdir() if (path/'complete.json').exists())
        assert len(runs)==9
        entries={}
        for folder in runs:
            status=json.loads((folder/'status.json').read_text())
            assert status['state']=='completed'
            entries[folder.name]={'path':str(folder/'best.pt'),'sha256':digest(folder/'best.pt'),
                                 'step':status['best_step'],'validation_l1':status['best_valid']}
        protocol={'version':1,'created_unix':time.time(),'selection':'validation reconstruction L1 only; best.pt fixed before test inference',
                  'checkpoints':entries,'test_count':400,'geometry':'native aspect, content/target long edge 384; reference long edge 384; no crop',
                  'metrics':'imagewise mean RGB float [0,1] L1/MSE/PSNR (MSE floor 1e-12); RGB SSIM gaussian 11 sigma 1.5 valid window C1=.01^2 C2=.03^2, equal channel mean; gradient L1',
                  'precision':'same FP16 autocast forward as validation; metrics FP32; TF32 disabled',
                  'baseline':'identity content to generated target','no_test_selection':True,
                  'code_sha256':{p.name:digest(p) for p in code.glob('*.py')}}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        subprocess.run(['tmux','new-session','-d','-s','mae-test-eval',f'{python} -u {Path(__file__)} --worker > {out}/supervisor.log 2>&1'],check=True)
        print('test_evaluation_launched',len(entries),flush=True);return
    protocol=json.loads((out/'protocol.json').read_text());jobs=[]
    for gpu,name in enumerate(protocol['checkpoints']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=(out/(name+'.log')).open('w')
        process=subprocess.Popen([str(python),'-u',str(out/'code/evaluate_mae_test.py'),'--root',str(root),'--experiment',str(base),'--run',name],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,process,log));print('started',name,gpu,process.pid,flush=True)
    codes={}
    for name,process,log in jobs:codes[name]=process.wait();log.close();print('exited',name,codes[name],flush=True)
    (out/'exit_codes.json').write_text(json.dumps(codes,indent=2))
    destination=Path('/data3/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/test_eval_v1')
    shutil.copytree(out,destination,dirs_exist_ok=True)
    print('evaluation_finished_and_archived',codes,flush=True)

if __name__=='__main__':main()
