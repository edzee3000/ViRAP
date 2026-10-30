"""Bounded compute-node orchestration. All live IO stays node-local."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from train import guard_experiment_paths
from prepare_encoder_experiment import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','pilot','train'],required=True)
    ap.add_argument('--gpus',default='0,1,2,3,4,5,6,7');args=ap.parse_args()
    root=args.root.resolve();base=root/'Experiments/encoder_probe_20261030';cache=base/'cache';logs=base/'logs'
    guard_experiment_paths(root,base);base.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free<150*1024**3:raise RuntimeError('Preserve 150 GiB local reserve')
    gpus=[int(x) for x in args.gpus.split(',')]
    def free_gpu(gpu):
        value=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(value.strip())>700:raise RuntimeError('GPU is occupied: '+str(gpu))
    def run(label,command,gpu=None):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if gpu is not None:free_gpu(gpu);env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (logs/(label+'.log')).open('a') as log:
            print('start',label,'gpu',gpu,flush=True)
            subprocess.run([sys.executable,*map(str,command)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('complete',label,flush=True)
    if args.phase=='prepare':
        run('prepare_images',[root/'Model/prepare_encoder_experiment.py','--root',root,'--cache',cache,'--images'])
        names=['dinov2_s','dinov3_s','mae_b','dinov2_b']
        with ThreadPoolExecutor(max_workers=min(len(gpus),len(names))) as pool:
            futures=[pool.submit(run,'features_'+name,[root/'Model/prepare_encoder_experiment.py','--root',root,'--cache',cache,'--backbone',name,'--batch-size','16'],gpus[i%len(gpus)]) for i,name in enumerate(names)]
            for future in as_completed(futures):future.result()
        save_json(base/'prepared.json',{'status':'complete','backbones':names,'cache':str(cache)})
    else:
        if not (base/'prepared.json').exists():raise RuntimeError('Preparation not complete')
        groups=['A','B','C','D','E','F'];seeds=[17] if args.phase=='pilot' else [17,29,43]
        if args.phase=='train':
            for group in groups:
                marker=base/'pilot'/(group+'_17')/'complete.json'
                if not marker.exists():raise RuntimeError('Pilot incomplete for '+group)
                if not json.loads(marker.read_text())['pilot']:raise RuntimeError('Invalid pilot')
        jobs=[(group,seed) for seed in seeds for group in groups]
        # One worker per GPU, serial jobs within each worker; no GPU overcommit.
        def worker(slot):
            for group,seed in jobs[slot::len(gpus)]:
                output=base/args.phase/(group+'_'+str(seed))
                if (output/'complete.json').exists():continue
                cmd=[root/'Model/run_encoder_probe.py','--cache',cache,'--output',output,'--group',group,'--seed',seed]
                if args.phase=='pilot':cmd+=['--pilot','--max-steps','60','--validation-limit','64']
                if (output/'config.json').exists():cmd+=['--resume']
                run(args.phase+'_'+group+'_'+str(seed),cmd,gpus[slot])
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,i) for i in range(len(gpus))]
            for future in as_completed(futures):future.result()
        save_json(base/(args.phase+'_complete.json'),{'status':'complete','groups':groups,'seeds':seeds})
        if args.phase=='train':
            from summarize_encoder_probe import summarize
            summarize(base)
            # A single deliberate end-of-run archive, no live checkpoint writes to NAS.
            destination=Path('/data3/shihangz/photo-tone-transfer/Experiments/encoder_probe_20261030')
            destination.mkdir(parents=True,exist_ok=True)
            for name in ('train','pilot','logs'):
                subprocess.run(['rsync','-a',str(base/name)+'/',str(destination/name)+'/'],check=True)
            for path in base.glob('*.json'):shutil.copy2(path,destination/path.name)
            for path in base.glob('*.md'):shutil.copy2(path,destination/path.name)
            provenance=destination/'cache_provenance';provenance.mkdir(exist_ok=True)
            for path in cache.glob('*.json'):shutil.copy2(path,provenance/path.name)
            subprocess.run(['rsync','-a',str(root/'Model')+'/',str(destination/'Model')+'/'],check=True)
    print('phase_complete',args.phase,flush=True)


if __name__=='__main__':main()
