"""Compute-node-only deployment: gallery assets and isolated preview worker."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile

assert socket.gethostname().split('.')[0] == 'trinity-2-23'
root = Path('/tmp/shihangz/photo-tone-transfer')
base = root/'Experiments/mae_full_20261030'
python = root/'Runtime/venv/bin/python'
assert shutil.disk_usage('/tmp').free > 150*1024**3
source = base/'gallery_deploy'
source.mkdir(exist_ok=True)
with tarfile.open(root/'Research/gallery_upgrade.tar') as archive:
    archive.extractall(source, filter='data')
monitor = base/'code/Model'
for name in ('mae_training_dashboard.py','mae_training_dashboard.html','mae_gallery.py','mae_gallery_ui.js'):
    shutil.copy2(source/'Model'/name, monitor/name)
worker = base/'gallery_code'
worker.mkdir(exist_ok=True)
for name in ('preview_gallery_worker.py','mae_gallery.py'):
    shutil.copy2(source/'Model'/name, worker/name)
subprocess.run([str(python),'-m','py_compile',str(worker/'preview_gallery_worker.py'),str(monitor/'mae_training_dashboard.py')],check=True)
if subprocess.run(['tmux','has-session','-t','mae-gallery-preview'],capture_output=True).returncode:
    usage = subprocess.check_output(['nvidia-smi','--id=9','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(usage.strip()) < 100, usage
    command = f'cd {root} && CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {python} -u {worker}/preview_gallery_worker.py --root {root} --experiment {base} --model-code {base}/addon_code/hybrid_lora/Model --count 48 --poll 30 > {base}/gallery_worker.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','mae-gallery-preview',command],check=True)
# Only the task-owned read-only monitor is restarted. Training sessions untouched.
subprocess.run(['tmux','kill-session','-t','mae-full-monitor'],check=True)
subprocess.run(['tmux','new-session','-d','-s','mae-full-monitor',f'{python} -u {monitor}/mae_training_dashboard.py --root {base} --port 55318 > {base}/monitor.log 2>&1'],check=True)
print('gallery_worker_and_monitor_deployed')
