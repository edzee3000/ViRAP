import shutil
from pathlib import Path
import socket
import subprocess
assert socket.gethostname().split('.')[0]=='trinity-2-23'
root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030'
addon=base/'addon_code/adaptation_sweep/Model';python=root/'Runtime/venv/bin/python'
assert (addon/'tone_model/mae_adaptation.py').exists()
shutil.copy2(root/'Research/adaptation_dashboard.html',base/'code/Model/mae_training_dashboard.html')
shutil.copy2(addon/'preview_gallery_worker.py',base/'gallery_code/preview_gallery_worker.py')
for session in ('mae-full-monitor','mae-gallery-preview'):
    subprocess.run(['tmux','kill-session','-t',session],check=True)
monitor=f'cd {root} && {python} -u {base}/code/Model/mae_training_dashboard.py --root {base} --port 55318 >> {base}/monitor.log 2>&1'
preview=f'cd {root} && CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {python} -u {base}/gallery_code/preview_gallery_worker.py --root {root} --experiment {base} --model-code {addon} --count 48 --poll 30 >> {base}/gallery_worker.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s','mae-full-monitor',monitor],check=True)
subprocess.run(['tmux','new-session','-d','-s','mae-gallery-preview',preview],check=True)
print('monitor_and_gallery_refreshed_no_training_restart')
