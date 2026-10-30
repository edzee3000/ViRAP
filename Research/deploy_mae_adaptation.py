"""Prepare isolated code and reclaim only hash-verified archived checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile

root=Path('/tmp/shihangz/photo-tone-transfer');base=root/'Experiments/mae_full_20261030'
assert socket.gethostname().split('.')[0]=='trinity-2-23'
addon=base/'addon_code/adaptation_sweep'
assert not addon.exists()
shutil.copytree(base/'addon_code/hybrid_lora/Model',addon/'Model')
with tarfile.open(root/'Research/mae_adaptation_code.tar') as archive:
    archive.extractall(addon,filter='data')

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()

cleanup=[]
destination=Path('/data3/shihangz/photo-tone-transfer/Experiments/mae_full_20261030')
for folder in sorted((base/'runs').iterdir()):
    if not (folder/'complete.json').exists():continue
    source=folder/'latest.pt';backup=destination/'runs'/folder.name/'latest.pt'
    if not source.exists():continue
    if not backup.exists() or source.stat().st_size!=backup.stat().st_size:
        raise RuntimeError('Missing or wrong-sized archived checkpoint '+folder.name)
    sha=digest(source)
    if sha!=digest(backup):raise RuntimeError('Archived checkpoint hash mismatch '+folder.name)
    assert source.resolve().is_relative_to((base/'runs').resolve()) and source.name=='latest.pt'
    cleanup.append({'run':folder.name,'bytes':source.stat().st_size,'sha256':sha,'archive':str(backup)})
    source.unlink()
    print('verified_archive_reclaimed',folder.name,flush=True)
(base/'archived_checkpoint_cleanup.json').write_text(json.dumps(cleanup,indent=2))
assert shutil.disk_usage('/tmp').free > 162*1024**3, 'Need reserve for eight new runs'
print('isolated_sweep_ready_free_GiB',shutil.disk_usage('/tmp').free/1024**3,flush=True)
python=root/'Runtime/venv/bin/python'
command=f'cd {root} && {python} -u {addon}/Model/adaptation_supervisor.py --root {root} --experiment {base} > {base}/adaptation_supervisor.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s','mae-adaptation-sweep',command],check=True)
print('parallel_preflights_then_eight_runs_launched',flush=True)
