"""Reclaim a completed probe's unused feature cache after verified archival."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

assert socket.gethostname().split('.')[0]=='trinity-2-23'
root=Path('/tmp/shihangz/photo-tone-transfer')
probe=root/'Experiments/encoder_probe_20261030'
assert (probe/'train_complete.json').is_file()
source=probe/'cache/dinov2_b.npy'
assert source.resolve().is_relative_to(root.resolve()) and source.name=='dinov2_b.npy'
assert subprocess.run(['fuser',str(source)],capture_output=True).returncode!=0, 'Cache in use'
target=Path('/data3/shihangz/photo-tone-transfer/Experiments/encoder_probe_20261030/cache/dinov2_b.npy')
target.parent.mkdir(parents=True,exist_ok=True)
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()
if not target.exists():
    temporary=target.with_suffix('.npy.archiving');shutil.copy2(source,temporary);temporary.replace(target)
assert source.stat().st_size==target.stat().st_size
sha=digest(source);assert sha==digest(target)
record={'source':str(source),'archive':str(target),'bytes':source.stat().st_size,'sha256':sha}
source.unlink()
(root/'Experiments/mae_full_20261030/probe_cache_archival.json').write_text(json.dumps(record,indent=2))
print('verified_unused_probe_cache_archived',record,flush=True)
print('free_GiB',shutil.disk_usage('/tmp').free/1024**3,flush=True)
