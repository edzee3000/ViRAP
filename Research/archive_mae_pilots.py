"""Archive only task-owned, stopped preflight checkpoints to free local headroom."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

assert socket.gethostname().split('.')[0]=='trinity-2-23'
base=Path('/tmp/shihangz/photo-tone-transfer/Experiments/mae_full_20261030').resolve()
destination=Path('/data3/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/preflight')
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):value.update(data)
    return value.hexdigest()
freed=0
for name in ('pilot_amp','pilot_resume','pilot_resume_v2'):
    folder=(base/name).resolve()
    assert folder.is_relative_to(base)
    status_path=folder/'status.json'
    if not status_path.exists():continue
    status=json.loads(status_path.read_text())
    try:os.kill(status['pid'],0);raise RuntimeError('Preflight process still alive; preserve '+name)
    except ProcessLookupError:pass
    target=destination/name;target.mkdir(parents=True,exist_ok=True)
    for path in folder.iterdir():
        if path.is_file():
            temp=target/(path.name+'.copying');shutil.copy2(path,temp);temp.replace(target/path.name)
            if path.suffix=='.pt':
                assert sha(path)==sha(target/path.name)
                freed+=path.stat().st_size;path.unlink()
print('archived_preflight_checkpoints_freed_gib',round(freed/1024**3,3),flush=True)
