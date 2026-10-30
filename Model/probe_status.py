"""Small read-only status report for one explicitly named compute-local run."""
from collections import Counter
import json
from pathlib import Path
import shutil
import socket
import subprocess

root=Path('/tmp/shihangz/photo-tone-transfer/Experiments/encoder_probe_20261030')
assert socket.gethostname().split('.')[0]=='trinity-2-23'
print('free_tmp_gib',round(shutil.disk_usage(root).free/1024**3,2))
for phase in ('prepare','pilot','train'):
    path=root/(phase+'_orchestration.log')
    if path.exists():print(phase,'orchestration',path.read_text()[-1400:])
for path in sorted((root/'logs').glob('*.log')):
    if path.name.startswith(('prepare','features')):
        print(path.name, path.read_text()[-650:])
for phase in ('pilot','train'):
    for folder in sorted((root/phase).glob('*')):
        if (folder/'complete.json').exists():
            result=json.loads((folder/'complete.json').read_text())
            print(phase,folder.name,'DONE',result['steps'],'val_l1',result['validation']['l1'],'seconds',result['elapsed_seconds'])
        elif (folder/'metrics.jsonl').exists():
            lines=(folder/'metrics.jsonl').read_text().splitlines()
            print(phase,folder.name,lines[-1] if lines else 'starting')
subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],check=True)
