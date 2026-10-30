"""Persist a lightweight status artifact, without copying live checkpoints."""
import datetime
import json
from pathlib import Path
import socket
from prepare_encoder_experiment import save_json

root=Path('/tmp/shihangz/photo-tone-transfer/Experiments/encoder_probe_20261030')
assert socket.gethostname().split('.')[0]=='trinity-2-23'
status={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),
        'root':str(root),'pilot':{},'train':{},'prepared':(root/'prepared.json').exists()}
for phase in ('pilot','train'):
    for folder in sorted((root/phase).glob('*')):
        if (folder/'complete.json').exists():status[phase][folder.name]=json.loads((folder/'complete.json').read_text())
        elif (folder/'metrics.jsonl').exists():
            lines=(folder/'metrics.jsonl').read_text().splitlines()
            status[phase][folder.name]={'state':'running','latest':json.loads(lines[-1]) if lines else None}
save_json(root/'status_snapshot.json',status)
print(json.dumps({'pilot_complete':len(status['pilot']),'training_started':len(status['train']),'updated_utc':status['updated_utc']}))
