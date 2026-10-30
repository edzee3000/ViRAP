"""Snapshot completed validation-only results, never checkpoints/test metrics."""
import datetime
import json
from pathlib import Path
import socket
import zipfile

root=Path('/tmp/shihangz/photo-tone-transfer/Experiments/encoder_probe_20261030')
assert socket.gethostname().split('.')[0]=='trinity-2-23'
out=root/'validation_snapshot.zip'
summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{}}
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in sorted((root/'train').glob('*')):
        marker=folder/'complete.json'
        if not marker.exists():continue
        data=json.loads(marker.read_text());data.pop('test',None)
        summary['runs'][folder.name]=data
        for name in ('validation_per_pair.json','metrics.jsonl','config.json'):
            path=folder/name
            archive.write(path,folder.name+'/'+name)
        if folder.name.endswith('_17'):
            for path in sorted((folder/'validation_examples').glob('*.png'))[:8]:
                archive.write(path,folder.name+'/validation_examples/'+path.name)
    archive.writestr('validation_summary.json',json.dumps(summary,indent=2))
print(json.dumps({'runs':len(summary['runs']),'snapshot_bytes':out.stat().st_size,'file':str(out)}))
