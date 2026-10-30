"""Record imported model/data source identities against training snapshots."""
import hashlib
import json
from pathlib import Path

base=Path('/tmp/shihangz/photo-tone-transfer/Experiments/mae_full_20261030')
code=base/'addon_code/hybrid_lora/Model'
audit={}
for path in sorted((base/'runs').glob('*/config.json')):
    config=json.loads(path.read_text());mismatches=[];checked={}
    for name,expected in config['code_sha256'].items():
        if not name.startswith('tone_model/'):continue
        actual=hashlib.sha256((code/name).read_bytes()).hexdigest()
        checked[name]={'training_sha256':expected,'evaluation_sha256':actual}
        if actual!=expected:mismatches.append(name)
    audit[path.parent.name]={'checked':checked,'mismatches':mismatches}
(base/'test_eval_v1/model_source_audit.json').write_text(json.dumps(audit,indent=2))
print({name:row['mismatches'] for name,row in audit.items()})
assert not any(row['mismatches'] for row in audit.values()), 'Model/data source differs from training'
