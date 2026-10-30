"""Validate and assemble generated-unreviewed triplets on the compute node."""
import collections,hashlib,json,sys,datetime
from pathlib import Path
from PIL import Image

def main(root):
    root=Path(root);manifest=root/'Dataset/Photo_Pairs_60000_v1/pairs.json'
    pairs=json.loads(manifest.read_text())['pairs'];records={}
    for run in ['preflight','pilot','bulk60000']:
        for path in (root/'Logs'/run).glob('worker_*.jsonl'):
            for line in path.read_text().splitlines():
                r=json.loads(line)
                if r['status']=='generated_unreviewed':records[r['pair_id']]=r
    triplets=[];failures=[]
    for p in pairs:
        try:
            r=records[p['pair_id']];path=(manifest.parent/r['target_path']).resolve()
            assert path==(manifest.parent/p['planned_target_path']).resolve()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_sha256']
            with Image.open(path) as im:im.verify()
            triplets.append({**p,**r,'status':'generated_unreviewed'})
        except Exception as e:failures.append({'pair_id':p['pair_id'],'error':repr(e)})
    out=root/'Dataset/Photo_Triplets_v1';out.mkdir(exist_ok=True)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    (out/'triplets.json').write_text(json.dumps({'path_base':'directory_containing_this_json','quality_state':'generated_unreviewed','triplets':triplets},ensure_ascii=False,indent=2))
    (out/'triplets.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in triplets))
    summary={'planned':len(pairs),'generated_verified':len(triplets),'failed_or_missing':len(failures),'by_split':dict(collections.Counter(r['split'] for r in triplets)),'accepted_after_human_review':0,'outputs_folder':'../Generated_Outputs','provider':'PETAL released model; see per-record generation settings','source_images_copied_per_pair':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    metadata_path=root/'Logs/run_metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text());metadata.update(status='incomplete' if failures else 'generation_complete',generated_verified=len(triplets),failed_or_missing=len(failures),generation_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());metadata_path.write_text(json.dumps(metadata,indent=2))
    if failures:raise RuntimeError('Incomplete run; see failures.json')

if __name__=='__main__':main(sys.argv[1])
