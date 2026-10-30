"""Apply one explicit review export against the trusted unified inventory."""
import collections,csv,hashlib,json,shutil,sys
from pathlib import Path
import collect_photos as c
from apply_review_deletions import raw_images,EXT

def main():
    input_path=Path(sys.argv[1]).resolve()
    obj=json.loads(input_path.read_text(encoding='utf-8-sig'))
    assert obj['collection']=='nju-all-images-default-keep-v2'
    assert obj['default_decision']=='keep'
    rejected={r['key'] for r in obj['reviewed'] if r['decision']=='reject'}
    assert len(rejected)==obj['rejected_count']
    out=c.BASE/'_collection'
    inventory=[json.loads(x) for x in (out/'all_images_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    known={r['key']:r for r in inventory}
    assert len(known)==len(inventory)
    report=out/('review_changes_'+input_path.stem)
    if report.exists():raise RuntimeError('Batch audit already exists; do not apply twice')
    plan=[]
    for key in sorted(rejected):
        r=known[key];p=(c.BASE/r['path']).resolve()
        p.relative_to(c.BASE.resolve())
        assert p.suffix.lower() in EXT and not p.is_symlink() and p.is_file(),key
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],key
        plan.append(r)
    before={p.resolve() for p in raw_images()}
    report.mkdir();backup=report/'before';backup.mkdir()
    for name in ['all_images_manifest.jsonl','all_images_manifest.csv','all_images_summary.json','accepted_manifest.jsonl','accepted_manifest.csv','summary.json','integrity_check.json']:
        if (out/name).exists():shutil.copy2(out/name,backup/name)
    shutil.copy2(input_path,backup/input_path.name)
    c.save_json(report/'deletion_plan.json',{'input':str(input_path),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'targets':plan})
    records={r['source']+'/'+r['id']:r for r in c.load_records()}
    for r in plan:
        p=(c.BASE/r['path']).resolve();p.relative_to(c.BASE.resolve());p.unlink()
        old=records.get(r['key'])
        if old and old.get('path')==r['path']:
            c.write_record({**old,'status':'user_deleted','user_review_files':[str(input_path)],'user_decision':'reject'})
    after={p.resolve() for p in raw_images()}
    assert before-after=={(c.BASE/r['path']).resolve() for r in plan}
    assert after-before==set()
    result={'input':str(input_path),'deleted_this_run':len(plan),'deleted_by_source':dict(collections.Counter(r['source'] for r in plan)),'raw_images_before':len(before),'raw_images_after':len(after),'unmentioned_images_preserved':True,'deleted':plan}
    c.save_json(report/'applied_deletions.json',result)
    accepted=[r for r in c.load_records() if r['status']=='accepted']
    (out/'accepted_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    fields=['source','id','path','width','height','bytes','sha256','category_hint','photographer','license','source_page','quality_flags']
    with (out/'accepted_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(accepted)
    for source in {r['source'] for r in accepted}:
        (c.BASE/source/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted if r['source']==source),encoding='utf-8')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary.update(accepted=len(accepted),accepted_by_source=dict(collections.Counter(r['source'] for r in accepted)),accepted_bytes=sum(r['bytes'] for r in accepted),human_review='Explicit reject decisions applied; unmentioned retained',latest_review_report=str(report))
    c.save_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in result.items() if k!='deleted'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
