"""Apply only explicit user reject decisions, resolving files through trusted inventories."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
import collect_photos as c

REPORT=c.BASE/'_collection'/'review_changes_20261029'
EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.heic','.avif'}

def raw_images():
    excluded={'metadata','review_thumbnails','contact_sheets','thumbnails','review_previews','versions'}
    for folder in c.BASE.iterdir():
        if not folder.is_dir() or folder.name=='_collection':continue
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in EXT and not (set(p.relative_to(folder).parts[:-1])&excluded):yield p
    for name in ['rejected','reserve']:
        for p in (c.BASE/'_collection'/name).rglob('*'):
            if p.is_file() and p.suffix.lower() in EXT:yield p

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    inputs=[];decisions=[]
    for path in sorted((c.ROOT/'Dataset'/'Review Datas').rglob('*.json')):
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(obj,dict) or not isinstance(obj.get('reviewed'),list):continue
        inputs.append({'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exported_at':obj.get('exported_at')})
        decisions+=obj['reviewed']
    rejected={r['key'] for r in decisions if r.get('decision')=='reject'}
    known={r['source']+'/'+r['id']:r for r in c.load_records()}
    backup=REPORT/'before'
    backup.mkdir(exist_ok=True)
    for name in ['accepted_manifest.jsonl','accepted_manifest.csv','summary.json','integrity_check.json','review.html']:
        original=c.BASE/'_collection'/name
        if original.exists() and not (backup/name).exists():shutil.copy2(original,backup/name)
    for path in sorted((c.ROOT/'Dataset'/'Review Datas').rglob('*.json')):
        if not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
    plan=[]
    # Validate every target before deleting any file. Do not execute JSON paths.
    for key in sorted(rejected):
        if key not in known:raise ValueError('Unresolved reject key: '+key)
        record=known[key];path=(c.BASE/record['path']).resolve()
        path.relative_to(c.BASE.resolve())
        if path.suffix.lower() not in EXT or path.is_symlink():raise ValueError('Unexpected image target: '+str(path))
        if record['status']=='user_deleted':
            if path.exists():raise ValueError('Previously deleted image has reappeared: '+key)
            plan.append({'key':key,'record':record,'already_applied':True});continue
        if not path.is_file():raise ValueError('Missing reject image: '+str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed reject image: '+key)
        plan.append({'key':key,'record':record,'already_applied':False})
    before={p.resolve() for p in raw_images()}
    c.save_json(REPORT/'deletion_plan.json',{'review_inputs':inputs,'reject_count':len(rejected),'keep_entries':sum(r.get('decision')=='keep' for r in decisions),'targets':plan})
    removed=[]
    for item in plan:
        if item['already_applied']:continue
        r=item['record'];path=(c.BASE/r['path']).resolve()
        path.relative_to(c.BASE.resolve())
        path.unlink()  # One validated, explicitly rejected image; never a recursive operation.
        thumbnail=c.BASE/'_collection'/'thumbnails'/(r['source']+'__'+r['id']+'.jpg')
        if thumbnail.exists():
            thumbnail.resolve().relative_to((c.BASE/'_collection'/'thumbnails').resolve());thumbnail.unlink()
        c.write_record({**r,'status':'user_deleted','user_review_files':[x['file'] for x in inputs],'user_decision':'reject'})
        removed.append({'key':item['key'],'path':r['path'],'sha256':r['sha256'],'bytes':r['bytes']})
    after={p.resolve() for p in raw_images()}
    expected={(c.BASE/r['path']).resolve() for r in removed}
    if before-after!=expected:raise RuntimeError('Unexpected image removal; inspect audit immediately')
    report={'explicit_reject_count':len(rejected),'deleted_this_run':len(removed),'already_applied':sum(p['already_applied'] for p in plan),'raw_images_before':len(before),'raw_images_after':len(after),'unmentioned_images_preserved':True,'deleted':removed,'review_inputs':inputs,'remaining_raw_by_top_folder':dict(Counter(p.relative_to(c.BASE.resolve()).parts[0] for p in after))}
    c.save_json(REPORT/'applied_deletions.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['deleted','review_inputs']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
