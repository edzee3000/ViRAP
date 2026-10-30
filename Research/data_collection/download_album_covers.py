"""Download the user-selected album-cover directory without modifying the photo pool."""
import concurrent.futures as cf
import hashlib,io,json,time,urllib.parse
from pathlib import Path
from collections import Counter
from PIL import Image,ImageOps,ImageDraw
import collect_photos as c

BASE=c.BASE/'Album_Covers'
META=BASE/'metadata'
IMAGES=BASE/'originals'
IMAGES.mkdir(parents=True,exist_ok=True)
INDEX=json.loads((META/'source_index.json').read_text(encoding='utf-8'))
REV=INDEX['revision']
LOG=META/'download_log.jsonl'

def download(entry):
    name=Path(entry['path']).name
    url='https://huggingface.co/datasets/edzee3000/GithubData/resolve/'+REV+'/'+urllib.parse.quote(entry['path'],safe='/')
    path=IMAGES/name
    record={'source':'Album_Covers','id':path.stem,'filename':name,'repository':INDEX['repository'],'revision':REV,'repository_path':entry['path'],'source_page':'https://huggingface.co/datasets/edzee3000/GithubData/blob/'+REV+'/'+urllib.parse.quote(entry['path'],safe='/'),'download_url':url,'expected_bytes':entry['size'],'expected_sha256':entry.get('lfs',{}).get('oid'),'license':'No unified image license declared by source repository; album-art rights not independently verified','download_transform':'None; original repository file bytes retained','review_status':'unreviewed'}
    try:
        data=path.read_bytes() if path.exists() else c.get(url).content
        digest=hashlib.sha256(data).hexdigest()
        if len(data)!=entry['size']:raise ValueError('file size mismatch')
        if record['expected_sha256'] and digest!=record['expected_sha256']:raise ValueError('SHA256 mismatch')
        with Image.open(io.BytesIO(data)) as im:
            im.load();w,h=im.size;fmt=im.format
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
        flags=[]
        if min(w,h)<720:flags.append('cover_below_photo_resolution')
        if min(w,h)<400:flags.append('cover_very_low_resolution')
        return {**record,'status':'downloaded','path':'originals/'+name,'width':w,'height':h,'format':fmt,'bytes':len(data),'sha256':digest,'quality_flags':flags}
    except Exception as e:return {**record,'status':'failed','error':type(e).__name__+': '+str(e)[:500]}

def main():
    items=[x for x in INDEX['entries'] if x['type']=='file']
    records=[]
    start=time.time()
    with cf.ThreadPoolExecutor(max_workers=6) as pool,LOG.open('a',encoding='utf-8',buffering=1) as log:
        for n,record in enumerate(pool.map(download,items),1):
            log.write(json.dumps(record,ensure_ascii=False)+'\n');records.append(record)
            if n%20==0 or n==len(items):
                print('ALBUM_DOWNLOAD',n,'/',len(items),'ok',sum(r['status']=='downloaded' for r in records),'elapsed_s',round(time.time()-start),flush=True)
                c.save_json(META/'progress.json',{'processed':n,'total':len(items),'downloaded':sum(r['status']=='downloaded' for r in records),'elapsed_s':round(time.time()-start)})
    records.sort(key=lambda r:r['filename'])
    (BASE/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    good=[r for r in records if r['status']=='downloaded']
    hashes={};duplicates=[]
    for r in good:
        if r['sha256'] in hashes:
            r['duplicate_of']=hashes[r['sha256']];duplicates.append({'id':r['id'],'duplicate_of':r['duplicate_of']})
        else:hashes[r['sha256']]=r['id']
    (BASE/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    summary={'requested':len(items),'downloaded':len(good),'failed':[r for r in records if r['status']=='failed'],'bytes':sum(r['bytes'] for r in good),'unique_sha256':len(hashes),'exact_duplicates':duplicates,'min_short_edge':min(min(r['width'],r['height']) for r in good) if good else None,'resolution_short_edge_bins':dict(Counter('<400' if min(r['width'],r['height'])<400 else '400-719' if min(r['width'],r['height'])<720 else '720-1279' if min(r['width'],r['height'])<1280 else '>=1280' for r in good)),'integrity_check':'File byte size, SHA256 against repository LFS OID where provided, and full image decoding checked for every downloaded file','revision':REV,'kept_all_originals':True,'previous_photo_collection_modified':False}
    c.save_json(BASE/'summary.json',summary)
    thumbs=BASE/'review_thumbnails';thumbs.mkdir(exist_ok=True)
    sheets=BASE/'contact_sheets';sheets.mkdir(exist_ok=True)
    gallery=[]
    for n,r in enumerate(good):
        thumb=thumbs/(r['id']+'.jpg')
        with Image.open(BASE/r['path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((360,360));im.save(thumb,quality=85)
        gallery.append({'key':'Album_Covers/'+r['id'],'source':'Album_Covers','id':r['id'],'path':r['path'],'thumbnail':'review_thumbnails/'+thumb.name,'width':r['width'],'height':r['height'],'flags':r['quality_flags'],'hint':'专辑封面；摄影/插画/文字设计待人工判断','photographer':'','source_page':r['source_page'],'license':r['license']})
    for start in range(0,len(good),100):
        batch=good[start:start+100];canvas=Image.new('RGB',(1600,((len(batch)+9)//10)*174),'#141922');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(batch):
            with Image.open(thumbs/(r['id']+'.jpg')) as im:
                im.thumbnail((154,146));x=j%10*160;y=j//10*174;canvas.paste(im,(x+(154-im.width)//2,y));draw.text((x+2,y+148),str(start+j)+' '+r['id'][:10],fill='white')
        canvas.save(sheets/f'covers_{start//100+1:02d}.jpg',quality=90)
    import build_review as b
    page=b.HTML.replace('__DATA__',json.dumps(gallery,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('nju-photo-review-20261029-v1','nju-album-review-20261029-v1').replace('摄影数据初筛 · 人工审核','专辑封面 · 摄影候选审核')
    page=page.replace("'monochrome_or_low_saturation':'黑白／低饱和：保留风格多样性'", "'monochrome_or_low_saturation':'黑白／低饱和：保留风格多样性','cover_below_photo_resolution':'短边不足720：建议先作参考图候选','cover_very_low_resolution':'短边不足400：低分辨率封面'")
    (BASE/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
