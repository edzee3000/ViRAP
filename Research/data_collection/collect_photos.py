"""Resumable photographic-image collection. Never modifies pre-existing Live Photo."""
from __future__ import annotations
import argparse, concurrent.futures as cf, csv, hashlib, io, json, math, os, random, re, struct, threading, time, urllib.parse, warnings, zipfile, zlib
from collections import Counter
from pathlib import Path
import requests
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'Dataset' / 'Original Data'
META = BASE / '_collection' / 'metadata'
META.mkdir(parents=True, exist_ok=True)
LOG = BASE / '_collection' / 'attempts.jsonl'
THREAD = threading.local()
cv2.setNumThreads(1)
SEED = 20261029

def session():
    if not hasattr(THREAD, 'session'):
        THREAD.session = requests.Session()
        THREAD.session.headers.update({'User-Agent': 'PhotoResearchCollection/1.0 (academic image dataset preparation)'})
    return THREAD.session

def get(url, **kw):
    last = None
    for attempt in range(3):
        try:
            r = session().get(url, timeout=(15, 80), **kw)
            if r.status_code in (403,404,429):
                r.raise_for_status()
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if isinstance(e,requests.HTTPError) and e.response is not None and e.response.status_code in (403,404,429):
                raise
            if attempt < 2: time.sleep(1 + attempt * 2)
    raise last

class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = get(url, headers={'Range': 'bytes=0-0'})
        if r.status_code != 206: raise ValueError('server does not support ranges')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.tail_start = max(0, self.size - 2 * 1024 * 1024)
        self.tail = self.fetch(self.tail_start, self.size - 1)
    def fetch(self, start, end):
        r = get(self.url, headers={'Range': f'bytes={start}-{end}'})
        expected = f'bytes {start}-{end}/'
        if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(expected):
            raise ValueError(f'range mismatch: {r.status_code} {r.headers.get("Content-Range")}')
        if len(r.content) != end - start + 1: raise ValueError('truncated range')
        return r.content
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        n = min(n if n >= 0 else self.size-self.pos, self.size-self.pos)
        if n <= 0: return b''
        start, self.pos = self.pos, self.pos+n
        if start >= self.tail_start: return self.tail[start-self.tail_start:start-self.tail_start+n]
        return self.fetch(start, start+n-1)

ZIP_SOURCES = {
 'DIV2K': {'url':'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', 'page':'https://data.vision.ee.ethz.ch/cvl/DIV2K/', 'license':'Academic research only; image copyrights remain with original owners', 'match':lambda n:n.lower().endswith('.png')},
 'Flickr2K': {'url':'https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/790432e75172aaf1c2c350825fb7b2b821603881/Flickr2K.zip', 'page':'https://cv.snu.ac.kr/research/EDSR/', 'license':'Original per-photo rights not provided by archive; research provenance recorded; mirror Apache tag is NOT treated as image license', 'match':lambda n:bool(re.search(r'(?:^|/)Flickr2K_HR/[^/]+\.png$', n))},
}

def make_indexes():
    for source, conf in ZIP_SOURCES.items():
        path = META / f'{source}_index.json'
        if path.exists(): continue
        remote = RangeReader(conf['url'])
        with zipfile.ZipFile(remote) as z:
            items = [{'name':i.filename,'offset':i.header_offset,'compressed_size':i.compress_size,'size':i.file_size,'crc':i.CRC,'method':i.compress_type} for i in z.infolist() if conf['match'](i.filename)]
            print(source, 'all', len(z.infolist()), 'HR', len(items), 'examples', [i['name'] for i in items[:3]], flush=True)
        if not items: raise ValueError(f'no HR images for {source}')
        path.write_text(json.dumps({'url':conf['url'],'archive_bytes':remote.size,'entries':items},indent=2),encoding='utf-8')
    documents = {
      'unsplash_dataset_README.md':'https://raw.githubusercontent.com/unsplash/datasets/master/README.md',
      'unsplash_dataset_TERMS.md':'https://raw.githubusercontent.com/unsplash/datasets/master/TERMS.md',
      'unsplash_dataset_DOCS.md':'https://raw.githubusercontent.com/unsplash/datasets/master/DOCS.md',
      'div2k_source.html':'https://data.vision.ee.ethz.ch/cvl/DIV2K/',
      'flickr2k_original_project.html':'https://cv.snu.ac.kr/research/EDSR/',
    }
    for name,url in documents.items():
        p=META/name
        if not p.exists():
            try: p.write_bytes(get(url).content)
            except Exception as e: print('documentation warning',name,str(e)[:120],flush=True)

def zip_image(item):
    off, cs = item['entry']['offset'], item['entry']['compressed_size']
    # Local filename and extra headers are normally <1KB. Header is read independently
    # so no assumption is made about ZIP64 or unusual local extra-field lengths.
    h = get(item['archive_url'],headers={'Range':f'bytes={off}-{off+29}'} )
    if h.status_code !=206 or len(h.content)!=30: raise ValueError('invalid local header range')
    sig,*fields = struct.unpack('<4s5H3I2H',h.content)
    if sig != b'PK\x03\x04': raise ValueError('bad ZIP header')
    nl, el = fields[-2:]
    start=off+30+nl+el
    r=get(item['archive_url'],headers={'Range':f'bytes={start}-{start+cs-1}'})
    if r.status_code !=206 or len(r.content)!=cs: raise ValueError('bad compressed image range')
    data=zlib.decompress(r.content,-15) if item['entry']['method']==8 else r.content
    if len(data)!=item['entry']['size'] or zlib.crc32(data)!=item['entry']['crc']: raise ValueError('ZIP CRC mismatch')
    return data

def category(text):
    text=text.lower()
    rules=[('portrait',r'\b(woman|man|girl|boy|person|people|portrait|couple|bride|wedding|child|model)\b'),('food',r'\b(food|coffee|cake|fruit|meal|bread|cooking|restaurant|tea|dish)\b'),('interior',r'\b(interior|room|bedroom|living room|kitchen|furniture|sofa|bed|indoors)\b'),('night',r'\b(night|neon|milky way|stars|nighttime|aurora)\b'),('architecture',r'\b(building|city|architecture|street|bridge|skyscraper|house|urban|tower|temple|church)\b'),('animals',r'\b(dog|cat|bird|horse|animal|wildlife|fox|deer|elephant|lion)\b'),('landscape',r'\b(mountain|forest|lake|ocean|beach|tree|sea|landscape|river|nature|sunset|field|desert|snow|waterfall)\b')]
    return next((c for c,p in rules if re.search(p,text)),'other')

def unsplash_candidates():
    p=META/'unsplash_photos.tsv'
    if not p.exists():p.write_bytes(get('https://huggingface.co/datasets/image-search-2/unsplash_lite_image_dataset/resolve/c9ab866576b08dd92819e413fc0b3853757da304/photos.tsv000').content)
    rows=list(csv.DictReader(p.open(encoding='utf-8'),delimiter='\t'))
    rng=random.Random(SEED); rng.shuffle(rows)
    groups={}
    excluded=[]
    for row in rows:
        w,h=int(row['photo_width']),int(row['photo_height'])
        if min(w,h)<900 or max(w,h)<1600 or max(w,h)/min(w,h)>2.5: continue
        text=row.get('ai_description','')+' '+row.get('photo_description','')
        if re.search(r'\b(illustration|rendering|3d render|screenshot|typography|digital art|collage|wallpaper design)\b',text,re.I):continue
        cat=category(text)
        row['_cat']=cat
        # Editorial promotion is a useful initial prior, not an aesthetic guarantee.
        row['_priority']=(row.get('photo_featured')=='t',math.log1p(int(row.get('stats_downloads') or 0))+rng.random()*2)
        groups.setdefault(cat,[]).append(row)
    for rs in groups.values():rs.sort(key=lambda r:r['_priority'],reverse=True)
    # Weighted round-robin prevents a nature-only first batch. Metadata categories
    # remain explicitly heuristic and are intended for the human reviewer.
    weights={'portrait':3,'food':1,'interior':1,'night':1,'architecture':3,'animals':1,'landscape':5,'other':2}
    order=[];authors=Counter()
    while any(groups.values()):
        for cat,n in weights.items():
            for _ in range(n):
                rs=groups.get(cat,[])
                if not rs:break
                row=rs.pop(0)
                author=row['photographer_username']
                if authors[author]>=6:continue
                authors[author]+=1
                url=row['photo_image_url']
                parts=urllib.parse.urlsplit(url); query=dict(urllib.parse.parse_qsl(parts.query))
                query.update({'fm':'jpg','q':'95','fit':'max','w':'2560','h':'2560','dpr':'1'})
                url=urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
                order.append({'source':'Unsplash_Lite','id':row['photo_id'],'filename':row['photo_id']+'.jpg','download_url':url,'source_page':row['photo_url'],'photographer':author,'category_hint':cat,'description':text if False else row.get('ai_description',''),'original_width':int(row['photo_width']),'original_height':int(row['photo_height']),'metadata_snapshot':'image-search-2/unsplash_lite_image_dataset@c9ab866576b08dd92819e413fc0b3853757da304','license':'Unsplash Dataset Terms; research collection; no blanket redistribution permission','download_transform':'Unsplash CDN fit=max, max 2560x2560, JPEG q95; no crop; not full camera original'})
    return order

def candidates(source):
    if source=='Unsplash_Lite':return unsplash_candidates()
    if source=='Wikimedia_Commons':
        rows=json.loads((META/'commons_candidates.json').read_text(encoding='utf-8'))
        random.Random(SEED).shuffle(rows)
        return rows
    conf=ZIP_SOURCES[source];index=json.loads((META/f'{source}_index.json').read_text(encoding='utf-8'))
    entries=index['entries'][:];random.Random(SEED).shuffle(entries)
    return [{'source':source,'id':Path(e['name']).stem,'filename':Path(e['name']).name,'entry':e,'archive_url':index['url'],'source_page':conf['page'],'license':conf['license'],'download_transform':'Original HR PNG archive member, byte-for-byte decompressed; CRC verified','category_hint':'unclassified'} for e in entries]

def analyze(data):
    with Image.open(io.BytesIO(data)) as raw:
        raw.load();w,h=raw.size;mode=raw.mode
        im=ImageOps.exif_transpose(raw).convert('RGB')
        im.thumbnail((512,512))
        arr=np.asarray(im)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        g32=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
        dct=cv2.dct(g32)[:8,:8].ravel(); bits=dct>np.median(dct[1:]);bits[0]=False
        ph=int.from_bytes(np.packbits(bits).tobytes(),'big')
        d=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
        dh=int.from_bytes(np.packbits(d[:,1:]>d[:,:-1]).tobytes(),'big')
        small=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)
        lap=float(cv2.Laplacian(gray,cv2.CV_32F).var())
        edge=float(np.mean(cv2.Canny(gray,60,120)>0))
        std=float(gray.std());dark=float(np.mean(gray<4));bright=float(np.mean(gray>251))
        entropy=float(-np.sum((hist:=np.bincount(gray.ravel(),minlength=256)/gray.size)[hist>0]*np.log2(hist[hist>0])))
    reject=[];flags=[]
    if min(w,h)<720 or max(w,h)<1280:reject.append('insufficient_resolution')
    if max(w,h)/min(w,h)>3:reject.append('extreme_aspect_ratio')
    if std<2 or entropy<1:reject.append('nearly_blank')
    if lap<5 and edge<0.003 and std<14:reject.append('severe_low_detail')
    if lap<35:flags.append('low_detail_or_shallow_depth_of_field')
    if dark>0.15:flags.append('large_dark_region')
    if bright>0.15:flags.append('large_bright_region')
    if float(np.mean(np.max(arr,axis=2)-np.min(arr,axis=2)))<5:flags.append('monochrome_or_low_saturation')
    return {'width':w,'height':h,'mode':mode,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'phash':f'{ph:016x}','dhash':f'{dh:016x}','laplacian_variance_512':round(lap,3),'edge_fraction_512':round(edge,5),'gray_std':round(std,3),'gray_entropy':round(entropy,3),'dark_fraction':round(dark,5),'bright_fraction':round(bright,5),'quality_flags':flags,'reject_reasons':reject},small

def fetch_item(item):
    try:
        data=zip_image(item) if 'entry' in item else get(item['download_url']).content
        metrics,small=analyze(data)
        return item,data,metrics,small,None
    except Exception as e:return item,None,None,None,type(e).__name__+': '+str(e)[:400]

def load_records():
    if not LOG.exists():return []
    latest={}
    malformed=[]
    for line_number,line in enumerate(LOG.read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try:r=json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number);continue
            latest[(r['source'],r['id'])]=r
    if malformed:print('MANIFEST_NEEDS_RECONCILIATION malformed_lines',malformed,flush=True)
    return list(latest.values())

def save_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(10):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(0.15)

def write_record(rec):
    # Serialize appends across source processes. Windows append writes do not
    # provide a portable multi-process JSONL record-atomicity guarantee.
    with LOG.with_suffix('.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            with LOG.open('ab') as f:
                f.seek(0,2);f.write((json.dumps(rec,ensure_ascii=False)+'\n').encode('utf-8'));f.flush()
        finally:
            lock.seek(0)
            if os.name=='nt':msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def collect(source,target,workers=6,pilot=False):
    records=load_records();done={(r['source'],r['id']) for r in records if r['status']!='download_failed'}
    accepted=[r for r in records if r['status']=='accepted']
    hashes={r['sha256']:r for r in accepted}
    phashes=[(int(r['phash'],16),int(r['dhash'],16),r) for r in accepted]
    thumbs={}
    count=sum(r['source']==source for r in accepted)
    if count>=target:return
    tasks=[i for i in candidates(source) if (source,i['id']) not in done]
    outdir=BASE/source/'images';outdir.mkdir(parents=True,exist_ok=True)
    rejectdir=BASE/'_collection'/'rejected'/source;rejectdir.mkdir(parents=True,exist_ok=True)
    print('START',source,'existing',count,'target',target,'available',len(tasks),flush=True)
    iterator=iter(tasks);pending={};processed=0;started=time.time()
    with LOG.open('a',encoding='utf-8',buffering=1) as log,cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            try:i=next(iterator);pending[pool.submit(fetch_item,i)]=i
            except StopIteration:break
        while pending:
            completed,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for future in completed:
                pending.pop(future);item,data,m,small,error=future.result();processed+=1
                rec={**item,'collected_at':'2026-10-29','status':'download_failed' if error else 'accepted'}
                if error:rec['error']=error
                else:
                    rec.update(m);duplicate=None
                    if m['sha256'] in hashes:duplicate=hashes[m['sha256']];rec['reject_reasons'].append('exact_duplicate')
                    if not duplicate and not rec['reject_reasons']:
                        ph,dh=int(m['phash'],16),int(m['dhash'],16)
                        for op,od,other in phashes:
                            if (ph^op).bit_count()<=4 and (dh^od).bit_count()<=8:
                                # Similar hashes alone are insufficient for skies, snow or monochrome scenes.
                                key=other['path']
                                if key not in thumbs:
                                    with Image.open(BASE/key) as old:
                                        old=ImageOps.exif_transpose(old).convert('L').resize((64,64),Image.Resampling.BOX)
                                        thumbs[key]=np.asarray(old,dtype=np.float32)
                                a=small.ravel();b=thumbs[key].ravel()
                                corr=float(np.corrcoef(a,b)[0,1]) if a.std()>1 and b.std()>1 else 0
                                if corr>0.985:
                                    duplicate=other;rec['reject_reasons'].append('near_duplicate');rec['duplicate_correlation']=round(corr,6);break
                    if duplicate:rec['duplicate_of']=duplicate['source']+'/'+duplicate['id']
                    if rec['reject_reasons']:
                        rec['status']='rejected';path=rejectdir/item['filename']
                    elif count>=target:
                        rec['status']='reserve';path=BASE/'_collection'/'reserve'/source/item['filename'];path.parent.mkdir(parents=True,exist_ok=True)
                    else:
                        path=outdir/item['filename'];count+=1
                    path.write_bytes(data);rec['path']=path.relative_to(BASE).as_posix()
                    if rec['status']=='accepted':
                        hashes[m['sha256']]=rec;phashes.append((int(m['phash'],16),int(m['dhash'],16),rec));thumbs[rec['path']]=small
                write_record(rec)
                if processed%20==0 or count>=target:
                    print(source,'accepted',count,'processed',processed,'elapsed_s',round(time.time()-started),'last',rec['status'],flush=True)
                    save_json(BASE/'_collection'/f'progress_{source}.json',{'source':source,'accepted':count,'target':target,'processed_this_run':processed,'elapsed_s':round(time.time()-started)})
                if count<target:
                    try:i=next(iterator);pending[pool.submit(fetch_item,i)]=i
                    except StopIteration:pass
    print('DONE',source,count,flush=True)

def commons_index():
    import html
    old={}
    for p in sorted(META.glob('commons_stage*_previous_manifest.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);old[row['pageid']]=row
    ids=sorted(old);items=[]
    cache=META/'commons_live_metadata';cache.mkdir(exist_ok=True)
    s=requests.Session();s.headers['User-Agent']='GraduationPhotoResearch/1.0 (academic; local personal collection)'
    for start in range(0,len(ids),40):
        batch=ids[start:start+40];cp=cache/f'{start:05d}.json'
        if cp.exists():data=json.loads(cp.read_text(encoding='utf-8'))
        else:
            for attempt in range(4):
                r=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','prop':'imageinfo','pageids':'|'.join(map(str,batch)),'iiprop':'url|size|extmetadata','iiurlwidth':'1920','iiextmetadatafilter':'LicenseShortName|LicenseUrl|Artist|Credit|Restrictions'},timeout=(15,45))
                if r.status_code !=429:break
                delay=min(60,max(15,int(r.headers.get('Retry-After','30'))))
                print('COMMONS_RATE_LIMIT waiting',delay,flush=True);time.sleep(delay)
            r.raise_for_status();data=r.json()
            if 'error' in data:raise ValueError(data['error'])
            save_json(cp,data);time.sleep(3)
        for page in data.get('query',{}).get('pages',{}).values():
            if not page.get('imageinfo'):continue
            info=page['imageinfo'][0];previous=old[page['pageid']]
            ext=info.get('extmetadata',{});lic=ext.get('LicenseShortName',{}).get('value','')
            if not (lic.startswith(('CC BY','CC0')) or 'Public domain' in lic):continue
            if min(info['width'],info['height'])<900:continue
            if not re.search(r'\.(jpg|jpeg|png)(?:\?|$)',info['url'],re.I):continue
            title=page['title']
            if re.search(r'\b(painting|lithograph|engraving|illustration|drawing|map of|photomontage|composite image)\b',title,re.I):continue
            clean=lambda x:html.unescape(re.sub('<[^>]*>','',x)).strip()
            ident='commons-'+str(page['pageid'])
            extn=Path(urllib.parse.urlsplit(info.get('thumburl',info['url'])).path).suffix.lower()
            items.append({'source':'Wikimedia_Commons','id':ident,'filename':ident+extn,'pageid':page['pageid'],'title':title,'download_url':info.get('thumburl',info['url']),'original_url':info['url'],'source_page':info['descriptionurl'],'photographer':clean(ext.get('Artist',{}).get('value','')),'credit':clean(ext.get('Credit',{}).get('value','')),'license':lic,'license_url':ext.get('LicenseUrl',{}).get('value',''),'restrictions':ext.get('Restrictions',{}).get('value',''),'license_review':'live Commons API metadata refreshed 2026-10-29; not a manual ownership audit','category_hint':'; '.join(previous['categories']),'original_width':info['width'],'original_height':info['height'],'download_transform':'Commons API thumbnail requested at width 1920; actual decoded dimensions recorded; no local crop or recompression'})
        save_json(META/'commons_candidates.json',items)
        print('COMMONS_METADATA',start+len(batch),'/',len(ids),'eligible',len(items),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['index','commons-index','collect']);p.add_argument('--source',choices=['DIV2K','Flickr2K','Unsplash_Lite','Wikimedia_Commons']);p.add_argument('--target',type=int,default=20);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.phase=='index':make_indexes()
    elif a.phase=='commons-index':commons_index()
    else:collect(a.source,a.target,a.workers)
if __name__=='__main__':main()
