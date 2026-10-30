"""Append missing public NetEase playlist artwork without replacing existing originals."""
import concurrent.futures as cf
import csv, hashlib, io, json, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import requests
from PIL import Image

BASE = Path(__file__).resolve().parents[2] / 'Dataset/Original Data/Album_Covers'
META = BASE / 'metadata'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://music.163.com/'}

def get(url, **kwargs):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=45, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt + 1)

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    snapshot = json.loads((META/'playlist_8114560070_latest.json').read_text(encoding='utf-8'))
    playlist = snapshot['playlist']
    ids = [x['id'] for x in playlist['trackIds']]
    assert len(ids) == len(set(ids))
    def batch(values):
        j = get('https://music.163.com/api/song/detail', params={'ids':json.dumps(values)}).json()
        assert j['code'] == 200, j
        return j['songs']
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        songs = [s for group in pool.map(batch, [ids[i:i+50] for i in range(0,len(ids),50)]) for s in group]
    by_id = {s['id']:s for s in songs}
    missing_ids = sorted(set(ids)-by_id.keys())
    save(META/'playlist_8114560070_tracks.json', {'playlist_id':8114560070,'advertised_count':playlist['trackCount'],'returned_ids':len(ids),'missing_ids':missing_ids,'tracks':[by_id[i] for i in ids if i in by_id]})
    if missing_ids: raise RuntimeError(f'Missing song metadata: {missing_ids}')
    rows = [json.loads(s) for s in (BASE/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s]
    original_count = len(rows)
    known_albums = {}
    known_urls = {}
    for row in rows:
        for t in row.get('tracks',[]): known_albums[t['album_id']] = row
        url = row.get('upstream_cover_url','').split('?')[0]
        if url: known_urls[url.rsplit('/',1)[-1]] = row
    groups = {}
    mapped = []
    for sid in ids:
        s = by_id[sid]; a = s.get('album',s.get('al',{}))
        artists = [x['name'] for x in s.get('artists',s.get('ar',[]))]
        track = {'song':s['name'],'song_id':sid,'album_id':a['id'],'album':a['name'],'artists':artists}
        url = a.get('picUrl','').split('?')[0].replace('http://','https://')
        old = known_albums.get(a['id']) or known_urls.get(url.rsplit('/',1)[-1])
        if old:
            if sid not in {t['song_id'] for t in old.get('tracks',[])}: old.setdefault('tracks',[]).append(track)
            mapped.append({'song_id':sid,'cover_id':old['id'],'existing':True})
        else:
            g = groups.setdefault(a['id'], {'album':a,'url':url,'tracks':[]})
            g['tracks'].append(track)
    print(json.dumps({'resolved_songs':len(songs),'existing_files':len(rows),'missing_album_covers':len(groups)}),flush=True)
    hashes = {r['sha256']:r for r in rows}
    def download(group):
        url = group['url']
        if not url: return {'error':'No artwork URL','tracks':group['tracks']}
        try:
            data = get(url).content
            with Image.open(io.BytesIO(data)) as im:
                im.load(); width,height = im.size; fmt=im.format
            digest = hashlib.sha256(data).hexdigest()
            ext = {'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(fmt)
            if not ext: raise ValueError(f'Unexpected image format {fmt}')
            return {'group':group,'data':data,'sha256':digest,'width':width,'height':height,'format':fmt,'ext':ext}
        except Exception as e: return {'error':str(e),'tracks':group['tracks'],'url':url}
    added=[]; reused=[]; failures=[]
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(download,groups.values()):
            if 'error' in result:
                failures.append(result); continue
            g=result['group']; digest=result['sha256']
            if digest in hashes:
                row=hashes[digest]
                for t in g['tracks']:
                    if t['song_id'] not in {x['song_id'] for x in row['tracks']}: row['tracks'].append(t)
                reused.append({'album_id':g['album']['id'],'cover_id':row['id']})
            else:
                ident='netease_'+str(g['album']['id'])+'_'+digest[:12]
                filename=ident+result['ext']; path=BASE/'originals'/filename
                assert not path.exists(),path
                path.write_bytes(result['data'])
                flags=[]
                if min(result['width'],result['height'])<720: flags.append('cover_below_photo_resolution')
                if min(result['width'],result['height'])<400: flags.append('cover_very_low_resolution')
                row={'source':'Album_Covers','id':ident,'filename':filename,'path':'originals/'+filename,'status':'downloaded','source_page':'https://music.163.com/album?id='+str(g['album']['id']),'download_url':g['url'],'upstream_cover_url':g['url'],'download_transform':'None; original server bytes retained','review_status':'unreviewed','visual_preselection':'unclassified','width':result['width'],'height':result['height'],'format':result['format'],'bytes':len(result['data']),'sha256':digest,'quality_flags':flags,'albums':[],'artists':[],'tracks':g['tracks'],'playlist_id':8114560070,'license':'Album-art rights remain with original rights holders'}
                rows.append(row); hashes[digest]=row; added.append(row['id'])
            for t in g['tracks']: mapped.append({'song_id':t['song_id'],'cover_id':row['id'],'existing':row['id'] not in added})
    for row in rows:
        row['albums']=list(dict.fromkeys(row.get('albums',[])+[t['album'] for t in row.get('tracks',[])]))
        row['artists']=list(dict.fromkeys(row.get('artists',[])+[a for t in row.get('tracks',[]) for a in t['artists']]))
    (BASE/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    fields=['id','filename','path','albums','artists','width','height','bytes','sha256','quality_flags','visual_preselection','upstream_cover_url','source_page']
    with (BASE/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for r in rows: writer.writerow({**r,'albums':' / '.join(r['albums']),'artists':' / '.join(r['artists']),'quality_flags':'; '.join(r['quality_flags'])})
    site=json.loads(Path('E:/Trivia/Homepage/edzee3000.github.io/_data/music.json').read_text(encoding='utf-8'))
    site_ids={s['id'] for s in site}
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'playlist_id':8114560070,'advertised_track_count':playlist['trackCount'],'returned_track_ids':len(ids),'resolved_tracks':len(songs),'count_difference':playlist['trackCount']-len(ids),'homepage_tracks':len(site),'songs_new_since_homepage':[{'id':s['id'],'name':s['name']} for s in songs if s['id'] not in site_ids],'previous_cover_files':original_count,'missing_album_groups':len(groups),'added_cover_files':len(added),'added_cover_ids':added,'reused_identical_images':reused,'failed':failures,'total_cover_files':len(rows),'mapped_songs':len(mapped),'song_cover_mapping':mapped}
    save(META/'playlist_cover_sync_report.json',report)
    summary=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
    summary.update({'requested':len(rows)+len(failures),'downloaded':len(rows),'failed':failures,'bytes':sum(r['bytes'] for r in rows),'unique_sha256':len({r['sha256'] for r in rows}),'album_names_mapped':sum(bool(r['albums']) for r in rows),'resolution_short_edge_bins':dict(Counter('<400' if min(r['width'],r['height'])<400 else '400-719' if min(r['width'],r['height'])<720 else '720-1279' if min(r['width'],r['height'])<1280 else '>=1280' for r in rows)),'latest_playlist_sync':{k:v for k,v in report.items() if k not in ['song_cover_mapping','songs_new_since_homepage','added_cover_ids']},'integrity_check':'Original upstream files retained; newly fetched images fully decoded and SHA256 recorded'})
    save(BASE/'summary.json',summary)
    print(json.dumps({k:v for k,v in report.items() if k not in ['song_cover_mapping','songs_new_since_homepage','added_cover_ids']},ensure_ascii=True,indent=2),flush=True)
    if failures: raise RuntimeError('Some covers failed; inspect sync report')

if __name__=='__main__': main()
