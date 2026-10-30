"""Create an entirely local, source-grouped human review gallery and audit."""
import collections, concurrent.futures as cf, csv, html, json, math, random, shutil, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import collect_photos as c

OUT=c.BASE/'_collection';THUMBS=OUT/'thumbnails';THUMBS.mkdir(exist_ok=True)

def append(rec):
    c.write_record(rec)

def quarantine(rec,reason,extra=None):
    source=(c.BASE/rec['path']).resolve();source.relative_to(c.BASE.resolve())
    dest=(OUT/'rejected'/rec['source']/source.name).resolve();dest.relative_to(c.BASE.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest!=source:raise RuntimeError('quarantine destination exists')
    source.rename(dest)
    rec={**rec,'status':'rejected','path':dest.relative_to(c.BASE).as_posix(),'reject_reasons':rec.get('reject_reasons',[])+[reason],**(extra or {})}
    append(rec)

def audit():
    records=c.load_records();accepted=[r for r in records if r['status']=='accepted']
    seen={};near=[];cache={};counts=collections.Counter()
    for r in sorted(accepted,key=lambda r:(-r['width']*r['height'],r['source'],r['id'])):
        p=c.BASE/r['path']
        if not p.exists():raise RuntimeError('missing '+str(p))
        if r['sha256'] in seen:
            quarantine(r,'global_exact_duplicate',{'duplicate_of':seen[r['sha256']]['source']+'/'+seen[r['sha256']]['id']});counts['exact']+=1;continue
        ph,dh=int(r['phash'],16),int(r['dhash'],16);match=None
        for op,od,other in near:
            if (ph^op).bit_count()<=8 and (dh^od).bit_count()<=12:
                for rec in (r,other):
                    key=rec['path']
                    if key not in cache:
                        with Image.open(c.BASE/key) as im:
                            a=np.asarray(ImageOps.exif_transpose(im).convert('L'))
                            cache[key]=cv2.resize(a,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
                a,b=cache[r['path']],cache[other['path']]
                corr=float(np.corrcoef(a,b)[0,1]) if a.std()>1 and b.std()>1 else 0
                if corr>0.985:match=(other,corr);break
        if match:
            other,corr=match
            quarantine(r,'global_near_duplicate',{'duplicate_of':other['source']+'/'+other['id'],'duplicate_correlation':round(corr,6)});counts['near']+=1;continue
        seen[r['sha256']]=r;near.append((ph,dh,r))
    print('GLOBAL_DEDUP',dict(counts),flush=True)

def thumb(r):
    name=r['source']+'__'+r['id']+'.jpg';path=THUMBS/name
    if not path.exists():
        with Image.open(c.BASE/r['path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((420,300));im.save(path,quality=85)
    return {'key':r['source']+'/'+r['id'],'source':r['source'],'id':r['id'],'path':'../'+r['path'],'thumbnail':'thumbnails/'+name,'width':r['width'],'height':r['height'],'flags':r.get('quality_flags',[]),'hint':r.get('category_hint',''),'photographer':r.get('photographer',''),'source_page':r.get('source_page',''),'license':r.get('license','')}

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>摄影数据初筛 · 人工审核</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101318;color:#e8edf2;font:15px system-ui,"Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:2;background:#181e27ef;padding:16px 24px;border-bottom:1px solid #33404e}h1{font-size:22px;margin:0 0 8px}.note{color:#aebdc9;font-size:13px;margin:5px 0 12px}select,input,button{font:inherit;border:1px solid #435268;border-radius:7px;padding:7px 10px;background:#232d3b;color:inherit}button{cursor:pointer}button:hover{background:#364760}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}#stats{margin:10px 0 0;color:#8ddbc6}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;padding:20px 24px}.card{background:#1b222c;border:2px solid #303a47;border-radius:9px;overflow:hidden}.card[data-decision=reject]{border-color:#e87575}.card[data-decision=keep]{border-color:#5ed5a6}.card[data-decision=uncertain]{border-color:#e3b85e}.pic{height:210px;width:100%;object-fit:contain;background:#0b0e12;cursor:zoom-in}.info{padding:10px;overflow-wrap:anywhere}.small{font-size:12px;color:#abb9c8}.tags{color:#e8bd76;font-size:12px;min-height:18px}.actions{display:flex;gap:5px;margin-top:8px}.actions button{font-size:12px;padding:5px 8px}.nav{display:flex;justify-content:center;gap:15px;padding:14px}a{color:#90c9f6}.modal{position:fixed;inset:0;z-index:5;background:#080b0ff5;display:none;padding:14px}.modal.open{display:flex;flex-direction:column;align-items:center}.modal img{max-width:96vw;max-height:80vh;object-fit:contain;min-height:0}.modal .controls{margin:10px;justify-content:center}#detail{text-align:center;font-size:13px}.empty{padding:50px;text-align:center}</style>
<header><h1>摄影数据初筛 · 人工审核</h1><p class="note">本地审核副本；不上传图片。点击缩略图查看大图。标记暂存在当前浏览器，请定期导出审核 JSON。标记不会删除原图。</p><div class="controls"><select id="source"><option value="">全部来源</option></select><select id="decision"><option value="">全部审核状态</option><option value="unreviewed">未审核</option><option value="keep">保留</option><option value="reject">剔除</option><option value="uncertain">待定</option></select><select id="flag"><option value="">全部质量提示</option><option value="any">有提示</option></select><input id="search" placeholder="文件名、摄影师或类别"><button id="export">导出审核 JSON</button><button id="import">导入审核 JSON</button><input id="importfile" type="file" accept=".json" hidden></div><div id="stats"></div></header><div class="nav"><button class="prev">上一页</button><span class="pagenum"></span><button class="next">下一页</button></div><main id="grid" class="grid"></main><div class="nav"><button class="prev">上一页</button><span class="pagenum"></span><button class="next">下一页</button></div>
<div id="modal" class="modal"><div class="controls"><button id="close">关闭 Esc</button><button id="mPrev">← 上一张</button><button id="mNext">下一张 →</button><button data-mark="keep">1 保留</button><button data-mark="reject">2 剔除</button><button data-mark="uncertain">3 待定</button><button data-mark="unreviewed">清除标记</button><a id="original" target="_blank">打开原始下载图</a></div><img id="large"><p id="detail"></p></div>
<script>const DATA=__DATA__;const KEY='nju-photo-review-20261029-v1';let marks={},page=0,filtered=[],active=-1;try{marks=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}const $=s=>document.querySelector(s);const FLAG_NAMES={'low_detail_or_shallow_depth_of_field':'低细节／浅景深：需人工判断','large_dark_region':'较大暗部：可能是正常低调摄影','large_bright_region':'较大亮部：可能是正常高调摄影','monochrome_or_low_saturation':'黑白／低饱和：保留风格多样性'};for(const s of [...new Set(DATA.map(x=>x.source))])$('#source').add(new Option(s,s));for(const f of [...new Set(DATA.flatMap(x=>x.flags))])$('#flag').add(new Option(FLAG_NAMES[f]||f,f));function decision(r){return marks[r.key]||'unreviewed'}function persist(){try{localStorage.setItem(KEY,JSON.stringify(marks))}catch(e){alert('浏览器未能保存，请立即导出 JSON')}}function mark(key,value){if(value==='unreviewed')delete marks[key];else marks[key]=value;persist();render();if(active>=0)show(active)}function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text)e.textContent=text;return e}function render(){const src=$('#source').value,dec=$('#decision').value,flag=$('#flag').value,q=$('#search').value.toLowerCase();filtered=DATA.filter(x=>(!src||x.source===src)&&(!dec||decision(x)===dec)&&(!flag||(flag==='any'?x.flags.length:x.flags.includes(flag)))&&(!q||[x.id,x.photographer,x.hint].join(' ').toLowerCase().includes(q)));const pages=Math.max(1,Math.ceil(filtered.length/80));page=Math.min(page,pages-1);$('#grid').replaceChildren();filtered.slice(page*80,(page+1)*80).forEach((r,j)=>{const card=el('article','card');card.dataset.decision=decision(r);const img=el('img','pic');img.src=r.thumbnail;img.loading='lazy';img.alt=r.id;img.onclick=()=>show(page*80+j);card.append(img);const info=el('div','info');info.append(el('b','',r.source+' / '+r.id),el('div','small',r.width+' × '+r.height+' · '+(r.photographer||'作者见来源记录')),el('div','small',r.hint),el('div','tags',r.flags.map(f=>FLAG_NAMES[f]||f).join('；')));const controls=el('div','actions');[['keep','保留'],['reject','剔除'],['uncertain','待定'],['unreviewed','清除']].forEach(([v,t])=>{const b=el('button','',t);b.onclick=()=>mark(r.key,v);controls.append(b)});info.append(controls);card.append(info);$('#grid').append(card)});const count=v=>DATA.filter(r=>decision(r)===v).length;$('#stats').textContent=`总计 ${DATA.length} 张 · 当前筛选 ${filtered.length} 张 · 保留 ${count('keep')} · 剔除 ${count('reject')} · 待定 ${count('uncertain')} · 未审核 ${count('unreviewed')}`;document.querySelectorAll('.pagenum').forEach(e=>e.textContent=`${page+1} / ${pages}`);document.querySelectorAll('.prev').forEach(e=>e.disabled=page===0);document.querySelectorAll('.next').forEach(e=>e.disabled=page>=pages-1)}function show(i){if(!filtered.length)return;active=Math.max(0,Math.min(filtered.length-1,i));const r=filtered[active];$('#large').src=r.path;$('#original').href=r.path;$('#detail').textContent=r.source+' / '+r.id+' · '+r.width+'×'+r.height+' · '+decision(r)+' · '+r.license;$('#modal').classList.add('open')}function close(){$('#modal').classList.remove('open');active=-1}for(const s of ['#source','#decision','#flag','#search'])$(s).addEventListener(s==='#search'?'input':'change',()=>{page=0;render()});document.querySelectorAll('.prev').forEach(e=>e.onclick=()=>{page--;render();scrollTo(0,0)});document.querySelectorAll('.next').forEach(e=>e.onclick=()=>{page++;render();scrollTo(0,0)});$('#close').onclick=close;$('#mPrev').onclick=()=>show(active-1);$('#mNext').onclick=()=>show(active+1);document.querySelectorAll('[data-mark]').forEach(e=>e.onclick=()=>{const key=filtered[active]?.key;if(key)mark(key,e.dataset.mark)});document.addEventListener('keydown',e=>{if(e.target.matches('input,select'))return;if(e.key==='Escape')close();if(active<0)return;if(e.key==='ArrowRight')show(active+1);if(e.key==='ArrowLeft')show(active-1);if(['1','2','3'].includes(e.key)){const r=filtered[active];if(r)mark(r.key,{'1':'keep','2':'reject','3':'uncertain'}[e.key])}});$('#export').onclick=()=>{const payload={collection:'nju-photo-20261029-v1',exported_at:new Date().toISOString(),total:DATA.length,reviewed:DATA.filter(x=>marks[x.key]).map(x=>({key:x.key,source:x.source,id:x.id,path:x.path,decision:marks[x.key]}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='photo-review-'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};$('#import').onclick=()=>$('#importfile').click();$('#importfile').onchange=async e=>{try{const obj=JSON.parse(await e.target.files[0].text());const keys=new Set(DATA.map(x=>x.key));for(const r of obj.reviewed||[])if(keys.has(r.key)&&['keep','reject','uncertain'].includes(r.decision))marks[r.key]=r.decision;persist();render()}catch(err){alert('导入失败：'+err.message)}};render();</script></html>'''

def build():
    records=c.load_records();accepted=[r for r in records if r['status']=='accepted']
    accepted.sort(key=lambda r:(r['source'],r['id']))
    with cf.ThreadPoolExecutor(max_workers=8) as pool:gallery=list(pool.map(thumb,accepted))
    (OUT/'review.html').write_text(HTML.replace('__DATA__',json.dumps(gallery,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    (OUT/'accepted_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    fields=['source','id','path','width','height','bytes','sha256','category_hint','photographer','license','source_page','quality_flags']
    with (OUT/'accepted_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in accepted:w.writerow({**r,'quality_flags':'; '.join(r.get('quality_flags',[]))})
    counts=collections.Counter((r['source'],r['status']) for r in records)
    summary={'collection_date':'2026-10-29','accepted':len(accepted),'accepted_by_source':dict(collections.Counter(r['source'] for r in accepted)),'accepted_bytes':sum(r['bytes'] for r in accepted),'attempt_status_by_source':{s:{status:n for (source,status),n in counts.items() if source==s} for s in sorted({r['source'] for r in records})},'rejection_reasons':dict(collections.Counter(reason for r in records if r['status']=='rejected' for reason in r.get('reject_reasons',[]))),'quality_flag_counts':dict(collections.Counter(x for r in accepted for x in r.get('quality_flags',[]))),'resolution_min_short_edge':min(min(r['width'],r['height']) for r in accepted),'resolution_min_long_edge':min(max(r['width'],r['height']) for r in accepted),'semantic_filter':'Metadata prefilter and visual spot-check; no claim of exhaustive watermark/illustration detection','dedup':'SHA256 plus pHash<=8,dHash<=12 and 64x64 grayscale correlation>0.985 globally; not guaranteed crop/scene dedup','existing_personal_photos':'Pre-existing personal photos excluded from new collection and never modified by collection scripts; folder initially Live Photo, observed as Phone Photo at final inventory','human_review':'pending user review; flags do not mean rejection'}
    c.save_json(OUT/'summary.json',summary)
    # Source-separated manifests put provenance beside the downloaded files.
    for source in summary['accepted_by_source']:
        rs=[r for r in accepted if r['source']==source]
        (c.BASE/source/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
    # Stable small contact sheets permit a quick visual spot-check and offline overview.
    sheets=OUT/'contact_sheets';sheets.mkdir(exist_ok=True)
    for source in summary['accepted_by_source']:
        rs=[r for r in gallery if r['source']==source]
        rng=random.Random(c.SEED);rng.shuffle(rs)
        for start in range(0,len(rs),100):
            batch=rs[start:start+100];canvas=Image.new('RGB',(1600,math.ceil(len(batch)/10)*145),'#151a22');draw=ImageDraw.Draw(canvas)
            for k,r in enumerate(batch):
                with Image.open(OUT/r['thumbnail']) as im:
                    im.thumbnail((154,118));x=(k%10)*160;y=(k//10)*145;canvas.paste(im,(x+(154-im.width)//2,y));draw.text((x+2,y+120),r['id'][:20],fill='white')
            canvas.save(sheets/f'{source}_{start//100+1:03d}.jpg',quality=88)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    if '--visual' in sys.argv:
        latest={(r['source'],r['id']):r for r in c.load_records()}
        for decision in json.loads((c.META/'visual_rejections.json').read_text(encoding='utf-8')):
            r=latest.get((decision['source'],decision['id']))
            if r and r['status']=='accepted':quarantine(r,decision['reason'],{'visual_review_method':decision['review_method']})
    if '--audit' in sys.argv:audit()
    build()
