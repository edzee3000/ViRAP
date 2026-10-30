"""Inventory every remaining original image; render an offline default-keep single-image reviewer."""
from pathlib import Path
from collections import Counter
import concurrent.futures as cf,csv,hashlib,json,urllib.parse
from PIL import Image,ImageOps
import collect_photos as c
from apply_review_deletions import raw_images

OUT=c.BASE/'_collection'
PREVIEWS=OUT/'review_previews'
PREVIEWS.mkdir(exist_ok=True)
SOURCE_ORDER={'DIV2K':0,'Unsplash_Lite':1,'Wikimedia_Commons':2,'Album_Covers':3,'Phone Photo':4}

def main():
    known={r['path']:r for r in c.load_records() if r.get('path')}
    album_manifest=c.BASE/'Album_Covers'/'manifest.jsonl'
    if album_manifest.exists():
        for line in album_manifest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);known['Album_Covers/'+r['path']]=r
    paths=list(raw_images())
    def row(path):
        resolved=path.resolve();resolved.relative_to(c.BASE.resolve());relative=path.relative_to(c.BASE).as_posix();parts=path.relative_to(c.BASE).parts
        record=known.get(relative,{})
        if parts[0]=='_collection':
            source=parts[2];area='此前自动隔离的原图' if parts[1]=='rejected' else '备用原图'
        else:
            source=parts[0];area='专辑封面原文件' if source=='Album_Covers' else '个人手机照片' if source=='Phone Photo' else '已收集的摄影原图'
        ident=record.get('id',path.stem);key=source+'/'+ident
        width,height=record.get('width'),record.get('height')
        digest=record.get('sha256');flags=list(record.get('quality_flags',[]))
        if not width or not height or not digest:
            with Image.open(path) as im:width,height=im.size
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
        href='../'+urllib.parse.quote(relative,safe='/')
        preview=href
        if max(width,height)>3200:
            destination=PREVIEWS/(hashlib.sha256((relative+digest).encode('utf-8')).hexdigest()[:24]+'.jpg')
            if not destination.exists():
                with Image.open(path) as im:
                    icc=im.info.get('icc_profile') if im.mode in ('RGB','RGBA') else None
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((3200,3200),Image.Resampling.LANCZOS)
                    options={'quality':93}
                    if icc:options['icc_profile']=icc
                    im.save(destination,**options)
            preview='review_previews/'+destination.name
        title=' / '.join(record.get('albums',[])) or path.name
        artists=' / '.join(record.get('artists',[])) or record.get('photographer',record.get('author',''))
        hint=record.get('category_hint','')
        if source=='Album_Covers':hint=('初步摄影候选' if record.get('visual_preselection')=='photographic_candidate' else '封面类型待审核')+'；'+hint
        if record.get('duplicate_of'):flags.append('有重复对应图：'+record['duplicate_of'])
        if record.get('reject_reasons'):flags.append('此前自动初筛理由：'+'；'.join(record['reject_reasons']))
        return {'key':key,'source':source,'id':ident,'path':relative,'href':href,'preview':preview,'width':width,'height':height,'sha256':digest,'bytes':path.stat().st_size,'area':area,'title':title,'artists':artists,'hint':hint,'flags':flags,'source_page':record.get('source_page',record.get('file_page','')),'default_decision':'keep'}
    with cf.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(row,paths))
    rows.sort(key=lambda r:(SOURCE_ORDER.get(r['source'],99),r['id'],r['path']))
    # Rare same-stem files in a personal folder must retain distinct review keys.
    groups=Counter(r['key'] for r in rows)
    for r in rows:
        if groups[r['key']]>1:r['key']+='__'+hashlib.sha256(r['path'].encode('utf-8')).hexdigest()[:8]
    assert len({r['key'] for r in rows})==len(rows)
    seed=[];review_files=[]
    for p in sorted((c.ROOT/'Dataset'/'Review Datas').rglob('*.json')):
        obj=json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(obj,dict):
            seed.extend(obj.get('seen_keys',[]))
            seed.extend(r['key'] for r in obj.get('reviewed',[]) if r.get('decision')=='keep');review_files.append(str(p.resolve()))
    config={'seedSeen':seed,'reviewFilesApplied':review_files,'defaultDecision':'keep'}
    template=(Path(__file__).parent/'unified_review.html').read_text(encoding='utf-8')
    page=template.replace('__DATA__',json.dumps(rows,ensure_ascii=False).replace('</','<\\/')).replace('__CONFIG__',json.dumps(config,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'review.html').write_text(page,encoding='utf-8')
    (OUT/'all_images_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    fields=['key','source','id','path','title','artists','width','height','area','default_decision','sha256','bytes']
    with (OUT/'all_images_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={'total_original_images':len(rows),'by_source':dict(Counter(r['source'] for r in rows)),'by_area':dict(Counter(r['area'] for r in rows)),'default_decision':'keep','automatic_quality_deletions_in_this_run':0,'includes_prior_quarantine_and_reserve':True,'includes_all_album_files_even_exact_duplicates':True,'derived_thumbnails_and_contact_sheets_excluded':True,'missing_images':[],'large_preview_count':sum(r['preview']!=r['href'] for r in rows),'review_page':'review.html'}
    missing=[r['path'] for r in rows if not (c.BASE/r['path']).is_file()]
    if missing:raise RuntimeError('Files changed during inventory: '+repr(missing))
    c.save_json(OUT/'all_images_summary.json',summary)
    # Existing entrypoints lead to the same reviewer and the same local progress.
    redirect='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐张图片审核</title><meta http-equiv="refresh" content="0;url={url}"><p><a href="{url}">打开全部图片逐张审核页</a></p></html>'
    (c.ROOT/'Dataset'/'review.html').write_text(redirect.format(url='Original%20Data/_collection/review.html'),encoding='utf-8')
    (c.BASE/'Album_Covers'/'review.html').write_text(redirect.format(url='../_collection/review.html'),encoding='utf-8')
    (OUT/'逐张审核说明.md').write_text(f'''# 全部图片逐张审核

当前共有 {len(rows)} 张实际原图，范围包括各摄影集、全部专辑封面文件、手机照片，以及此前自动隔离和备用区中的原图。缩略图和拼图不重复纳入。各图片仍保留在原目录中。

默认全部保留。无需点击“保留”，使用右箭头或空格看下一张即可。只有不想要的图片才按X/Delete或点击“剔除并下一张”。左箭头返回，Ctrl+Z撤销，F全屏；“放大”和“查看原图”用于核查细节。列表到末尾会停止，不会自动从头循环。

完整构图以适应窗口方式显示，不裁切。超过3200像素的图使用保持完整构图的大尺寸预览来避免卡顿，原文件不变，可随时打开原图。

位置、已浏览记录和剔除标记保存在当前浏览器。请定期导出审核JSON；导出只列出明确剔除项，并声明其他图片默认保留。网页按钮不会即时删除磁盘图片，之后应用该JSON时才删除。可导入之前的JSON继续审核。

已按用户提交的审核JSON删除明确reject的原图，其余未提及图片全部保留。每批操作记录见_collection/review_changes_*/applied_deletions.json，历史记录分别保存。

全量清单：all_images_manifest.jsonl和all_images_manifest.csv。它与只记录原摄影集的accepted_manifest.jsonl范围不同，后者不包含手机、封面、隔离和备用原图。
''',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
