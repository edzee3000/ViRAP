"""Enrich album covers with the source playlist's album/artist mapping."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json
import collect_photos as c
import build_review as b

BASE=c.BASE/'Album_Covers'

def main():
    upstream=json.loads((BASE/'metadata/upstream_source-manifest.json').read_text(encoding='utf-8'))
    tracks=defaultdict(list)
    for track in upstream['tracks']:tracks[Path(track['file']).name].append(track)
    assets={Path(asset['file']).name:{**asset,'upstream_cover_url':url} for url,asset in upstream['assets'].items()}
    records=[json.loads(line) for line in (BASE/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    selection_path=BASE/'metadata/visual_photo_candidates.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else []
    selected={row['id']:row for row in selection}
    good=[];gallery=[];mismatches=[]
    for n,r in enumerate(records):
        if r['status']!='downloaded':continue
        matches=tracks.get(r['filename'],[])
        r['albums']=list(dict.fromkeys(t['album'] for t in matches))
        r['artists']=list(dict.fromkeys(a for t in matches for a in t['artists']))
        r['tracks']=[{'song':t['song'],'song_id':t['song_id'],'album_id':t['album_id'],'album':t['album'],'artists':t['artists']} for t in matches]
        asset=assets.get(r['filename'],{})
        r['upstream_cover_url']=asset.get('upstream_cover_url')
        r['upstream_manifest_sha256_match']=asset.get('sha256')==r['sha256']
        if asset and not r['upstream_manifest_sha256_match']:mismatches.append(r['id'])
        r['visual_preselection']='photographic_candidate' if r['id'] in selected else 'unclassified'
        if r.get('duplicate_of'):
            r['quality_flags']=list(dict.fromkeys(r['quality_flags']+['exact_duplicate_cover']))
        if r['id'] in selected:r['visual_review_note']=selected[r['id']]['note']
        good.append(r)
        if r.get('duplicate_of'):continue
        hint=('初步摄影候选；' if r['id'] in selected else '封面类型待审核；')+' / '.join(r['albums'])
        label=(' / '.join(r['albums']) or r['id'])+' · '+r['id']
        gallery.append({'key':'Album_Covers/'+r['id'],'source':'Album_Covers','id':label,'path':r['path'],'thumbnail':'review_thumbnails/'+r['id']+'.jpg','width':r['width'],'height':r['height'],'flags':r['quality_flags'],'hint':hint,'photographer':' / '.join(r['artists']),'source_page':r['source_page'],'license':r['license']})
    (BASE/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in good),encoding='utf-8')
    (BASE/'photographic_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in good if r['id'] in selected and not r.get('duplicate_of')),encoding='utf-8')
    fields=['id','filename','path','albums','artists','width','height','bytes','sha256','quality_flags','visual_preselection','upstream_cover_url','source_page']
    with (BASE/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for r in good:writer.writerow({**r,'albums':' / '.join(r['albums']),'artists':' / '.join(r['artists']),'quality_flags':'; '.join(r['quality_flags'])})
    page=b.HTML.replace('__DATA__',json.dumps(gallery,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('nju-photo-review-20261029-v1','nju-album-review-20261029-v1').replace('摄影数据初筛 · 人工审核','专辑封面 · 摄影候选审核').replace('文件名、摄影师或类别','专辑、歌手、文件名或摄影候选')
    page=page.replace("'monochrome_or_low_saturation':'黑白／低饱和：保留风格多样性'", "'monochrome_or_low_saturation':'黑白／低饱和：保留风格多样性','cover_below_photo_resolution':'短边不足720：建议先作参考图候选','cover_very_low_resolution':'短边不足400：低分辨率封面'")
    page=page.replace('<button id="export">','<button onclick="setAlbumView(true)">只看初步摄影候选</button><button onclick="setAlbumView(false)">全部封面</button><button id="export">')
    page=page.replace('</script></html>',";window.setAlbumView=(onlyPhotos)=>{for(const sel of ['#source','#decision','#flag'])document.querySelector(sel).value='';document.querySelector('#search').value=onlyPhotos?'初步摄影候选':'';page=0;render()};</script></html>")
    (BASE/'review.html').write_text(page,encoding='utf-8')
    summary=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
    summary.update({'album_names_mapped':sum(bool(r['albums']) for r in good),'upstream_sha_mismatches':mismatches,'review_gallery_unique_images':len(gallery),'photographic_candidates_visual':sum(r['id'] in selected and not r.get('duplicate_of') for r in good),'visual_preselection_scope':'Conservative thumbnail-based candidate selection; all original files retained; unselected means unclassified, not rejected'})
    c.save_json(BASE/'summary.json',summary)
    readme=f'''# 喜爱专辑的封面素材

从用户指定的Hugging Face目录下载：
https://huggingface.co/datasets/edzee3000/GithubData/tree/main/homepage-assets/music/atlas/originals

- 仓库固定版本：`{summary['revision']}`。
- 下载 {summary['downloaded']} / {summary['requested']} 张，总计 {summary['bytes']/1e6:.1f} MB。全部文件保留原名、原尺寸和原始字节；未裁切、重压缩或去文字。
- `originals/` 为原文件；`review.html`为本地人工审核页；`manifest.csv`可按专辑、歌手、尺寸查看。
- 718份文件中有{summary['unique_sha256']}份独立图像，{summary['downloaded']-summary['unique_sha256']}份为字节完全一致的重复封面。全部原文件保留，审核页默认只展示{len(gallery)}张去重图；重复对应关系记录在manifest中。
- {summary['album_names_mapped']}张匹配到了上游歌单的专辑/歌手信息。信息来自source-manifest，可能一张封面对应多首曲目或多个条目；没有把表演者误当封面摄影师。
- 短边尺寸分布：{json.dumps(summary['resolution_short_edge_bins'],ensure_ascii=False)}。低分辨率仅作标记，没有放大图片或直接删除。
- 原始718条来源文件清单、SHA256、图片解码与尺寸逐图核验；每个有LFS SHA256的文件都与仓库记录一致。
- 初步视觉摄影候选 {summary['photographic_candidates_visual']} 张，名单在`photographic_candidates.jsonl`。这是保守的缩略图初选，并非最终接受集；未选入的封面仍待判断。
- 审核页可搜索专辑、歌手和“初步摄影候选”；点击图片查看大图，1保留、2剔除、3待定，左右键翻图。网页标记不删除文件，请导出审核JSON保存进度。
- `contact_sheets/`按manifest顺序提供带编号总览。原图名是来源文件哈希，未依据专辑名重命名，以免丢失映射。

封面可能包含摄影、插画、文字、拼贴或组合设计。大面积文字、边框和拼贴会影响颜色统计，不能默认整张封面都适合作摄影调色输入。初选只标记候选，不自动去字、裁切或重绘。

本目录独立于上一批5,000张摄影候选，未修改其图片、清单和审核状态。专辑图像属于各原始创作者；上游README明确说明了这一点，未提供统一训练或再分发授权。本地记录来源，不把公开下载等同于开源图片许可。
'''
    (BASE/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='exact_duplicates'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
