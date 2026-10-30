"""Apply the train/validation gallery layout without changing trainer code."""
from pathlib import Path
import re

p=Path(__file__).resolve().parents[1]/'Model/mae_training_dashboard.html'
text=p.read_text(encoding='utf-8')
if 'id="validationGallery"' not in text:
    panels=[]
    for split,title,note in [('validation','验证样例','固定验证集，未参与参数更新。'),('train','训练样例','固定训练集样本，用于观察拟合情况，不能作为泛化证据。')]:
        panels.append(f'''<section class="panel gallery-panel"><div class="gallery-title"><div><strong>{title}</strong><p>{note} 每组48对，按来源分散选取。点击图片放大。</p></div><div class="gallery-pager"><button id="{split}Prev" type="button">上一页</button><span id="{split}Page">第1页</span><button id="{split}Next" type="button">下一页</button></div></div><p id="{split}GalleryInfo">正在读取检查点样例…</p><div id="{split}Gallery"></div>{'<img id="preview" alt="已有验证预览" hidden>' if split=='validation' else ''}</section>''')
    pattern=r'(<div class="runs" id="runs"></div>).*?(<div class="foot" id="updated">)'
    text,count=re.subn(pattern,lambda match:match.group(1)+''.join(panels)+match.group(2),text,flags=re.S)
    assert count==1
    old="if(s.preview){$('preview').hidden=false;$('preview').src='/preview/'+encodeURIComponent(run.name)+'?v='+run.valid.at(-1)?.[0];}else $('preview').hidden=true;"
    assert old in text;text=text.replace(old,'renderGalleries(run);')
    text=text.replace('<script>','<script src="/gallery-ui.js"></script>\n<script>',1)
    css='''.gallery-panel{margin-top:22px}.gallery-title{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.gallery-title strong{font-size:20px}.gallery-pager{display:flex;align-items:center;gap:10px;white-space:nowrap}.gallery-pager button:disabled{opacity:.35;cursor:default}.sample-row{border-top:1px solid var(--line);padding:18px 0}.sample-label{display:flex;justify-content:space-between;gap:14px;margin-bottom:10px;font-size:12px}.sample-label span{color:var(--muted)}.sample-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.sample-grid figure{margin:0;min-width:0}.sample-grid figcaption{color:var(--muted);margin:0 0 6px;font-size:12px}.gallery-image{padding:0;display:block;width:100%;height:230px;background:#0c141f;border:1px solid #253449;overflow:hidden;cursor:zoom-in}.gallery-image img{width:100%;height:100%;object-fit:contain;display:block}dialog{border:1px solid #405672;border-radius:14px;background:#111b29;color:#e9f1fc;max-width:95vw;max-height:95vh;padding:20px}dialog::backdrop{background:#03070de6}#zoomImage{display:block;max-width:88vw;max-height:78vh;min-width:320px;object-fit:contain;margin:auto}#zoomCaption{margin:0 45px 14px 0}#zoomClose{position:absolute;right:12px;top:12px;font-size:18px}.gallery-help{color:var(--muted);font-size:12px}@media(max-width:650px){.gallery-title{flex-direction:column}.gallery-pager{width:100%;justify-content:space-between}.sample-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.gallery-image{height:205px}.sample-label{flex-direction:column;gap:2px}#zoomImage{min-width:0;max-width:80vw}.sample-row{padding:14px 0}}'''
    text=text.replace('</style>',css+'\n</style>')
    text=text.replace('</main>','<dialog id="galleryZoom"><button id="zoomClose" aria-label="关闭">×</button><p id="zoomCaption"></p><img id="zoomImage" alt="放大预览"><p class="gallery-help">样例来自标注步数的检查点推理；图片保持完整比例。</p></dialog></main>')
    p.write_text(text,encoding='utf-8')
print('two_split_gallery_layout_ready')
