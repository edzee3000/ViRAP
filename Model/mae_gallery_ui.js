// Fixed checkpoint galleries. Validation above, training below. Read-only.
let galleryRun='',galleryPages={validation:0,train:0},galleryKeys={},galleryPending={},galleryRequests={};
const gallerySize=6;
const galleryEscape=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function showGalleryImage(url,caption){
  const dialog=document.getElementById('galleryZoom');
  document.getElementById('zoomCaption').textContent=caption;
  document.getElementById('zoomImage').src=url;dialog.showModal();
}
async function loadGallery(run,split){
  const summary=run.gallery||{},page=galleryPages[split];
  const key=[run.name,summary.version,summary.ready?.[split]||0,page].join('|');
  if(galleryKeys[split]===key||galleryPending[split]===key)return;
  galleryPending[split]=key;const request=(galleryRequests[split]||0)+1;galleryRequests[split]=request;
  const host=document.getElementById(split+'Gallery'),info=document.getElementById(split+'GalleryInfo');
  try{
    const response=await fetch('/api/gallery?run='+encodeURIComponent(run.name)+'&split='+split+'&page='+page+'&size='+gallerySize,{cache:'no-store'});
    if(!response.ok)throw Error('HTTP '+response.status);
    const data=await response.json();
    if(request!==galleryRequests[split]||selected!==run.name||galleryPages[split]!==page)return;
    const label=split==='validation'?'验证集':'训练集';
    const pages=Math.max(1,Math.ceil(data.total/gallerySize));
    document.getElementById(split+'Page').textContent=`第 ${page+1} / ${pages} 页`;
    document.getElementById(split+'Prev').disabled=page===0;
    document.getElementById(split+'Next').disabled=(page+1)*gallerySize>=data.ready;
    info.textContent=data.version?`${label}固定样例 · 检查点第 ${data.checkpoint_step.toLocaleString()} 步${data.checkpoint_kind==='best'?'（最优断点）':''} · 已生成 ${data.ready}/${data.total} 组 · ${new Date(data.updated_unix*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'})}`:`正在从已保存检查点生成${label}样例，训练照常运行。`;
    host.innerHTML=data.rows.map((row,index)=>{
      const caption=`${label} ${row.pair_id} · 检查点 ${data.checkpoint_step}`;
      const cells=[['content','输入图'],['reference','参考图'],['target','生成目标'],['output','模型输出']].map(([role,title])=>`<figure><figcaption>${title}</figcaption><button class="gallery-image" type="button" data-url="${galleryEscape(row[role+'_url'])}" data-caption="${galleryEscape(caption+' / '+title)}" aria-label="放大${title}"><img loading="lazy" src="${galleryEscape(row[role+'_url'])}" alt="${galleryEscape(title+' '+row.pair_id)}"></button></figure>`).join('');
      return `<article class="sample-row"><div class="sample-label"><strong>${page*gallerySize+index+1}. ${galleryEscape(row.pair_id)}</strong><span>${galleryEscape(row.content_source)} → ${galleryEscape(row.reference_source)}</span></div><div class="sample-grid">${cells}</div></article>`;
    }).join('');
    if(!data.rows.length)host.innerHTML='<p class="muted">预览进程正在生成更多图片，请稍候；页面会自动更新。</p>';
    host.querySelectorAll('.gallery-image').forEach(button=>button.onclick=()=>showGalleryImage(button.dataset.url,button.dataset.caption));
    galleryKeys[split]=key;
    if(split==='validation'){
      const legacy=document.getElementById('preview');legacy.hidden=data.rows.length>0||!run.status.preview;
      if(!legacy.hidden){const source='/preview/'+encodeURIComponent(run.name)+'?v='+run.valid.at(-1)?.[0];if(legacy.getAttribute('src')!==source)legacy.src=source;}
    }
  }catch(error){if(selected===run.name)info.textContent='样例暂时无法载入，稍后自动重试。';}
  finally{if(galleryPending[split]===key)galleryPending[split]='';}
}
function renderGalleries(run){
  if(galleryRun!==run.name){galleryRun=run.name;galleryPages={validation:0,train:0};galleryKeys={};}
  loadGallery(run,'validation');loadGallery(run,'train');
}
for(const split of ['validation','train']){
  document.getElementById(split+'Prev').onclick=()=>{galleryPages[split]=Math.max(0,galleryPages[split]-1);galleryKeys[split]='';const run=state?.runs.find(r=>r.name===selected);if(run)loadGallery(run,split);};
  document.getElementById(split+'Next').onclick=()=>{galleryPages[split]++;galleryKeys[split]='';const run=state?.runs.find(r=>r.name===selected);if(run)loadGallery(run,split);};
}
document.getElementById('zoomClose').onclick=()=>document.getElementById('galleryZoom').close();
document.getElementById('galleryZoom').onclick=event=>{if(event.target===event.currentTarget)event.currentTarget.close();};
