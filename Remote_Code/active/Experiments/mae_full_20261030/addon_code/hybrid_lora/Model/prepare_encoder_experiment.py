"""Compute-local preparation and public frozen-feature extraction for the probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train import guard_experiment_paths
from tone_model.data import load_triplets,read_rgb


WEIGHTS={
 'dinov2_s':('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',None),
 'dinov2_b':('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',None),
 'dinov3_s':('https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/resolve/3bf4720a82ec2066db88137180ff1f83a675cef0/model.safetensors','2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'),
 'mae_b':('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',None),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def prepare_images(root,cache,threads=8):
    manifest=root/'Dataset/Photo_Triplets_v1/triplets.json'
    records=load_triplets(manifest)
    if any(r['generation']['max_image_edge']!=768 for r in records):raise ValueError('Unexpected generator geometry')
    fingerprint=hashlib.sha256(json.dumps([{k:r[k] for k in ('pair_id','split','content_key','reference_key','output_sha256')} for r in records],sort_keys=True).encode()).hexdigest()
    marker=cache/'images.json'
    if marker.exists():
        meta=json.loads(marker.read_text())
        if meta['fingerprint']!=fingerprint:raise ValueError('Cache manifest changed')
        print('Images already prepared',flush=True);return
    sources={}
    for r in records:
        for role in ('content','reference'):sources[r[role+'_key']]=str(r['_'+role+'_path'])
    keys=sorted(sources);indices={k:i for i,k in enumerate(keys)}
    photos=np.lib.format.open_memmap(cache/'photos.npy',mode='w+',dtype='uint8',shape=(len(keys),2,3,224,224))
    render=np.lib.format.open_memmap(cache/'render.npy',mode='w+',dtype='uint8',shape=(len(keys),3,128,128))
    def source(index):
        image=read_rgb(sources[keys[index]])
        reference=image.resize((224,224),Image.Resampling.LANCZOS)
        image.thumbnail((768,768),Image.Resampling.LANCZOS)
        photos[index,0]=np.asarray(image.resize((224,224),Image.Resampling.LANCZOS)).transpose(2,0,1)
        photos[index,1]=np.asarray(reference).transpose(2,0,1)
        render[index]=np.asarray(image.resize((128,128),Image.Resampling.LANCZOS)).transpose(2,0,1)
    with ThreadPoolExecutor(threads) as pool:
        for n,_ in enumerate(pool.map(source,range(len(keys))),1):
            if n%500==0:print('source_images',n,len(keys),flush=True)
    photos.flush();render.flush()
    targets=np.lib.format.open_memmap(cache/'targets.npy',mode='w+',dtype='uint8',shape=(len(records),3,128,128))
    def target(index):
        row=records[index];image=read_rgb(row['_target_path'])
        # Full raster shape is checked against recorded generator dimensions;
        # content is EXIF-transposed and teacher-thumbnail aligned above.
        if image.size!=(row['width'],row['height']):raise ValueError('Target raster mismatch '+row['pair_id'])
        targets[index]=np.asarray(image.resize((128,128),Image.Resampling.LANCZOS)).transpose(2,0,1)
    with ThreadPoolExecutor(threads) as pool:
        for n,_ in enumerate(pool.map(target,range(len(records))),1):
            if n%5000==0:print('target_images',n,len(records),flush=True)
    targets.flush()
    save_json(marker,{'fingerprint':fingerprint,'source_keys':keys,'source_paths':[sources[k] for k in keys],
       'photo_size':224,'render_size':128,'photo_roles':['content_teacher_thumbnail768','reference_original'],
       'geometry':'Full image square resize (no crop), fixed low-resolution representation probe; NOT full-resolution final evaluation',
       'rows':[{'pair_id':r['pair_id'],'split':r['split'],'content_key':r['content_key'],'reference_key':r['reference_key'],
                'content_index':indices[r['content_key']],'reference_index':indices[r['reference_key']]} for r in records]})


def load_backbone(name,root,cache):
    import timm
    from safetensors.torch import load_file
    url,expected=WEIGHTS[name];weights=cache/(name+('.safetensors' if name=='dinov3_s' else '.pth'))
    if not weights.exists():
        temp=weights.with_suffix(weights.suffix+'.download')
        with urllib.request.urlopen(url,timeout=90) as source,temp.open('wb') as dest:
            while True:
                data=source.read(4*1024*1024)
                if not data:break
                dest.write(data)
        temp.replace(weights)
    sha=digest(weights)
    if expected and sha!=expected:raise ValueError('Public mirror SHA256 mismatch')
    state=load_file(str(weights)) if name=='dinov3_s' else torch.load(weights,map_location='cpu',weights_only=True)
    if name.startswith('dinov2'):
        sys.path.insert(0,str(root/'Research/External_Code/dinov2'))
        from dinov2.hub import backbones
        model=getattr(backbones,'dinov2_vits14' if name.endswith('_s') else 'dinov2_vitb14')(pretrained=False)
        model.load_state_dict(state,strict=True)
    elif name=='dinov3_s':
        model=timm.create_model('vit_small_patch16_dinov3.lvd1689m',pretrained=False,num_classes=0,img_size=224)
        model.load_state_dict(state,strict=True)
    else:
        model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
        encoder={k:v for k,v in state['model'].items() if not k.startswith('decoder') and k!='mask_token'}
        model.load_state_dict(encoder,strict=True)
    model.eval().requires_grad_(False)
    return model,{'url':url,'sha256':sha,'weight_file':str(weights),'pretrained':True,'parameters':sum(p.numel() for p in model.parameters()),
                  'implementation':'official DINOv2' if name.startswith('dinov2') else 'timm 1.0.22; official MAE encoder state only' if name=='mae_b' else 'timm 1.0.22; timm public DINOv3 weight conversion'}


@torch.inference_mode()
def extract_features(name,root,cache,batch_size,device):
    marker=cache/(name+'.json');image_meta=json.loads((cache/'images.json').read_text())
    if marker.exists():
        meta=json.loads(marker.read_text())
        if meta['image_fingerprint']!=image_meta['fingerprint']:raise ValueError('Features refer to different images')
        print('Feature cache already complete',name,flush=True);return
    model,provenance=load_backbone(name,root,cache);model=model.to(device)
    photos=np.load(cache/'photos.npy',mmap_mode='r');count=photos.shape[0];dim=384 if name.endswith('_s') else 768
    bank=np.lib.format.open_memmap(cache/(name+'.npy'),mode='w+',dtype='float16',shape=(count,2,3,197,dim))
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    stages={};handles=[]
    if not name.startswith('dinov2'):
        for index in (3,7,11):
            def capture(module,args,value,index=index):stages[index]=value
            handles.append(model.blocks[index].register_forward_hook(capture))
    started=time.monotonic()
    for role in range(2):
        for start in range(0,count,batch_size):
            end=min(count,start+batch_size)
            image=torch.from_numpy(np.array(photos[start:end,role])).to(device).float()/255
            image=(image-mean)/std
            if name.startswith('dinov2'):
                extracted=model.get_intermediate_layers(image,n=(3,7,11),return_class_token=True,norm=True)
            else:
                model.forward_features(image)
                prefix=model.num_prefix_tokens
                extracted=[]
                for index in (3,7,11):
                    layer=model.norm(stages[index]);extracted.append((layer[:,prefix:],layer[:,0]))
            outputs=[]
            for patch,cls in extracted:
                side=int(patch.shape[1]**.5)
                if side*side!=patch.shape[1]:raise ValueError('Patch grid is not square')
                patch=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(image.shape[0],dim,side,side),(14,14)).flatten(2).transpose(1,2)
                outputs.append(torch.cat([cls[:,None],patch],1))
            bank[start:end,role]=torch.stack(outputs,1).half().cpu().numpy()
            if start%(batch_size*20)==0:print('features',name,role,end,count,round(time.monotonic()-started,1),flush=True)
    bank.flush()
    for handle in handles:handle.remove()
    save_json(marker,{**provenance,'image_fingerprint':image_meta['fingerprint'],'shape':list(bank.shape),
                     'feature_dim':dim,'layers':[4,8,12],'input_size':224,'patch_grid_after_pooling':[14,14],
                     'extraction_dtype':'float32 forward, float16 storage','elapsed_seconds':time.monotonic()-started})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--backbone',choices=list(WEIGHTS));ap.add_argument('--images',action='store_true');ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--device',default='cuda');ap.add_argument('--threads',type=int,default=8);args=ap.parse_args()
    guard_experiment_paths(args.root,args.cache);args.cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.images:prepare_images(args.root,args.cache,args.threads)
    if args.backbone:extract_features(args.backbone,args.root,args.cache,args.batch_size,args.device)


if __name__=='__main__':main()
