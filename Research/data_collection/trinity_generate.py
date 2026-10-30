"""Compute-node-only black-box image generation; no encoder features are exported."""
import argparse,hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--run',default='pilot');ap.add_argument('--limit',type=int);args=ap.parse_args()
    assert socket.gethostname().startswith('trinity-2-23'),socket.gethostname()
    root=Path(args.root).resolve();assert str(root).startswith('/tmp/shihangz/')
    import torch
    sys.path.insert(0,str(root/'Runtime'/'petal'))
    from infer import load_models,inference
    torch.set_num_threads(4);torch.manual_seed(20261029)
    assert torch.cuda.is_available()
    device=torch.device('cuda');dtype=torch.float32
    manifest=Path(args.manifest);data=json.loads(manifest.read_text());pairs=data['pairs']
    if args.limit:pairs=pairs[:args.limit]
    pairs=pairs[args.worker::args.workers]
    out=root/'Dataset'/'Generated_Outputs';out.mkdir(exist_ok=True)
    logs=root/'Logs'/args.run;logs.mkdir(parents=True,exist_ok=True)
    log=logs/f'worker_{args.worker}.jsonl';summary=logs/f'worker_{args.worker}_summary.json'
    model,head=load_models(str(root/'Runtime/weights/ckpts/style_encoder.pt'),str(root/'Runtime/weights/ckpts/lut_model.pt'),device,dtype)
    def load(path):
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');original=im.size
            im.thumbnail((768,768),Image.Resampling.LANCZOS)
            arr=np.asarray(im,dtype=np.float32)/255.
        return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device),original
    done=failed=skipped=0;start=time.time()
    for i,p in enumerate(pairs):
        target_relative=p.get('planned_target_path') or f"../Generated_Outputs/{p['split']}/{int(p['pair_id'].split('_')[-1])//1000:04d}/{p['pair_id']}.png"
        target=(manifest.parent/target_relative).resolve();target.relative_to(out)
        if target.exists():
            with Image.open(target) as im:im.verify()
            skipped+=1;continue
        began=time.time()
        try:
            # Keep a large reserve on tmpfs; stop rather than consume shared RAM unchecked.
            if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<150*1024**3:raise RuntimeError('Local free space below 150GiB reserve')
            content,original=load((manifest.parent/p['content_path']).resolve());reference,_=load((manifest.parent/p['reference_path']).resolve())
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():pred=inference(model,head,content,reference,guidance_scale=1.,style_input_size=512)
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite model output')
            a=(pred[0].float().permute(1,2,0).cpu().numpy().clip(0,1)*255+.5).astype(np.uint8)
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.tmp')
            Image.fromarray(a).save(temporary,format='PNG',compress_level=3);temporary.replace(target)
            record={**p,'target_path':target_relative,'status':'generated_unreviewed','width':a.shape[1],'height':a.shape[0],'original_content_size':original,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-began,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'mean_abs_input_change':float((pred-content).abs().mean().item()),'output_std':float(a.std()),'worker':args.worker,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'generation':{'provider':'PETAL released model','dtype':'float32','max_image_edge':768,'style_input_size':512,'guidance_scale':1.,'exif_transpose':True,'rgb_conversion':'Pillow convert RGB; embedded ICC not transformed'}}
            done+=1
        except Exception as e:
            failed+=1;record={**p,'status':'failed','error':repr(e),'traceback':traceback.format_exc()};torch.cuda.empty_cache()
        with log.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if (i+1)%10==0 or record['status']=='failed':print(f'worker={args.worker} {i+1}/{len(pairs)} done={done} failed={failed} {time.time()-start:.1f}s',flush=True)
        summary.write_text(json.dumps({'worker':args.worker,'total':len(pairs),'done':done,'skipped':skipped,'failed':failed,'elapsed_seconds':time.time()-start,'finished':i+1==len(pairs)}))
        if failed>=3:raise RuntimeError('Three failures; stop this worker for investigation')
    if failed:raise RuntimeError(f'{failed} generation failures')

if __name__=='__main__':main()
