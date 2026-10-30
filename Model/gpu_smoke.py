"""Three-step real-data CUDA compatibility check, not a quality experiment."""
import argparse,json,socket,time
from pathlib import Path
import torch
import numpy as np
from tone_model.data import load_triplets,TripletDataset,read_rgb,image_tensor,shrink
from tone_model.model import ToneTransferModel
from tone_model.losses import ToneLoss
from train import guard_experiment_paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    guard_experiment_paths(args.manifest,args.output)
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(17)
    records=load_triplets(args.manifest)
    dataset=TripletDataset(records,'train',256,256,verify_files=False);sample=dataset[0]
    x=sample['content'][None].cuda();r=sample['reference'][None].cuda();target=sample['target'][None].cuda()
    model=ToneTransferModel().cuda();criterion=ToneLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    losses=[];times=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad()
        prediction,aux=model(x,r,True);pieces=criterion(prediction,target,x,aux)
        assert torch.isfinite(pieces['total']);pieces['total'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(pieces['total'].detach()))
    train_peak=torch.cuda.max_memory_allocated()/1024**2
    model.eval();source=read_rgb(dataset.records[0]['_content_path'])
    native=torch.from_numpy(np.array(source)).permute(2,0,1)[None]
    with torch.inference_mode():
        plan=model.prepare(image_tensor(shrink(source,256))[None].cuda(),r)
        torch.cuda.reset_peak_memory_stats();rows=0
        for start,end,block in model.iter_render_rows(native,plan):
            assert block.device.type=='cpu' and torch.isfinite(block).all();rows+=end-start
        assert rows==source.height
    result={'node':socket.gethostname(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'parameters':sum(p.numel() for p in model.parameters()),'real_triplet':sample['pair_id'],
            'debug_steps':3,'debug_losses':losses,'debug_step_seconds':times,'training_peak_allocated_mib':train_peak,
            'native_dimensions':[source.width,source.height],'streaming_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'finite_gradients':True,'warning':'Compatibility only, one training photo, no convergence/generalization/quality conclusion. No checkpoint published.'}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
