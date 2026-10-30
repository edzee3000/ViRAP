"""Target-access oracle diagnostic, NOT a deployed method or benchmark score.

Fits a global LUT and optionally a free low-resolution RGB field per image.
The local field is a loose representation diagnostic, not the exact model class.
Fitting and held-out spatial blocks are separated; only train/validation allowed.
"""
import argparse,json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tone_model.data import TripletDataset,load_triplets
from tone_model.model import identity_lut,sample_lut,color_basis
from train import guard_experiment_paths


def fit_oracle(content,target,mode='color',steps=100,lut_size=9,field_size=16,seed=17):
    torch.manual_seed(seed)
    identity=identity_lut(lut_size).to(content)
    delta=nn.Parameter(torch.zeros_like(identity));field=nn.Parameter(torch.zeros(1,3,field_size,field_size,device=content.device))
    params=[delta] if mode=='global' else [delta,field]
    optimizer=torch.optim.Adam(params,lr=.03)
    height,width=content.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(height,device=content.device),torch.arange(width,device=content.device),indexing='ij')
    mask=(((yy//16+xx//16)%2)==0)[None,None].to(content)
    if not (1-mask).sum():raise ValueError('Need at least two held-out spatial blocks')
    basis=color_basis(content.detach()).flatten(1,2)
    with torch.no_grad():inverse=torch.linalg.pinv(basis,rtol=1e-4)
    def predict():
        global_delta=.3*delta.tanh();result=sample_lut(identity+global_delta,content)
        if mode!='global':
            raw=.1*F.interpolate(field.tanh(),(height,width),mode='bilinear',align_corners=False)
            values=raw.flatten(2).transpose(1,2)
            coefficients=values.new_zeros(1,10,3)
            if mode=='color':coefficients=inverse@values
            elif mode=='mean':coefficients[:,0]=values.mean(1)
            elif mode!='none':raise ValueError(mode)
            residual=(values-basis@coefficients).transpose(1,2).reshape_as(raw)
            scale=(.1/(.1+coefficients.abs().sum(1).amax(1))).clamp(max=1)
            result=result+residual*scale[:,None,None,None]
        return result.clamp(0,1),global_delta
    for _ in range(steps):
        optimizer.zero_grad();prediction,global_delta=predict()
        loss=((prediction-target).abs()*mask).sum()/(mask.sum()*3)
        smooth=sum(global_delta.diff(dim=d).square().mean() for d in [-1,-2,-3])/3
        (loss+.001*smooth).backward();optimizer.step()
    with torch.no_grad():
        prediction,_=predict();error=(prediction-target).square()
        return {'fit_mae':float(((prediction-target).abs()*mask).sum()/(mask.sum()*3)),
                'heldout_mse':float((error*(1-mask)).sum()/((1-mask).sum()*3)),
                'heldout_psnr':float(-10*torch.log10(((error*(1-mask)).sum()/((1-mask).sum()*3)).clamp_min(1e-12)))}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--split',choices=['train','validation'],default='train');ap.add_argument('--limit',type=int,default=8)
    ap.add_argument('--steps',type=int,default=100);ap.add_argument('--long-edge',type=int,default=128)
    ap.add_argument('--device',default='cpu');args=ap.parse_args()
    if args.steps<1 or args.limit<1:raise ValueError('steps and limit must be positive')
    guard_experiment_paths(args.manifest,args.output)
    rows=load_triplets(args.manifest)
    guard_experiment_paths(*(row['_'+role+'_path'].parent for row in rows for role in ['content','reference','target']))
    dataset=TripletDataset(rows,args.split,args.long_edge,args.long_edge)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);results=[]
    for i in range(min(args.limit,len(dataset))):
        example=dataset[i];content=example['content'].unsqueeze(0).to(args.device);target=example['target'].unsqueeze(0).to(args.device)
        for mode in ['global','none','mean','color']:
            record={'pair_id':example['pair_id'],'mode':mode,**fit_oracle(content,target,mode,args.steps)}
            results.append(record);print(json.dumps(record),flush=True)
    (out/'oracle_diagnostic.json').write_text(json.dumps({'warning':'Uses target at fit time. Not deployable. Free local RGB field differs from model parametrization. No test split used.','results':results},indent=2))

if __name__=='__main__':main()
