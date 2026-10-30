"""Instantiate, count and shape-check v0.2; never starts a training run."""
import argparse,json,subprocess,time
from dataclasses import asdict
from pathlib import Path
import torch
from tone_model.v2 import V2Config,ReferenceToneModelV2


def count(module):return sum(p.numel() for p in module.parameters())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]/'Research/External_Code/dinov2'))
    ap.add_argument('--weights');ap.add_argument('--structure-only',action='store_true')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default=str(Path(__file__).parent/'v2_1_architecture_report.json'))
    ap.add_argument('--without-pair-attention',action='store_true');ap.add_argument('--readout-layers',type=int,default=2)
    ap.add_argument('--spatial-conditioning',choices=['none','pair'],default='pair')
    ap.add_argument('--local-rank',type=int,default=2)
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261030)
    config=V2Config(pair_attention=not args.without_pair_attention,readout_layers=args.readout_layers,
                    spatial_conditioning=args.spatial_conditioning,local_rank=args.local_rank)
    model=ReferenceToneModelV2.from_official(args.repo,args.weights,config,args.structure_only)
    total=count(model)
    report={'version':'0.2.1 reference-conditioned spatial maps','config':asdict(config),'total_parameters':total,
            'pretrained_loaded':model.pretrained_loaded,'weights_downloaded_by_this_script':False,
            'backbone_commit':subprocess.check_output(['git','-C',str(Path(args.repo).resolve()),'rev-parse','HEAD'],text=True).strip(),
            'modules':{name:count(module) for name,module in model.named_children()},
            'encoder_modules':{name:count(module) for name,module in model.encoder.named_children()},
            'encoder_direct_parameters':{name:p.numel() for name,p in model.encoder.named_parameters(recurse=False)},
            'backbone_position_shape':list(model.encoder.backbone.pos_embed.shape),
            'stages':{}}
    for stage in ['encoder','heads','joint','full']:
        model.set_stage(stage)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        report['stages'][stage]={'trainable':trainable,'frozen':total-trainable}
    model.set_stage('heads');model.eval()
    if args.smoke:
        with torch.inference_mode():
            content=torch.rand(1,3,73,101);reference=torch.rand(1,3,89,67)
            begin=time.perf_counter();output,aux=model(content,reference,True)
            assert torch.isfinite(output).all()
            assert output.shape==content.shape
            assert (output-content).abs().max()<1e-6
            report['structure_smoke']={'output_shape':list(output.shape),'condition':list(aux['condition'].shape),
                    'style_tokens':list(aux['content_tokens'].shape),'global':list(aux['content_global'].shape),
                    'memory_tokens_per_image':aux['memory_tokens'],'maps':list(aux['maps'].shape),
                    'global_lut':list(aux['global_lut'].shape),'local_luts':list(aux['local_luts'].shape),
                    'finite':True,'initial_identity_max_error':float((output-content).abs().max()),
                    'cpu_seconds_not_benchmark':time.perf_counter()-begin}
    report['warning']='Structure counting only when pretrained_loaded=false. No evidence of style quality, convergence, or novelty. V2 is not wired into the V1 training CLI.'
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
