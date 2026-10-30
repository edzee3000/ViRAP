"""Instantiate MAE-tone, count actual parameters, optionally run CPU structure checks."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import torch
from tone_model.mae_tone import MAEToneConfig,MAEToneModel


def count(module):return sum(p.numel() for p in module.parameters())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights');ap.add_argument('--structure-only',action='store_true')
    ap.add_argument('--local-rank',type=int,default=4);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--output',type=Path,default=Path(__file__).parent/'mae_tone_architecture_report.json')
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261030)
    cfg=MAEToneConfig(local_rank=args.local_rank)
    model=MAEToneModel.from_pretrained(args.weights,cfg,allow_uninitialized=args.structure_only)
    report={'architecture':'MAE-tone M1 (K=4) / M0 (K=2) design','config':asdict(cfg),
            'total_parameters_including_fixed_positions':count(model),'backbone_parameters_including_fixed_positions':count(model.encoder.backbone),
            'fixed_sincos_position_elements':model.encoder.backbone.pos_embed.numel(),
            'pretrained_loaded':model.pretrained_loaded,'packages':{k:importlib.metadata.version(k) for k in ('torch','timm')},
            'modules':{name:count(module) for name,module in model.named_children()},
            'encoder_modules':{name:count(module) for name,module in model.encoder.named_children()},
            'encoder_direct':{name:p.numel() for name,p in model.encoder.named_parameters(recurse=False)},
            'spatial_modules':{name:count(module) for name,module in model.content_context.named_children()},
            'backbone_block_count':len(model.encoder.backbone.blocks),'backbone_parameters_per_block':count(model.encoder.backbone.blocks[0]),
            'renderer_generated_values':17**3*3*(1+cfg.local_rank),'stages':{}}
    for stage in ('heads','joint','full'):
        model.set_stage(stage)
        report['stages'][stage]={'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    model.set_stage('heads');model.eval()
    if args.smoke:
        with torch.inference_mode():
            x=torch.rand(1,3,73,101);r=torch.rand(1,3,89,67)
            output,aux=model(x,r,True)
        report['structure_smoke']={'output_shape':list(output.shape),'maps':list(aux['maps'].shape),
             'content_tokens':list(aux['content_tokens'].shape),'memory_tokens':aux['memory_tokens'],
             'global_lut':list(aux['global_lut'].shape),'local_luts':list(aux['local_luts'].shape),
             'identity_error':float((output-x).abs().max()),'finite':bool(torch.isfinite(output).all())}
    report['scope']='Architecture and structure checks, not evidence of full MAE-tone quality. No masking or MAE pixel reconstruction decoder.'
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
