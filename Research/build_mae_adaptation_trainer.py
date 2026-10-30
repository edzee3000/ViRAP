"""Derive the controlled sweep trainer without changing the original trainer."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'Model/train_mae_hybrid.py').read_text(encoding='utf-8')
source=source.replace('from tone_model.mae_hybrid_lora import MAEHybridLoRAModel,adaptation_counts',
                     'from tone_model.mae_hybrid_lora import adaptation_counts\nfrom tone_model.mae_adaptation import build_adaptation')
source=source.replace("ap.add_argument('--lora-rank'", "ap.add_argument('--adaptation',choices=['all_lora','full','hybrid'],required=True)\n    ap.add_argument('--lora-rank'")
source=source.replace('MAEHybridLoRAModel.from_pretrained(args.weights,model_cfg,lora_rank=args.lora_rank,lora_alpha=args.lora_alpha)',
                      'build_adaptation(args.weights,model_cfg,args.adaptation,lora_rank=args.lora_rank,lora_alpha=args.lora_alpha)')
source=source.replace("stage='hybrid'",'stage=args.adaptation').replace("'stage':'hybrid'","'stage':args.adaptation")
source=source.replace("'lora_rank':args.lora_rank", "'lora_rank':args.lora_rank if args.adaptation!='full' else 0")
source=source.replace('all four epochs hybrid','all four epochs use the configured adaptation')
# Equal validation coverage across batch sizes: once per 8,000 consumed pairs,
# plus step 25 and final, using validation L1 only. Same schedule for all 8 runs.
source=source.replace("if step%args.validate_every==0 or step in (min(25,total),total):validate();checkpoint()",
    "if consumed//8000 > (consumed-batch_count)//8000 or step in (min(25,total),total):validate();checkpoint()")
source=source.replace("config=json.loads(json.dumps(config));step=0", "config['validation_schedule']='every 8000 consumed training pairs, step 0/25 and final'\n    config=json.loads(json.dumps(config));step=0")
(root/'Model/train_mae_adaptation.py').write_text(source,encoding='utf-8')
print('adaptation_trainer_ready')
