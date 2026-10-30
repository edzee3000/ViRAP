"""Write the two explicit architecture configs; does not start training."""
from dataclasses import asdict
import json
from pathlib import Path
from tone_model.mae_tone import MAEToneConfig

for name,rank in [('mae_tone_m0',2),('mae_tone_m1',4)]:
    config={'architecture':'MAEToneModel','model':asdict(MAEToneConfig(local_rank=rank)),
            'backbone':'official MAE pretrain ViT-B/16 encoder, no decoder/masking',
            'checkpoint_required_for_real_use':True,'timm_version':'1.0.22',
            'checkpoint_sha256_used_by_encoder_probe':'aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d',
            'note':'Architecture config, no trained full MAE-tone checkpoint; not a V1 train.py config'}
    path=Path(__file__).parent/'configs'/(name+'.json')
    path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(path)
