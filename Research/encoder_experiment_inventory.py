"""Read-only compute-node inventory; never prints authentication credentials."""
import importlib.metadata as metadata
import json
from pathlib import Path
import socket
import urllib.request
import urllib.error

assert socket.gethostname().split('.')[0] == 'trinity-2-23'
for package in ['torch', 'torchvision', 'timm', 'transformers', 'huggingface-hub', 'safetensors']:
    try: print(package, metadata.version(package), flush=True)
    except metadata.PackageNotFoundError: print(package, 'NOT INSTALLED', flush=True)
for base in [Path('/home/shihangz/.cache/huggingface/hub'), Path('/home/shihangz/.cache/torch/hub/checkpoints'), Path('/data3/shihangz/petal-demo/weights')]:
    if base.exists():
        for path in base.iterdir():
            if 'dino' in path.name.lower() or 'mae' in path.name.lower(): print('candidate', path, flush=True)
root=Path('/tmp/shihangz/photo-tone-transfer')
manifest=json.loads((root/'Dataset/Photo_Triplets_v1/summary.json').read_text())
print('triplets',json.dumps(manifest),flush=True)
for name,url in {
    'dinov2_s':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'mae_b':'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
}.items():
    try:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=20) as response:
            print('public_weight',name,response.status,response.headers.get('Content-Length'),flush=True)
    except Exception as error: print('public_weight',name,type(error).__name__,str(error)[:180],flush=True)
