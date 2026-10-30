"""Gallery selection and safe read-only access, shared by monitor and sidecar."""
from collections import defaultdict
import json
from pathlib import Path
import random
import re


def fixed_selection(records,count=48,seed=20261030):
    result={}
    for split in ('validation','train'):
        groups=defaultdict(list)
        for row in records:
            if row['split']==split and row.get('_decision')!='reject':
                groups[row['content_key'].split('/')[0]].append(row)
        rng=random.Random(seed+(split=='train'))
        for values in groups.values():rng.shuffle(values)
        selected=[];seen=set();sources=sorted(groups)
        while len(selected)<count:
            advanced=False
            for source in sources:
                while groups[source] and groups[source][-1]['content_key'] in seen:groups[source].pop()
                if groups[source]:
                    row=groups[source].pop();seen.add(row['content_key']);selected.append(row['pair_id']);advanced=True
                    if len(selected)==count:break
            if not advanced:break
        result[split]=selected
    return result


def safe_run(root,name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',name):raise ValueError('Invalid run')
    path=(Path(root)/'runs'/name).resolve()
    if not path.is_relative_to((Path(root)/'runs').resolve()) or not path.is_dir():raise ValueError('Unknown run')
    return path


def read_gallery(root,name):
    path=safe_run(root,name)/'gallery.json'
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def gallery_payload(root,name,split,page=0,size=6):
    if split not in ('train','validation'):raise ValueError('Only train and validation galleries are available')
    if not 0<=page<=1000 or not 1<=size<=12:raise ValueError('Invalid page size')
    manifest=read_gallery(root,name);rows=manifest.get(split,[]);total=manifest.get('requested',{}).get(split,48)
    return {'run':name,'split':split,'page':page,'page_size':size,'total':total,'ready':len(rows),
            'version':manifest.get('version'),'checkpoint_step':manifest.get('checkpoint_step'),
            'checkpoint_kind':manifest.get('checkpoint_kind'),'updated_unix':manifest.get('updated_unix'),
            'rows':rows[page*size:(page+1)*size]}


def safe_asset(root,relative):
    base=(Path(root)/'gallery_assets').resolve();path=(base/relative).resolve()
    if not path.is_relative_to(base) or path.suffix.lower() not in ('.png','.jpg','.webp') or not path.is_file():
        raise ValueError('Unknown gallery image')
    return path
