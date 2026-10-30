import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw

def main(root):
    root=Path(root);folder=root/'Dataset/Photo_Pairs_v1'
    pairs=json.loads((folder/'pilot_200.json').read_text())['pairs']
    records={}
    for path in (root/'Logs/pilot').glob('worker_*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line);records[r['pair_id']]=r
    assert len(records)==200,len(records)
    assert all(r['status']=='generated_unreviewed' for r in records.values())
    for r in records.values():
        path=(folder/r['target_path']).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_sha256']
        with Image.open(path) as im:im.verify()
        assert r['output_std']>1,r['pair_id']
    times=[r['elapsed_seconds'] for r in records.values()]
    result={'pairs':200,'success':200,'failed':0,'output_hashes_verified':True,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),'max_gpu_mib':max(r['peak_gpu_mib'] for r in records.values()),'mean_input_change':float(np.mean([r['mean_abs_input_change'] for r in records.values()])),'minimum_output_std':min(r['output_std'] for r in records.values()),'quality_state':'Technical checks only; visual sample provided; all outputs remain unreviewed.'}
    (root/'Logs/pilot/audit.json').write_text(json.dumps(result,indent=2))
    sample=pairs[::25];canvas=Image.new('RGB',(960,8*250+34),(20,24,30));draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(['INPUT','REFERENCE','OUTPUT']):draw.text((j*320+10,8),label,fill='white')
    for i,p in enumerate(sample):
        r=records[p['pair_id']]
        for j,path in enumerate([folder/p['content_path'],folder/p['reference_path'],folder/r['target_path']]):
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((312,220));canvas.paste(im,(j*320+(320-im.width)//2,34+i*250+(220-im.height)//2))
        draw.text((8,34+i*250+224),p['pair_id'],fill='white')
    canvas.save(root/'Logs/pilot/contact_sheet.jpg',quality=90)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main(sys.argv[1])
