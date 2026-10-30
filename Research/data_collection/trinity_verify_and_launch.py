"""Bounded staging gate; verify every original before starting the bulk job."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path

ROOT=Path('/tmp/shihangz/photo-tone-transfer')
DEST=Path('/data3/shihangz/photo-tone-transfer')

def main():
    assert socket.gethostname()=='trinity-2-23'
    rows=json.loads((ROOT/'Dataset/Photo_Pairs_v1/source_images.json').read_text())['images']
    deadline=time.time()+3600
    while True:
        pending=[r for r in rows if not (ROOT/'Dataset/Original Data'/r['path']).is_file() or (ROOT/'Dataset/Original Data'/r['path']).stat().st_size!=r['bytes']]
        print(f'Staging: {len(rows)-len(pending)}/{len(rows)} original files complete by size',flush=True)
        if not pending:break
        if time.time()>deadline:raise RuntimeError('Upload timeout after one hour; no bulk job started')
        time.sleep(45)
    for r in rows:
        path=ROOT/'Dataset/Original Data'/r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'],r['key']
    result={'files':len(rows),'all_sha256_match':True,'node':socket.gethostname(),'head_node_compute':False}
    (ROOT/'Logs/upload_integrity.json').write_text(json.dumps(result,indent=2))
    audit=json.loads((ROOT/'Logs/pilot/audit.json').read_text());assert audit['success']==200 and audit['failed']==0
    subprocess.run([str(ROOT/'Runtime/venv/bin/python'),str(ROOT/'Research/data_collection/expand_pairs_60000.py'),str(ROOT)],check=True)
    DEST.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--',str(ROOT/'Dataset'),str(ROOT/'Research'),str(ROOT/'Logs'),str(DEST)+'/'],check=True)
    print('Originals verified and archived. Starting 8-GPU bulk generation.',flush=True)
    os.execv('/bin/bash',['bash',str(ROOT/'Research/data_collection/trinity_bulk.sh')])

if __name__=='__main__':main()
