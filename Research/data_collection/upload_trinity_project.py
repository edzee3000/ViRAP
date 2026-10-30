"""Stream selected project assets to the compute node, with no local archive copy."""
import json,subprocess,tarfile,time,concurrent.futures as cf
from pathlib import Path
import collect_photos as c

def main():
    inventory=c.BASE/'_collection'/'all_images_manifest.jsonl'
    rows=[json.loads(s) for s in inventory.read_text(encoding='utf-8').splitlines()]
    paths=[c.BASE/r['path'] for r in rows]
    paths+=list((c.ROOT/'Dataset'/'Photo_Pairs_v1').glob('*'))
    paths += [inventory,c.BASE/'_collection'/'all_images_summary.json']
    paths+=list((c.ROOT/'Research'/'data_collection').glob('*.py'))
    paths+=list((c.ROOT/'Research'/'data_collection').glob('*.sh'))
    paths+=list((c.ROOT/'Research'/'PETAL').glob('*.md'))
    paths=[p for p in paths if p.is_file()]
    pilot=json.loads((c.ROOT/'Dataset/Photo_Pairs_v1/pilot_200.json').read_text(encoding='utf-8'))['pairs']
    priority={(c.ROOT/'Dataset/Photo_Pairs_v1'/p[k]).resolve() for p in pilot for k in ['content_path','reference_path']}
    paths.sort(key=lambda p:(0 if p.resolve() in priority else 1 if p.suffix in ['.json','.jsonl','.py','.sh'] else 2,str(p)))
    start=time.time()
    def transfer(worker):
        subset=paths[worker::4]
        proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','trinity-2-23','tar -xf - -C /tmp/shihangz/photo-tone-transfer'],stdin=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdin,mode='w|') as archive:
            for i,p in enumerate(subset):
                archive.add(p,arcname=p.relative_to(c.ROOT).as_posix(),recursive=False)
                if i%100==0:print(f'Stream {worker}: uploaded {i}/{len(subset)} files; {time.time()-start:.0f}s',flush=True)
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('SSH transfer failed')
    with cf.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(transfer,range(4)))
    print(f'DONE {len(paths)} files, {time.time()-start:.1f}s',flush=True)

if __name__=='__main__':main()
