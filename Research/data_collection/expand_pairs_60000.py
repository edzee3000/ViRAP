"""Expand frozen random pairs without changing source splits or pilot identifiers."""
import collections,hashlib,json,random
from pathlib import Path

def main(root):
    dataset=Path(root)/'Dataset';old=dataset/'Photo_Pairs_v1';out=dataset/'Photo_Pairs_60000_v1'
    if out.exists():raise RuntimeError('Frozen manifest already exists')
    images=json.loads((old/'source_images.json').read_text(encoding='utf-8'))['images']
    pairs=json.loads((old/'pairs.json').read_text(encoding='utf-8'))['pairs']
    pools=[r for r in images if r['split']=='train' and r['pair_representative']]
    used={(p['content_key'],p['reference_key']) for p in pairs};counts=collections.Counter(p['reference_key'] for p in pairs if p['split']=='train')
    rng=random.Random(20261030);number=0
    while len(pairs)<60000:
        rng.shuffle(pools)
        for content in pools:
            if len(pairs)>=60000:break
            for attempt in range(10000):
                reference=rng.choice(pools)
                key=(content['key'],reference['key'])
                if key[0]!=key[1] and key not in used and counts[key[1]]<30:break
            else:raise RuntimeError('Reference quota exhausted')
            number+=1;used.add(key);counts[key[1]]+=1
            pairs.append({'pair_id':f'train_extra_{number:06d}','split':'train','strategy':'seeded_random_reference_cap30','content_key':key[0],'reference_key':key[1],'content_path':content['dataset_relative_path'],'reference_path':reference['dataset_relative_path'],'status':'pending_generation','target_path':None})
    assert len(used)==60000
    assignment={r['key']:r['split'] for r in images}
    for p in pairs:
        assert assignment[p['content_key']]==assignment[p['reference_key']]
        p['planned_target_path']=f"../Generated_Outputs/{p['split']}/{int(p['pair_id'].split('_')[-1])//1000:04d}/{p['pair_id']}.png"
    out.mkdir();(out/'pairs.json').write_text(json.dumps({'version':1,'seed':20261030,'path_base':'directory_containing_this_json','inherited_source_manifest':'../Photo_Pairs_v1/source_images.json','pairs':pairs},ensure_ascii=False,indent=2),encoding='utf-8')
    stats={'total_pairs':len(pairs),'by_split':dict(collections.Counter(p['split'] for p in pairs)),'source_images':len(images),'train_representatives':len(pools),'max_training_reference_uses':max(counts.values()),'unique_directed_pairs':len(used),'split_overlap':0,'includes_original_8090':True,'pilot_outputs_reused':True,'generation_resolution_max_edge':768,'images_copied':0}
    (out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    (out/'README.md').write_text('60,000 candidate queries, including 59,100 training, 400 validation, 400 test and 100 same-image diagnostics. Source splits remain frozen. Original pilot and pair identifiers are retained, so already generated outputs are reused. Paths are relative to this directory. Outputs are generated-unreviewed, not accepted training data. No encoder embeddings are exported. The generation backend is the released PETAL model and must be credited in provenance.\n',encoding='utf-8')
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    import sys
    main(sys.argv[1])
