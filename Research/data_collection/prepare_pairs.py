"""Freeze grouped source splits and inexpensive reproducible directed random pairs."""
from pathlib import Path
import collections,concurrent.futures as cf,hashlib,json,random
import cv2
import numpy as np
from PIL import Image,ImageOps
import collect_photos as c

SEED=20261029
OUT=c.ROOT/'Dataset'/'Photo_Pairs_v1'

def main():
    OUT.mkdir(exist_ok=False)
    inventory=c.BASE/'_collection'/'all_images_manifest.jsonl'
    rows=[json.loads(s) for s in inventory.read_text(encoding='utf-8').splitlines()]
    def feature(r):
        with Image.open(c.BASE/r['path']) as im:
            im=ImageOps.exif_transpose(im).convert('L')
            a=np.asarray(im.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
            dct=cv2.dct(a)[:8,:8].flatten();bits=dct>np.median(dct[1:])
            ph=sum(int(b)<<i for i,b in enumerate(bits))
            a9=np.asarray(im.resize((9,8),Image.Resampling.LANCZOS))
            dh=sum(int(b)<<i for i,b in enumerate((a9[:,1:]>a9[:,:-1]).flatten()))
            gray=np.asarray(im.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32).flatten()
        return ph,dh,gray
    with cf.ThreadPoolExecutor(max_workers=4) as pool:features=list(pool.map(feature,rows))
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[root(i)]=root(j)
    exact={};links=[]
    for i,r in enumerate(rows):
        if r['sha256'] in exact:union(i,exact[r['sha256']]);links.append([i,exact[r['sha256']],'exact_sha256'])
        else:exact[r['sha256']]=i
        ph,dh,a=features[i]
        for j in range(i):
            op,od,b=features[j]
            if (ph^op).bit_count()<=8 and (dh^od).bit_count()<=12 and a.std()>1 and b.std()>1:
                if float(np.corrcoef(a,b)[0,1])>0.985:
                    union(i,j);links.append([i,j,'conservative_near_duplicate'])
    groups=collections.defaultdict(list)
    for i in range(len(rows)):groups[root(i)].append(i)
    ordered=list(groups.values());rng=random.Random(SEED);rng.shuffle(ordered)
    split_groups={'train':[],'validation':[],'test':[]}
    # Group-wise random split; counts approximate 80/10/10, never split a group.
    n=len(ordered);cut1=round(n*.8);cut2=round(n*.9)
    split_groups['train']=ordered[:cut1];split_groups['validation']=ordered[cut1:cut2];split_groups['test']=ordered[cut2:]
    assignments={};pools={};images=[]
    for split,gs in split_groups.items():
        pool=[]
        for members in gs:
            representative=min(members,key=lambda i:rows[i]['key'])
            group_id='g_'+hashlib.sha256('\n'.join(sorted(rows[i]['key'] for i in members)).encode()).hexdigest()[:16]
            pool.append(rows[representative])
            for i in members:
                r=rows[i];assignments[r['key']]=split
                images.append({**r,'split':split,'duplicate_group':group_id,'pair_representative':i==representative,'dataset_relative_path':'../Original Data/'+r['path']})
        rng.shuffle(pool);pools[split]=pool
    pairs=[]
    for split,pool in pools.items():
        contents=pool if split=='train' else pool[:min(400,len(pool))]
        refs=collections.Counter();used=set()
        for round_id in range(2 if split=='train' else 1):
            for content in contents:
                eligible=[r for r in pool if r['key']!=content['key'] and (content['key'],r['key']) not in used]
                # Random matching with a cap, to avoid a few references dominating.
                capped=[r for r in eligible if refs[r['key']]<6]
                reference=rng.choice(capped or eligible)
                refs[reference['key']]+=1;used.add((content['key'],reference['key']))
                pairs.append({'pair_id':f'{split}_{len(used):06d}','split':split,'strategy':'seeded_random_reference_cap6','content_key':content['key'],'reference_key':reference['key'],'content_path':'../Original Data/'+content['path'],'reference_path':'../Original Data/'+reference['path'],'status':'pending_generation','target_path':None})
    # Separate diagnostics: outputs are not assumed to be identical to the input.
    for i,r in enumerate(pools['train'][:100]):
        pairs.append({'pair_id':f'diagnostic_{i+1:06d}','split':'diagnostic','source_split':'train','strategy':'same_image_diagnostic','content_key':r['key'],'reference_key':r['key'],'content_path':'../Original Data/'+r['path'],'reference_path':'../Original Data/'+r['path'],'status':'pending_generation','target_path':None})
    train=[p for p in pairs if p['split']=='train']
    pilot=rng.sample(train,min(200,len(train)))
    # Put a dispersed pilot first; subsequent stages reuse these outputs.
    pilot_ids={p['pair_id'] for p in pilot}
    train_order=pilot+[p for p in train if p['pair_id'] not in pilot_ids]
    stages={'pilot_200':[p['pair_id'] for p in pilot],'train_first_1000':[p['pair_id'] for p in train_order[:1000]],'train_first_3000':[p['pair_id'] for p in train_order[:3000]],'train_all':[p['pair_id'] for p in train_order]}
    assert len({p['pair_id'] for p in pairs})==len(pairs)
    assert len({(p['content_key'],p['reference_key']) for p in pairs})==len(pairs)
    for p in pairs:
        assert assignments[p['content_key']]==assignments[p['reference_key']]
        for field in ['content_path','reference_path']:
            path=(OUT/p[field]).resolve();path.relative_to(c.BASE.resolve());assert path.is_file()
    stats={'seed':SEED,'source_images':len(rows),'duplicate_groups':n,'nonrepresentative_files_not_paired':len(rows)-n,'images_by_source':dict(collections.Counter(r['source'] for r in rows)),'split_image_counts':dict(collections.Counter(assignments.values())),'split_representative_counts':{s:len(p) for s,p in pools.items()},'pairs_by_split':dict(collections.Counter(p['split'] for p in pairs)),'total_pairs':len(pairs),'pilot_pairs':len(pilot),'all_paths_exist':True,'group_split_overlap':0,'original_images_copied':0,'source_inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),'limitations':['Near-duplicate grouping is conservative, not a proof of scene or photographer isolation.','No semantic/style scoring; random pairing does not ensure successful tone transfer.','Generated outputs require quality review; pair counts are not usable triplet counts.']}
    def save(name,obj):c.save_json(OUT/name,obj)
    save('pairs.json',{'version':1,'seed':SEED,'path_base':'directory_containing_this_json','role_definition':{'content':'输入图，保留其人物、物体与结构','reference':'参考图，提供色调风格'},'pairs':pairs})
    save('pilot_200.json',{'path_base':'directory_containing_this_json','pairs':pilot})
    save('source_images.json',{'images':images})
    save('generation_stages.json',stages);save('summary.json',stats)
    save('duplicate_groups.json',{'groups':[[rows[i]['key'] for i in g] for g in ordered if len(g)>1],'rule':'SHA256 or pHash distance<=8 + dHash distance<=12 + grayscale correlation>0.985'})
    (OUT/'README.md').write_text(f'''# 摄影色调迁移配对 v1

共{len(rows)}张现有原图，按重复组约80/10/10划分，构建{len(pairs)}对；各集合数量见summary.json。只保存路径，不复制图片，不调用模型。

pairs.json中的每条记录是一组有方向的配对：content_path为输入图，reference_path为参考图。相对路径以本JSON所在目录为基准，用Path(json_file).parent / pair['content_path']解析。请整体保留Dataset目录关系，避免重命名原文件。

训练集中每张代表图作为输入恰好两次，参考不同；不按来源预设角色。不做场景或风格分类，采用固定seed随机配对，参考使用次数上限6。验证、测试在各自源图集合内配对，每集最多400对。100对同图诊断独立记录，不并入常规训练/测试。

先运行pilot_200.json的200对，检查吞吐、输出和筛选通过率；通过后按generation_stages.json扩到1000、3000及全部训练对。阶段为嵌套子集，已有输出不重复生成。最终训练可用数量取决于生成成功率和审核，不预先保证这些配对都有可用目标。

重复文件仍保留在原目录，配对只选每组一个代表；分组内所有原图归同一集合。精确和保守近重复检查无法保证排除全部同场景、连拍和裁剪版本，未执行摄影师隔离，后续人工发现应整组修正并重新冻结划分。

这版为降低准备成本，将原方案的场景分层简化为随机配对。不能把本批结果写成已经验证分层策略有效；若要比较分层与随机，应另建等预算受控实验。所有输出均尚未生成，status为pending_generation。
''',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
