"""Aggregate completed, predeclared runs without selecting a backbone on test."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from prepare_encoder_experiment import save_json


def summarize(base):
    groups={};runs={}
    for path in sorted((base/'train').glob('*/complete.json')):
        run=json.loads(path.read_text());runs[(run['group'],run['seed'])]=run
    expected={(g,s) for g in 'ABCDEF' for s in (17,29,43)}
    if set(runs)!=expected:raise ValueError('Expected all 6 groups x 3 seeds')
    for seed in (17,29,43):
        if len({runs[(g,seed)]['train_schedule_sha256'] for g in 'ABCDEF'})!=1:raise ValueError('Pair schedules differ across groups')
    for group in 'ABCDEF':
        selected=[runs[(group,s)] for s in (17,29,43)]
        groups[group]={'runs':3,'head_parameters':selected[0]['head_parameters'],'backbone_parameters':selected[0]['backbone_parameters']}
        for split in ('validation','test'):
            groups[group][split]={metric:{'mean':float(np.mean([r[split][metric] for r in selected])),
                                         'std_across_seeds':float(np.std([r[split][metric] for r in selected],ddof=1))}
                                  for metric in ('l1','psnr','ssim','delta_e76')}
    comparisons={}
    for left,right in [('C','B'),('C','E'),('D','F')]:
        # Average seeds per pair, then aggregate by content image before bootstrap.
        # This is content-cluster sensitivity, not full two-way reference/content CI.
        values=defaultdict(list)
        for seed in (17,29,43):
            lrows=json.loads((base/'train'/f'{left}_{seed}'/'test_per_pair.json').read_text())
            rrows={r['pair_id']:r for r in json.loads((base/'train'/f'{right}_{seed}'/'test_per_pair.json').read_text())}
            for row in lrows:
                values[row['content_key']].append(row['l1']-rrows[row['pair_id']]['l1'])
        cluster=np.array([np.mean(v) for v in values.values()]);rng=np.random.default_rng(20261030)
        boot=np.mean(cluster[rng.integers(len(cluster),size=(2000,len(cluster)))],axis=1)
        comparisons[left+' minus '+right]={'metric':'L1, negative favors first group','content_clusters':len(cluster),
           'mean_of_content_means':float(cluster.mean()),'content_bootstrap_95_percentile':np.quantile(boot,[.025,.975]).tolist(),
           'limitation':'Repeated references remain dependent; this is not an unconditional significance test or a causal pretraining-objective comparison.'}
    report={'groups':groups,'paired_content_sensitivity':comparisons,
            'scope':'Frozen encoder + shared global LUT, 128-square output. Targets are unreviewed synthetic outputs, not human aesthetic ground truth.'}
    save_json(base/'summary.json',report)
    lines=['# Frozen encoder probe results','',
           'All groups: 59,100 train / 400 validation / 400 test, 3 seeds, 3 epochs, global-only LUT. Test is evaluated after validation checkpoint selection.','',
           '|Group|Backbone parameters|Trainable parameters|Test L1|Test PSNR|Test SSIM|Test ΔE76|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for group,value in groups.items():
        cells=[]
        for metric in ('l1','psnr','ssim','delta_e76'):
            score=value['test'][metric];cells.append(f"{score['mean']:.5f} ± {score['std_across_seeds']:.5f}")
        lines.append('|'+ '|'.join([group,str(value['backbone_parameters']),str(value['head_parameters']),*cells])+'|')
    lines+=['','These results measure reproduction of generated targets. They do not establish human preference, pure style disentanglement, or superiority of a pretraining objective.','',
            'A: Lab statistics; B: DINOv2-S + photo; C: DINOv3-S + photo; D: MAE-B + photo; E: DINOv3-S only; F: DINOv2-B + photo.','',
            'D/F controls approximate model scale, but pretraining data, patch size and training recipe still differ. C/E also changes trainable capacity and token count.']
    (base/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('base',type=Path);args=ap.parse_args();summarize(args.base)
