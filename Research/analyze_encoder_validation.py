"""Validation-only interim analysis; does not open test results or checkpoints."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


ROOT=Path(__file__).resolve().parent/'Experiments/encoder_validation_20261030'


def main():
    snapshot=json.loads((ROOT/'validation_summary.json').read_text());runs=snapshot['runs'];summary={};curves={};pairs={}
    for name,run in runs.items():
        events=[json.loads(line) for line in (ROOT/name/'metrics.jsonl').read_text().splitlines()]
        curves[name]=[e for e in events if 'validation' in e]
        pairs[name]={r['pair_id']:r for r in json.loads((ROOT/name/'validation_per_pair.json').read_text())}
    for group in 'ABCDEF':
        selected=[(n,r) for n,r in runs.items() if n.startswith(group+'_')]
        metrics={key:{'mean':float(np.mean([r['validation'][key] for n,r in selected])),
                      'std':float(np.std([r['validation'][key] for n,r in selected],ddof=1))}
                 for key in ('l1','psnr','ssim','delta_e76')}
        sources=defaultdict(list)
        for name,_ in selected:
            per_source=defaultdict(list)
            for row in pairs[name].values():per_source[row['content_key'].split('/')[0]].append(row['l1'])
            for source,values in per_source.items():sources[source].append(np.mean(values))
        summary[group]={'n_seeds':len(selected),'validation':metrics,
            'shuffled_reference_l1':float(np.mean([r['validation_shuffled_reference']['l1'] for n,r in selected])),
            'per_seed':{n:{'best_step':r['best_step'],'l1':r['validation']['l1'],
                           'delta_e76':r['validation']['delta_e76'],
                           'curve':[[e['step'],e['validation']['l1']] for e in curves[n]]} for n,r in selected},
            'content_source_l1':{source:float(np.mean(values)) for source,values in sources.items()}}
    comparisons={}
    for left,right in [('B','A'),('C','B'),('C','E'),('D','F'),('F','B')]:
        seeds=[s for s in (17,29,43) if f'{left}_{s}' in runs and f'{right}_{s}' in runs]
        differences=defaultdict(list);wins=[]
        for seed in seeds:
            lr=pairs[f'{left}_{seed}'];rr=pairs[f'{right}_{seed}']
            for key,row in lr.items():
                differences[row['content_key']].append(row['l1']-rr[key]['l1']);wins.append(row['l1']<rr[key]['l1'])
        groups=np.array([np.mean(v) for v in differences.values()]);rng=np.random.default_rng(20261030)
        sampled=groups[rng.integers(len(groups),size=(3000,len(groups)))].mean(1)
        lm=float(np.mean([runs[f'{left}_{s}']['validation']['l1'] for s in seeds]));rm=float(np.mean([runs[f'{right}_{s}']['validation']['l1'] for s in seeds]))
        comparisons[left+'-'+right]={'matched_seeds':seeds,'mean_l1_difference':lm-rm,'relative_l1_reduction_percent':100*(rm-lm)/rm,
            'fraction_pair_seed_better':float(np.mean(wins)),'content_clusters':len(groups),
            'equal_content_weight_difference':float(groups.mean()),'content_bootstrap95':np.quantile(sampled,[.025,.975]).tolist(),
            'caution':'Seed dependence and repeated reference dependence are not fully handled. This is exploratory validation sensitivity, not a formal significance conclusion.'}
    report={'updated_utc':snapshot['updated_utc'],'groups':summary,'comparisons':comparisons,
            'identity':{key:runs['A_17']['validation']['identity_'+key] for key in ('l1','psnr','ssim','delta_e76')}}
    (ROOT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=dict(zip('ABCDEF',['#7f7f7f','#2878b5','#e07a21','#218653','#b3428e','#7161ba']))
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.6,1]})
        for group in 'ABCDEF':
            group_curves=[c for name,c in curves.items() if name.startswith(group+'_')]
            xs=[e['step'] for e in group_curves[0]]
            matrix=np.array([[e['validation']['l1'] for e in curve] for curve in group_curves])
            for row in matrix:axes[0].plot(xs,row,color=colors[group],alpha=.22,lw=.8)
            axes[0].plot(xs,matrix.mean(0),color=colors[group],lw=2,label=group+(' (2 seeds)' if group=='F' else ''))
        axes[0].set(xlabel='Optimizer steps',ylabel='Validation RGB L1 (lower is better)',title='Frozen encoder + global LUT: learning curves')
        axes[0].grid(alpha=.2);axes[0].legend(ncol=3,frameon=False)
        means=[summary[g]['validation']['l1']['mean'] for g in 'ABCDEF'];std=[summary[g]['validation']['l1']['std'] for g in 'ABCDEF']
        axes[1].bar(list('ABCDEF'),means,yerr=std,color=[colors[g] for g in 'ABCDEF'],capsize=4)
        axes[1].set(ylabel='Best-checkpoint validation RGB L1',title='Mean +/- sample SD across seeds',ylim=(0,max(means)*1.25))
        axes[1].grid(axis='y',alpha=.2)
        fig.text(.5,.005,'A: stats | B: DINOv2-S+photo | C: DINOv3-S+photo | D: MAE-B+photo | E: DINOv3-S | F: DINOv2-B+photo (incomplete)',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(ROOT/'validation_curves.png',dpi=180);plt.close(fig)
    except ImportError:print('Matplotlib unavailable; numeric report saved.')


if __name__=='__main__':main()
