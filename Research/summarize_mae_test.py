"""Summarize precommitted test results without selecting a model on test."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[1]
folder=root/'Model/Results/test_eval_v1'
protocol=json.loads((folder/'protocol.json').read_text())
results=[json.loads((folder/(name+'.json')).read_text()) for name in protocol['checkpoints']]
assert len({r['test_fingerprint'] for r in results})==1
assert all(r['count']==400 for r in results)
assert len({tuple(row['pair_id'] for row in r['rows']) for r in results})==1
metrics=('l1','psnr','ssim','gradient_l1')
baseline=results[0]['mean_per_image']['identity']
assert all(abs(r['mean_per_image']['identity'][m]-baseline[m])<1e-8 for r in results for m in metrics)
rows=[{'run':r['run'],'checkpoint_step':r['checkpoint']['step'],**r['mean_per_image']['model']} for r in results]
with (folder/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
groups={}
for prefix in ('M0','M1'):
    group=[r for r in results if r['run'].startswith(prefix+'_s')]
    groups[prefix]={m:{'mean':statistics.mean(r['mean_per_image']['model'][m] for r in group),
                       'std':statistics.stdev(r['mean_per_image']['model'][m] for r in group)} for m in metrics}
source_rows=[]
for result in results:
    for source in sorted({r['content_key'].split('/')[0] for r in result['rows']}):
        subset=[r for r in result['rows'] if r['content_key'].split('/')[0]==source]
        source_rows.append({'run':result['run'],'content_source':source,'count':len(subset),
                            **{m:statistics.mean(r['model'][m] for r in subset) for m in metrics}})
with (folder/'by_content_source.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(source_rows[0]));writer.writeheader();writer.writerows(source_rows)
summary={'protocol':protocol,'identity':baseline,'runs':rows,'four_seed_groups':groups,'source_breakdown':source_rows}
(folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# MAE摄影调色模型：固定测试集评测','',
       '评测对象为固定的400组测试三元组；所有检查点均在测试推理前固定，使用验证集重建L1最低的 best.pt。未使用测试结果选择检查点。','',
       '| 模型 | 验证集选择的步数 | 测试L1 ↓ | 测试PSNR ↑ | 测试SSIM ↑ | 梯度L1 ↓ |',
       '|---|---:|---:|---:|---:|---:|',
       f"| 保持输入不变 | — | {baseline['l1']:.5f} | {baseline['psnr']:.3f} | {baseline['ssim']:.5f} | {baseline['gradient_l1']:.5f} |"]
for r in rows:
    lines.append(f"| {r['run']} | {r['checkpoint_step']} | {r['l1']:.5f} | {r['psnr']:.3f} | {r['ssim']:.5f} | {r['gradient_l1']:.5f} |")
lines+=['','四种子均值±样本标准差：','']
for name,g in groups.items():
    lines.append(f"- {name}：L1 {g['l1']['mean']:.5f} ± {g['l1']['std']:.5f}；PSNR {g['psnr']['mean']:.3f} ± {g['psnr']['std']:.3f} dB；SSIM {g['ssim']['mean']:.5f} ± {g['ssim']['std']:.5f}。")
lines+=['','评测口径：原始长宽比，输入和目标长边384，参考图长边384；不裁边。前向FP16与验证一致，指标FP32。指标先逐图计算再等权平均。SSIM在RGB各通道采用11×11、σ=1.5高斯有效窗口，最后对通道和空间平均。PSNR的MSE下限为1e-12。',
        '', '结论边界：目标图为生成数据，指标衡量对生成目标的拟合程度，不等同于人类调色偏好、内容保真性或真实摄影泛化的完整评估。LoRA仅一个种子，且batch和训练日程与M1不同，不能将差异独立归因于LoRA。',
        '', '所有逐图结果、冻结协议、检查点SHA256和代码位于本目录；summary.csv为逐模型结果，by_content_source.csv为内容来源分组结果。']
(folder/'测试集评测报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'identity':baseline,'runs':rows,'groups':groups},indent=2))
