# MAE补充实验（8组）

监控：http://127.0.0.1:55318/ 。节点 trinity-2-23，tmux 会话
`mae-adaptation-sweep`；八个独立训练进程使用 GPU 0–7，GPU 9 生成预览。
关闭本机不影响训练。原有9组结果保留。

| 组别 | batch | 种子 | 可训练参数（含调色模块） | 优化步数/种子 |
|---|---:|---|---:|---:|
| 冻结MAE＋全层LoRA | 16 | 17、29 | 6,796,279 | 15,328 |
| MAE全参数微调 | 16 | 17、29 | 91,402,999 | 15,328 |
| 冻结MAE＋全层LoRA | 8 | 17、29 | 6,796,279 | 30,044 |
| 末4层及末端LN全量＋前8层LoRA | 8 | 17、29 | 34,756,087 | 30,044 |

四组都从官方MAE-B/16预训练权重重新开始，使用相同的K=4调色模型、损失、
59,100训练对及400验证对，共4个epoch，累计236,400对样本。测试集不参与训练
或选检查点。按原长宽比和精确尺寸分桶，保留尾批；batch是最大物理批量，
不是每一个尾批的大小。实际平均batch约为15.42和7.87。无需梯度累积。

全层LoRA在12个block的qkv、attention projection、MLP fc1/fc2及patch projection
共49处权重矩阵上添加适配器，rank=8、alpha=16，共1,191,936个LoRA参数。
原MAE的所有基础权重、bias、LN、CLS及位置向量冻结；不对一维向量应用矩阵LoRA。
摄影表征readout、光度分支、调色条件及映射模块正常训练。

全参数组不添加LoRA，解冻整个MAE，包含CLS、位置向量、patch embedding及全部LN。
末4层＋LoRA组沿用已有实现：前8层及patch矩阵LoRA共33处，末4层及末端LN全量更新，
前8层的基础参数、bias及LN冻结，位置与CLS向量冻结。

训练器派生自已有原生batch训练器，不覆盖旧训练代码。
AdamW、weight decay .01；调色模块LR 2e-4、LoRA LR 1e-4、全量骨干LR 1e-5；
沿用原warmup/cosine调度、FP16 AMP与梯度裁剪。
为统一batch 8/16的验证覆盖频率，新的8组每消耗8,000训练对验证一次，另在0、25步
及结束时验证；只按验证重建L1保存best.pt。原有单组batch16实验是每500步验证，
因此新8组与该旧实验的验证候选检查点集合并非严格一致。

检查：全层LoRA冻结边界及首末层梯度测试、全参数解冻测试通过；三种配置均在
384×384最大原生batch上进行了3次前向/反向和验证。全参数batch16峰值约4.73GiB，
全层LoRA batch16峰值约4.27GiB。Chrome已检查17组实验、模型切换、batch标注、
移动端布局，无脚本错误。

运行目录：`/tmp/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/`。
新代码快照为`addon_code/adaptation_sweep/Model/`；配置、日志、checkpoint分别在
`runs/<name>/config.json`、`logs/<name>.log`、`runs/<name>/`。
监督进程记录在`additional_supervisors/adaptation_sweep.json`。
进程失败时有checkpoint则最多重试3次；每2小时恢复备份，完成后归档至同结构的
`/data3/shihangz/photo-tone-transfer/Experiments/mae_full_20261030/`。

空间处理：9个已完成旧实验的latest.pt在与长期归档逐文件SHA256核对后移除本地副本，
保留本地best.pt及现有预览；DINOv2-B旧probe特征缓存同样在核验归档后回收，
不影响MAE训练。原数据没有改变。记录见archived_checkpoint_cleanup.json及probe_cache_archival.json。
