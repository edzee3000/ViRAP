# 参考条件摄影调色研究原型 v0.1

完整设计见Research/PETAL/模型架构完整设计与自审.md，审稿式质询见Research/PETAL/architecture_peer_critique.md。

全模型1,779,605参数；共享风格集合编码器＋输入专属空间网络＋全局3D LUT＋两个局部颜色分支＋代理空间最终RGB残差投影/整体幅度保护。无扩散、无下载骨干。没有发布已训练权重，当前不能展示有效调色结果或声称优于已有方法。

## 环境与测试

Python>=3.10，依赖见requirements.txt。使用已有PyTorch环境即可，Windows可CPU功能验证。集群实验必须在计算节点，本地暂存数据；train.py会检查Trinity主机和IO挂载。不要在head运行任何测试、服务或训练。

```powershell
python -m unittest discover -s Model/tests -v
```

## 训练（从项目根目录执行）

本地尚未下载完整输出，真实数据位于Trinity；下例应在计算节点/tmp项目目录运行，环境用该节点Runtime/venv/bin/python。CLI默认batch1保持纵横比，默认步数不是有效训练预算承诺。

```bash
python Model/train.py --manifest Dataset/Photo_Triplets_v1/triplets.json --output Runs/tone_global --global-only --max-steps 1000
python Model/train.py --manifest Dataset/Photo_Triplets_v1/triplets.json --output Runs/tone_joint --init-checkpoint Runs/tone_global/latest.pt --max-steps 10000
python Model/train.py --manifest Dataset/Photo_Triplets_v1/triplets.json --output Runs/tone_selected --init-checkpoint Runs/tone_joint/latest.pt --review Dataset/Review\ Datas/triplet-review-EXAMPLE.json --high-quality-only --learning-rate 0.00002 --max-steps 2000
```

最后一条只有实际存在人工优质标记时才可使用。审核只影响三元组选择，不删除原图。若未指定优质子集，未剔除的合成候选仍可参与训练；不将它们称为人工真值。

恢复同一阶段：增加max-steps并使用--resume和原output，保留原配置与global-only设置。新数据选择、损失或训练阶段用新output及--init-checkpoint。仅加载自己可信的checkpoint。恢复优化器和RNG状态不等于保证跨平台或非确定性CUDA算子的逐位一致性。

真实训练建议另加`--cache-dir Dataset/Training_Cache`，在计算节点本地按需缓存256长边lossless PNG，避免每步重复解码超大原图。该选项默认关闭；缓存由三元组文件stat和处理参数派生，修改源文件会失效。首版缓存按三元组分开，可能重复保存小尺寸content/reference，但不会复制完整原图；目录按hash分片。缓存仅优化IO，不改变训练张量。

对照配置位于configs：global_only、local_unprojected、local_mean、local_soft、without_style_aux、unbounded_projection。通过--config切换；固定训练预算与数据，记录参数差异。默认cfg是color投影，并不意味着已经优于对照。

## 原分辨率推理

```bash
python Model/infer.py --checkpoint Runs/tone_joint/latest.pt --content input.jpg --reference reference.jpg --output result.png
```

强制提供至少进行过训练更新的checkpoint，拒绝覆盖原文件。输出PNG，EXIF转正，保持输入尺寸。流式接口将CPU原图分块上传；CPU仍存原图/输出，并非零内存。完整Tensor的model.forward/render适合训练代理，显存仍随全图尺寸增长。

## 表达能力诊断

```bash
python Model/diagnose.py --manifest Dataset/Photo_Triplets_v1/triplets.json --output Runs/oracle_check --split train --limit 8 --steps 100 --device cuda
```

该工具读取目标拟合映射并在留出空间块测误差，是oracle诊断，不能当作实际模型推理性能；自由局部RGB场也不等同于本模型的精确参数族。

## 性质与限制

固定patch排列不变≠纯风格；输入产生空间图≠零幻觉；投影仅约束代理分辨率下有限颜色基的数值列空间；原分辨率和最终clamp后不保证同样正交。局部整体幅度界不约束全局映射的所有伪影。监督目标来自现有生成器，独立真实评测和人工审核仍不可省略。
