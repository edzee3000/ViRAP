# MAE-tone：完整架构设计实例

使用公开MAE-B/16的编码器，无随机mask、无MAE像素解码器。共享输入/参考摄影编码器，16个查询读取，参考条件空间图和少量局部3D LUT。

- M0：K=2，91,401,581参数，骨干替换对照。
- M1：K=4，91,402,999参数，推荐待验证候选。
- 骨干85,798,656参数含151,296个固定sin/cos位置元素；共享参数仅计一次，通常执行两次。
- M1：heads训练5,604,343参数；joint训练33,957,367参数；full训练91,251,703参数，位置仍固定。

```powershell
python Model/inspect_mae_tone.py --structure-only --smoke
python Model/inspect_mae_tone.py --structure-only --local-rank 2 --output Model/mae_tone_m0_report.json
python -m unittest discover -s Model/tests -v
```

`MAEToneModel.from_pretrained(weights=...)`读取官方MAE预训练checkpoint，仅取encoder并严格加载，不使用分类微调权重。不会自动下载权重。`allow_uninitialized=True`仅用于结构检查。

配置：`configs/mae_tone_m0.json`、`configs/mae_tone_m1.json`。这些不是旧V1 train.py可读取的配置；尚未提供完整MAE-tone训练CLI或完整模型训练结果。

已打通ToneLoss所需的global_delta/preclamp_output接口。训练以三元组重建为主，不额外宣称纯风格解耦。独立encoder-only阶段被拒绝，避免冻结零输出头导致无重建梯度。

部署仍需要整个约9140万参数模型；冻结参数不是免费参数。原比例内容按像素查表输出，但编码分支仍缩放到224方形，空间权重仍64×64双线性查询，没有实现高分辨率边缘细化或显式区域对应。

完整说明：`Research/PETAL/MAE共享摄影编码器完整架构_20261030.md`。源码：`tone_model/mae_tone.py`及复用的`tone_model/v2.py`。

官方来源：https://github.com/facebookresearch/mae 。公开MAE预训练checkpoint的来源/hash沿用已完成的编码器探针记录；当前本地结构检查未加载该权重，不代表新完整模型效果已获验证。
