# V2.1：参考条件化空间权重

默认总参数27,315,437，空间模块358,498。新增82,240参数，通过256维pair条件在96/64通道空间特征上实施FiLM。正式画质仍需训练验证。

新配置：`configs/v2_1.json`。原版：`V2Config(spatial_conditioning='none')`、`configs/v2.json`，保留原参数键及27,233,197总数。旧权重须显式选none再严格加载，不要静默忽略新参数。

```powershell
python Model/inspect_v2.py --structure-only --smoke
python Model/inspect_v2.py --structure-only --spatial-conditioning none --output Model/v2_legacy_recheck.json
python -m unittest discover -s Model/tests -v
```

默认统计输出`v2_1_architecture_report.json`；旧`v2_architecture_report.json`保留历史记录。`--local-rank 4` / `8`支持更多局部分量。

完整说明：`Research/PETAL/V2.1参考条件空间调色修订_20261030.md`。源码仍为`Model/tone_model/v2.py`。

`--structure-only`明确允许随机初始化的结构检查；真实使用必须提供公开预训练权重。共享参数计一次、输入与参考通常各运行一次。V2.1尚未接入旧V1训练CLI或原图CPU流式部署。

本次只修改本地V2.1，未改动远端正在运行的全局LUT编码器对照实验。
