> Historical V2 fixed-map architecture. Current defaults and counts: [V2_1_README.md](V2_1_README.md). Explicit legacy config: configs/v2.json.

# v0.2 架构设计实例

精确总参数：27,233,197。共享摄影编码器25,162,624；正式设计为官方DINOv2-S/14初始化＋Lab光度支路＋多层query读出。当前仅结构验证，未加载预训练权重、未训练，不能用于调色效果展示。

完整模块尺寸/参数表：Research/PETAL/v2完整架构与参数表_20261030.md。

文件：tone_model/v2.py（网络），inspect_v2.py（实例计数与烟测），v2_architecture_report.json（实际统计），tests/test_v2.py（7项测试）。

官方源码位于Research/External_Code/dinov2，commit7764ea0f912e53c92e82eb78a2a1631e92725fc8。干净环境可先git clone该仓库并固定commit；不要改官方构造img_size=518。运行尺寸280是位置插值，不是替换存储位置参数。

```powershell
python Model/inspect_v2.py --structure-only --smoke
python -m unittest discover -s Model/tests -p test_v2.py -v
```

--structure-only明确允许随机结构检查。正常工厂默认要求提供官方本地权重路径；不会自动下载或把随机权重叫预训练。共享参数只计一次，输入/参考通常各编码一次，所有冻结参数计入完整模型。

V2尚未接入V1训练/推理CLI，不要将其当作已有正式训练入口。forward支持小尺寸验证，尚未接CPU原图流式部署；默认关闭显式颜色投影，作为独立对照保留。不得以参数变大、已通过形状测试或采用预训练结构就声称方法更好或满足投稿贡献要求。
