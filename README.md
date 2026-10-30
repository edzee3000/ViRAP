# ViRAP

参考图引导的摄影调色研究代码：数据准备与审核、模型训练、评测，以及 Trinity 实验监控工具。

## 代码结构

- `Model/`：调色网络、MAE/LoRA 适配、训练与推理、配置、测试和监控。
- `Research/data_collection/`：图像收集、配对、目标生成及人工审核工具。
- `Research/`：实验启动、部署脚本和研究记录。
- `Prompts/`：研究提示与设计记录。
- `Dataset/*.ps1`：本地审核与监控入口脚本。
- `Remote_Code/`：远端实验使用的代码快照及与本地不同的版本，按远端来源保留。

本地代码作为主目录版本；远端独有文件补入对应目录，不同版本保存在 `Remote_Code/`。同步来源与 SHA256 见 `Research/Code_Sync_20261030.json`，包括 Trinity 活跃目录和长期存储目录。快照反映同步时的代码，不表示对应实验均已完成。

## 使用

使用 Python 3.10+，根据实验安装依赖：

```bash
pip install -r Model/requirements.txt
# 编码器实验需要额外依赖：
pip install -r Model/requirements-encoder-probe.txt
python -m unittest discover -s Model/tests -v
```

模型与基础训练说明见 [Model/README.md](Model/README.md)。MAE 适配实验说明见 [Research/MAE_Adaptation_Sweep_20261030.md](Research/MAE_Adaptation_Sweep_20261030.md)。部分脚本保留原 Trinity 路径、节点与端口，其他环境需要修改；监控脚本需要已有 SSH 配置。

训练和推理需要自行准备数据与预训练权重。数据构建、研究记录及部分脚本使用 PETAL 生成目标图，相关来源和方法请参阅对应文档。下载的第三方源码未纳入仓库；需要 DINOv2/DINOv3 的实验请按配置准备官方依赖。

## 上传范围

本仓库只保存源码、模型配置和研究文档，不包含原始照片、生成图片、配对/三元组清单、人工审核导出、模型权重、训练缓存、结果、日志、压缩包或论文 PDF。这些文件保留在原本的本地/远端位置，`.gitignore` 会排除它们。

GitHub 代码仓库与同名的外部数据集仓库用途不同。不要将整个数据目录或运行环境加入 Git。
