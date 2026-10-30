# Trinity 数据生成运行记录

日期：2026-10-29。远端项目：`/data3/shihangz/photo-tone-transfer/`。计算节点：`trinity-2-23`。节点内临时项目：`/tmp/shihangz/photo-tone-transfer/`。本地对应目录：`E:\大学课程\毕业论文`。head 仅作 SSH 跳板，不运行实验。

## 数据与配对

上传4519张审核后保留的原图，保持Dataset/Original Data下来源目录；每张原图只保存一次，配对与三元组均使用相对路径。冻结源图分组与训练/验证/测试划分，沿用Photo_Pairs_v1/source_images.json。

Photo_Pairs_60000_v1/pairs.json共60000组：59100训练、400验证、400测试、100同图诊断。包含原8090组，原200组试运行结果复用。原图、近重复组不会跨划分；保守分组不保证全部连拍/场景/摄影师隔离。随机配对，不依据来源限定图片角色；训练参考使用最多30次。

## 首批200组结果

200/200成功，失败0；输出哈希和解码检查通过。单对中位0.636秒，P95 0.799秒，最大峰值显存6064MiB。抽查8组三元组，内容结构保留且输出受参考影响，但有部分高光过亮或偏色案例，不能把技术成功等同于通过人工质量审核。抽查图：Research/data_collection/trinity_pilot_contact.jpg。

## 生成配置

PETAL代码提交162bad662d5a108285bb308699faed2888e3d782；现有权重来源/data3/shihangz/petal-demo/weights/ckpts。仅生成图像监督，不向自有模型暴露编码器特征或权重。

PyTorch 2.8.0+cu129，torchvision 0.23.0+cu129，kornia 0.8.2，Pillow 11.3.0，numpy 2.2.6。FP32，模型style_input_size=512，guidance_scale=1。输入和参考均先EXIF转正、转RGB，再等比例限制长边768，不裁剪、不放大。未执行嵌入ICC到sRGB的转换。PNG compress_level=3。输出大小匹配经过上述预处理的输入图；原始照片保持不变。

8个独立进程使用GPU0–7，每个进程模型仅加载一次。GPU8–9保留空闲。tmux会话photo-bulk-20261029，日志Logs/bulk_controller.log及Logs/bulk60000/。批量开始前重新检查GPU占用，如被占用则停止，不抢卡。

## 存储与故障保护

/scratch仅55GB空闲，不使用。/ssd0和/ssd1是系统根分区下空目录，并非独立大盘，不使用。采用节点本地/tmp（tmpfs，启动时249GB空闲），每次生成前检查并保留至少150GiB空闲；不在/data3上运行推理。跨网使用现有SSH路由，没有假定或虚构InfiniBand SSH路径。

全部原图传完后在计算节点逐文件SHA256核验，归档原图和元数据到/data3，再运行全量生成。生成期间输出和日志只写节点本地；结束后rsync统一归档。3次错误会停止对应worker，并保存失败信息；批量结束时即使有失败也整理并归档已有结果。

输出：Dataset/Generated_Outputs/{split}/{shard}/{pair_id}.png，按目录分片。最终清单：Dataset/Photo_Triplets_v1/triplets.json和triplets.jsonl，逐条记录输入、参考、输出路径及生成参数、哈希和状态。所有状态为generated_unreviewed，尚未通过人工筛选；本阶段不训练自有迁移模型。

全量完成判据：Logs/bulk60000/final_exit_status为0，且Dataset/Photo_Triplets_v1/summary.json中generated_verified=60000、failed_or_missing=0。启动不等于完成，应以这些文件为准。
