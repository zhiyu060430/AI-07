# 验收报告

本报告把可重复的离线计算和需要外部依赖的现场验收分开。回退仿真、确定性图像嵌入和规则标注用于开发测试；它们不能证明 SRS 中的真实数据指标。

2026-10-17 起，现行版本不再运行 FDS。下面记录的是移除之前的现场测量。

## 本地基线

在仓库根目录执行：

```bash
python -m pytest -q
python -m compileall -q firephantom
docker compose config --quiet
cd web && npm run build && cd ..
git diff --check
```

测试命令会检查覆盖率门槛（至少 70%）、三种基础场景、四种增强、三种导出格式、DVC 增量/冲突/回滚、加密、HTTP API、批处理和版本归档。`python -m firephantom.probe` 是轻量连通性诊断：Blender 没有输入文件时会显示未运行，Label Studio 需提供 token；完整外部工具验收应通过下方 API 端到端流程。

本机是 16 GB 统一内存，检测任务按低内存配置运行：YOLO 训练/评估默认 `batch=1`、`imgsz=320`、`workers=0`，启动时至少保留 3 GiB 可用内存；Tiny-SD 和 GAN 训练在检测任务期间必须停止。曾用 `batch=16,imgsz=640` 评估时出现可用内存不足和 NMS 超时，已停止该进程，后续报告均使用低内存策略。

本地验收环境为 Apple M4、16 GB 统一内存，项目环境使用 Python 3.11.15、PyTorch 2.10.0、Ultralytics 8.4.177、MPS；运行时、模型、数据集和生成文件均放在 `/Volumes/T7/FirePhantom-runtime`。2026-10-15 最新全量测试为 80 项通过、覆盖率 74.54%；`compileall`、`git diff --check`、`docker compose config --quiet` 和 Vite 生产构建也通过。服务重启后 API 与 Label Studio 健康；`/api/v1/system/status` 可返回 16 GiB 机器的实时可用内存、swap 和四个外部集成状态。Tiny-SD 最后一次冷启动复验用 256×256、4 步请求生成 2 张有效图片，冷态首张 8.215 秒，热态 0.706 秒，后端报告 MPS。此前用 `scripts/benchmark-sd.py` 连续向本机 MPS 服务提交 1,000 次：所有 1,000 张输出均通过尺寸/非空检查，总耗时 735.246 秒（12 分 15 秒），首请求 11.99 秒，999 次热态请求均值 0.722 秒、P95 0.741 秒。完整 JSON 在 T7 的 `reports/sd-1000-benchmark-20261015.json`。这验证了纯 txt2img 图片生成端点在该配置下满足 2 小时时限，不包含 FDS、Blender、其他合成模态、标注、质量评估和版本存储。单样本全流程实测 30.9 秒，线性估算 1000 项约 8.6 小时，不能满足 4 小时门槛；这个估算也不是批量验收。当时的作业结果和版本目录由 SQLite 分页读取。现行存储是 PostgreSQL，不再导入旧 SQLite 或 `index.json`。

版本目录索引在 T7 的项目 Python 环境中使用 `scripts/benchmark-version-catalog.py --count 100000` 测试，写入 100,000 条合成元数据耗时 0.406 秒（约 246,504 条/秒）；COUNT 为 1.021 ms，倒序 50 条首页为 0.549 ms，倒序 50 条深页为 1.652 ms，主键查询为 0.400 ms。命令与完整结果可复现；该测试当时只验证 SQLite 元数据目录，不写入图像、不调用 DVC、不运行生成/标注/导出流水线，也不构成 SRS 的 10 万张数据集容量验收。现行目录在 PostgreSQL 上，这组数字不能当作 PostgreSQL 的成绩。

在仓库根目录复现该目录压测：

```bash
PYTHONPATH=. /Volumes/T7/FirePhantom-runtime/conda-envs/firephantom/bin/python \
  scripts/benchmark-version-catalog.py --count 100000 \
  --parent /Volumes/T7/FirePhantom-runtime
```

## 最新复验（2026-10-15）

本轮使用 T7 上的 Python 3.11.15 环境执行完整测试集：80 项通过、覆盖率 74.54%。`compileall`、`docker compose config --quiet`、`git diff --check` 和 Vite 生产构建均通过。M4 低内存配置下，YOLO 置信度评估使用 MPS、320 像素、batch 1；validation 与独立 test 各按 fire、smoke、both、background 抽取 128 张，confidence 在 validation 上选择为 0.3，test 框级 F1 为 0.583（precision 0.658、recall 0.523），耗时 65.344 秒。报告文件：`/Volumes/T7/FirePhantom-runtime/reports/yolo-confidence-tuning-512-lowmem-20261015.json`。该结果仍低于 85% 门槛，未把 checkpoint 设为默认模型。本轮评估期间系统 memory pressure 显示约 80% 可用，未观察到 MPS 进程残留。新增作业取消接口、T7 默认数据根目录、16 GB macOS 单 worker 保护、GAN/Tiny-SD 冷加载内存保护均通过测试；默认冷加载保护要求至少 3 GiB 可用空间。

本机训练和评估默认 batch 1、imgsz 320、workers 0，并在可观测可用内存低于 3 GiB 时拒绝启动；GPU 模型重任务需串行执行。完整 1000 样本端到端时限、物理误差、FID/IS/SSIM 验收阈值、人工标注准确率、下游 mAP 增益、可用性和合规批准仍未验收。

## 本地端到端演示

2026-10-15 使用 `scripts/start-local.sh` 启动 API、Tiny-SD 和 Label Studio 后，提交一个 30 秒房间火灾场景（风速 1.5 m/s、室内、晴、侧视、四种增强）。API 返回 HTTP 200，端到端耗时 30.9 秒。FDS 生成并成功求解 30 秒场景，Blender 4.5.9 成功渲染 PNG；Tiny-SD 经本地 MPS 服务生成图像；Label Studio 项目 1 接收任务并返回 201。数据集版本 `7726cf56350eafda8e2ebf73b19afd9d530781682461323c5ef20219967a7b0d` 已落盘，包含 FDS/Blender 产物、图像、8 帧视频、369 点点云、标注任务、质量报告及 COCO、Pascal VOC、YOLO 导出。

FDS Plot3D 接入和作业 SQL 迁移后的异步复验中，`POST /api/v1/jobs` 完成一个真实样本；FDS、Blender 均成功，Label Studio 创建任务返回 201，独立预览返回 PNG，31 个版本文件包含 `.q`/`.xyz`、字段、标注提交状态及 COCO/VOC/YOLO。作业 `494615daa1e24f43adae1273fe5d52f4` 对应版本 `f4028ffe1e0a0c6e172367c0e78c8b63025c98f781e3845ef56113d8ad60f4ee`。服务启动同时将 T7 上两条旧作业结果迁入 SQL 表并验证可读取；这仍不是 1000 张吞吐或大规模压测。

这次演示中的质量结果为 FID 386.351（FASDD_CV 验证集 128 张参考图，Inception v3 权重已加载）、IS 1.417、SSIM 0.149；SQL 迁移后的最新作业记录 FID 388.704。系统因默认宽松阈值保留了该样本；这不代表质量达到 SRS。GAN checkpoint 已在 128 张 FASDD_CV 训练图上完成 3 个 epoch 的 MPS smoke training，加载后实际生成路径为 PyTorch DCGAN；该小样本模型不代表正式质量。当前默认标注仍为物理启发式；新的 YOLO checkpoint 已训练并通过独立测试，但还没有人工标注准确率验收，因此只作为可选的 Label Studio 预标注来源。

随后已把 FDS 6.11.1 的 Plot3D 快照接入字段流水线。使用相同 30 秒房间火灾工况运行真实求解器，解析得到 `[16,16,8]` 温度与烟尘场、`[16,16,8,3]` 速度场；实际后端标记为 `fds_plot3d`，温度范围为 28.02–358.63 °C，SOOT 质量分数范围为 4.67e-8–6.69e-4。原始 `.q` 和坐标 `.xyz` 现随保留的数据版本一起归档。此次快照验证证明数据接线正确，不提供与真实火场的误差验证。

YOLO11n 的 MPS smoke run 完成 1 个 epoch，仅使用 0.1% 训练数据；此前 595 张验证图上的 mAP50 为 0.00051。之后 1%/1 epoch checkpoint 对完整 9,512 张 test split（10,383 个实例）的 mAP50 为 0.01447。为验证本机训练能力，另用 10% 训练集完成 5 个 epoch（训练日志 850.7 秒），并对同一完整测试集评估 7 分 49 秒：mAP50 0.51091、mAP50-95 0.29590、precision 0.60084、recall 0.46718。验证日志给出的每图预处理、推理、后处理分别为 0.2、5.4、27.6 ms；这不包括 Label Studio 网络提交或人工复核。另在独立 test split 按 fire、smoke、both、background 各抽 128 张，以 confidence 0.25、IoU 0.5 计算框级标注指标：512 张的 precision 0.568、recall 0.601、F1 0.584；输出记录在 T7 的 `reports/yolo-annotation-accuracy-512-20261015.json`。新 checkpoint 位于 T7 的 `models/yolo-fire-smoke-10pct-5epoch-v1/fire-smoke/weights/best.pt`。它是可选的人工作业预标注候选，未达到 85% 标注准确率，因此没有设为默认模型并仍需人工复核。

## 真实验收输入

现场验收环境应登记以下输入的来源、许可证、版本和校验值：

| 项目 | 配置 | 说明 |
| --- | --- | --- |
| FDS | `fds` 可执行文件 | 运行生成的 `.fds` deck，保存 CSV/JSON 探针输出 |
| Blender | `blender` 可执行文件 | 运行生成的场景脚本，保存 PNG 渲染 |
| Stable Diffusion | `SD_API_URL` | 固定模型、采样器、seed 和服务版本 |
| Inception | `INCEPTION_WEIGHTS` | 只从本地权重加载，不在运行时下载 |
| 真实参考集 | `FIRE_REFERENCE_DATASET` | 图片目录或每行一个路径的 manifest |
| Label Studio | `LABEL_STUDIO_URL` | 记录人工复核后的 task 导出 |

## 指标记录

物理场误差使用 `firephantom.evaluation.physics_error`，要求温度、烟雾和流场的 `mean_percent` 分别记录，并与真实或经过校准的基准场比较。标注准确率使用 `annotation_accuracy` 的 precision、recall、F1；检测模型 mAP 使用 `detection_map`。示例：

```bash
python -m firephantom.evaluate predictions.json ground_truth.json --iou 0.5
```

每次验收应保存命令输出、数据集版本 ID、模型 profile、运行时长和硬件。只有真实参考集、真实权重和独立验证集上的结果才能填写下面的 SRS 门槛：

| SRS 门槛 | 当前结果 | 状态与所需证据 |
| --- | --- | --- |
| 物理仿真误差不超过 5% | FDS 完成一组 30 秒场景；没有校准基准场误差 | 未验收；需真实或经校准的基准场，以及 `physics_error` 输出 |
| FID 对应相似度至少 80% | FID 386.351（最新作业 388.704），128 张 FASDD_CV 验证图作参考 | 未达标；SRS 需定义“80% 相似度”如何映射到 FID 距离，并使用独立真实参考集验收 |
| 标注平台每小时处理至少 50 张 | YOLO 验证器记录每图预处理、推理、后处理合计约 33.2 ms；无 Label Studio/人工任务吞吐测试 | 未验收；当前仅证明模型推理有余量，不代表完整任务处理或人工标注速度 |
| 自动标注准确率至少 85% | 最新分层复验：validation/test 各 512 张，IoU 0.5，confidence 由 validation 调至 0.3；test F1 0.583（P 0.658、R 0.523） | 未达标；需提升模型并由领域人员审核独立真值 |
| 合成数据使下游 mAP 提升至少 15% | 10%/5 epoch checkpoint 在完整 test split 的 mAP50 为 0.51091；没有 real-only 对照 | 未验收；需同一模型/训练预算的 real-only 与 real+synthetic 对照实验 |
| 单批次合成 1000 张不超过 2 小时 | MPS txt2img 实际连续生成并检查 1,000 张，735.246 秒（12 分 15 秒） | SD 图片生成端点在本机通过；含多模态产物、标注和版本归档的完整批次尚未验收 |
| 端到端批次不超过 4 小时 | 单样本全链路 30.9 秒 | 未验收；单样本时长不能替代批次性能测试 |
| 可用性至少 99.5% | 已验证单次本地启动与健康检查 | 未验收；需持续运行/故障恢复测试 |
| 数据来源、PIA、伦理审查、脱敏合规 | 已记录数据/模型来源和许可；实现了脱敏工具 | 未验收；需项目级 PIA、伦理审查记录和实际数据脱敏复核 |

SRS 的“支持 >10 万张”仍未进行真实图像文件、DVC 缓存和全流程的规模验收；已完成的 100,000 条 SQLite 目录压测仅证明元数据分页层可处理该记录数。M4 本地已能在约 12 分钟内完成 1000 次 Tiny-SD 图像生成请求，适合本地开发、模型验证、短场景 FDS/Blender、千张级纯图像生成与交互式标注。完整端到端 1000 样本若保持当前单 worker 单样本耗时会超过 4 小时；大规模训练、并发全流程和 >10 万张文件存储压测应评估云端 GPU/存储。

## 可运行演示

启动 API 后打开 `/`，提交一个场景即可看到物理场摘要、PNG、MP4、点云、Label Studio 任务、质量指标和版本文件。PPT 与演示视频按项目范围不制作；API 文档和可运行 Web 工作台用于本地演示。
