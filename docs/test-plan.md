# 测试计划与报告

## 自动化范围

`tests/` 覆盖物理参数敏感性、三类场景、Blender 脚本与帧缓存、img2img 保留初始火焰、扩散失败时视频等于引擎帧、环境/天气/视角的 18 组展开、四类增强、Label Studio 任务、COCO/VOC/YOLO、一致性指标、质量筛选、DVC 冲突/回滚、脱敏、作业 SQL 分页/旧数据迁移/部分失败状态和 HTTP/OpenAPI。

## 执行

运行 `python -m pytest`。当前基线要求覆盖率至少 70%；CI 会执行同一命令。外部 Blender、Stable Diffusion、Label Studio、真实 Inception 权重和真实火焰验证集属于集成/验收环境；使用真实配置提交 API 端到端请求并保存版本/质量报告。`python -m firephantom.probe` 仅作连通性诊断，不会在没有输入文件时运行 Blender，也不会替代真实数据验收。

本地 M4 机器只有 16 GB 统一内存。YOLO 训练和评估采用 `imgsz=320`、`batch=1`、`workers=0`，启动前检查 macOS 可用内存，默认至少保留 3 GiB；Tiny-SD 和 YOLO 不得同时运行。评估或训练期间持续观察 `vm_stat`、`vm.swapusage` 和进程 RSS；可用内存不足或 swap 持续增长时立即停止任务。`--allow-low-memory` 只用于已确认风险的短 smoke run，不能作为正式验收配置。

## 验收证据

性能报告应记录单场景耗时、1000 张批次耗时、标注吞吐量和端到端耗时；质量报告应记录真实 Inception FID/IS/SSIM、人工/基准标注准确率及下游检测 mAP。`python -m firephantom.evaluate predictions.json ground_truth.json` 输出 IoU、precision/recall/F1 和 mAP。检测训练器的 `--fraction` 只限制训练集，`--test-fraction` 独立记录测试请求；Ultralytics 4.177 的验证器可能仍扫描完整 test split，因此报告应以实际输出的图像数为准。没有对应真实基准时不得宣称 SRS 的 5%、80%、85% 或 15% 数字已经达标。
