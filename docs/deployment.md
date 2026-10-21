# 部署手册

## 本地与 Anaconda/PyTorch

T7 工作站依赖位于 `/Volumes/T7/FirePhantom-runtime`：Python 环境为 `conda-envs/firephantom`，数据和模型位于 `data/`、`models/`，Blender 位于 `tools/`。重新创建环境时使用 Python 3.11+ 并运行 `pip install -r requirements-dev.txt -r requirements-ml.txt`。前端依赖使用 `cd web && npm ci && npm run build`。

Apple Silicon 自动使用 MPS，训练器也可在 CUDA/CPU 上运行。本机是 Apple M4、16 GB 统一内存；启动 YOLO 训练或评估前必须停止 Tiny-SD 和其他 MPS 作业，保留至少 3 GiB 可用内存。训练器默认 `--batch 1`、`--imgsz 320`、`workers=0`，评估器也默认 `--batch 1`；低于阈值会拒绝启动，只有确认 swap 和内存压力可控时才使用 `--allow-low-memory`。YOLO11n 检测器 smoke run 可用 `python -m firephantom.annotation.train /Volumes/T7/FirePhantom-runtime/data/FASDD_CV/data.yaml /Volumes/T7/FirePhantom-runtime/models/yolo-evaluation --epochs 1 --fraction 0.001 --test-fraction 0.001 --imgsz 320 --batch 1 --device mps`。本机另有 10%/5 epoch 候选 checkpoint：`/Volumes/T7/FirePhantom-runtime/models/yolo-fire-smoke-10pct-5epoch-v1/fire-smoke/weights/best.pt`，对完整 test split 的 mAP50 为 0.511；由于 precision/recall 和人工准确率均未达验收要求，不设为默认。需要在 Label Studio 中试用时，在冷启动 API 前设置 `ANNOTATION_MODEL` 为该 checkpoint，并设置 `ANNOTATION_DEVICE=mps`，提交任务后由人工审核修正。快速本地模式跳过每 epoch 的重复验证，训练完成后仍执行独立测试；Ultralytics 4.177 可能忽略测试 fraction 并扫描完整 split，报告以实际输出为准。Stable Diffusion 服务和 Label Studio 分别由本地脚本启动，也可覆盖 `SD_API_URL`、`LABEL_STUDIO_URL`。工具由 `BLENDER_BIN` 配置。

本地 Tiny-SD 服务是 `python -m firephantom.synthesis.diffusion_server`，默认使用 FP32、256 像素、20 步和 MPS。4 步只用于速度基准。Apple M4 16 GB 已实测可以生成非空图像；FP16 在该环境曾输出全黑图像，因此默认使用 FP32。Tiny-SD 运行时会占用统一内存，启动 YOLO 训练/评估前应运行 `scripts/stop-local.sh` 释放它。服务首次加载时 4 步请求耗时 7.09 秒；4 步热态基准的三次请求为 0.703–0.713 秒，均值 0.708 秒。可运行 `python scripts/benchmark-sd.py --samples 10 --output /Volumes/T7/FirePhantom-runtime/reports/sd-benchmark.json` 重测（`--samples` 是请求总数）；报告会区分实测请求耗时与线性外推，并说明结果不包含物理仿真、渲染、标注、质量筛选和存储。需要云端兼容端点时设置 `SD_API_URL`；若供应商要求 Bearer 认证，设置 `SD_API_TOKEN`，如网络要求代理再设置 `SD_HTTP_PROXY`（例如 `http://127.0.0.1:10808`）。令牌只通过请求头发送，不写入日志或返回值。具体下载来源和许可见 `docs/data-and-model-provenance.md`。

视频导出优先使用本机或容器中的 ffmpeg 编码为 H.264 `avc1`、`yuv420p`、`faststart`，保证 Safari/Chromium 的网页播放和颜色通道兼容；没有 ffmpeg 时才回退到 OpenCV `mp4v`。版本文件接口支持 HTTP Range 请求，浏览器可以边下载边播放。容器镜像已安装 ffmpeg；本地可用 `FFMPEG_BIN` 指定路径。

`GET /api/v1/system/status` 会返回 Apple Silicon 的可用统一内存、swap 概况以及 Blender、Tiny-SD、Label Studio 的可达性。默认 `FIREPHANTOM_MIN_AVAILABLE_GB=3`；Tiny-SD 冷加载前若低于该值会拒绝启动，避免把系统推入更严重的 swap。需要在已确认风险的短 smoke run 中关闭保护时才设置 `FIREPHANTOM_MEMORY_GUARD=0`。

使用 `scripts/start-local.sh` 启动本地 API 和可用的模型/标注服务；默认前台监管本次创建的进程，按 Ctrl-C 会一并停止。也可用 `scripts/stop-local.sh` 停止记录在 T7 PID 文件中的服务。外接盘未挂载时不要将 `DATASET_ROOT` 改回仓库内目录，以免生成大文件写入内置盘。

工作台账号存在 FirePhantom 的 PostgreSQL 库 `firephantom`。第一次打开 `http://127.0.0.1:8765/` 时创建管理员，之后由管理员添加账号。打开标注页不会再要求 Label Studio 密码。Label Studio 1.23 社区版没有 OAuth；服务用影子账号签发浏览器会话。`firephantom@localhost` 只给启动脚本的 `--init` 和服务令牌使用，密码在 `/Volumes/T7/FirePhantom-runtime/label-studio.password`。服务令牌在同目录的 `label-studio.token`，长度是 40 个字符，因为 Label Studio 的 `authtoken_token.key` 是 `varchar(40)`。人的密码不会写入 Label Studio。

本机关系数据使用一个 PostgreSQL 16：`firephantom` 存账号、作业、版本和样本索引，`labelstudio` 存 Label Studio。图片、视频、点云和 DVC 仍在磁盘上。数据库密码在 `/Volumes/T7/FirePhantom-runtime/postgres.password`（权限 0600），会话签名密钥在同目录的 `firephantom-session.secret`。`scripts/start-local.sh` 在 5432 没有监听时用 Docker 启动 PostgreSQL，`shared_buffers` 设为 128MB。若本机 5432 已经有 PostgreSQL，脚本会直接使用它，密码必须与该文件一致。旧的 MySQL 数据和 SQLite 标注库不再导入。

启动脚本会在 T7 未挂载或运行时目录不存在时直接退出，不会自动在内置盘创建模型、DVC 或数据目录。直接运行 API 时也优先使用 `/Volumes/T7/FirePhantom-runtime/data/dataset` 作为默认数据根目录。

持久作业默认使用一个 worker，避免本地 Blender 与其他重任务争用内存和 I/O；在 16 GB macOS 上即使把 `FIREPHANTOM_JOB_WORKERS` 设得更高，系统也会强制限制为 1。云端 GPU 或多实例部署可设置 `FIREPHANTOM_JOB_WORKERS=2` 到 `8`；只有明确确认统一内存余量时才设置 `FIREPHANTOM_ALLOW_CONCURRENT_MPS=true`。每个 worker 的结果写入 PostgreSQL `job_results`，服务重启后会恢复排队作业。

跨域前端需显式设置 `CORS_ALLOW_ORIGINS`（逗号分隔）；`LABEL_STUDIO_URL` 所在的源（本机时同时含 `127.0.0.1` 与 `localhost`）会自动加入，供 Label Studio 跨域读取样本图片；生产传输可设置 `REQUIRE_HTTPS=true`，由反向代理提供 TLS。

## Docker Compose

运行 `docker compose config` 和 `docker compose up --build`。`DATASET_ROOT=/data` 使用持久卷。生产环境应把 `DATASET_ENCRYPTION_KEY` 放到 Secret 管理器，通过 TLS 反向代理暴露服务，并限制版本文件接口的访问角色。

## Kubernetes

`deploy/k8s.yaml` 提供 PVC、Secret、Deployment、Service 和健康探针。运行 `kubectl apply -f deploy/k8s.yaml` 后用 `kubectl rollout status deployment/firephantom` 等待就绪。CI 流水线位于 `.github/workflows/ci.yml`，执行单元测试、覆盖率门槛和 Compose 配置检查。

## Git Flow

建议使用 `main` 保存可发布版本，功能从 `codex/feature-*` 或 `feature/*` 分支开发，修复使用 `fix/*`，合并前必须通过 CI。发布时创建带版本号的 tag，并把对应 DatasetStore 的版本 ID 和验收报告一并归档。
