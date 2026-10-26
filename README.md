# 火焰数据集仿体系统（SE2026-AI-07）

物理仿真、多视角渲染、自动标注、质量筛选和 DVC 数据集版本。

流程：

1. **场景库**：一个场景就是一套燃料摆放，按版本保存，可以复制、复用。作业记下用的是哪个场景的哪一版。
2. **烘焙与渲染**：一次作业用 Blender Mantaflow 烘焙火，每种天气一次。每次烘焙按所选视角分别渲染，每个视角一台真实相机，每组渲染 `frame_count` 帧。天空、雨雪粒子和湿地、积雪都在 Blender 场景里完成。
3. **样本**：每一帧都是一个样本。火焰掩码来自 Blender 的发光通道（Blender 没产出画面时，参数化回退用来自温度场的投影），框、轮廓和关键点由掩码算出。火焰太小或画面过暗、过曝的帧标为「已筛除」，不能送审。噪声、模糊、光照变化另存为派生样本。
4. **送审**：在结果抽屉里选帧，送到顶栏选中的数据集，同时推给 Label Studio。审核结果自动回写，已发布的样本不再改动。
5. **SD 增强**（可选）：在数据集页对已审核的原始样本重绘背景，每个样本 1–4 份，按室内/室外轮换背景风格。火焰像素不变，派生样本直接是已审核、带同样的框，和原样本同一划分。
6. **发布**：版本是累积的，包含数据集里全部已审核样本，并记录父版本。同一段序列的帧固定在同一个 train/val/test 划分。导出 `data.yaml`、`images/<split>`、`labels/<split>`、COCO 和 VOC，可以直接训练。回滚是把某个版本设为当前版本，下次发布以它为基础。

房间火灾、林区火灾、工业区火灾是三套预设，用来填入结构、场地和默认摆放；墙、天花板这类方块和场地尺寸都可以再改。环境（室内/室外）由顶盖决定：不燃烧的方块顶到场地高度并盖住雨雪落下的范围时是室内，雨雪落在顶上，地面保持干燥；否则是室外，雨雪进到场地里。晴、雨、雪都可以选。每种天气单独烘焙：雨雪压低火焰、增加烟，雨雪是受重力和风驱动的粒子。环境温度和湿度改变浮力、入流温度和烟量。Stable Diffusion 不参与生成，只在数据集里对已审核样本重绘背景（第 5 步）。架构、部署、用户、测试、合规和验收交付物在 `docs/`，答辩提纲和技术博客也一并提供。

## 运行环境

使用 Python 3.11+ 与 PyTorch。运行数据默认在仓库的 `./data`（可用 `FIREPHANTOM_RUNTIME` 改到其他磁盘或目录），macOS、Linux 和 Windows 同一套路径。React/Vite 源码位于 `web/`，构建产物由 FastAPI 提供。

## 启动服务

macOS 和 Linux 在仓库根目录执行：

```bash
scripts/start-local.sh
```

Windows 使用同一套 Python 启动器：

```powershell
py -3.11 -m firephantom.local
```

也可以运行 `scripts\start-local.ps1`。停止服务时，macOS/Linux 用 `scripts/stop-local.sh`，Windows 用 `scripts\stop-local.ps1`。

人只在工作台登录：打开 `http://127.0.0.1:8765/`。第一次访问创建管理员，之后由管理员添加账号。工作台里的「打开 Label Studio」会直接进入标注页，不再出现第二套登录。`firephantom@localhost` 只作为 Label Studio 服务账号，密码在运行目录的 `label-studio.password`（默认 `./data/label-studio.password`，不要提交到 Git，也不要用它登录工作台）。

- 健康检查：`GET http://127.0.0.1:8765/health`
- 响应式工作台：`GET http://127.0.0.1:8765/`
- OpenAPI：`GET http://127.0.0.1:8765/openapi.json`
- 同步跑一个序列（调试用，烘焙要几分钟）：`POST /api/v1/pipeline`
- 场景库：`GET/POST /api/v1/scenes`、`GET/PUT/DELETE /api/v1/scenes/{id}`、`POST /api/v1/scenes/{id}/duplicate`、`PUT/GET /api/v1/scenes/{id}/thumbnail`
- 作业：`POST /api/v1/jobs`（每行一个序列，可带 `scene_id`）、`GET /api/v1/jobs/{job_id}`、`GET /api/v1/jobs/{job_id}/stages`、`POST /api/v1/jobs/{job_id}/continue`
- 序列文件（每个视角的视频、物理场、点云、求解器输入）：`GET /api/v1/sequences/{id}/files/{path}`
- 作业取消：`POST /api/v1/jobs/{job_id}/cancel`
- 本机资源与外部集成状态：`GET /api/v1/system/status`
- 能力目录：`GET /api/v1/catalog`
- Label Studio 回读：`GET /api/v1/annotations/label-studio`
- SD 增强：`GET/POST /api/v1/datasets/{id}/sd-augment`、`GET /api/v1/sd-augment/{job_id}`、`POST /api/v1/sd-augment/{job_id}/cancel`
- 样本：`GET /api/v1/samples`、`POST /api/v1/samples/sync-reviews`、`POST /api/v1/samples/{id}/review`
- 发布数据集版本：`POST /api/v1/datasets/{id}/publish`，发布预览：`GET /api/v1/datasets/{id}/plan`
- 版本和文件：`GET /api/v1/versions?dataset_id=&offset=0&limit=50`（默认 50 条，最多 200 条）、`GET /api/v1/versions/compare?base=&other=`、`GET /api/v1/versions/{id}/files`、`GET /api/v1/versions/{id}/archive`、`POST /api/v1/versions/{id}/rollback`、`POST /api/v1/versions/{id}/evaluations`

请求示例：

```bash
curl -s -X POST http://127.0.0.1:8765/api/v1/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"scenario":"房间火灾","wind_speed":1.5,"temperature":28,"humidity":45,"fuel_type":"木材","weathers":["晴"],"viewpoints":["侧视","俯视"],"frame_count":8}'
```

React 工作台有三页：场景库（打开场景后，左边摆放、右边调参数生成、结果就地查看）、作业、数据集与版本。地址用 hash 路由，刷新不丢页面。样本文件写入 `DATASET_ROOT/samples`，序列级文件写入 `DATASET_ROOT/sequences`；只有发布动作才把冻结后的清单和标注导出交给本地 DVC，未加密时版本里的图像是硬链接，不重复占空间。账号、作业、版本目录和样本索引用本机 PostgreSQL 库 `firephantom`，查询分页。图片、视频、点云和 DVC 缓存仍在磁盘上。

MP4 默认由 ffmpeg 编码为 H.264 `avc1/yuv420p` 并启用 `faststart`；版本文件接口支持 HTTP Range，前端可直接播放。没有 ffmpeg 时才使用 OpenCV `mp4v` 回退编码。

本地默认单 worker；部署到 GPU 云主机时可设置 `FIREPHANTOM_JOB_WORKERS=2` 到 `8` 并重新进行吞吐验收。并发数不会替代 SRS 要求的完整 1000 张和 10 万张实测。

16 GB macOS 会自动把作业 worker 限制为 1；仅在确认统一内存余量后设置 `FIREPHANTOM_ALLOW_CONCURRENT_MPS=true` 才允许并发 MPS 作业。

本机 Apple M4 只有 16 GB 统一内存，YOLO 训练/评估默认使用 `batch=1`、`imgsz=320`、`workers=0`，且要求至少 3 GiB 可用内存；运行检测任务前先停止 Tiny-SD 和其他 MPS 作业。低内存时工具会拒绝启动，`--allow-low-memory` 仅供短 smoke run 使用。

Tiny-SD 冷加载受 `FIREPHANTOM_MEMORY_GUARD` 保护；`GET /api/v1/system/status` 和工作台顶栏会显示当前可用内存。16 GB macOS 默认强制单 worker。

设置 `DATASET_ENCRYPTION_KEY` 可启用 Fernet 静态加密。`ANNOTATION_MODEL` 启用本地 YOLO 自动标注。`FIRE_REFERENCE_DATASET` 指向合法的真实火焰参考集，FID 固定抽样并缓存。下载资产的来源与许可见 `docs/data-and-model-provenance.md`。

渲染默认 256×256、每个视角 16 帧。本机实测（M4，16 GB）：房间火灾一次烘焙加 2 个视角 × 2 种天气 × 4 帧共 16 张，端到端 80 秒，产出 16 个带掩码标注的样本。开启背景重绘时每个组合多一次 Tiny-SD 调用。云端 txt2img 兼容端点可通过 `SD_API_URL` 配置，并可选 `SD_API_TOKEN` 与 `SD_HTTP_PROXY`。

离线验收清单可用 `python -m firephantom.evaluate predictions.json ground_truth.json` 计算标注 F1 和检测 mAP。

验收环境记录方式见 `docs/acceptance-report.md`；PIA、伦理审查和脱敏表分别见 `docs/PIA.md`、`docs/ethics-review.md` 和 `docs/anonymization-plan.md`。答辩提纲见 `docs/presentation-outline.md`，技术说明见 `docs/technical-blog.md`。

## 外部工具

每次仿真启动 Blender。每种天气烘焙一次，再按视角 × 帧渲染，同时写出发光通道作为掩码；超时按天气数和渲染张数放宽，可用 `BLENDER_TIMEOUT_SECONDS` 覆盖。种子会微调起火点，换种子就是另一团火。Blender 不可用时退回参数化羽流，并在结果里写明原因。温度、烟和速度网格来自这套羽流。

Stable Diffusion 只在数据集的 SD 增强里调用：每个派生样本一次带掩码的 img2img（`SD_API_URL` 默认 `http://127.0.0.1:7861/sdapi/v1/txt2img`，实际调同服务的 `/img2img`）。Label Studio 仍用 `LABEL_STUDIO_URL`；任务里的图片是 `FIREPHANTOM_PUBLIC_URL`（默认 `http://127.0.0.1:8765`）下的样本地址，不再内嵌 base64；API 自动对 `LABEL_STUDIO_URL` 的源开放 CORS，Label Studio 才能读取这些图片。

```bash
python -m firephantom.probe
```

FID 和 IS 在发布时对整个版本计算（FID 需要 `FIRE_REFERENCE_DATASET`），不再逐样本拿自己的渲染当参考。FID 使用 Fréchet 距离（越小越近），不换算成相似度百分比。Inception 权重不可用时，FID 和 IS 在同一组函数里改用固定的确定性嵌入和 4 类分类头，SSIM 不依赖权重。

无真实 Blender、模型权重或验证集时，系统会明确记录回退后端；SRS 中的物理误差、FID 相似度、标注准确率和 mAP 数字必须在真实验收数据上测量，不能由回退模式推断。

## 测试

```bash
python -m pytest
```

覆盖率门槛是应用包的 70%。

## React 前端

```bash
cd web
npm ci
npm run build
```

开发期使用 `npm run dev`，生产构建从 FastAPI 根路径提供。

## Docker Compose

```bash
docker compose config
docker compose up --build
```

容器内服务同样监听 `8765`，数据目录是 `/data`。
