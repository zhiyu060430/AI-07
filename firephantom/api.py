"""HTTP API for the flame-dataset phantom pipeline."""

from __future__ import annotations

import logging
import os
import io
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from firephantom import __version__
from firephantom.annotation.labelstudio import endpoint_from_env, latest_annotation, objects_from_annotation, pull_tasks
from firephantom.images import browser_compatible_video
from firephantom.augment_jobs import AugmentJobs
from firephantom.jobs import JobManager
from firephantom.pipeline import SCENARIOS, SCENES, VIEWPOINTS, WEATHERS, run_sequence
from firephantom.runtime import safe_job_workers, system_status
from firephantom.synthesis.augment import AUGMENTATIONS
from firephantom.versioning.store import VersionConflict

logger = logging.getLogger("firephantom")


def _byte_response(data: bytes, media_type: str, request: Request) -> Response:
    """Return bytes with HTTP range support required by browser video players."""
    total = len(data)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-cache",
        "Content-Length": str(total),
    }
    range_header = request.headers.get("range", "").strip()
    if not range_header:
        return Response(content=data, media_type=media_type, headers=headers)
    if not range_header.startswith("bytes="):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})
    value = range_header[6:].split(",", 1)[0].strip()
    try:
        start_text, end_text = value.split("-", 1)
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start, end = max(total - suffix_length, 0), total - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else total - 1
            if start < 0 or start >= total or end < start:
                raise ValueError
            end = min(end, total - 1)
    except (TypeError, ValueError):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})
    body = data[start : end + 1]
    headers.update(
        {
            "Content-Length": str(len(body)),
            "Content-Range": f"bytes {start}-{end}/{total}",
        }
    )
    return Response(status_code=206, content=body, media_type=media_type, headers=headers)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


@asynccontextmanager
async def lifespan(application: FastAPI):
    yield
    for name in ("job_manager", "augment_jobs"):
        manager = getattr(application.state, name, None)
        if manager is not None:
            manager.close()


def default_dataset_root() -> Path:
    """Dataset directory. ``DATASET_ROOT`` wins, otherwise ``<runtime>/dataset``."""
    from firephantom.paths import dataset_root

    return dataset_root()


def _cors_origins() -> list[str]:
    """Configured origins plus Label Studio, which loads sample images cross-origin."""
    origins = [origin.strip() for origin in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if origin.strip()]
    target = urlsplit(endpoint_from_env())
    if target.scheme and target.hostname:
        port = f":{target.port}" if target.port else ""
        hosts = [target.hostname]
        if target.hostname in {"127.0.0.1", "localhost"}:
            hosts = ["127.0.0.1", "localhost"]
        origins.extend(f"{target.scheme}://{host}{port}" for host in hosts)
    return list(dict.fromkeys(origins))


class SdAugmentRequest(BaseModel):
    """SD background restyle of reviewed samples. Flame pixels and reviewed boxes stay."""

    sample_ids: list[str] | None = Field(default=None, description="留空表示数据集里全部已审核、非 SD 派生的样本")
    variants: int = Field(1, ge=1, le=4, description="每个样本生成几份，背景风格依次轮换")
    styles: list[str] | None = Field(default=None, description="背景风格；留空用环境对应的全部风格")
    strength: float = Field(0.2, gt=0.0, le=0.6)
    steps: int = Field(20, ge=1, le=20)
    cfg_scale: float = Field(7.0, ge=0.0, le=20.0)


class SequenceRequest(BaseModel):
    """One bake. Every environment x weather x viewpoint is rendered from it."""

    scenario: str = Field("房间火灾", description="基础火灾场景：房间火灾、林区火灾、工业区火灾", examples=["房间火灾"])
    scene_id: str | None = Field(default=None, description="场景库里的场景；给出时覆盖 scenario、fuel_type 和 placements")
    scene_revision: int | None = Field(default=None, ge=1)
    wind_speed: float = Field(..., description="风速，米每秒", ge=0.0, le=100.0)
    temperature: float = Field(..., description="环境温度，摄氏度", ge=-80.0, le=100.0)
    humidity: float = Field(..., description="相对湿度，百分比", ge=0.0, le=100.0)
    fuel_type: str = Field("木材", description="起火点燃料", examples=["木材"])
    scene: str | None = Field(default=None, description="已废弃：环境由顶盖决定。给出时必须和这一烘推导出的室内或室外一致")
    weather: str | None = None
    viewpoint: str | None = None
    scenes: list[str] | None = Field(default=None, description="已废弃，同 scene")
    weathers: list[str] | None = Field(default=None, description="晴、雨、雪；每种天气单独烘焙一次，雨雪会压低火焰、增加烟。顶盖盖住时雨雪落在顶上")
    viewpoints: list[str] | None = Field(default=None, description="俯视、侧视、第一人称；每个视角一台相机")
    frame_count: int = Field(16, ge=1, le=48, description="每个视角渲染的帧数；每帧一个样本")
    fire_engine: str = Field("blender", description="仿真引擎。只接受 blender，留空也是 Blender", examples=["blender"])
    placements: list[dict[str, Any]] = Field(default_factory=list, description="燃料包摆放。空列表表示不加额外燃料。")
    lights: list[dict[str, Any]] | None = Field(default=None, description="场景灯。留空用自动补光和太阳，空列表表示没有灯。")
    augmentations: list[str] = Field(default_factory=list, description="噪声、模糊、光照变化；每种增强另存一份派生样本")
    diffusion: dict[str, Any] | None = Field(
        default=None, description="已废弃：生成阶段不再调用 SD；请用 /api/v1/datasets/{id}/sd-augment"
    )
    min_flame_area: float = Field(0.002, ge=0.0, le=0.5, description="火焰像素占比低于它的帧标为已筛除")
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, description="随机种子；留空时按参数推导。种子会微调起火点，换种子就是另一团火")


class JobItem(SequenceRequest):
    """One workbench row: a sequence, repeated ``count`` times with new seeds."""

    count: int = Field(1, ge=1, le=100)
    wind_step: float = Field(0.0, ge=0.0, le=100.0)


class ReviewRequest(BaseModel):
    objects: list[dict] = Field(default_factory=list)
    decision: str = Field(..., description="reviewed 或 rejected")
    rereview: bool = Field(False, description="已审核但未发布的样本重新审核")


class DetectionEvaluation(BaseModel):
    map50: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    map50_95: float | None = Field(default=None, ge=0.0, le=1.0)


class JobBatchRequest(BaseModel):
    items: list[JobItem] = Field(..., min_length=1, max_length=100)
    staged: bool = True


class ContinueRequest(BaseModel):
    from_stage: str = "engine"
    combo_index: int | None = None
    items: list[JobItem] = Field(..., min_length=1, max_length=100)


class SceneCreate(BaseModel):
    name: str
    scenario: str
    origin_fuel: str = "木材"
    placements: list[dict[str, Any]] = Field(default_factory=list)
    origin: list[float] | None = None
    cameras: dict[str, Any] | None = None
    gas_domain: dict[str, Any] | None = None
    lights: list[dict[str, Any]] | None = None
    simulation: dict[str, Any] | None = None
    structures: list[dict[str, Any]] | None = None
    domain: list[float] | None = None


class SceneUpdate(BaseModel):
    name: str | None = None
    scenario: str | None = None
    origin_fuel: str | None = None
    placements: list[dict[str, Any]] | None = None
    origin: list[float] | None = None
    cameras: dict[str, Any] | None = None
    gas_domain: dict[str, Any] | None = None
    lights: list[dict[str, Any]] | None = None
    simulation: dict[str, Any] | None = None
    structures: list[dict[str, Any]] | None = None
    domain: list[float] | None = None
    base_revision: int | None = Field(default=None, ge=1)


class LayoutPreview(BaseModel):
    scenario: str
    origin_fuel: str = "木材"
    placements: list[dict[str, Any]] = Field(default_factory=list)
    origin: list[float] | None = None
    cameras: dict[str, Any] | None = None
    gas_domain: dict[str, Any] | None = None
    viewpoints: list[str] | None = None
    lights: list[dict[str, Any]] | None = None
    simulation: dict[str, Any] | None = None
    frame_count: int = Field(16, ge=1, le=48)
    look_count: int = Field(1, ge=1, le=6)
    structures: list[dict[str, Any]] | None = None
    domain: list[float] | None = None


class SceneDuplicate(BaseModel):
    name: str | None = None


class DatasetCreate(BaseModel):
    name: str


class DatasetSamples(BaseModel):
    sample_ids: list[str] = Field(..., min_length=1)


class AccountRequest(BaseModel):
    email: str
    password: str


def _current_user(request: Request) -> dict | None:
    from firephantom.auth.sessions import COOKIE, read
    from firephantom.auth.store import get_user

    user_id = read(request.cookies.get(COOKIE))
    if user_id is None:
        return None
    return get_user(user_id)


def _with_session(payload: dict, user_id: int) -> JSONResponse:
    from firephantom.auth.sessions import COOKIE, MAX_AGE_SECONDS, issue

    response = JSONResponse(payload)
    response.set_cookie(
        COOKIE,
        issue(user_id),
        max_age=MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


def _account_error(exc: Exception) -> HTTPException:
    detail = str(exc)
    status = 409 if "已经注册" in detail else 400
    return HTTPException(status_code=status, detail=detail)


def _dataset_store(application: FastAPI):
    from firephantom.versioning.store import DatasetStore

    return DatasetStore(application.state.dataset_root)


def _sample_catalog(application: FastAPI):
    from firephantom.versioning.samples import SampleCatalog

    return SampleCatalog(application.state.dataset_root)


def _media_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".xml":
        return "application/xml"
    if suffix == ".png":
        return "image/png"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix in {".txt", ".fds", ".py"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _browser_file(file_path: str, data: bytes) -> bytes:
    """Convert legacy MP4 payloads before returning them to browser clients."""
    if Path(file_path).suffix.lower() == ".mp4":
        return browser_compatible_video(data)
    return data


def _public_sample(record: dict, *, detail: bool) -> dict:
    body = {
        "sample_id": record["sample_id"],
        "status": record["status"],
        "seed": record["seed"],
        "generator_version": record["generator_version"],
        "generator_profile": record["generator_profile"],
        "scene_params": record["scene_params"],
        "quality": record["quality"],
        "annotation_error": record["annotation_error"],
        "label_studio_task_id": record["label_studio_task_id"],
        "sequence_id": record["sequence_id"],
        "split": record["split"],
        "derived_from": record["derived_from"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }
    if detail:
        body["auto_objects"] = record["auto_objects"]
        body["reviewed_objects"] = record["reviewed_objects"]
        body["filter"] = record["filter_decision"]
    return body


def _same_boxes(stored: list[dict] | None, pulled: list[dict]) -> bool:
    def key(objects):
        return sorted(
            (obj.get("category"), tuple(round(float(v), 2) for v in obj.get("bbox", [])), tuple(obj.get("attributes") or []))
            for obj in objects or []
        )

    return key(stored) == key(pulled)


def create_app(dataset_root: Path | str | None = None) -> FastAPI:
    application = FastAPI(
        title="火焰数据集仿体系统",
        version=__version__,
        description="SE2026-AI-07 物理仿真、多模态合成、标注、质量筛选与 DVC 版本导出。",
        lifespan=lifespan,
    )
    application.add_middleware(SecurityHeadersMiddleware)
    if os.environ.get("REQUIRE_HTTPS", "").lower() in {"1", "true", "yes"}:
        application.add_middleware(HTTPSRedirectMiddleware)
    origins = _cors_origins()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization"],
        )
    application.state.dataset_root = Path(dataset_root) if dataset_root is not None else default_dataset_root()
    try:
        configured_workers = int(os.environ.get("FIREPHANTOM_JOB_WORKERS", "1"))
    except ValueError:
        configured_workers = 1
    application.state.job_manager = JobManager(
        application.state.dataset_root,
        max_workers=safe_job_workers(configured_workers),
    )
    application.state.augment_jobs = AugmentJobs(application.state.dataset_root)
    web_root = Path(__file__).resolve().parent.parent / "web"
    web_dist = web_root / "dist"
    static_root = web_dist if (web_dist / "index.html").is_file() else web_root
    application.mount("/static", StaticFiles(directory=static_root), name="static")
    from firephantom.physics.placements import assets_root

    try:
        asset_dir = assets_root()
        asset_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        asset_dir = None
    if asset_dir is not None and asset_dir.is_dir():
        application.mount("/assets", StaticFiles(directory=asset_dir), name="assets")

    @application.get("/", include_in_schema=False)
    def home() -> FileResponse:
        index = web_dist / "index.html" if (web_dist / "index.html").is_file() else web_root / "index.html"
        return FileResponse(index)

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @application.get("/api/v1/system/status", operation_id="get_system_status", tags=["system"])
    def get_system_status() -> dict:
        """Report local memory headroom and configured integration reachability."""
        return system_status()

    @application.post("/api/v1/pipeline", operation_id="run_sequence", tags=["pipeline"])
    def pipeline(body: SequenceRequest, http_request: Request) -> dict:
        """Bake one fire and store every rendered frame as a sample, synchronously.

        Use jobs for anything longer than a quick check: a Blender bake takes minutes.
        """
        from firephantom.scenes import resolve_scene_items

        root = http_request.app.state.dataset_root
        request = resolve_scene_items(root, [body.model_dump()])[0]
        outcome = run_sequence(request, root)
        for combo in outcome["combos"]:
            combo.pop("preview", None)
            combo["video_url"] = f"/api/v1/sequences/{outcome['sequence_id']}/files/{combo['video']}"
        return outcome

    @application.post("/api/v1/jobs", operation_id="submit_batch_job", tags=["jobs"])
    def submit_batch_job(body: JobBatchRequest, http_request: Request) -> dict:
        """Queue a persistent job and return immediately.

        Each row is one bake (``count`` repeats make more). Its environment x
        weather x viewpoint combinations are rendered from that bake, and each
        rendered frame becomes a sample.
        """
        return http_request.app.state.job_manager.submit(
            [item.model_dump() for item in body.items],
            staged=body.staged,
        )

    @application.get("/api/v1/jobs/{job_id}/stages", operation_id="get_job_stages", tags=["jobs"])
    def get_job_stages(job_id: str, http_request: Request) -> dict:
        from firephantom.runs import stage_view
        from firephantom.versioning.datasets import DatasetCatalog
        from firephantom.versioning.samples import SampleCatalog

        if http_request.app.state.job_manager.get(job_id, include_results=False) is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        view = stage_view(http_request.app.state.dataset_root, job_id)
        ids = [item["sample_id"] for combo in view["combos"] for item in combo.get("samples") or []]
        statuses = SampleCatalog(http_request.app.state.dataset_root).statuses(ids)
        owners = DatasetCatalog(http_request.app.state.dataset_root).datasets_of(ids)
        for combo in view["combos"]:
            for item in combo.get("samples") or []:
                item["status"] = statuses.get(item["sample_id"], item.get("status"))
                item["dataset_id"] = owners.get(item["sample_id"])
        return view

    @application.get("/api/v1/jobs/{job_id}/artifacts/{file_path:path}", operation_id="get_job_artifact", tags=["jobs"])
    def get_job_artifact(job_id: str, file_path: str, http_request: Request) -> FileResponse:
        from firephantom.runs import artifact_file

        if http_request.app.state.job_manager.get(job_id, include_results=False) is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        try:
            path = artifact_file(http_request.app.state.dataset_root, job_id, file_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if path is None:
            raise HTTPException(status_code=404, detail="stage artifact is unavailable")
        return FileResponse(path, media_type="image/png")

    @application.post("/api/v1/jobs/{job_id}/continue", operation_id="continue_batch_job", tags=["jobs"])
    def continue_batch_job(job_id: str, body: ContinueRequest, http_request: Request) -> dict:
        """Redraw from one stage, reusing the pictures already stored above it."""
        job = http_request.app.state.job_manager.continue_job(
            job_id,
            body.from_stage,
            [item.model_dump() for item in body.items],
            body.combo_index,
        )
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return job

    @application.get("/api/v1/datasets", operation_id="list_datasets", tags=["datasets"])
    def list_datasets(http_request: Request) -> dict:
        from firephantom.versioning.datasets import DatasetCatalog

        return {"datasets": DatasetCatalog(http_request.app.state.dataset_root).list_all()}

    @application.post("/api/v1/datasets", operation_id="create_dataset", tags=["datasets"])
    def create_dataset(body: DatasetCreate, http_request: Request) -> dict:
        from firephantom.versioning.datasets import DatasetCatalog

        return DatasetCatalog(http_request.app.state.dataset_root).create(body.name)

    @application.post("/api/v1/datasets/{dataset_id}/samples", operation_id="admit_dataset_samples", tags=["datasets"])
    def admit_dataset_samples(dataset_id: str, body: DatasetSamples, http_request: Request) -> dict:
        from firephantom.versioning.datasets import DatasetCatalog

        try:
            return DatasetCatalog(http_request.app.state.dataset_root).admit(dataset_id, body.sample_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/v1/datasets/{dataset_id}/samples/remove", operation_id="remove_dataset_samples", tags=["datasets"])
    def remove_dataset_samples(dataset_id: str, body: DatasetSamples, http_request: Request) -> dict:
        from firephantom.versioning.datasets import DatasetCatalog

        try:
            return DatasetCatalog(http_request.app.state.dataset_root).remove(dataset_id, body.sample_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/api/v1/jobs", operation_id="list_batch_jobs", tags=["jobs"])
    def list_batch_jobs(http_request: Request) -> dict:
        return {"jobs": http_request.app.state.job_manager.list()}

    @application.get("/api/v1/jobs/{job_id}", operation_id="get_batch_job", tags=["jobs"])
    def get_batch_job(
        job_id: str,
        http_request: Request,
        result_offset: int = 0,
        result_limit: int = 50,
    ) -> dict:
        """Return job status with a bounded window of the most recent results."""
        job = http_request.app.state.job_manager.get(
            job_id,
            include_details=False,
            result_offset=result_offset,
            result_limit=result_limit,
        )
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return job

    @application.post("/api/v1/jobs/{job_id}/cancel", operation_id="cancel_batch_job", tags=["jobs"])
    def cancel_batch_job(job_id: str, http_request: Request) -> dict:
        """Request cancellation before the next item in a running batch."""
        job = http_request.app.state.job_manager.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return job

    @application.get("/api/v1/jobs/{job_id}/results", operation_id="list_batch_job_results", tags=["jobs"])
    def list_batch_job_results(
        job_id: str,
        http_request: Request,
        offset: int = 0,
        limit: int = 50,
        before_index: int | None = Query(default=None, ge=0),
    ) -> dict:
        """Page through a job's results from newest to oldest using offset or cursor."""
        job = http_request.app.state.job_manager.get(
            job_id,
            include_details=False,
            result_offset=offset,
            result_limit=limit,
            before_index=before_index,
        )
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return {
            "job_id": job_id,
            "offset": job["result_offset"],
            "limit": job["result_limit"],
            "total": job["total_results"],
            "has_more": job["has_more_results"],
            "next_before": job["next_before"],
            "results": job["results"],
        }

    @application.get("/api/v1/jobs/{job_id}/results/{result_index}", operation_id="get_batch_job_result", tags=["jobs"])
    def get_batch_job_result(job_id: str, result_index: int, http_request: Request) -> dict:
        result = http_request.app.state.job_manager.get_result(job_id, result_index)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown job result")
        return result

    @application.get("/api/v1/jobs/{job_id}/results/{result_index}/preview", operation_id="get_batch_job_preview", tags=["jobs"])
    def get_batch_job_preview(job_id: str, result_index: int, http_request: Request) -> FileResponse:
        path = http_request.app.state.job_manager.get_preview_path(job_id, result_index)
        if path is None:
            raise HTTPException(status_code=404, detail="preview is unavailable")
        return FileResponse(
            path,
            media_type="image/png",
            filename=f"firephantom-{job_id}-{result_index}.png",
            headers={"Cache-Control": "private, no-cache"},
        )

    @application.get("/api/v1/catalog", operation_id="get_catalog", tags=["catalog"])
    def catalog() -> dict:
        from firephantom.physics.placements import catalog_extras
        from firephantom.synthesis.sd_augment import styles_for

        payload = {
            "scenarios": list(SCENARIOS),
            "scenes": list(SCENES),
            "weathers": list(WEATHERS),
            "sd_styles": {name: styles_for(name) for name in SCENES},
            "viewpoints": list(VIEWPOINTS),
            "fire_engines": ["blender"],
            "augmentations": list(AUGMENTATIONS),
            "exports": ["coco", "pascal_voc", "yolo"],
            "frame_count": {"default": 16, "max": 48},
        }
        payload.update(catalog_extras())
        return payload

    @application.post("/api/v1/assets/uploads", operation_id="upload_asset", tags=["catalog"])
    async def upload_asset(file: UploadFile = File(...)) -> dict:
        from firephantom.physics.placements import MAX_GLB_BYTES, store_upload

        name = Path(file.filename or "").name.lower()
        if not name.endswith(".glb"):
            raise ValueError("只接受 .glb")
        data = await file.read(MAX_GLB_BYTES + 1)
        return store_upload(data, label=Path(file.filename or "").stem)

    @application.get("/api/v1/assets/uploads", operation_id="list_uploads", tags=["catalog"])
    def list_uploads() -> dict:
        from firephantom.physics.placements import list_uploads as stored_uploads

        return {"uploads": stored_uploads()}

    @application.post("/api/v1/assets/uploads/batch", operation_id="upload_assets_batch", tags=["catalog"])
    async def upload_assets_batch(files: list[UploadFile] = File(...)) -> dict:
        """Accept many .glb files, or a .zip of them. One bad file stays in rejected."""
        from firephantom.physics.placements import (
            MAX_BATCH_FILES,
            MAX_GLB_BYTES,
            MAX_ZIP_BYTES,
            store_upload_many,
        )

        items: list[tuple[str, bytes]] = []
        rejected: list[dict] = []
        for upload in files:
            name = Path(upload.filename or "file").name
            data = await upload.read()
            lower = name.lower()
            if lower.endswith(".zip"):
                if len(data) > MAX_ZIP_BYTES:
                    rejected.append({"name": name, "reason": "压缩包不能超过 80 MB"})
                    continue
                try:
                    archive = zipfile.ZipFile(io.BytesIO(data))
                except zipfile.BadZipFile:
                    rejected.append({"name": name, "reason": "压缩包打不开"})
                    continue
                found = False
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    inner = Path(info.filename).name
                    if not inner or inner.startswith("."):
                        continue
                    if not inner.lower().endswith(".glb"):
                        continue
                    found = True
                    if info.file_size > MAX_GLB_BYTES:
                        rejected.append({"name": inner, "reason": "模型不能超过 8 MB"})
                        continue
                    items.append((inner, archive.read(info)))
                if not found:
                    rejected.append({"name": name, "reason": "压缩包里没有 .glb"})
                continue
            if not lower.endswith(".glb"):
                rejected.append({"name": name, "reason": "只接受 .glb"})
                continue
            items.append((name, data))
        if len(items) > MAX_BATCH_FILES:
            raise ValueError(f"一次最多 {MAX_BATCH_FILES} 个模型")
        result = store_upload_many(items)
        result["rejected"] = rejected + result["rejected"]
        return result

    @application.delete("/api/v1/assets/uploads/{token}", operation_id="delete_uploaded_asset", tags=["catalog"])
    def delete_uploaded_asset(token: str) -> dict:
        from firephantom.physics.placements import delete_upload

        return delete_upload(token)

    @application.get("/api/v1/integrations", operation_id="get_integrations", tags=["catalog"])
    def integrations() -> dict:
        return {
            "label_studio_url": os.environ.get("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks"),
            "stable_diffusion_url": os.environ.get("SD_API_URL", "http://127.0.0.1:7861/sdapi/v1/txt2img"),
            "blender_bin": os.environ.get("BLENDER_BIN"),
        }

    @application.get("/api/v1/annotations/label-studio", operation_id="pull_label_studio_tasks", tags=["annotation"])
    def label_studio_tasks(limit: int = 100) -> dict:
        """Fetch reviewed/annotated tasks for the semi-automatic workflow."""
        return pull_tasks(limit=limit)

    @application.get("/api/v1/samples/summary", operation_id="summarize_samples", tags=["samples"])
    def summarize_samples(http_request: Request) -> dict:
        return _sample_catalog(http_request.app).summary()

    @application.get("/api/v1/samples", operation_id="list_samples", tags=["samples"])
    def list_samples(
        http_request: Request,
        status: str | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        records, total = _sample_catalog(http_request.app).list_samples(status=status, offset=offset, limit=limit)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "samples": [_public_sample(record, detail=False) for record in records],
        }

    @application.post("/api/v1/samples/sync-reviews", operation_id="sync_sample_reviews", tags=["samples"])
    def sync_sample_reviews(http_request: Request) -> dict:
        """Pull every Label Studio task and write the newest annotation back.

        A cancelled (skipped) annotation rejects the sample. A sample already
        in a published version is frozen and skipped. A decided, unpublished
        sample is updated when its annotation changed.
        """
        catalog = _sample_catalog(http_request.app)
        pulled = pull_tasks()
        updated: list[str] = []
        frozen: list[str] = []
        if pulled.get("success"):
            for task in pulled.get("tasks") or []:
                sample_id = str((task.get("data") or {}).get("sample_id") or "")
                annotation = latest_annotation(task)
                if not sample_id or annotation is None:
                    continue
                sample = catalog.get(sample_id)
                if sample is None or sample["status"] not in {"pending_review", "reviewed", "rejected"}:
                    continue
                width = int(sample["auto_objects"][0]["image_width"]) if sample["auto_objects"] else 256
                height = int(sample["auto_objects"][0]["image_height"]) if sample["auto_objects"] else 256
                objects = objects_from_annotation(annotation, width, height, f"samples/{sample_id}/image.png")
                decision = "rejected" if annotation.get("was_cancelled") else "reviewed"
                decided = sample["status"] != "pending_review"
                if decided and sample["status"] == decision and _same_boxes(sample["reviewed_objects"], objects):
                    continue
                if catalog.is_published(sample_id):
                    frozen.append(sample_id)
                    continue
                task_id = task.get("id")
                catalog.record_review(
                    sample_id,
                    objects,
                    decision,
                    task_id=None if task_id is None else str(task_id),
                    rereview=decided,
                )
                updated.append(sample_id)
        return {
            "updated": updated,
            "frozen": frozen,
            "label_studio": {
                "called": pulled.get("called"),
                "success": pulled.get("success"),
                "error": pulled.get("error"),
                "count": pulled.get("count", 0),
            },
        }

    @application.get("/api/v1/samples/{sample_id}", operation_id="get_sample", tags=["samples"])
    def get_sample(sample_id: str, http_request: Request) -> dict:
        record = _sample_catalog(http_request.app).get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown sample {sample_id}")
        return _public_sample(record, detail=True)

    @application.post("/api/v1/samples/{sample_id}/review", operation_id="review_sample", tags=["samples"])
    def review_sample(sample_id: str, body: ReviewRequest, http_request: Request) -> dict:
        if body.decision not in {"reviewed", "rejected"}:
            raise ValueError("decision must be reviewed or rejected")
        try:
            record = _sample_catalog(http_request.app).record_review(
                sample_id, body.objects, body.decision, rereview=body.rereview
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _public_sample(record, detail=True)

    @application.get(
        "/api/v1/samples/{sample_id}/files/{file_path:path}",
        operation_id="read_sample_file",
        tags=["samples"],
    )
    def read_sample_file(sample_id: str, file_path: str, http_request: Request) -> Response:
        try:
            data = _sample_catalog(http_request.app).read_file(sample_id, file_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _byte_response(_browser_file(file_path, data), _media_type(file_path), http_request)

    @application.post("/api/v1/datasets/{dataset_id}/publish", operation_id="publish_dataset", tags=["datasets"])
    def publish_named_dataset(dataset_id: str, http_request: Request) -> dict:
        """Publish the dataset as a new cumulative version on top of its head."""
        from firephantom.versioning.publish import publish_dataset

        try:
            return publish_dataset(http_request.app.state.dataset_root, dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/api/v1/datasets/{dataset_id}/sd-augment", operation_id="list_sd_augment_jobs", tags=["datasets"])
    def list_sd_augment_jobs(dataset_id: str, http_request: Request) -> dict:
        """SD augmentation jobs of this dataset and how many reviewed samples are eligible now."""
        from firephantom.synthesis.sd_augment import eligible_sources
        from firephantom.versioning.datasets import DatasetCatalog

        root = http_request.app.state.dataset_root
        if DatasetCatalog(root).get(dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
        return {
            "dataset_id": dataset_id,
            "eligible": len(eligible_sources(root, dataset_id)),
            "jobs": http_request.app.state.augment_jobs.list(dataset_id),
        }

    @application.post("/api/v1/datasets/{dataset_id}/sd-augment", operation_id="submit_sd_augment", tags=["datasets"])
    def submit_sd_augment(dataset_id: str, body: SdAugmentRequest, http_request: Request) -> dict:
        """Queue SD background restyles of reviewed samples. Children are stored as reviewed with the same boxes."""
        options = body.model_dump(exclude={"sample_ids"})
        return http_request.app.state.augment_jobs.submit(dataset_id, body.sample_ids, options)

    @application.get("/api/v1/sd-augment/{job_id}", operation_id="get_sd_augment_job", tags=["datasets"])
    def get_sd_augment_job(job_id: str, http_request: Request) -> dict:
        job = http_request.app.state.augment_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown augment job {job_id}")
        return job

    @application.post("/api/v1/sd-augment/{job_id}/cancel", operation_id="cancel_sd_augment_job", tags=["datasets"])
    def cancel_sd_augment_job(job_id: str, http_request: Request) -> dict:
        job = http_request.app.state.augment_jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown augment job {job_id}")
        return job

    @application.get("/api/v1/datasets/{dataset_id}/plan", operation_id="plan_dataset_publish", tags=["datasets"])
    def plan_dataset_publish(dataset_id: str, http_request: Request) -> dict:
        """What publishing now would add to and remove from the head version."""
        from firephantom.versioning.publish import plan_members

        plan = plan_members(http_request.app.state.dataset_root, dataset_id)
        return {
            "dataset_id": dataset_id,
            "parent_id": plan["parent_id"],
            "members": len(plan["members"]),
            "added": plan["added"][:500],
            "removed": plan["removed"][:500],
        }

    @application.get("/api/v1/versions", operation_id="list_versions", tags=["versions"])
    def list_versions(
        http_request: Request,
        dataset_id: str | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        store = _dataset_store(http_request.app)
        total = store.version_count(dataset_id)
        versions = store.list_versions(offset=offset, limit=limit, newest_first=True, dataset_id=dataset_id)
        head = store.head()
        if dataset_id is not None:
            from firephantom.versioning.datasets import DatasetCatalog

            dataset = DatasetCatalog(http_request.app.state.dataset_root).get(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
            head = dataset["head_version_id"]
        return {
            "head": head,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(versions) < total,
            "versions": versions,
        }

    @application.get("/api/v1/versions/compare", operation_id="compare_versions", tags=["versions"])
    def compare_versions(http_request: Request, base: str, other: str) -> dict:
        from firephantom.versioning.compare import compare_versions as build_comparison
        from firephantom.versioning.publish import version_members

        store = _dataset_store(http_request.app)
        catalog = _sample_catalog(http_request.app)
        left = store.get(base)
        right = store.get(other)
        if left is None:
            raise HTTPException(status_code=404, detail=f"unknown version {base}")
        if right is None:
            raise HTTPException(status_code=404, detail=f"unknown version {other}")
        return build_comparison(
            left,
            right,
            catalog.latest_evaluation(base),
            catalog.latest_evaluation(other),
            version_members(base),
            version_members(other),
        )

    @application.get("/api/v1/versions/{version_id}", operation_id="get_version", tags=["versions"])
    def get_version(version_id: str, http_request: Request) -> dict:
        record = _dataset_store(http_request.app).get(version_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown version {version_id}")
        return record

    @application.post("/api/v1/versions/{version_id}/rollback", operation_id="rollback_version", tags=["versions"])
    def rollback_version(version_id: str, http_request: Request) -> dict:
        """Make this version its dataset's head; the next publish builds on it."""
        from firephantom.versioning.datasets import DatasetCatalog

        store = _dataset_store(http_request.app)
        try:
            record = store.rollback(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        dataset = None
        if record.get("dataset_id"):
            dataset = DatasetCatalog(http_request.app.state.dataset_root).rollback(record["dataset_id"], version_id)
        return {"head": version_id, "version": record, "dataset": dataset}

    @application.post(
        "/api/v1/versions/{version_id}/evaluations",
        operation_id="record_version_evaluation",
        tags=["versions"],
    )
    def record_version_evaluation(
        version_id: str,
        body: DetectionEvaluation,
        http_request: Request,
    ) -> dict:
        record = _dataset_store(http_request.app).get(version_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown version {version_id}")
        if record.get("metadata", {}).get("类型") != "dataset":
            raise ValueError("只能为已发布的数据集版本记录检测效果")
        metrics = _sample_catalog(http_request.app).record_evaluation(version_id, body.model_dump())
        return {"version_id": version_id, "detection": metrics}

    @application.get("/api/v1/versions/{version_id}/files", operation_id="list_version_files", tags=["versions"])
    def list_version_files(version_id: str, http_request: Request) -> dict:
        try:
            files = _dataset_store(http_request.app).list_files(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"version_id": version_id, "files": files}

    @application.get("/api/v1/versions/{version_id}/archive", operation_id="download_version_archive", tags=["versions"])
    def download_version_archive(version_id: str, http_request: Request) -> Response:
        try:
            files = _dataset_store(http_request.app).files(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, payload in files.items():
                bundle.writestr(name, _browser_file(name, payload))
        return Response(
            content=output.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="firephantom-{version_id[:16]}.zip"'},
        )

    @application.get("/api/v1/versions/{version_id}/files/{file_path:path}", operation_id="read_version_file", tags=["versions"])
    def read_version_file(version_id: str, file_path: str, http_request: Request) -> Response:
        try:
            data = _dataset_store(http_request.app).read_file(version_id, file_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if data is None:
            raise HTTPException(status_code=404, detail=f"unknown file {file_path}")
        return _byte_response(_browser_file(file_path, data), _media_type(file_path), http_request)

    @application.get(
        "/api/v1/sequences/{sequence_id}/files/{file_path:path}",
        operation_id="read_sequence_file",
        tags=["samples"],
    )
    def read_sequence_file(sequence_id: str, file_path: str, http_request: Request) -> Response:
        """Sequence-level files: per-view videos, physical fields, point cloud, solver inputs."""
        from firephantom.pipeline import sequence_dir

        if not sequence_id.isalnum():
            raise HTTPException(status_code=400, detail="invalid sequence id")
        root = sequence_dir(http_request.app.state.dataset_root, sequence_id).resolve()
        target = (root / file_path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail=f"unknown file {file_path}")
        return _byte_response(_browser_file(file_path, target.read_bytes()), _media_type(file_path), http_request)

    @application.post("/api/v1/layout/preview", operation_id="preview_layout", tags=["scenes"])
    def preview_layout(body: LayoutPreview) -> dict:
        from firephantom.physics.placements import layout_preview

        return layout_preview(
            body.scenario,
            body.placements,
            body.origin_fuel,
            body.origin,
            body.cameras,
            body.gas_domain,
            body.viewpoints,
            body.lights,
            body.simulation,
            body.frame_count,
            body.look_count,
            body.structures,
            body.domain,
        )

    @application.get("/api/v1/scenes", operation_id="list_scenes", tags=["scenes"])
    def list_scenes(http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        return {"scenes": SceneStore(http_request.app.state.dataset_root).list_all()}

    @application.post("/api/v1/scenes", operation_id="create_scene", tags=["scenes"])
    def create_scene(body: SceneCreate, http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        user = _current_user(http_request)
        return SceneStore(http_request.app.state.dataset_root).create(
            body.name,
            body.scenario,
            body.origin_fuel,
            body.placements,
            created_by=None if user is None else user.get("email"),
            origin=body.origin,
            cameras=body.cameras,
            gas_domain=body.gas_domain,
            lights=body.lights,
            simulation=body.simulation,
            structures=body.structures,
            domain=body.domain,
        )

    @application.get("/api/v1/scenes/{scene_id}", operation_id="get_scene", tags=["scenes"])
    def get_scene(scene_id: str, http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        scene = SceneStore(http_request.app.state.dataset_root).get(scene_id)
        if scene is None:
            raise HTTPException(status_code=404, detail=f"unknown scene {scene_id}")
        return scene

    @application.put("/api/v1/scenes/{scene_id}", operation_id="update_scene", tags=["scenes"])
    def update_scene(scene_id: str, body: SceneUpdate, http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        # Omitted fields stay unchanged. An explicit null clears that override.
        return SceneStore(http_request.app.state.dataset_root).update(scene_id, **body.model_dump(exclude_unset=True))

    @application.delete("/api/v1/scenes/{scene_id}", operation_id="delete_scene", tags=["scenes"])
    def delete_scene(scene_id: str, http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        SceneStore(http_request.app.state.dataset_root).delete(scene_id)
        return {"scene_id": scene_id, "deleted": True}

    @application.post("/api/v1/scenes/{scene_id}/duplicate", operation_id="duplicate_scene", tags=["scenes"])
    def duplicate_scene(scene_id: str, body: SceneDuplicate, http_request: Request) -> dict:
        from firephantom.scenes import SceneStore

        user = _current_user(http_request)
        return SceneStore(http_request.app.state.dataset_root).duplicate(
            scene_id,
            body.name,
            created_by=None if user is None else user.get("email"),
        )

    @application.put("/api/v1/scenes/{scene_id}/thumbnail", operation_id="put_scene_thumbnail", tags=["scenes"])
    async def put_scene_thumbnail(scene_id: str, http_request: Request) -> dict:
        from firephantom.scenes import MAX_THUMBNAIL_BYTES, SceneStore

        data = await http_request.body()
        if len(data) > MAX_THUMBNAIL_BYTES:
            raise ValueError("缩略图不能超过 2 MB")
        return SceneStore(http_request.app.state.dataset_root).save_thumbnail(scene_id, data)

    @application.get("/api/v1/scenes/{scene_id}/thumbnail", operation_id="get_scene_thumbnail", tags=["scenes"])
    def get_scene_thumbnail(scene_id: str, http_request: Request) -> FileResponse:
        from firephantom.scenes import SceneStore

        path = SceneStore(http_request.app.state.dataset_root).thumbnail_path(scene_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="scene has no thumbnail")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @application.get("/api/v1/auth/me", operation_id="current_account", tags=["auth"])
    def current_account(http_request: Request) -> dict:
        from firephantom.auth.store import count_users

        user = _current_user(http_request)
        if user is None:
            return {"authenticated": False, "needs_bootstrap": count_users() == 0}
        return {"authenticated": True, "needs_bootstrap": False, **user}

    @application.post("/api/v1/auth/bootstrap", operation_id="bootstrap_account", tags=["auth"])
    def bootstrap_account(body: AccountRequest) -> JSONResponse:
        from firephantom.auth.store import AccountError, count_users, create_user

        if count_users() != 0:
            raise HTTPException(status_code=409, detail="管理员已经创建")
        try:
            user = create_user(body.email, body.password, is_admin=True)
        except AccountError as exc:
            raise _account_error(exc) from exc
        return _with_session({"authenticated": True, **user}, user["id"])

    @application.post("/api/v1/auth/login", operation_id="login_account", tags=["auth"])
    def login_account(body: AccountRequest) -> JSONResponse:
        from firephantom.auth.store import authenticate

        user = authenticate(body.email, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="邮箱或密码不正确")
        return _with_session({"authenticated": True, **user}, user["id"])

    @application.post("/api/v1/auth/logout", operation_id="logout_account", tags=["auth"])
    def logout_account(http_request: Request) -> JSONResponse:
        from firephantom.auth.labelstudio_bridge import LabelStudioAuthError
        from firephantom.auth.labelstudio_bridge import logout as logout_label_studio
        from firephantom.auth.sessions import COOKIE

        warning = None
        try:
            logout_label_studio(http_request.cookies.get("sessionid"))
        except LabelStudioAuthError as exc:
            warning = str(exc)
        response = JSONResponse({"authenticated": False, "label_studio_error": warning})
        response.delete_cookie(COOKIE, path="/")
        response.delete_cookie("sessionid", path="/")
        return response

    @application.post("/api/v1/auth/users", operation_id="create_account", tags=["auth"])
    def create_account(body: AccountRequest, http_request: Request) -> dict:
        from firephantom.auth.store import AccountError, create_user

        user = _current_user(http_request)
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录")
        if not user["is_admin"]:
            raise HTTPException(status_code=403, detail="只有管理员可以添加账号")
        try:
            created = create_user(body.email, body.password, is_admin=False)
        except AccountError as exc:
            raise _account_error(exc) from exc
        return created

    @application.get("/auth/label-studio", include_in_schema=False)
    def open_label_studio(http_request: Request) -> RedirectResponse:
        from firephantom.auth.labelstudio_bridge import LabelStudioAuthError, browser_url, open_session
        from firephantom.auth.store import label_studio_secret

        user = _current_user(http_request)
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录工作台")
        credentials = label_studio_secret(user["id"])
        if credentials is None:
            raise HTTPException(status_code=401, detail="请先登录工作台")
        email, shadow = credentials
        try:
            session_id = open_session(email, shadow)
        except LabelStudioAuthError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response = RedirectResponse(browser_url(http_request.url.hostname or "127.0.0.1"), status_code=302)
        response.set_cookie("sessionid", session_id, httponly=True, samesite="lax", path="/")
        return response

    @application.exception_handler(VersionConflict)
    def conflict(_request: Request, exc: VersionConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @application.exception_handler(ValueError)
    def bad_request(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.exception_handler(KeyError)
    def missing_resource(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return application


class _LazyApp:
    """Build the ASGI app on first use so importing this module does not open PostgreSQL."""

    def __init__(self):
        self._app = None

    def _get(self):
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope, receive, send):
        await self._get()(scope, receive, send)

    def __getattr__(self, name):
        return getattr(self._get(), name)


app = _LazyApp()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "firephantom.api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8765")),
    )


if __name__ == "__main__":
    main()
