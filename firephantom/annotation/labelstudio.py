"""Build a Label Studio task and optionally push it to the HTTP API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urljoin

import httpx

from firephantom.synthesis.stable_diffusion import stable_error

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/api/tasks"

# The one copy of the labeling interface. Label Studio projects use this file.
LABEL_CONFIG = (Path(__file__).with_name("label_config.xml")).read_text(encoding="utf-8")


def endpoint_from_env() -> str:
    return os.environ.get("LABEL_STUDIO_URL", DEFAULT_ENDPOINT)


def pull_endpoint_from_env() -> str:
    endpoint = os.environ.get("LABEL_STUDIO_PULL_URL", "").strip()
    if endpoint:
        return endpoint
    target = endpoint_from_env().rstrip("/")
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID", "").strip()
    if target.endswith("/api/tasks") and project_id:
        base_url = target[: -len("/api/tasks")]
        return f"{base_url}/api/projects/{project_id}/tasks/"
    return target


def _headers() -> dict[str, str]:
    token = os.environ.get("LABEL_STUDIO_TOKEN", "").strip() or os.environ.get("LABEL_STUDIO_API_KEY", "").strip()
    return {"Authorization": f"Token {token}"} if token else {}


def public_base_url() -> str:
    """Where Label Studio's browser can load FirePhantom sample files from."""
    return os.environ.get("FIREPHANTOM_PUBLIC_URL", "http://127.0.0.1:8765").rstrip("/") + "/"


def resolve_image_url(task: dict) -> dict:
    """Copy of ``task`` whose image is an absolute URL instead of a route."""
    data = dict(task.get("data") or {})
    image = str(data.get("image") or "")
    if image.startswith("/"):
        data["image"] = urljoin(public_base_url(), image.lstrip("/"))
    return {**task, "data": data}


def build_task(
    image_url: str,
    width: int,
    height: int,
    obj: dict | list[dict],
    sample_id: str | None = None,
    attributes: list[str] | None = None,
) -> dict:
    """Label Studio task plus a prediction (box, keypoints, multi-label choices).

    ``image_url`` is a URL or a FirePhantom route; pixels are never inlined.
    """
    objects = obj if isinstance(obj, list) else [obj]
    results = []
    for index, item in enumerate(objects):
        x, y, box_w, box_h = item["bbox"]
        results.append({
            "id": f"bbox-{index}",
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": x / width * 100.0,
                "y": y / height * 100.0,
                "width": box_w / width * 100.0,
                "height": box_h / height * 100.0,
                "rotation": 0,
                "rectanglelabels": [item["category"]],
            },
        })
        for point in item.get("keypoints", []):
            results.append({
                "id": f"kp-{index}-{point['name']}",
                "type": "keypointlabels",
                "from_name": "kp",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "value": {
                    "x": point["x"] / width * 100.0,
                    "y": point["y"] / height * 100.0,
                    "width": 0.4,
                    "keypointlabels": [point["name"]],
                },
            })
    if attributes is None:
        attributes = objects[0].get("attributes", []) if objects else []
    results.append(
        {
            "id": "attrs",
            "type": "choices",
            "from_name": "attrs",
            "to_name": "image",
            "value": {"choices": list(attributes)},
        }
    )
    data = {"image": image_url}
    if sample_id:
        data["sample_id"] = sample_id
    return {
        "data": data,
        "predictions": [
            {
                "model_version": "firephantom-1.0.0",
                "score": 1.0,
                "result": results,
            }
        ],
    }


def _collection_endpoint(tasks_endpoint: str, name: str) -> str:
    """``/api/tasks`` and ``/api/predictions`` live on the same host."""
    base = tasks_endpoint.rstrip("/")
    if base.endswith("/api/tasks"):
        return base[: -len("tasks")] + name + "/"
    return base.rsplit("/", 1)[0] + "/" + name + "/"


def _prediction_body(task_id: int, project_id: int | None, prediction: dict) -> dict:
    body = {
        "task": int(task_id),
        "result": prediction.get("result") or [],
        "score": prediction.get("score", 1.0),
        "model_version": prediction.get("model_version") or "firephantom-1.0.0",
    }
    if project_id is not None:
        body["project"] = project_id
    return body


def _post_predictions(client, tasks_endpoint: str, task_id: int, project_id: int | None, predictions: list) -> str | None:
    """POST each prediction. Returns an error string, or None when all were stored."""
    endpoint = _collection_endpoint(tasks_endpoint, "predictions")
    for prediction in predictions:
        try:
            response = client.post(
                endpoint,
                json=_prediction_body(task_id, project_id, prediction),
                headers=_headers(),
            )
            response.raise_for_status()
        except Exception as exc:
            return stable_error(exc)
    return None


def _delete_task(client, tasks_endpoint: str, task_id: int) -> str | None:
    url = tasks_endpoint.rstrip("/") + "/" + str(task_id) + "/"
    try:
        response = client.delete(url, headers=_headers())
        response.raise_for_status()
    except Exception as exc:
        return stable_error(exc)
    return None


def _failure(endpoint: str, error: str) -> dict:
    return {
        "called": True,
        "success": False,
        "endpoint": endpoint,
        "error": error,
        "status_code": None,
        "task_id": None,
    }


def push_task(task: dict, endpoint: str | None = None, timeout: float = 1.5) -> dict:
    """Create one task, then store its predictions.

    ``POST /api/tasks`` ignores a nested ``predictions`` list, so the boxes
    go to ``/api/predictions`` afterwards. If that call fails, the new task
    is deleted and this returns a failure, so a sample is not marked sent.
    """
    target = endpoint or endpoint_from_env()
    payload = resolve_image_url(task)
    predictions = list(payload.pop("predictions", None) or [])
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID", "").strip()
    project_number = None
    if project_id:
        try:
            project_number = int(project_id)
            payload["project"] = project_number
        except ValueError:
            return _failure(target, "ValueError: LABEL_STUDIO_PROJECT_ID must be an integer")
    try:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            response = client.post(target, json=payload, headers=_headers())
            response.raise_for_status()
            task_id = None
            try:
                body = response.json()
            except Exception:
                body = None
            if isinstance(body, dict) and body.get("id") is not None:
                task_id = body.get("id")
            if predictions:
                if task_id is None:
                    return _failure(target, "Label Studio 没有返回任务编号，预标注没有写入")
                error = _post_predictions(client, target, int(task_id), project_number, predictions)
                if error:
                    removed = _delete_task(client, target, int(task_id))
                    if removed:
                        error = f"{error}；删除任务也失败：{removed}"
                    return _failure(target, error)
            return {
                "called": True,
                "success": True,
                "endpoint": target,
                "error": None,
                "status_code": response.status_code,
                "task_id": task_id,
            }
    except Exception as exc:
        return _failure(target, stable_error(exc))


def backfill_missing_predictions(root: Path | str, timeout: float = 5.0) -> dict:
    """Write stored predictions onto Label Studio tasks that have none.

    Matches ``data.sample_id`` to ``samples/<id>/label_studio.json``. Tasks
    that already have predictions, or have no local file, are left alone.
    """
    pulled = pull_tasks(timeout=timeout)
    if not pulled.get("success"):
        return {**pulled, "filled": 0, "skipped": 0, "errors": []}
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID", "").strip()
    try:
        project_number = int(project_id) if project_id else None
    except ValueError:
        return _failure(endpoint_from_env(), "ValueError: LABEL_STUDIO_PROJECT_ID must be an integer") | {
            "filled": 0,
            "skipped": 0,
            "errors": [],
        }
    root = Path(root)
    filled = 0
    skipped = 0
    errors = []
    target = endpoint_from_env()
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        for task in pulled["tasks"]:
            if task.get("predictions"):
                skipped += 1
                continue
            sample_id = str((task.get("data") or {}).get("sample_id") or "")
            path = root / "samples" / sample_id / "label_studio.json"
            if not sample_id or not path.is_file():
                skipped += 1
                continue
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append({"task_id": task.get("id"), "error": stable_error(exc)})
                continue
            predictions = stored.get("predictions") or []
            if not predictions or task.get("id") is None:
                skipped += 1
                continue
            error = _post_predictions(client, target, int(task["id"]), project_number, predictions)
            if error:
                errors.append({"task_id": task.get("id"), "error": error})
            else:
                filled += 1
    return {
        "called": True,
        "success": not errors,
        "endpoint": target,
        "error": None if not errors else errors[0]["error"],
        "filled": filled,
        "skipped": skipped,
        "errors": errors,
    }


def _task_rows(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tasks", "results", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def pull_tasks(
    endpoint: str | None = None,
    timeout: float = 2.0,
    limit: int | None = None,
    page_size: int = 100,
) -> dict:
    """Read Label Studio tasks page by page, up to ``limit`` (all when None).

    Newer Label Studio versions expose reads through a project-scoped endpoint
    even though task creation uses ``/api/tasks``. Override the read URL when a
    deployment uses a different route. Annotation payloads stay unchanged.
    """
    target = endpoint or pull_endpoint_from_env()
    size = max(1, min(int(page_size), 1000))
    tasks: list = []
    status_code = None
    try:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            page = 1
            while limit is None or len(tasks) < limit:
                response = client.get(target, params={"page": page, "page_size": size}, headers=_headers())
                if page > 1 and response.status_code == 404:
                    break
                response.raise_for_status()
                status_code = response.status_code
                payload = response.json()
                rows = _task_rows(payload)
                tasks.extend(rows)
                total = payload.get("total") if isinstance(payload, dict) else None
                # A plain list response is one unpaged dump.
                if not isinstance(payload, dict) or len(rows) < size or (total is not None and len(tasks) >= total):
                    break
                page += 1
        if limit is not None:
            tasks = tasks[:limit]
        return {
            "called": True,
            "success": True,
            "endpoint": target,
            "error": None,
            "status_code": status_code,
            "count": len(tasks),
            "tasks": tasks,
        }
    except Exception as exc:
        return {
            "called": True,
            "success": False,
            "endpoint": target,
            "error": stable_error(exc),
            "status_code": None,
            "count": 0,
            "tasks": [],
        }


def latest_annotation(task: dict) -> dict | None:
    """The newest submitted annotation of a task, by update time."""
    annotations = [item for item in task.get("annotations") or [] if isinstance(item, dict)]
    if not annotations:
        return None
    return max(annotations, key=lambda item: (str(item.get("updated_at") or item.get("created_at") or ""), item.get("id") or 0))


def objects_from_annotation(
    annotation: dict,
    image_width: int,
    image_height: int,
    image_file: str,
) -> list[dict]:
    """Convert one Label Studio annotation into pixel boxes.

    Keypoints are attached to the box that contains them, and the chosen
    attributes go on every box, so a review keeps what the reviewer set.
    """
    from firephantom.annotation.classes import canonical

    results = annotation.get("result") or []
    attributes: list[str] = []
    points: list[dict] = []
    objects = []
    for item in results:
        value = item.get("value") or {}
        width = float(item.get("original_width") or image_width or 0)
        height = float(item.get("original_height") or image_height or 0)
        kind = item.get("type")
        if kind == "choices":
            attributes = [str(choice) for choice in value.get("choices") or []]
            continue
        if width <= 0 or height <= 0:
            continue
        if kind == "keypointlabels":
            labels = value.get("keypointlabels") or []
            if labels:
                points.append({
                    "name": str(labels[0]),
                    "x": float(value.get("x", 0.0)) / 100.0 * width,
                    "y": float(value.get("y", 0.0)) / 100.0 * height,
                })
            continue
        if kind != "rectanglelabels":
            continue
        labels = value.get("rectanglelabels") or ["flame"]
        objects.append(
            {
                "category": canonical(labels[0]),
                "bbox": [
                    float(value.get("x", 0.0)) / 100.0 * width,
                    float(value.get("y", 0.0)) / 100.0 * height,
                    float(value.get("width", 0.0)) / 100.0 * width,
                    float(value.get("height", 0.0)) / 100.0 * height,
                ],
                "image_width": int(width),
                "image_height": int(height),
                "image_file": image_file,
                "keypoints": [],
            }
        )
    for point in points:
        for obj in objects:
            x, y, w, h = obj["bbox"]
            if x <= point["x"] <= x + w and y <= point["y"] <= y + h:
                obj["keypoints"].append(point)
                break
    for obj in objects:
        obj["attributes"] = list(attributes)
    return objects
