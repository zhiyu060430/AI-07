"""Regression coverage for SRS-facing batch, export, security, and tool APIs."""

import json
import time
import zipfile
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.images import browser_compatible_video, encode_video, video_encoder
from firephantom.jobs import JobManager
from firephantom.pipeline import combo_list, run_sequence
from firephantom.security import decrypt, encrypt
from firephantom.validation import expand_job_items


@pytest.fixture(autouse=True)
def _jobs_have_headroom(monkeypatch):
    """Job tests assert queue behavior. Host memory near 3 GiB must not fail them."""
    monkeypatch.setattr(
        "firephantom.jobs.memory_status",
        lambda: {"state": "ok", "available_gib": 8.0, "minimum_available_gib": 3.0},
    )


REQUEST = {
    "scenario": "房间火灾",
    "wind_speed": 1.5,
    "temperature": 28.0,
    "humidity": 45.0,
    "fuel_type": "木材",
    "scene": "室内",
    "weather": "晴",
    "viewpoint": "侧视",
    "frame_count": 2,
}


def fake_sequence(calls=None, fail_wind=None, delay=0.0):
    """Stand-in for run_sequence: one finished combination per rendered view, no bake."""
    preview = np.zeros((4, 4, 3), dtype=np.uint8)

    def run(sequence, _root, only=None, on_combo=None, stop=None, simulated=None):
        if calls is not None:
            calls.append(sequence["wind_speed"])
        if fail_wind is not None and sequence["wind_speed"] == fail_wind:
            raise RuntimeError("simulated sequence failure")
        stopped = False
        for index, combo in enumerate(combo_list(sequence)):
            if only is not None and index not in only:
                continue
            if stop is not None and stop():
                stopped = True
                break
            time.sleep(delay)
            on_combo(index, {
                **combo,
                "sequence_id": f"seq{sequence['wind_speed']}",
                "camera": None,
                "frames": sequence["frame_count"],
                "samples": [{"sample_id": f"s{sequence['wind_speed']}-{index}-{f}", "frame_index": f, "augmentation": None, "kept": True, "status": "generated", "boxes": 1, "reasons": []} for f in range(sequence["frame_count"])],
                "kept": sequence["frame_count"],
                "dropped": 0,
                "video": f"videos/{combo['key']}.mp4",
                "preview": preview,
                "contributors": [],
                "ssim_to_render": None,
            })
        return {"renderer": "parameterized", "physical_backend": "parameterized-plume", "physical_field_source": {}, "stopped": stopped, "combos": []}

    return run


def _wait(read, done=("succeeded", "failed", "partial", "cancelled"), tries=200):
    for _ in range(tries):
        job = read()
        if job["state"] in done:
            return job
        time.sleep(0.02)
    return job


def test_video_roundtrip_and_sequence_files(tmp_path):
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    first[..., 0] = 240
    second = np.zeros((16, 16, 3), dtype=np.uint8)
    second[..., 2] = 220
    payload = encode_video([first, second], fps=4)
    assert payload.startswith(b"\x00\x00\x00")
    capture_path = tmp_path / "roundtrip.mp4"
    capture_path.write_bytes(payload)
    import cv2

    capture = cv2.VideoCapture(str(capture_path))
    if capture.isOpened():
        ok_first, decoded_first = capture.read()
        ok_second, decoded_second = capture.read()
        capture.release()
        assert ok_first and ok_second
        # OpenCV returns BGR; verify the encoded RGB channels survive the
        # browser-compatible conversion instead of turning into a green cast.
        assert float(decoded_first[..., 2].mean()) > float(decoded_first[..., 1].mean()) * 2
        assert float(decoded_second[..., 0].mean()) > float(decoded_second[..., 1].mean()) * 2
    result = run_sequence(REQUEST, tmp_path)
    sequence = tmp_path / "sequences" / result["sequence_id"]
    assert (sequence / "videos" / "v0_l0.mp4").is_file()
    assert {"fields.json", "point_cloud.json", "provenance.json", "blender_scene.py"} <= {
        path.name for path in sequence.iterdir()
    }
    sample_id = result["combos"][0]["samples"][0]["sample_id"]
    sample_dir = tmp_path / "samples" / sample_id
    assert {path.name for path in sample_dir.iterdir()} == {
        "image.png", "mask.png", "annotation.json", "label_studio.json", "quality_report.json", "provenance.json",
    }
    assert video_encoder()

def test_legacy_mp4v_is_upgraded_for_browser_delivery(tmp_path):
    import cv2

    path = tmp_path / "legacy.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (16, 16)
    )
    assert writer.isOpened()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[..., 2] = 220
    writer.write(frame)
    writer.release()
    legacy = path.read_bytes()
    upgraded = browser_compatible_video(legacy)
    assert upgraded
    if video_encoder().startswith("h264:"):
        assert upgraded != legacy
        assert b"avc1" in upgraded
    capture_path = tmp_path / "upgraded.mp4"
    capture_path.write_bytes(upgraded)
    capture = cv2.VideoCapture(str(capture_path))
    assert capture.isOpened()
    ok, decoded = capture.read()
    capture.release()
    assert ok
    assert float(decoded[..., 2].mean()) > float(decoded[..., 1].mean()) * 2


def test_catalog_pipeline_and_version_file_api(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "firephantom.versioning.datasets.push_task",
        lambda task: {"success": True, "task_id": 7, "called": True},
    )
    with TestClient(create_app(tmp_path)) as client:
        catalog = client.get("/api/v1/catalog")
        assert catalog.status_code == 200
        assert {"房间火灾", "林区火灾", "工业区火灾"} <= set(catalog.json()["scenarios"])
        assert "视角变换" not in catalog.json()["augmentations"]
        run = client.post("/api/v1/pipeline", json=REQUEST)
        assert run.status_code == 200, run.text
        body = run.json()
        assert body["combos"][0]["video_url"].startswith(f"/api/v1/sequences/{body['sequence_id']}/files/videos/")
        assert client.get(body["combos"][0]["video_url"]).status_code == 200
        sample_id = body["combos"][0]["samples"][0]["sample_id"]
        dataset = client.post("/api/v1/datasets", json={"name": "回归"}).json()
        assert client.post(f"/api/v1/datasets/{dataset['dataset_id']}/samples", json={"sample_ids": [sample_id]}).status_code == 200
        detail = client.get(f"/api/v1/samples/{sample_id}").json()
        reviewed = client.post(
            f"/api/v1/samples/{sample_id}/review",
            json={"decision": "reviewed", "objects": detail["auto_objects"]},
        )
        assert reviewed.status_code == 200, reviewed.text
        assert client.post("/api/v1/datasets/publish", json={}).status_code in {404, 405}
        published = client.post(f"/api/v1/datasets/{dataset['dataset_id']}/publish")
        assert published.status_code == 200, published.text
        version_id = published.json()["version_id"]
        files = client.get(f"/api/v1/versions/{version_id}/files").json()["files"]
        names = {item["path"] for item in files}
        split = client.get(f"/api/v1/samples/{sample_id}").json()["split"]
        image_path = f"images/{split}/{sample_id}.png"
        assert {
            "manifest.json",
            "statistics.json",
            "quality_report.json",
            "data.yaml",
            f"annotations/coco_{split}.json",
            f"labels/{split}/{sample_id}.txt",
            image_path,
        } <= names
        image = client.get(f"/api/v1/versions/{version_id}/files/{image_path}")
        assert image.status_code == 200 and image.headers["content-type"].startswith("image/png")
        assert image.headers["accept-ranges"] == "bytes"
        ranged = client.get(
            f"/api/v1/versions/{version_id}/files/{image_path}",
            headers={"Range": "bytes=0-31"},
        )
        assert ranged.status_code == 206
        assert ranged.headers["content-range"].startswith("bytes 0-31/")
        assert ranged.content == image.content[:32]
        archive = client.get(f"/api/v1/versions/{version_id}/archive")
        assert archive.status_code == 200 and archive.headers["content-type"].startswith("application/zip")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert "manifest.json" in bundle.namelist()
            assert image_path in bundle.namelist()
        assert client.get("/api/v1/versions/missing").status_code == 404

def test_job_worker_count_is_bounded_for_cloud_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREPHANTOM_JOB_WORKERS", "99")
    monkeypatch.setattr("firephantom.runtime.platform.system", lambda: "Linux")
    app = create_app(tmp_path)
    try:
        assert app.state.job_manager._executor._max_workers == 8
    finally:
        app.state.job_manager.close()


def test_macos_unified_memory_defaults_to_one_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREPHANTOM_JOB_WORKERS", "8")
    monkeypatch.setattr("firephantom.runtime.platform.system", lambda: "Darwin")
    monkeypatch.setattr("firephantom.runtime.total_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.delenv("FIREPHANTOM_ALLOW_CONCURRENT_MPS", raising=False)
    app = create_app(tmp_path)
    try:
        assert app.state.job_manager._executor._max_workers == 1
    finally:
        app.state.job_manager.close()


def test_persistent_job_api_reports_completion(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/v1/jobs", json={"items": [REQUEST]})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        job = _wait(lambda: client.get(f"/api/v1/jobs/{job_id}").json())
        assert job["state"] == "succeeded", job.get("error")
        assert job["completed"] == 1 and job["total"] == 1
        assert job["kept"] + job["dropped"] == 2
        assert job["request"][0]["scenario"] == "房间火灾"
        first = job["results"][0]
        assert first["frames"] == 2
        assert "samples" not in first
        assert first["preview_url"].endswith("/results/0/preview")
        assert first["video_url"].endswith("/files/videos/v0_l0.mp4")
        result = client.get(f"/api/v1/jobs/{job_id}/results/0")
        assert result.status_code == 200
        assert result.json()["job_id"] == job_id
        assert len(result.json()["samples"]) == 2
        preview = client.get(f"/api/v1/jobs/{job_id}/results/0/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("image/png")
        assert client.get(f"/api/v1/jobs/{job_id}/results/1").status_code == 404
        stages = client.get(f"/api/v1/jobs/{job_id}/stages").json()
        assert stages["plate_url"]
        assert stages["combos"][0]["samples"][0]["status"] in {"generated", "dropped"}
        summary = client.get("/api/v1/jobs").json()["jobs"][0]
        assert summary["results"] == []
        assert summary["total_results"] == 1
        assert summary["has_more_results"] is False

def test_job_results_are_paged_by_combination(tmp_path, monkeypatch):
    monkeypatch.setattr("firephantom.runs.run_sequence", fake_sequence())
    app = create_app(tmp_path)
    # The forest is outdoors, so all three weathers apply: 9 combinations a bake, 14 bakes.
    many = {**REQUEST, "scenario": "林区火灾", "scene": None, "weather": None, "viewpoint": None, "weathers": ["晴", "雨", "雪"], "viewpoints": ["侧视", "俯视", "第一人称"], "count": 14, "wind_step": 0.5}
    with TestClient(app) as client:
        job_id = client.post("/api/v1/jobs", json={"items": [many]}).json()["job_id"]
        job = _wait(lambda: client.get(f"/api/v1/jobs/{job_id}").json())
        assert job["state"] == "succeeded", job.get("error")
        assert job["total_results"] == 126
        assert len(job["results"]) == 50
        assert [item["index"] for item in job["results"]] == list(range(76, 126))
        assert job["has_more_results"] is True
        middle = client.get(f"/api/v1/jobs/{job_id}/results?offset=50&limit=50").json()
        assert [item["index"] for item in middle["results"]] == list(range(26, 76))
        cursor = client.get(f"/api/v1/jobs/{job_id}/results?before_index=26&limit=50").json()
        assert [item["index"] for item in cursor["results"]] == list(range(26))
        assert cursor["has_more"] is False
        assert job["kept"] == 126 * 2
        last = client.get(f"/api/v1/jobs/{job_id}/results/125").json()
        assert last["sequence_index"] == 13

def test_running_job_recovery_reruns_from_the_payload(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    manager.close()
    timestamp = "2026-10-15T00:00:00+00:00"
    sequences = expand_job_items([REQUEST, {**REQUEST, "wind_speed": 2.5}])
    with manager._db() as db:
        db.execute(
            """INSERT INTO jobs(
                id,state,total,completed,kept,dropped,created_at,updated_at,payload,error,cancel_requested
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "recovery", "running", 2, 0, 0, 0, timestamp, timestamp,
                json.dumps({"items": [REQUEST], "sequences": sequences, "combo_index": None}), None, 0,
            ),
        )
        db.execute(
            """INSERT INTO jobs(
                id,state,total,completed,kept,dropped,created_at,updated_at,payload,error,cancel_requested
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            ("legacy", "queued", 1, 0, 0, 0, timestamp, timestamp, json.dumps([REQUEST]), None, 0),
        )
    calls = []
    monkeypatch.setattr("firephantom.runs.run_sequence", fake_sequence(calls))
    manager = JobManager(tmp_path)
    try:
        job = _wait(lambda: manager.get("recovery"))
        assert job["state"] == "succeeded"
        assert calls == [1.5, 2.5]
        assert [item["index"] for item in job["results"]] == [0, 1]
        legacy = _wait(lambda: manager.get("legacy"))
        assert legacy["state"] == "failed" and "旧版本" in legacy["error"]
    finally:
        manager.close()

def test_job_reports_partial_when_sequences_mix_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("firephantom.runs.run_sequence", fake_sequence(fail_wind=1.0))
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/v1/jobs",
            json={"items": [{**REQUEST, "wind_speed": 1.0}, {**REQUEST, "wind_speed": 2.0}]},
        )
        job_id = response.json()["job_id"]
        job = _wait(lambda: client.get(f"/api/v1/jobs/{job_id}").json())
        assert job["state"] == "partial"
        assert job["completed"] == 2
        assert [result["state"] for result in job["results"]] == ["failed", "succeeded"]
        assert "simulated sequence failure" in job["error"]

def test_integrations_and_react_static_build(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        integrations = client.get("/api/v1/integrations")
        assert integrations.status_code == 200
        assert integrations.json()["stable_diffusion_url"].endswith("/sdapi/v1/txt2img")
        index = client.get("/")
        assert index.status_code == 200


def test_system_status_exposes_memory_and_integrations(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"ok", "degraded"}
        assert {"memory", "integrations"} <= set(payload)
        assert {"blender", "stable_diffusion", "label_studio"} <= set(payload["integrations"])
        assert "fds" not in payload["integrations"]
        assert "state" in payload["memory"]


def test_system_status_marks_memory_warning_as_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "firephantom.runtime.memory_status",
        lambda: {"state": "warning", "available_gib": 4.5, "minimum_available_gib": 3.0},
    )
    with TestClient(create_app(tmp_path)) as client:
        payload = client.get("/api/v1/system/status").json()
    assert payload["status"] == "degraded"
    assert payload["memory"]["state"] == "warning"


def test_default_dataset_root_follows_runtime_directory(monkeypatch, tmp_path):
    import firephantom.api as api_module

    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.setenv("FIREPHANTOM_RUNTIME", str(tmp_path))
    assert api_module.default_dataset_root() == tmp_path / "dataset"

    monkeypatch.delenv("FIREPHANTOM_RUNTIME", raising=False)
    assert api_module.default_dataset_root() == Path(__file__).resolve().parents[1] / "data" / "dataset"


def test_queued_job_can_be_cancelled_without_running_the_next_item(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("firephantom.runs.run_sequence", fake_sequence(calls, delay=0.12))
    manager = JobManager(tmp_path, max_workers=1)
    try:
        first = manager.submit([{**REQUEST, "wind_speed": 1.0}])
        second = manager.submit([{**REQUEST, "wind_speed": 2.0}])
        cancelled = manager.cancel(second["job_id"])
        assert cancelled["state"] == "cancelled"
        assert _wait(lambda: manager.get(first["job_id"]))["state"] == "succeeded"
        assert manager.get(second["job_id"])["state"] == "cancelled"
        assert calls == [1.0]
    finally:
        manager.close()

def test_job_memory_guard_stops_before_starting_a_combination(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("firephantom.runs.run_sequence", fake_sequence(calls))
    monkeypatch.setattr(
        "firephantom.jobs.memory_status",
        lambda: {"state": "critical", "available_gib": 2.4, "minimum_available_gib": 3.0},
    )
    manager = JobManager(tmp_path)
    try:
        submitted = manager.submit([REQUEST])
        job = _wait(lambda: manager.get(submitted["job_id"]))
        assert job["state"] == "failed"
        assert "memory guard" in job["error"]
        assert job["completed"] == 0
    finally:
        manager.close()

def test_encryption_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_ENCRYPTION_KEY", "local-test-key")
    secret = b"confidential dataset"
    encrypted = encrypt(secret)
    assert encrypted != secret
    assert decrypt(encrypted) == secret


def test_validation_normalizes_and_rejects_bad_inputs():
    (normalized,) = expand_job_items([REQUEST])
    assert normalized["fuel_type"] == "木材"
    assert normalized["fire_engine"] == "blender"
    # SD left generation: nothing about it is stored on a sequence.
    assert "diffusion" not in normalized
    assert expand_job_items([{**REQUEST, "diffusion": {"enabled": False, "strength": 0.2}}])[0] == normalized
    assert normalized["frame_count"] == 2
    assert isinstance(normalized["seed"], int)
    assert expand_job_items([REQUEST])[0]["seed"] == normalized["seed"]
    for key, value in (
        ("wind_speed", -1),
        ("humidity", 101),
        ("temperature", float("nan")),
        ("fire_engine", "fds"),
        ("frame_count", 49),
        ("augmentations", ["视角变换"]),
        ("diffusion", {"enabled": True}),
        ("scene", "室外"),
    ):
        with pytest.raises(ValueError):
            expand_job_items([{**REQUEST, key: value}])
    # A room preset still has a lid, so the bake stays 室内, but rain is allowed.
    rainy = expand_job_items([{**REQUEST, "weather": "雨"}])[0]
    assert rainy["weathers"] == ["雨"]
    assert rainy["scenes"] == ["室内"]

def test_one_row_is_one_bake_with_every_view_inside():
    items = expand_job_items([{
        "scenario": "林区火灾",
        "wind_speed": 1.5,
        "temperature": 28.0,
        "humidity": 45.0,
        "fuel_type": "木材",
        "weathers": ["晴", "雨", "雪"],
        "viewpoints": ["俯视", "侧视", "第一人称"],
        "fire_engine": "blender",
        "count": 2,
        "wind_step": 1.0,
    }])
    assert len(items) == 2
    assert [item["wind_speed"] for item in items] == [1.5, 2.5]
    assert items[0]["seed"] != items[1]["seed"]
    combos = combo_list(items[0])
    assert len(combos) == 9
    assert (combos[0]["weather"], combos[0]["viewpoint"]) == ("晴", "俯视")
    assert {combo["key"] for combo in combos} == {f"v{v}_l{l}" for v in range(3) for l in range(3)}
    with pytest.raises(ValueError, match="at most"):
        expand_job_items([{**items[0], "frame_count": 48, "count": 100, "augmentations": ["噪声", "模糊"]}])
