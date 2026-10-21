"""Job runs, the confirm-then-annotate gate, and cumulative dataset versions."""

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.images import encode_png
from firephantom.physics.plate import render_scenario_plate
from firephantom.pipeline import run_sequence
from firephantom.versioning.datasets import DatasetCatalog
from firephantom.versioning.publish import publish_dataset, version_members
from firephantom.versioning.samples import SampleCatalog


REQUEST = {
    "scenario": "房间火灾",
    "wind_speed": 1.5,
    "temperature": 28.0,
    "humidity": 45.0,
    "fuel_type": "木材",
    "scene": "室内",
    "weather": "晴",
    "viewpoints": ["侧视", "俯视"],
    "frame_count": 2,
}


@pytest.fixture(autouse=True)
def _jobs_have_headroom(monkeypatch):
    monkeypatch.setattr(
        "firephantom.jobs.memory_status",
        lambda: {"state": "ok", "available_gib": 8.0, "minimum_available_gib": 3.0},
    )


def _objects(bbox):
    return [{
        "category": "flame",
        "bbox": bbox,
        "image_width": 8,
        "image_height": 8,
        "image_file": "image.png",
    }]


def _save(root, sample_id: str, status: str = "generated", sequence_id: str | None = None) -> None:
    SampleCatalog(root).save_generation(
        sample_id=sample_id,
        seed=3,
        generator_version="2.0.0",
        generator_profile={"annotation": "render-mask"},
        scene_params={"scenario": "房间火灾", "scene": "室内", "weather": "晴", "viewpoint": "侧视", "frame_index": 0},
        status=status,
        quality={"checks": {"kept": status == "generated"}},
        filter_decision={"kept": status == "generated"},
        auto_objects=_objects([0, 0, 4, 4]),
        files={
            "image.png": encode_png(np.zeros((8, 8, 3), dtype=np.uint8)),
            "label_studio.json": json.dumps({"data": {"sample_id": sample_id, "image": f"/api/v1/samples/{sample_id}/files/image.png"}}).encode("utf-8"),
        },
        sequence_id=sequence_id,
    )


def _reviewed(root, dataset_id, sample_ids, sequence_id=None):
    catalog = SampleCatalog(root)
    datasets = DatasetCatalog(root)
    for sample_id in sample_ids:
        _save(root, sample_id, sequence_id=sequence_id)
        datasets._link(dataset_id, sample_id)
        catalog.mark_pending_review(sample_id, "1")
        catalog.record_review(sample_id, _objects([0, 0, 4, 4]), "reviewed")


def test_scenario_plates_differ():
    room = render_scenario_plate("房间火灾")
    forest = render_scenario_plate("林区火灾")
    assert room.shape == (256, 256, 3)
    assert not np.array_equal(room, forest)


def test_generation_stays_out_of_label_studio(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "firephantom.annotation.labelstudio.push_task",
        lambda *_args, **_kwargs: calls.append("push") or {"success": True, "called": True},
    )
    outcome = run_sequence(REQUEST, tmp_path)
    sample_id = outcome["combos"][0]["samples"][0]["sample_id"]
    assert SampleCatalog(tmp_path).get(sample_id)["status"] == "generated"
    assert calls == []


def test_admit_checks_every_sample_before_sending_any(tmp_path, monkeypatch):
    pushed = []

    def fake_push(task, endpoint=None, timeout=1.5):
        pushed.append(task["data"]["sample_id"])
        return {"called": True, "success": True, "task_id": len(pushed)}

    monkeypatch.setattr("firephantom.versioning.datasets.push_task", fake_push)
    for sample_id in ("one", "two", "three"):
        _save(tmp_path, sample_id)
    _save(tmp_path, "blurry", status="dropped")
    catalog = DatasetCatalog(tmp_path)
    first = catalog.create("第一批")
    with pytest.raises(ValueError, match="没有通过质量检查"):
        catalog.admit(first["dataset_id"], ["one", "blurry"])
    assert pushed == []
    admitted = catalog.admit(first["dataset_id"], ["one", "two"])
    assert admitted["sample_ids"] == ["one", "two"]
    assert pushed == ["one", "two"]
    samples = SampleCatalog(tmp_path)
    assert samples.get("one")["status"] == "pending_review"
    assert samples.get("three")["status"] == "generated"
    assert catalog.datasets_of(["one", "three"]) == {"one": first["dataset_id"]}
    second = catalog.create("另一批")
    with pytest.raises(ValueError, match="已在别的数据集"):
        catalog.admit(second["dataset_id"], ["one"])
    with pytest.raises(ValueError, match="已经送去标注"):
        catalog.admit(first["dataset_id"], ["one"])


def test_label_studio_gets_an_absolute_image_url(tmp_path, monkeypatch):
    from firephantom.annotation import labelstudio

    monkeypatch.setenv("FIREPHANTOM_PUBLIC_URL", "http://lab.local:8765")
    task = labelstudio.resolve_image_url({"data": {"image": "/api/v1/samples/x/files/image.png"}})
    assert task["data"]["image"] == "http://lab.local:8765/api/v1/samples/x/files/image.png"


def test_failed_push_leaves_the_rest_unsent(tmp_path, monkeypatch):
    calls = []

    def flaky(task, **_kwargs):
        calls.append(task["data"]["sample_id"])
        if len(calls) == 2:
            return {"called": True, "success": False, "error": "Label Studio 没有响应"}
        return {"called": True, "success": True, "task_id": len(calls)}

    monkeypatch.setattr("firephantom.versioning.datasets.push_task", flaky)
    for sample_id in ("a", "b", "c"):
        _save(tmp_path, sample_id)
    catalog = DatasetCatalog(tmp_path)
    dataset = catalog.create("待确认")
    with pytest.raises(ValueError, match="已送出 1 个，剩下 2 个"):
        catalog.admit(dataset["dataset_id"], ["a", "b", "c"])
    samples = SampleCatalog(tmp_path)
    assert [samples.get(name)["status"] for name in ("a", "b", "c")] == ["pending_review", "generated", "generated"]
    assert catalog.datasets_of(["a", "b", "c"]) == {"a": dataset["dataset_id"]}


def test_review_state_machine_and_frozen_published_samples(tmp_path):
    datasets = DatasetCatalog(tmp_path)
    dataset = datasets.create("审核")
    catalog = SampleCatalog(tmp_path)
    _save(tmp_path, "new")
    with pytest.raises(ValueError, match="不能记录审核"):
        catalog.record_review("new", [], "reviewed")
    _reviewed(tmp_path, dataset["dataset_id"], ["done"])
    with pytest.raises(ValueError, match="不能记录审核"):
        catalog.record_review("done", [], "rejected")
    assert catalog.record_review("done", _objects([1, 1, 3, 3]), "reviewed", rereview=True)["status"] == "reviewed"
    publish_dataset(tmp_path, dataset["dataset_id"])
    with pytest.raises(ValueError, match="已在发布的版本里"):
        catalog.record_review("done", [], "rejected", rereview=True)


def test_versions_are_cumulative_split_by_sequence_and_roll_back(tmp_path):
    datasets = DatasetCatalog(tmp_path)
    dataset = datasets.create("累积")
    dataset_id = dataset["dataset_id"]
    with pytest.raises(ValueError, match="没有可发布"):
        publish_dataset(tmp_path, dataset_id)
    _reviewed(tmp_path, dataset_id, ["a1", "a2", "a3"], sequence_id="sequence-a")
    first = publish_dataset(tmp_path, dataset_id)
    assert first["parent_id"] is None
    assert sorted(version_members(first["version_id"])) == ["a1", "a2", "a3"]
    assert len(set(version_members(first["version_id"]).values())) == 1
    with pytest.raises(ValueError, match="没有变化"):
        publish_dataset(tmp_path, dataset_id)
    assert datasets.get(dataset_id)["ready_to_publish"] == 0

    _reviewed(tmp_path, dataset_id, ["b1"], sequence_id="sequence-b")
    assert datasets.get(dataset_id)["pending_added"] == 1
    second = publish_dataset(tmp_path, dataset_id)
    assert second["parent_id"] == first["version_id"]
    members = version_members(second["version_id"])
    assert sorted(members) == ["a1", "a2", "a3", "b1"]
    assert {members[name] for name in ("a1", "a2", "a3")} == set(version_members(first["version_id"]).values())
    assert second["metadata"]["变化"] == {"added": 1, "removed": 0}

    datasets.rollback(dataset_id, first["version_id"])
    assert datasets.get(dataset_id)["head_version_id"] == first["version_id"]
    _reviewed(tmp_path, dataset_id, ["c1"], sequence_id="sequence-c")
    third = publish_dataset(tmp_path, dataset_id)
    assert third["parent_id"] == first["version_id"]
    assert sorted(version_members(third["version_id"])) == ["a1", "a2", "a3", "c1"]

    other = datasets.create("别的")
    with pytest.raises(ValueError, match="不属于"):
        datasets.rollback(other["dataset_id"], first["version_id"])


def test_job_run_exposes_plate_and_rejects_a_second_continue(tmp_path, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()

    def slow_sequence(sequence, root, only=None, on_combo=None, stop=None, simulated=None):
        entered.set()
        assert gate.wait(5)
        return {"renderer": "parameterized", "physical_backend": "parameterized-plume", "physical_field_source": {}, "stopped": False, "combos": []}

    monkeypatch.setattr("firephantom.runs.run_sequence", slow_sequence)
    try:
        with TestClient(create_app(tmp_path)) as client:
            created = client.post("/api/v1/jobs", json={"items": [REQUEST]})
            assert created.status_code == 200
            job_id = created.json()["job_id"]
            assert created.json()["total"] == 2
            assert entered.wait(5)
            body = client.get(f"/api/v1/jobs/{job_id}/stages").json()
            assert body["plate_url"]
            assert [combo["viewpoint"] for combo in body["combos"]] == ["侧视", "俯视"]
            picture = client.get(body["plate_url"])
            assert picture.status_code == 200 and picture.headers["content-type"].startswith("image/png")
            blocked = client.post(f"/api/v1/jobs/{job_id}/continue", json={"from_stage": "diffusion", "items": [REQUEST]})
            assert blocked.status_code == 400
            assert "作业还在跑" in blocked.json()["detail"]
            gate.set()
            for _ in range(100):
                if client.get(f"/api/v1/jobs/{job_id}").json()["state"] != "running":
                    break
                time.sleep(0.02)
    finally:
        gate.set()


def test_continue_one_combination_reuses_the_bake(tmp_path, monkeypatch):
    import firephantom.physics.engine as engine

    calls = []
    original = engine._build
    monkeypatch.setattr(engine, "_build", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    with TestClient(create_app(tmp_path)) as client:
        job_id = client.post("/api/v1/jobs", json={"items": [REQUEST]}).json()["job_id"]
        for _ in range(200):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["state"] not in {"queued", "running"}:
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded", job["error"]
        again = client.post(
            f"/api/v1/jobs/{job_id}/continue",
            json={"from_stage": "diffusion", "combo_index": 1, "items": [REQUEST]},
        )
        assert again.status_code == 200
        for _ in range(200):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["state"] not in {"queued", "running"}:
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded"
        assert job["completed"] == 2
    assert calls == [1]
