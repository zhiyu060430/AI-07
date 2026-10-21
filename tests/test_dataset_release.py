"""Dataset publish, statistics, comparison, and rollback through the API."""

import json

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.images import encode_png
from firephantom.versioning.datasets import DatasetCatalog
from firephantom.versioning.publish import publish_dataset
from firephantom.versioning.samples import SampleCatalog
from firephantom.versioning.store import DatasetStore


def _png() -> bytes:
    return encode_png(np.zeros((100, 100, 3), dtype=np.uint8))


def _objects(bbox: list[float]) -> list[dict]:
    return [{"category": "flame", "bbox": bbox, "image_width": 100, "image_height": 100, "image_file": "image.png"}]


def _save(root, sample_id: str, *, weather: str, bbox: list[float], sequence: str | None = None) -> None:
    SampleCatalog(root).save_generation(
        sample_id=sample_id,
        seed=11,
        generator_version="2.0.0",
        generator_profile={"annotation": "render-mask"},
        scene_params={
            "scenario": "房间火灾",
            "scene": "室内",
            "weather": weather,
            "viewpoint": "侧视",
            "frame_index": 0,
        },
        status="generated",
        quality={"checks": {"kept": True}},
        filter_decision={"kept": True},
        auto_objects=_objects(bbox),
        files={"image.png": _png(), "annotation.json": json.dumps(_objects(bbox)).encode("utf-8")},
        sequence_id=sequence or f"seq-{sample_id}",
    )


def _admit_and_review(root, dataset_id: str, sample_id: str, bbox: list[float]) -> None:
    DatasetCatalog(root)._link(dataset_id, sample_id)
    catalog = SampleCatalog(root)
    catalog.mark_pending_review(sample_id, None)
    catalog.record_review(sample_id, _objects(bbox), "reviewed")


def test_publish_freezes_statistics_splits_and_training_layout(tmp_path):
    dataset = DatasetCatalog(tmp_path).create("统计")
    _save(tmp_path, "sunny", weather="晴", bbox=[0, 0, 10, 10])
    _save(tmp_path, "rainy", weather="雨", bbox=[0, 0, 10, 10])
    _admit_and_review(tmp_path, dataset["dataset_id"], "sunny", [0, 0, 10, 10])
    _admit_and_review(tmp_path, dataset["dataset_id"], "rainy", [0, 0, 40, 40])
    first = publish_dataset(tmp_path, dataset["dataset_id"])
    assert DatasetStore(tmp_path).version_count() == 1
    stats = first["metadata"]["数据统计"]
    assert stats["数据量"] == {"samples": 2, "images": 2, "boxes": 2, "sequences": 2}
    assert [item["count"] for item in stats["火焰尺寸分布"]["bins"]] == [1, 0, 1, 0]
    assert stats["天气比例"]["晴"]["ratio"] == 0.5
    assert stats["天气比例"]["雪"]["count"] == 0
    assert sum(item["count"] for item in stats["划分"].values()) == 2
    assert first["metadata"]["标注错误率"] == 0.5
    assert first["metadata"]["质量报告"]["method"] == "version_set"
    store = DatasetStore(tmp_path)
    files = {item["path"] for item in store.list_files(first["version_id"])}
    manifest = json.loads(store.read_file(first["version_id"], "manifest.json"))
    assert [item["sample_id"] for item in manifest["samples"]] == ["rainy", "sunny"]
    entry = manifest["samples"][0]
    assert entry["image"] in files and entry["labels"] in files
    config = yaml.safe_load(store.read_file(first["version_id"], "data.yaml"))
    assert config["names"] == {0: "flame", 1: "smoke", 2: "ember"}
    assert (config["train"], config["val"], config["test"]) == ("images/train", "images/val", "images/test")
    assert store.read_file(first["version_id"], entry["image"]) == _png()


def test_published_images_survive_a_regenerated_sample(tmp_path):
    dataset = DatasetCatalog(tmp_path).create("硬链接")
    _save(tmp_path, "frozen", weather="晴", bbox=[0, 0, 10, 10])
    _admit_and_review(tmp_path, dataset["dataset_id"], "frozen", [0, 0, 10, 10])
    record = publish_dataset(tmp_path, dataset["dataset_id"])
    store = DatasetStore(tmp_path)
    path = next(item["path"] for item in store.list_files(record["version_id"]) if item["path"].startswith("images/"))
    before = store.read_file(record["version_id"], path)
    SampleCatalog(tmp_path).write_files("frozen", {"image.png": encode_png(np.full((100, 100, 3), 255, dtype=np.uint8))})
    assert store.read_file(record["version_id"], path) == before


def test_unreviewed_and_dropped_members_are_not_published(tmp_path):
    datasets = DatasetCatalog(tmp_path)
    dataset = datasets.create("等待")
    _save(tmp_path, "waiting", weather="晴", bbox=[0, 0, 10, 10])
    datasets._link(dataset["dataset_id"], "waiting")
    with pytest.raises(ValueError, match="没有可发布"):
        publish_dataset(tmp_path, dataset["dataset_id"])
    assert DatasetStore(tmp_path).version_count() == 0


def test_compare_shows_the_sample_diff_and_rollback_moves_the_head(tmp_path):
    datasets = DatasetCatalog(tmp_path)
    dataset = datasets.create("对比")
    dataset_id = dataset["dataset_id"]
    _save(tmp_path, "left", weather="晴", bbox=[0, 0, 10, 10])
    _save(tmp_path, "right", weather="雪", bbox=[0, 0, 80, 80])
    _admit_and_review(tmp_path, dataset_id, "left", [0, 0, 10, 10])
    left = publish_dataset(tmp_path, dataset_id)
    _admit_and_review(tmp_path, dataset_id, "right", [0, 0, 80, 80])
    right = publish_dataset(tmp_path, dataset_id)
    catalog = SampleCatalog(tmp_path)
    catalog.record_evaluation(right["version_id"], {"map50": 0.4, "precision": 0.5, "recall": 0.6})
    with TestClient(create_app(tmp_path)) as client:
        body = client.get("/api/v1/versions/compare", params={"base": left["version_id"], "other": right["version_id"]}).json()
        assert body["data_volume"]["delta"]["samples"] == 1
        assert body["samples"]["counts"] == {"added": 1, "removed": 0, "split_changed": 0}
        assert body["samples"]["added"] == ["right"]
        assert body["scene_weather"]["other"]["天气比例"]["雪"]["count"] == 1
        assert body["detection"]["other"]["map50"] == 0.4
        assert client.get("/api/v1/versions/compare", params={"base": "missing", "other": right["version_id"]}).status_code == 404
        listed = client.get("/api/v1/versions", params={"dataset_id": dataset_id}).json()
        assert listed["head"] == right["version_id"] and listed["total"] == 2
        assert listed["versions"][0]["parent_id"] == left["version_id"]
        rolled = client.post(f"/api/v1/versions/{left['version_id']}/rollback")
        assert rolled.status_code == 200
        assert rolled.json()["dataset"]["head_version_id"] == left["version_id"]
        assert client.get("/api/v1/versions", params={"dataset_id": dataset_id}).json()["head"] == left["version_id"]
        plan = client.get(f"/api/v1/datasets/{dataset_id}/plan").json()
        assert plan["parent_id"] == left["version_id"] and plan["added"] == []
    assert SampleCatalog(tmp_path).get("right")["status"] == "reviewed"


def test_compare_rejects_legacy_sample_versions(tmp_path):
    store = DatasetStore(tmp_path)
    legacy = store.add({"a.txt": b"old"}, {"生成参数": {}, "质量报告": {}}, version_id="legacy")
    dataset = DatasetCatalog(tmp_path).create("新")
    _save(tmp_path, "kept", weather="晴", bbox=[0, 0, 10, 10])
    _admit_and_review(tmp_path, dataset["dataset_id"], "kept", [0, 0, 10, 10])
    published = publish_dataset(tmp_path, dataset["dataset_id"])
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/versions/compare", params={"base": legacy["version_id"], "other": published["version_id"]})
        assert response.status_code == 400
        recorded = client.post(
            f"/api/v1/versions/{legacy['version_id']}/evaluations",
            json={"map50": 0.1, "precision": 0.2, "recall": 0.3},
        )
        assert recorded.status_code == 400
