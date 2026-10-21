"""Saved scenes: revisions, thumbnails, and jobs that pin the revision they rendered."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.images import encode_png
from firephantom.physics.placements import default_placements
from firephantom.scenes import SceneStore, resolve_scene_items


@pytest.fixture(autouse=True)
def _jobs_have_headroom(monkeypatch):
    monkeypatch.setattr(
        "firephantom.jobs.memory_status",
        lambda: {"state": "ok", "available_gib": 8.0, "minimum_available_gib": 3.0},
    )


def test_catalog_placements_keep_a_size_when_the_editor_omits_it(tmp_path):
    store = SceneStore(tmp_path)
    layout = default_placements("房间火灾")
    slim = [{key: value for key, value in row.items() if key != "size"} for row in layout]
    scene = store.create("无尺寸", "房间火灾", "木材", slim)
    assert [row["size"] for row in scene["placements"]] == [row["size"] for row in layout]
    assert store.revision(scene["scene_id"])["placements"][0]["size"] == layout[0]["size"]


def test_content_changes_make_revisions_and_old_ones_stay(tmp_path):
    store = SceneStore(tmp_path)
    layout = default_placements("房间火灾")
    scene = store.create("客厅", "房间火灾", "木材", layout)
    assert scene["revision"] == 1
    assert scene["placements"] == [{key: row[key] for key in row if key in scene["placements"][0]} for row in layout]
    renamed = store.update(scene["scene_id"], name="客厅 A")
    assert renamed["revision"] == 1 and renamed["name"] == "客厅 A"
    emptied = store.update(scene["scene_id"], placements=[], base_revision=1)
    assert emptied["revision"] == 2
    with pytest.raises(ValueError, match="请先重新载入"):
        store.update(scene["scene_id"], origin_fuel="汽油", base_revision=1)
    assert store.revision(scene["scene_id"], 1)["placements"] == scene["placements"]
    assert store.revision(scene["scene_id"])["placements"] == []
    with pytest.raises(ValueError):
        store.create("坏", "房间火灾", "木材", [{"asset_id": "nope", "x": 1, "y": 1, "z": 0}])
    with pytest.raises(ValueError):
        store.create("", "房间火灾", "木材", [])
    store.delete(scene["scene_id"])
    assert store.get(scene["scene_id"]) is None
    assert store.revision(scene["scene_id"], 2)["revision"] == 2


def test_scene_stores_a_custom_origin_camera_and_gas_box(tmp_path):
    store = SceneStore(tmp_path)
    camera = {"侧视": {"location": [4.0, 1.0, 1.2], "target": [1.2, 1.2, 1.0], "focal_mm": 50}}
    gas = {"location": [2.5, 2.5, 1.2], "scale": [1.0, 1.0, 1.0], "resolution": 24}
    origin = [2.0, 2.0, 0.3]
    scene = store.create("自定义", "房间火灾", "木材", [], origin=origin, cameras=camera, gas_domain=gas)
    assert scene["origin"] == pytest.approx(origin)
    assert scene["cameras"]["侧视"]["focal_mm"] == 50
    assert scene["gas_domain"]["resolution"] == 24
    renamed = store.update(scene["scene_id"], name="自定义 A")
    assert renamed["revision"] == 1 and renamed["cameras"]["侧视"]["focal_mm"] == 50
    cleared = store.update(scene["scene_id"], cameras={}, base_revision=1)
    assert cleared["revision"] == 2 and cleared["cameras"] == {}
    assert store.revision(scene["scene_id"], 1)["cameras"]["侧视"]["location"] == pytest.approx(camera["侧视"]["location"])
    copy = store.duplicate(scene["scene_id"])
    assert copy["origin"] == pytest.approx(origin)
    assert copy["gas_domain"]["resolution"] == 24
    (resolved,) = resolve_scene_items(tmp_path, [{"scene_id": scene["scene_id"], "wind_speed": 1}])
    assert resolved["scene_revision"] == 2
    assert resolved["gas_domain"]["resolution"] == 24
    (pinned,) = resolve_scene_items(tmp_path, [{"scene_id": scene["scene_id"], "scene_revision": 1, "wind_speed": 1}])
    assert pinned["cameras"]["侧视"]["focal_mm"] == 50


def test_scene_stores_lamps_and_an_empty_list_is_not_automatic(tmp_path):
    store = SceneStore(tmp_path)
    scene = store.create("暗室", "房间火灾", "木材", [])
    assert scene["lights"] is None
    lamp = {
        "id": "L1",
        "name": "台灯",
        "type": "POINT",
        "color": [1, 0.9, 0.7],
        "energy": 200,
        "size": 0.4,
        "size_y": 1,
        "spot_size": 0.8,
        "spot_blend": 0.15,
        "location": [2, 2, 1.5],
        "rotation": [0, 0, 0],
        "enabled": True,
    }
    lit = store.update(scene["scene_id"], lights=[lamp], base_revision=1)
    assert lit["revision"] == 2
    assert lit["lights"][0]["energy"] == 200
    assert store.revision(scene["scene_id"], 1)["lights"] is None
    cleared = store.update(scene["scene_id"], lights=[], base_revision=2)
    assert cleared["lights"] == []
    assert cleared["revision"] == 3
    copy = store.duplicate(scene["scene_id"])
    assert copy["lights"] == []
    (resolved,) = resolve_scene_items(tmp_path, [{"scene_id": scene["scene_id"], "wind_speed": 1}])
    assert resolved["lights"] == []


def test_scene_stores_simulation_settings(tmp_path):
    store = SceneStore(tmp_path)
    scene = store.create("慢烧", "房间火灾", "木材", [], simulation={"warmup_s": 1.0})
    assert scene["simulation"] is None
    tuned = store.update(scene["scene_id"], simulation={"warmup_s": 4, "samples": 48}, base_revision=1)
    assert tuned["revision"] == 2
    assert tuned["simulation"]["warmup_s"] == 4
    assert tuned["simulation"]["noise"] == 2
    assert store.revision(scene["scene_id"], 1)["simulation"] is None
    assert store.duplicate(scene["scene_id"])["simulation"]["samples"] == 48
    (resolved,) = resolve_scene_items(tmp_path, [{"scene_id": scene["scene_id"], "wind_speed": 1}])
    assert resolved["simulation"]["warmup_s"] == 4


def test_layout_preview_route(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        preview = client.post("/api/v1/layout/preview", json={"scenario": "房间火灾", "origin_fuel": "木材", "placements": []})
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert set(body["cameras"]) == {"侧视", "俯视", "第一人称"}
        assert len(body["gas_domain"]["bounds"]) == 6
        assert body["gas_overridden"] is False
        refused = client.post("/api/v1/layout/preview", json={"scenario": "房间火灾", "origin": [0, 0, 0]})
        assert refused.status_code == 400


def test_job_rows_pin_the_scene_revision(tmp_path):
    store = SceneStore(tmp_path)
    scene = store.create("厂区", "工业区火灾", "汽油", default_placements("工业区火灾"))
    row = {"scene_id": scene["scene_id"], "scenario": "房间火灾", "fuel_type": "木材", "wind_speed": 1.0}
    (resolved,) = resolve_scene_items(tmp_path, [row])
    assert resolved["scenario"] == "工业区火灾"
    assert resolved["fuel_type"] == "汽油"
    assert resolved["scene_revision"] == 1
    store.update(scene["scene_id"], placements=[])
    (pinned,) = resolve_scene_items(tmp_path, [{**row, "scene_revision": 1}])
    assert pinned["placements"] == scene["placements"]
    (latest,) = resolve_scene_items(tmp_path, [row])
    assert latest["scene_revision"] == 2 and latest["placements"] == []


def test_scene_api_thumbnail_duplicate_and_job(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = client.post("/api/v1/scenes", json={"name": "仓库", "scenario": "房间火灾", "origin_fuel": "木材", "placements": []})
        assert created.status_code == 200, created.text
        scene = created.json()
        assert scene["thumbnail_url"] is None
        png = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
        assert client.put(f"/api/v1/scenes/{scene['scene_id']}/thumbnail", content=b"not a png").status_code == 400
        thumbed = client.put(f"/api/v1/scenes/{scene['scene_id']}/thumbnail", content=png)
        assert thumbed.status_code == 200
        assert client.get(thumbed.json()["thumbnail_url"]).content == png
        copy = client.post(f"/api/v1/scenes/{scene['scene_id']}/duplicate", json={}).json()
        assert copy["name"] == "仓库 副本" and copy["thumbnail_url"]
        assert [item["scene_id"] for item in client.get("/api/v1/scenes").json()["scenes"]][:2] == [copy["scene_id"], scene["scene_id"]]
        updated = client.put(f"/api/v1/scenes/{scene['scene_id']}", json={"origin_fuel": "塑料", "base_revision": 1})
        assert updated.json()["revision"] == 2
        job = client.post("/api/v1/jobs", json={"items": [{
            "scene_id": scene["scene_id"],
            "wind_speed": 1.0,
            "temperature": 20.0,
            "humidity": 40.0,
            "scene": "室内",
            "weather": "晴",
            "viewpoint": "侧视",
            "frame_count": 1,
        }]}).json()
        for _ in range(200):
            detail = client.get(f"/api/v1/jobs/{job['job_id']}").json()
            if detail["state"] not in {"queued", "running"}:
                break
            time.sleep(0.02)
        assert detail["state"] == "succeeded", detail["error"]
        sample_id = client.get(f"/api/v1/jobs/{job['job_id']}/results/0").json()["samples"][0]["sample_id"]
        params = client.get(f"/api/v1/samples/{sample_id}").json()["scene_params"]
        assert (params["scene_id"], params["scene_revision"], params["fuel_type"]) == (scene["scene_id"], 2, "塑料")
        assert client.delete(f"/api/v1/scenes/{copy['scene_id']}").status_code == 200
        assert client.get(f"/api/v1/scenes/{copy['scene_id']}").status_code == 404
