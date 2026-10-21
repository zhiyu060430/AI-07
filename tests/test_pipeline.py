"""End-to-end sequences and the HTTP service contract."""

import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from firephantom import __version__
from firephantom.api import create_app
from firephantom.pipeline import run_sequence
from firephantom.probe import collect_report
from firephantom.versioning.samples import SampleCatalog
from firephantom.versioning.store import DatasetStore

REQUEST = {
    "scenario": "工业区火灾",
    "wind_speed": 1.5,
    "temperature": 28.0,
    "humidity": 45.0,
    "fuel_type": "木材",
    "weathers": ["晴", "雨"],
    "viewpoints": ["侧视", "俯视"],
    "frame_count": 3,
}


def _ids(outcome):
    return [item["sample_id"] for combo in outcome["combos"] for item in combo["samples"]]


def test_every_frame_of_every_view_is_a_sample_and_repeats(tmp_path):
    first = run_sequence(REQUEST, tmp_path)
    second = run_sequence(REQUEST, tmp_path)
    assert _ids(first) == _ids(second)
    assert first["sequence_id"] == second["sequence_id"]
    assert len(first["combos"]) == 4
    assert len(set(_ids(first))) == 4 * 3
    assert DatasetStore(tmp_path).version_count() == 0
    catalog = SampleCatalog(tmp_path)
    combo = first["combos"][3]
    assert (combo["weather"], combo["viewpoint"]) == ("雨", "俯视")
    record = catalog.get(combo["samples"][1]["sample_id"])
    assert record["status"] == "generated"
    assert record["sequence_id"] == first["sequence_id"]
    assert record["generator_version"] == __version__
    assert record["scene_params"]["frame_index"] == 1
    assert record["scene_params"]["viewpoint"] == "俯视"
    assert record["scene_params"]["diffusion"] is None
    assert record["generator_profile"]["annotation"] == "render-mask"
    sample_dir = tmp_path / "samples" / record["sample_id"]
    mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    x, y, w, h = (int(v) for v in record["auto_objects"][0]["bbox"])
    ys, xs = np.nonzero(mask)
    assert (x, y) == (xs.min(), ys.min())
    task = json.loads((sample_dir / "label_studio.json").read_text(encoding="utf-8"))
    assert task["data"]["sample_id"] == record["sample_id"]
    assert not task["data"]["image"].startswith("data:")
    provenance = json.loads((sample_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["video"] == f"sequences/{first['sequence_id']}/videos/{combo['key']}.mp4"
    by_tool = {item["tool"]: item for item in first["tool_runs"]}
    assert set(by_tool) == {"blender"}
    assert "blender executable was not run" in by_tool["blender"]["message"]


def test_layout_seed_and_physics_change_the_samples(tmp_path):
    base = run_sequence(REQUEST, tmp_path)
    windy = run_sequence({**REQUEST, "wind_speed": 8.0}, tmp_path)
    seeded = run_sequence({**REQUEST, "seed": 7}, tmp_path)
    laid_out = run_sequence({**REQUEST, "placements": []}, tmp_path)
    from firephantom.physics.placements import default_placements

    furnished = run_sequence({**REQUEST, "placements": default_placements("房间火灾")}, tmp_path)
    assert set(_ids(base)).isdisjoint(_ids(windy))
    assert set(_ids(base)).isdisjoint(_ids(seeded))
    assert set(_ids(laid_out)).isdisjoint(_ids(furnished))
    assert SampleCatalog(tmp_path).get(_ids(seeded)[0])["seed"] == 7


def test_frames_without_a_visible_flame_are_dropped(tmp_path):
    outcome = run_sequence({**REQUEST, "min_flame_area": 0.5}, tmp_path)
    statuses = {item["status"] for combo in outcome["combos"] for item in combo["samples"]}
    assert statuses == {"dropped"}
    sample = outcome["combos"][0]["samples"][0]
    assert sample["reasons"] == ["flame_too_small"]


def test_augmentations_add_derived_samples(tmp_path):
    outcome = run_sequence({**REQUEST, "viewpoints": ["侧视"], "weathers": ["晴"], "augmentations": ["噪声", "模糊"]}, tmp_path)
    samples = outcome["combos"][0]["samples"]
    assert len(samples) == 3 * 3
    base = samples[0]
    derived = [item for item in samples if item["augmentation"]]
    assert {item["augmentation"] for item in derived} == {"噪声", "模糊"}
    record = SampleCatalog(tmp_path).get(derived[0]["sample_id"])
    assert record["derived_from"] == base["sample_id"]
    original = SampleCatalog(tmp_path).get(base["sample_id"])
    assert record["auto_objects"][0]["bbox"] == original["auto_objects"][0]["bbox"]


def test_http_pipeline_and_openapi(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        document = client.get("/openapi.json").json()
        assert document["paths"]["/api/v1/pipeline"]["post"]["operationId"] == "run_sequence"
        assert "/api/v1/scenes" in document["paths"]
        rejected = client.post("/api/v1/pipeline", json={**REQUEST, "scenario": "海上火灾"})
        assert rejected.status_code == 400
        accepted = client.post("/api/v1/pipeline", json={**REQUEST, "frame_count": 1})
        assert accepted.status_code == 200, accepted.text
        body = accepted.json()
        assert len(body["combos"]) == 4
        assert "preview" not in body["combos"][0]
        sample_id = body["combos"][0]["samples"][0]["sample_id"]
        mask = client.get(f"/api/v1/samples/{sample_id}/files/mask.png")
        assert mask.status_code == 200 and mask.headers["content-type"] == "image/png"
        fields = client.get(f"/api/v1/sequences/{body['sequence_id']}/files/fields.json")
        assert fields.status_code == 200
        assert client.get(f"/api/v1/sequences/{body['sequence_id']}/files/../../samples").status_code == 404


def test_probe_records_missing_external_tools():
    report = collect_report()
    assert "blender executable was not run" in report
    assert "stable diffusion endpoint" in report
    assert "call failed" in report
    assert "inception weights were not loaded" in report
