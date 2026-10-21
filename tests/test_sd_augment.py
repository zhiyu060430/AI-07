"""SD restyle of reviewed samples: reviewed children, same boxes, same split, published with the source."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.images import encode_mask_png, encode_png
from firephantom.synthesis.sd_augment import augment_source, eligible_sources, normalize_options
from firephantom.versioning.datasets import DatasetCatalog
from firephantom.versioning.publish import publish_dataset
from firephantom.versioning.samples import SampleCatalog


class _TintServer:
    """img2img stand-in: the init image with the blue channel raised."""

    def __init__(self):
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.calls += 1
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                init = cv2.imdecode(np.frombuffer(base64.b64decode(body["init_images"][0]), np.uint8), cv2.IMREAD_COLOR)
                init[..., 0] = np.clip(init[..., 0].astype(int) + 60, 0, 255)
                ok, png = cv2.imencode(".png", init)
                reply = json.dumps({"images": [base64.b64encode(png.tobytes()).decode("ascii")]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/sdapi/v1/txt2img"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()


def _frame():
    rng = np.random.default_rng(1)
    image = rng.integers(40, 120, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=bool)
    mask[120:200, 100:150] = True
    image[mask] = (255, 140, 40)
    return image, mask


BOX = [100.0, 120.0, 50.0, 80.0]


def _objects(sample_id: str | None = None) -> list[dict]:
    image_file = f"samples/{sample_id}/image.png" if sample_id else "image.png"
    return [{"category": "flame", "bbox": BOX, "image_width": 256, "image_height": 256, "image_file": image_file}]


def _reviewed(root, dataset_id: str, sample_id: str = "a" * 64, *, review: bool = True) -> dict:
    image, mask = _frame()
    catalog = SampleCatalog(root)
    catalog.save_generation(
        sample_id=sample_id,
        seed=5,
        generator_version="2.0.0",
        generator_profile={"annotation": "render-mask"},
        scene_params={"scenario": "林区火灾", "scene": "室外", "weather": "雨", "viewpoint": "侧视", "frame_index": 0},
        status="generated",
        quality={"checks": {"kept": True}},
        filter_decision={"kept": True},
        auto_objects=_objects(),
        files={"image.png": encode_png(image), "mask.png": encode_mask_png(mask)},
        sequence_id="seq-forest",
    )
    DatasetCatalog(root)._link(dataset_id, sample_id)
    if review:
        catalog.mark_pending_review(sample_id, None)
        catalog.record_review(sample_id, _objects(), "reviewed")
    return catalog.get(sample_id)


def test_children_are_reviewed_with_the_same_boxes_and_split(tmp_path):
    dataset = DatasetCatalog(tmp_path).create("SD")["dataset_id"]
    source = _reviewed(tmp_path, dataset)
    options = normalize_options({"variants": 2, "styles": ["dusk", "night"]})
    with _TintServer() as server:
        outcomes = augment_source(tmp_path, dataset, source, options, endpoint=server.endpoint, timeout=5.0)
        assert [item["state"] for item in outcomes] == ["created", "created"]
        assert [item["style"] for item in outcomes] == ["dusk", "night"]
        again = augment_source(tmp_path, dataset, source, options, endpoint=server.endpoint, timeout=5.0)
        # Same source, styles and options: the same children, no new SD calls.
        assert [item["state"] for item in again] == ["exists", "exists"]
        assert [item["sample_id"] for item in again] == [item["sample_id"] for item in outcomes]
        assert server.calls == 2
    catalog = SampleCatalog(tmp_path)
    image, mask = _frame()
    for outcome in outcomes:
        child = catalog.get(outcome["sample_id"])
        assert child["status"] == "reviewed"
        assert child["derived_from"] == source["sample_id"]
        assert child["sequence_id"] == source["sequence_id"]
        assert child["reviewed_objects"] == [
            {**obj, "image_file": f"samples/{child['sample_id']}/image.png"} for obj in source["reviewed_objects"]
        ]
        assert child["annotation_error"] == source["annotation_error"]
        assert child["scene_params"]["diffusion"]["style"] == outcome["style"]
        assert DatasetCatalog(tmp_path).dataset_of(child["sample_id"]) == dataset
        restyled = cv2.cvtColor(cv2.imdecode(np.frombuffer(catalog.read_file(child["sample_id"], "image.png"), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(restyled[mask], image[mask])
        assert not np.array_equal(restyled, image)
        provenance = json.loads(catalog.read_file(child["sample_id"], "provenance.json"))
        assert provenance["derived_from"] == source["sample_id"]
        assert "fire" not in provenance["prompt"]
    ids = [source["sample_id"]] + [item["sample_id"] for item in outcomes]
    assert len(set(catalog.assign_splits(ids).values())) == 1
    version = publish_dataset(tmp_path, dataset)
    stats = version["metadata"]["数据统计"]
    assert stats["数据量"]["samples"] == 3
    assert stats["SD 派生"] == {"count": 2, "ratio": pytest.approx(2 / 3, abs=1e-6)}


def test_only_reviewed_originals_are_eligible(tmp_path):
    dataset = DatasetCatalog(tmp_path).create("SD")["dataset_id"]
    source = _reviewed(tmp_path, dataset)
    _reviewed(tmp_path, dataset, "b" * 64, review=False)
    assert [item["sample_id"] for item in eligible_sources(tmp_path, dataset)] == [source["sample_id"]]
    with pytest.raises(ValueError, match="已审核"):
        eligible_sources(tmp_path, dataset, ["b" * 64])
    with _TintServer() as server:
        (outcome,) = augment_source(tmp_path, dataset, source, normalize_options({}), endpoint=server.endpoint, timeout=5.0)
    # An SD child is never restyled again.
    assert [item["sample_id"] for item in eligible_sources(tmp_path, dataset)] == [source["sample_id"]]
    with pytest.raises(ValueError):
        eligible_sources(tmp_path, dataset, [outcome["sample_id"]])


def test_failed_sd_stores_nothing(tmp_path):
    dataset = DatasetCatalog(tmp_path).create("SD")["dataset_id"]
    source = _reviewed(tmp_path, dataset)
    (outcome,) = augment_source(
        tmp_path, dataset, source, normalize_options({}), endpoint="http://127.0.0.1:9/sdapi/v1/txt2img", timeout=1.0
    )
    assert outcome["state"] == "failed" and outcome["sample_id"] is None
    assert SampleCatalog(tmp_path).summary()["counts"]["reviewed"] == 1


def test_api_runs_an_augment_job(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with _TintServer() as server, TestClient(app) as client:
        monkeypatch.setenv("SD_API_URL", server.endpoint)
        dataset = client.post("/api/v1/datasets", json={"name": "SD"}).json()["dataset_id"]
        _reviewed(tmp_path, dataset)
        listing = client.get(f"/api/v1/datasets/{dataset}/sd-augment").json()
        assert listing["eligible"] == 1 and listing["jobs"] == []
        job = client.post(f"/api/v1/datasets/{dataset}/sd-augment", json={"variants": 2}).json()
        for _ in range(100):
            job = client.get(f"/api/v1/sd-augment/{job['job_id']}").json()
            if job["state"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert job["state"] == "succeeded", job["error"]
        assert (job["completed"], job["created"], job["failed"]) == (1, 2, 0)
        assert client.get(f"/api/v1/datasets/{dataset}/sd-augment").json()["jobs"][0]["job_id"] == job["job_id"]
        assert client.post(f"/api/v1/datasets/{dataset}/sd-augment", json={"variants": 9}).status_code == 422
        assert client.get("/api/v1/datasets/missing/sd-augment").status_code == 404
        catalog = client.get("/api/v1/catalog").json()
        assert "indoor_weathers" not in catalog
        assert {item["key"] for item in catalog["sd_styles"]["室内"]} >= {"warehouse", "office"}
