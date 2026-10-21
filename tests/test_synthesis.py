"""Rendered views, SD background restyle of reviewed frames, and photometric augmentations."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from firephantom.physics.engine import simulate
from firephantom.synthesis.augment import AUGMENTATIONS, augment_image
from firephantom.synthesis.sd_augment import (
    STYLE_BANK,
    build_prompt,
    flame_guard,
    normalize_options,
    plan_variants,
    restyle,
)

PHYSICAL = simulate(
    "房间火灾",
    wind_speed=1.5,
    temperature=28.0,
    humidity=40.0,
    fuel_type="木材",
    viewpoints=["侧视", "俯视"],
    looks=[("室内", "晴"), ("室外", "雪")],
    frame_count=3,
)
RENDER = PHYSICAL["renders"][0]


class _EchoServer:
    """img2img stand-in that records requests.

    ``mode`` "tint" returns the init image with more blue, "noise" returns
    random pixels, so the restyle can be checked for both outcomes.
    """

    def __init__(self, mode: str = "tint"):
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.headers.append(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                outer.requests.append({"path": self.path, "body": body})
                init = cv2.imdecode(np.frombuffer(base64.b64decode(body["init_images"][0]), np.uint8), cv2.IMREAD_COLOR)
                if mode == "noise":
                    picture = np.random.default_rng(0).integers(0, 255, init.shape, dtype=np.uint8)
                else:
                    picture = init.copy()
                    picture[..., 0] = np.clip(picture[..., 0].astype(int) + 60, 0, 255)
                ok, png = cv2.imencode(".png", picture)
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


def test_every_view_and_look_is_its_own_render_with_masks():
    keys = [(item["scene"], item["weather"], item["viewpoint"]) for item in PHYSICAL["renders"]]
    assert keys == [
        ("室内", "晴", "侧视"), ("室内", "晴", "俯视"),
        ("室外", "雪", "侧视"), ("室外", "雪", "俯视"),
    ]
    side, top = PHYSICAL["renders"][0], PHYSICAL["renders"][1]
    assert not np.array_equal(side["masks"][0], top["masks"][0])
    for render in PHYSICAL["renders"]:
        assert len(render["frames"]) == len(render["masks"]) == 3
        assert all(int(mask.sum()) > 20 for mask in render["masks"])
    assert not np.array_equal(PHYSICAL["renders"][0]["frames"][0], PHYSICAL["renders"][2]["frames"][0])


def _restyle(endpoint, **overrides):
    frame, mask = RENDER["frames"][0], RENDER["masks"][0]
    options = {"prompt": build_prompt("室内", "晴", "侧视", "warehouse"), "strength": 0.3, "steps": 8,
               "cfg_scale": 3.5, "seed": 4, "endpoint": endpoint, "timeout": 5.0}
    return restyle(frame, mask, **{**options, **overrides})


def test_failed_diffusion_makes_no_image():
    result = _restyle("http://127.0.0.1:9/sdapi/v1/txt2img", timeout=1.0)
    assert result["image"] is None
    assert result["contributor"]["called"] is True and result["contributor"]["success"] is False


def test_masked_diffusion_restyles_only_the_background(monkeypatch):
    monkeypatch.setenv("SD_API_TOKEN", "test-token")
    with _EchoServer() as server:
        result = _restyle(server.endpoint)
    assert server.headers == ["Bearer test-token"]
    request = server.requests[0]
    assert request["path"].endswith("/sdapi/v1/img2img")
    body = request["body"]
    assert (body["steps"], body["cfg_scale"], body["seed"]) == (8, 3.5, 4)
    assert body["strength"] == pytest.approx(0.3)
    assert "flame" not in body["prompt"] and "fire" not in body["prompt"]
    assert "warehouse" in body["prompt"] and "室内" not in body["prompt"]
    frame, flame = RENDER["frames"][0], RENDER["masks"][0]
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(body["mask"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    guard = flame_guard(flame)
    assert np.array_equal(mask > 127, ~guard)
    image = result["image"]
    assert np.array_equal(image[guard], frame[guard])
    far = ~cv2.dilate(guard.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert float(image[far][:, 2].mean()) > float(frame[far][:, 2].mean()) + 20
    assert 0.35 <= result["ssim"] < 1.0


def test_a_collapsed_background_is_rejected():
    with _EchoServer(mode="noise") as server:
        result = _restyle(server.endpoint)
    assert result["image"] is None
    assert result["reason"].startswith("ssim")


def test_variants_cycle_styles_of_the_samples_environment():
    source = {"sample_id": "a" * 64, "scene_params": {"scene": "室外", "weather": "雨", "viewpoint": "侧视"}}
    options = normalize_options({"variants": 3, "styles": ["dusk", "night"]})
    plans = plan_variants(source, options)
    assert [plan["style"] for plan in plans] == ["dusk", "night", "dusk"]
    assert len({plan["seed"] for plan in plans}) == 3
    assert "rainy" in plans[0]["prompt"]
    # An indoor sample never gets an outdoor background.
    indoor = plan_variants({**source, "scene_params": {"scene": "室内", "weather": "晴"}}, options)
    assert {plan["style"] for plan in indoor} <= set(STYLE_BANK["室内"])


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_each_augmentation_changes_pixels_only(name):
    frame = RENDER["frames"][0]
    changed = augment_image(frame, [name], seed=3)
    assert changed.shape == frame.shape
    assert not np.array_equal(changed, frame)


def test_bad_options_are_rejected():
    with pytest.raises(ValueError):
        normalize_options({"strength": 0.9})
    with pytest.raises(ValueError):
        normalize_options({"variants": 5})
    with pytest.raises(ValueError):
        normalize_options({"styles": ["火山"]})
    with pytest.raises(ValueError):
        augment_image(RENDER["frames"][0], ["视角变换"])
