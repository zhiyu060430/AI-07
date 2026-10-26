"""Blender bakes, one camera per viewpoint, rendered masks, and sequence jobs."""

import ast
import base64
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from firephantom.images import encode_png
from firephantom.physics.blender_scene import (
    FLAME_HEIGHT,
    ambient_physics,
    camera_pose,
    look_settings,
    precipitation_box,
    precipitation_lifetime,
    render_key,
    seed_jitter,
    visible_obstacles,
    weather_physics,
)
from firephantom.physics.engine import simulate
from firephantom.physics.scenarios import SCENARIOS, get_scenario
from firephantom.pipeline import combo_list
from firephantom.validation import expand_job_items

BASE = {
    "wind_speed": 1.2,
    "temperature": 22.0,
    "humidity": 35.0,
    "fuel_type": "木材",
}
JOB = {
    "scenario": "林区火灾",
    "wind_speed": 1.5,
    "temperature": 28.0,
    "humidity": 45.0,
    "fuel_type": "木材",
    "weathers": ["晴", "雨", "雪"],
    "viewpoints": ["侧视", "俯视", "第一人称"],
    "count": 1,
    "fire_engine": "blender",
}


def _params(text: str) -> dict:
    for line in text.splitlines():
        if "PARAMS_JSON=" in line:
            return json.loads(line.split("PARAMS_JSON=", 1)[1])
    raise AssertionError("emitted artifact has no PARAMS_JSON")


def test_only_blender_starts(monkeypatch, tmp_path):
    import firephantom.physics.engine as engine

    calls = []

    def record(path, timeout=None):
        calls.append("blender")
        return {
            "tool": "blender",
            "executed": True,
            "success": False,
            "message": "blender called",
            "outputs": {},
        }

    monkeypatch.setattr(engine, "run_blender", record)
    simulate("房间火灾", work_dir=tmp_path / "blender", fire_engine="blender", **BASE)
    assert calls == ["blender"]
    with pytest.raises(ValueError, match="fire_engine must be blender"):
        simulate("房间火灾", work_dir=tmp_path / "rejected", fire_engine="fds", **BASE)
    assert calls == ["blender"]


def _inside(point: np.ndarray, box: list[float]) -> bool:
    x, y, z = (float(value) for value in point)
    x0, x1, y0, y1, z0, z1 = box
    return x0 < x < x1 and y0 < y < y1 and z0 < z < z1


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
@pytest.mark.parametrize("viewpoint", ("侧视", "俯视", "第一人称"))
def test_camera_looks_at_the_flame_from_inside_the_domain(scenario, viewpoint):
    spec = get_scenario(scenario)
    pose = camera_pose(spec, viewpoint)
    location = np.array(pose["location"], dtype=np.float64)
    target = np.array(pose["target"], dtype=np.float64)
    length, width, height = spec.domain
    assert 0.0 < location[0] < length
    assert 0.0 < location[1] < width
    assert 0.0 < location[2] < height
    for item in visible_obstacles(spec):
        assert not _inside(location, item["xb"])
        for step in np.linspace(0.05, 0.95, 80):
            point = (1.0 - step) * location + step * target
            assert not _inside(point, item["xb"])
    distance = float(np.linalg.norm(target - location))
    flame_angle = 2.0 * math.atan((FLAME_HEIGHT / 2.0) / distance)
    fov = 2.0 * math.atan(18.0 / float(pose["focal_mm"]))
    fraction = float(flame_angle / fov)
    assert fraction == pytest.approx(pose["frame_fraction"], abs=0.02)
    assert fraction >= 0.4
    if not pose["fallback"]:
        rise = (location[2] - target[2]) / distance
        if viewpoint == "俯视":
            assert rise > 0.7
        elif viewpoint == "侧视":
            assert abs(rise) < 0.95
    params = _params(simulate(scenario, viewpoints=[viewpoint], **BASE)["blender_script"])
    assert params["cameras"][0]["location"] == pytest.approx(pose["location"])
    assert params["cameras"][0]["focal_mm"] == pytest.approx(pose["focal_mm"])

def test_room_viewpoints_are_different_cameras():
    spec = get_scenario("房间火灾")
    poses = {view: camera_pose(spec, view) for view in ("侧视", "俯视", "第一人称")}
    assert not any(pose["fallback"] for pose in poses.values())
    locations = [tuple(round(v, 3) for v in pose["location"]) for pose in poses.values()]
    assert len(set(locations)) == 3


def test_blender_script_bakes_each_weather_and_renders_every_view_with_masks():
    result = simulate(
        "房间火灾",
        viewpoints=["侧视", "俯视"],
        looks=[("室内", "晴"), ("室外", "雪")],
        frame_count=5,
        seed=11,
        **BASE,
    )
    script = result["blender_script"]
    ast.parse(script)
    params = _params(script)
    assert params["frame_count"] == 5
    assert [camera["viewpoint"] for camera in params["cameras"]] == ["侧视", "俯视"]
    assert [look["precipitation"] for look in params["looks"]] == [None, "snow"]
    snow = look_settings("室外", "雪")
    snow["ground_covered"] = False
    snow["physics"]["motion"]["lifetime"] = precipitation_lifetime(params["warmup"])
    assert params["looks"][1] == {"key": "l1", **snow}
    assert params["looks"][0]["physics"] == weather_physics("晴")
    assert "visible_volume_scatter = True" in script
    assert "scene.cycles.volume_bounces = 2" in script
    assert "use_pass_emit = True" in script
    assert "use_pass_volume_direct = True" in script
    assert "use_pass_volume_indirect = True" in script
    assert 'aov.aov_name = "ember"' in script
    assert 'hot.operation = "GREATER_THAN"' in script
    assert "embers_hot" in script and "embers_ash" in script
    assert params["ambient"] == ambient_physics(BASE["temperature"], BASE["humidity"])
    assert params["jitter"] == seed_jitter(11)
    assert seed_jitter(11) != seed_jitter(12)
    assert seed_jitter(None) == {"dx": 0.0, "dy": 0.0, "velocity_scale": 1.0}
    # One bake call site, run once per weather into its own cache directory.
    assert script.count("bake_all") == 1
    assert "_bake(bpy, gas, look[\"key\"])" in script
    assert '"fluid_cache", key' in script
    assert "effector_weights.wind" in script
    assert 'stem + "_" + name + "_"' in script
    assert "CompositorNodeOutputFile" in script
    assert 'domain_type = "GAS"' in script
    assert "ShaderNodeVolumePrincipled" in script
    assert render_key(1, 0, 3) == "v1_l0_f003"
    assert len(result["renders"]) == 4

def test_a_stored_camera_inside_the_ceiling_bakes_the_searched_pose():
    buried = {"俯视": {"location": [7.0979, 5.2121, 2.95], "target": [1.2, 1.2, 0.7675], "focal_mm": 25.99}}
    params = _params(simulate("房间火灾", viewpoints=["俯视"], cameras=buried, **BASE)["blender_script"])
    assert params["cameras"][0]["location"] == pytest.approx(camera_pose(get_scenario("房间火灾"), "俯视")["location"])
    assert params["cameras"][0]["location"][2] < 2.8


def test_custom_camera_and_gas_box_are_baked():
    pose = {"location": [4.0, 1.0, 1.2], "target": [1.2, 1.2, 1.0], "focal_mm": 50}
    gas = {"location": [2.5, 2.5, 1.2], "scale": [1.0, 1.0, 1.0], "resolution": 24}
    result = simulate("房间火灾", viewpoints=["侧视"], cameras={"侧视": pose}, gas_domain=gas, origin=[2.0, 2.0, 0.3], **BASE)
    params = _params(result["blender_script"])
    assert params["origin"] == pytest.approx([2.0, 2.0, 0.3])
    assert params["cameras"][0]["location"] == pytest.approx(pose["location"])
    assert params["cameras"][0]["focal_mm"] == pytest.approx(50)
    assert params["gas_domain"]["resolution"] == 24
    untouched = _params(simulate("房间火灾", viewpoints=["侧视"], **BASE)["blender_script"])
    assert untouched["cameras"][0]["location"] == pytest.approx(camera_pose(get_scenario("房间火灾"), "侧视")["location"])
    assert untouched["gas_domain"]["resolution"] == 40


def test_custom_simulation_reaches_the_blender_script():
    from firephantom.physics.engine import _blender_timeout, _cache_key
    from firephantom.validation import physical_parameters

    settings = {"warmup_s": 3.0, "noise": 0, "vorticity": 0.5, "samples": 64}
    params = _params(simulate("房间火灾", viewpoints=["侧视"], simulation=settings, **BASE)["blender_script"])
    assert params["warmup"] == 36
    assert params["simulation"]["noise"] == 0
    assert params["simulation"]["vorticity"] == pytest.approx(0.5)
    assert params["simulation"]["samples"] == 64
    assert params["simulation"]["burning_rate"] == pytest.approx(0.6)
    defaults = _params(simulate("房间火灾", viewpoints=["侧视"], simulation={"noise": 2}, **BASE)["blender_script"])
    assert defaults["warmup"] == 12
    assert defaults["simulation"]["samples"] == 20
    fire = physical_parameters(1.5, 20, 40, "木材")
    plain = _cache_key("blender", "房间火灾", fire, [], ["侧视"], [("室内", "晴")], 8, None)
    assert plain == _cache_key("blender", "房间火灾", fire, [], ["侧视"], [("室内", "晴")], 8, None, {"simulation": None})
    assert plain != _cache_key("blender", "房间火灾", fire, [], ["侧视"], [("室内", "晴")], 8, None, {"simulation": {"noise": 0}})
    assert _blender_timeout(16) == pytest.approx(240 + 96)
    assert _blender_timeout(16, 2000) == pytest.approx(3120)


def test_job_over_the_bake_budget_is_refused():
    room = {"location": [4.0, 3.0, 1.5], "scale": [3.9, 2.9, 1.45], "resolution": 128}
    heavy = {**JOB, "scenario": "房间火灾", "weathers": ["晴"], "gas_domain": room, "simulation": {"noise": 3, "warmup_s": 10}}
    with pytest.raises(ValueError, match="超过上限"):
        expand_job_items([heavy])


def test_each_weather_is_a_bake_in_the_estimate_and_timeout():
    from firephantom.physics.engine import _blender_timeout
    from firephantom.physics import simulation
    from firephantom.physics.simulation import estimate_bake_seconds

    gas = {"location": [4.0, 3.0, 1.5], "scale": [1.0, 1.0, 1.5], "resolution": 64}
    one = estimate_bake_seconds(gas, None, 16, 16, bakes=1)
    three = estimate_bake_seconds(gas, None, 16, 16, bakes=3)
    render = one - estimate_bake_seconds(gas, None, 16, 0, bakes=1)
    solve = one - render - simulation._BASE_S
    assert three > one
    # Only the solve repeats; renders and the fixed start-up are paid once.
    assert three - one == pytest.approx(2 * solve, rel=1e-6)
    assert _blender_timeout(16, 0.0, 3) == pytest.approx(3 * 240 + 96)


def test_weather_changes_the_flame_and_ambient_air_changes_buoyancy():
    clear, rain, snow = (weather_physics(name) for name in ("晴", "雨", "雪"))
    assert clear["temperature_scale"] == clear["fuel_scale"] == 1.0 and clear["motion"] is None
    assert rain["temperature_scale"] < snow["temperature_scale"] < 1.0
    assert rain["fuel_scale"] < 1.0 and rain["smoke_scale"] > snow["smoke_scale"] > 1.0
    assert min(rain["smoke_color"]) > min(snow["smoke_color"]) > min(clear["smoke_color"]) >= 0.4
    assert rain["downdraft"] > snow["downdraft"] > 0.0
    assert (rain["surface"], snow["surface"]) == ("wet", "snow")
    # Snow drifts slower than rain falls, and the wind moves it at least as much.
    assert snow["motion"]["speed"] < rain["motion"]["speed"]
    assert snow["motion"]["wind"] >= rain["motion"]["wind"]
    cold, hot = ambient_physics(-20.0, 40.0), ambient_physics(40.0, 40.0)
    assert cold["beta"] > hot["beta"]
    dry, humid = ambient_physics(20.0, 10.0), ambient_physics(20.0, 90.0)
    assert humid["temperature_scale"] < dry["temperature_scale"]
    assert humid["smoke_scale"] < dry["smoke_scale"]
    with pytest.raises(ValueError):
        weather_physics("冰雹")


def test_precipitation_falls_over_the_fire_and_every_camera():
    gas = {"location": [20.0, 20.0, 2.0], "scale": [1.5, 1.5, 2.0]}
    cameras = [{"location": [12.0, 20.0, 1.6]}, {"location": [20.0, 26.0, 9.0]}]
    box = precipitation_box((40.0, 40.0, 20.0), gas, cameras)
    x0, x1 = box["center"][0] - box["size"][0] / 2, box["center"][0] + box["size"][0] / 2
    y0, y1 = box["center"][1] - box["size"][1] / 2, box["center"][1] + box["size"][1] / 2
    assert x0 < 12.0 and x1 > 21.5 and y0 < 18.5 and y1 > 26.0
    assert box["center"][2] > 9.0
    # Far less than the whole 40 m forest, so the drops stay dense in view.
    assert box["size"][0] * box["size"][1] < 0.25 * 40.0 * 40.0
    assert precipitation_lifetime(12) == 12


def test_same_fire_reuses_cached_engine_frames(monkeypatch, tmp_path):
    import firephantom.physics.engine as engine

    calls = []
    monkeypatch.setattr(
        engine,
        "run_blender",
        lambda *_args, **_kwargs: calls.append("blender") or {
            "tool": "blender",
            "executed": False,
            "success": False,
            "message": "blender executable was not run: not found on PATH",
            "outputs": {},
        },
    )
    first = simulate("房间火灾", work_dir=tmp_path, fire_engine="blender", **BASE)
    second = simulate("房间火灾", work_dir=tmp_path, fire_engine="blender", **BASE)
    assert calls == ["blender"]
    assert first["engine_cache"] == "miss"
    assert second["engine_cache"] == "hit"
    assert all(np.array_equal(left, right) for left, right in zip(first["engine_frames"], second["engine_frames"]))


def test_blender_frames_and_masks_come_from_rendered_pngs(monkeypatch, tmp_path):
    import firephantom.physics.engine as engine

    calls = []

    def fake_blender(script, timeout=60.0):
        calls.append(timeout)
        directory = script.parent / "room_fire_renders"
        (directory / "rgb").mkdir(parents=True, exist_ok=True)
        (directory / "mask").mkdir(parents=True, exist_ok=True)
        for camera in range(2):
            for frame in range(3):
                stem = render_key(camera, 0, frame)
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                image[..., 0] = 30 + frame * 25 + camera * 100
                cv2.imwrite(str(directory / "rgb" / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                emission = np.zeros((256, 256), dtype=np.uint8)
                emission[20:90, 40 + frame:80 + frame] = 200
                emission[200, 200] = 255  # a lone speck is not flame
                cv2.imwrite(str(directory / "mask" / f"{stem}_{12 + frame:04d}.png"), emission)
        return {"tool": "blender", "executed": True, "success": True, "returncode": 0, "message": "blender executable completed successfully", "outputs": {}}

    monkeypatch.setattr(engine, "run_blender", fake_blender)
    result = simulate("房间火灾", work_dir=tmp_path, fire_engine="blender", viewpoints=["侧视", "俯视"], frame_count=3, **BASE)
    assert len(calls) == 1 and calls[0] > 240
    assert result["engine_frames_source"] == "blender"
    assert result["physical_backend"] == "parameterized-plume"
    side, top = result["renders"]
    assert (side["viewpoint"], top["viewpoint"]) == ("侧视", "俯视")
    assert int(top["frames"][0][0, 0, 0]) == 130
    assert int(side["masks"][2].sum()) == 70 * 40
    assert side["masks"][1][50, 41] and not side["masks"][1][50, 40]
    assert not side["masks"][0][200, 200]
    cached = simulate("房间火灾", work_dir=tmp_path, fire_engine="blender", viewpoints=["侧视", "俯视"], frame_count=3, **BASE)
    assert cached["engine_cache"] == "hit"
    assert np.array_equal(cached["renders"][1]["masks"][2], top["masks"][2])


def test_all_choices_render_from_one_sequence():
    items = expand_job_items([JOB])
    assert len(items) == 1
    combos = combo_list(items[0])
    assert len(combos) == 9
    assert {(combo["scene"], combo["weather"], combo["viewpoint"]) for combo in combos} == {
        ("室外", weather, viewpoint)
        for weather in JOB["weathers"]
        for viewpoint in JOB["viewpoints"]
    }
    assert items[0]["fire_engine"] == "blender"
    single = expand_job_items([{
        "scenario": "房间火灾",
        "wind_speed": 1.5,
        "temperature": 28.0,
        "humidity": 45.0,
        "fuel_type": "木材",
        "scene": "室内",
        "weather": "晴",
        "viewpoint": "侧视",
    }])
    assert len(single) == 1
    assert single[0]["fire_engine"] == "blender"
    assert (single[0]["scenes"], single[0]["weathers"], single[0]["viewpoints"]) == (["室内"], ["晴"], ["侧视"])

def test_img2img_route_reuses_the_loaded_pipeline(monkeypatch):
    import firephantom.synthesis.diffusion_server as server

    source = type("Pipe", (), {})()
    source.vae = "shared-vae"
    source.unet = "shared-unet"
    source.text_encoder = "shared-text"
    source.tokenizer = "shared-tokenizer"
    source.scheduler = "shared-scheduler"
    source.feature_extractor = None
    source.image_encoder = None
    seen = {}

    class Fake:
        def __call__(self, **kwargs):
            seen["call"] = kwargs
            return type("Result", (), {"images": [Image.new("RGB", (16, 16), (240, 80, 10))]})()

        def set_progress_bar_config(self, **_kwargs):
            return None

    def build(pipe):
        seen["vae"] = pipe.vae
        seen["unet"] = pipe.unet
        return Fake()

    monkeypatch.setattr(server, "_load_pipeline", lambda: source)
    monkeypatch.setattr(server, "_build_img2img", build)
    monkeypatch.setattr(server, "_img2img", None)
    png = encode_png(np.full((16, 16, 3), (240, 80, 10), dtype=np.uint8))
    with TestClient(server.app) as client:
        response = client.post(
            "/sdapi/v1/img2img",
            json={
                "prompt": "inside a room, clear weather, side view",
                "init_images": [base64.b64encode(png).decode("ascii")],
                "strength": 0.4,
                "steps": 20,
            },
        )
    assert response.status_code == 200
    assert seen["vae"] == "shared-vae"
    assert seen["unet"] == "shared-unet"
    assert seen["call"]["strength"] == pytest.approx(0.4)
    assert {route.path for route in server.app.routes} >= {"/sdapi/v1/txt2img", "/sdapi/v1/img2img"}


def _runtime_tool(env_name: str, fallback: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured and Path(configured).is_file():
        return configured
    if Path(fallback).is_file():
        return str(Path(fallback))
    return shutil.which(env_name.lower())


def test_room_blender_frame_contains_a_flame(tmp_path, monkeypatch):
    from firephantom.paths import find_blender

    blender = _runtime_tool("BLENDER_BIN", "") or find_blender()
    if blender is None:
        pytest.skip("blender is not installed")
    monkeypatch.setenv("BLENDER_BIN", blender)
    result = simulate("房间火灾", work_dir=tmp_path, fire_engine="blender", frame_count=2, **BASE)
    assert result["engine_frames_source"] == "blender"
    frame = result["renders"][0]
    # Emit with the sparks removed is a tighter column than the old glow pass.
    # This frame is the moment the burner reaches full heat, so the plume of
    # combustion smoke has not filled the view yet.
    assert int(frame["masks"][0].sum()) >= 1000
    assert int(frame["smoke_masks"][0].sum()) < 400
    assert int(frame["ember_masks"][0].sum()) >= 20
