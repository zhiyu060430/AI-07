"""Bake one fire with Blender and render every view of it."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from firephantom.images import to_uint8
from firephantom.physics.adapters import run_blender
from firephantom.physics.blender_scene import (
    DEFAULT_FRAME_COUNT,
    MAX_FRAME_COUNT,
    camera_poses,
    render_blender,
    render_key,
)
from firephantom.physics.fields import compute_fields
from firephantom.physics.scenarios import spec_to_dict
from firephantom.serialize import plain
from firephantom.synthesis.render import (
    IMAGE_SIZE,
    SCENES,
    VIEWPOINTS,
    WEATHERS,
    dress_frame,
    project_frame,
    project_mask,
    project_smoke_mask,
)
from firephantom.validation import normalize_fire_engine, physical_parameters

ENGINE_CACHE_VERSION = "engine-renders-v15"
# Emission pass values above this are flame. Below it is glow on smoke.
_EMIT_THRESHOLD = 40
_MIN_MASK_PIXELS = 6


def _blender_timeout(render_count: int, estimate: float = 0.0, bakes: int = 1) -> float:
    configured = os.environ.get("BLENDER_TIMEOUT_SECONDS", "").strip()
    if configured:
        return float(configured)
    # The default bake takes about two minutes per weather; each 256 px frame a
    # few seconds. A larger box or finer grid gets half again its estimated time.
    return max(240.0 * max(1, int(bakes)) + 6.0 * render_count, 1.5 * float(estimate) + 120.0)


def normalize_views(
    viewpoints: list[str] | tuple[str, ...] | None,
    looks: list | tuple | None,
    frame_count: object,
) -> tuple[list[str], list[tuple[str, str]], int]:
    views = list(viewpoints or ["侧视"])
    for view in views:
        if view not in VIEWPOINTS:
            raise ValueError(f"未知视角 {view!r}")
    pairs = [(str(scene), str(weather)) for scene, weather in (looks or [("室内", "晴")])]
    for scene, weather in pairs:
        if scene not in SCENES:
            raise ValueError(f"未知场景 {scene!r}")
        if weather not in WEATHERS:
            raise ValueError(f"未知天气 {weather!r}")
    if len(set(views)) != len(views) or len(set(pairs)) != len(pairs):
        raise ValueError("视角和环境天气组合不能重复")
    count = int(DEFAULT_FRAME_COUNT if frame_count in (None, "") else frame_count)
    if not 1 <= count <= MAX_FRAME_COUNT:
        raise ValueError(f"frame_count must be between 1 and {MAX_FRAME_COUNT}")
    return views, pairs, count


def _cache_key(
    fire_engine: str,
    scenario: str,
    params: dict,
    placements: list | None,
    viewpoints: list[str],
    looks: list[tuple[str, str]],
    frame_count: int,
    seed: int | None,
    rig: dict | None = None,
) -> str:
    payload = {
        "version": ENGINE_CACHE_VERSION,
        "engine": fire_engine,
        "scenario": scenario,
        "wind_speed": round(float(params["wind_speed"]), 6),
        "temperature": round(float(params["temperature"]), 6),
        "humidity": round(float(params["humidity"]), 6),
        "fuel_type": str(params["fuel_type"]),
        "placements": placements or [],
        "viewpoints": viewpoints,
        "looks": looks,
        "frame_count": frame_count,
        "seed": seed,
    }
    # Absent rig keeps the key identical to bakes from before cameras were editable.
    # A missing lights entry is the automatic fill and sun. An empty list is none.
    for key in ("origin", "cameras", "gas_domain", "simulation"):
        value = None if not rig else rig.get(key)
        if value:
            payload[key] = value
    if rig and "lights" in rig and rig["lights"] is not None:
        payload["lights"] = rig["lights"]
    # Null structures and domain keep the preset, so they stay out of the key.
    # An empty structure list is a site with no boxes.
    if rig and "structures" in rig and rig["structures"] is not None:
        payload["structures"] = rig["structures"]
    if rig and rig.get("domain"):
        payload["domain"] = rig["domain"]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _read_png(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise OSError(f"could not read {path.name}")
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image


def emission_mask(emission: np.ndarray, min_pixels: int = _MIN_MASK_PIXELS) -> np.ndarray:
    """Threshold an 8-bit pass and drop regions smaller than ``min_pixels``."""
    mask = (np.asarray(emission) >= _EMIT_THRESHOLD).astype(np.uint8)
    if int(mask.sum()) == 0:
        return mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_pixels:
            keep |= labels == label
    return keep


def ember_mask(raw: np.ndarray) -> np.ndarray:
    """Grow each speck so a few bright pixels become a box, then drop dust."""
    spots = (np.asarray(raw) >= 20).astype(np.uint8)
    if int(spots.sum()) == 0:
        return spots.astype(bool)
    grown = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return emission_mask(grown * 255, min_pixels=8)


def _mask_path(directory: Path, stem: str, kind: str | None) -> Path | None:
    folder = directory / "mask"
    if kind:
        paths = sorted(folder.glob(f"{stem}_{kind}_*.png"))
        return paths[-1] if paths else None
    paths = [
        path
        for path in sorted(folder.glob(f"{stem}_*.png"))
        if not any(token in path.name for token in ("_flame_", "_smoke_", "_ember_"))
    ]
    return paths[-1] if paths else None


def _read_blender_renders(
    directory: Path,
    viewpoints: list[str],
    looks: list[tuple[str, str]],
    frame_count: int,
    cameras: list[dict],
) -> list[dict] | None:
    renders = []
    for look_index, (scene, weather) in enumerate(looks):
        for camera_index, viewpoint in enumerate(viewpoints):
            frames, masks, smoke_masks, ember_masks = [], [], [], []
            for frame_index in range(frame_count):
                stem = render_key(camera_index, look_index, frame_index)
                rgb_path = directory / "rgb" / f"{stem}.png"
                flame_path = _mask_path(directory, stem, "flame") or _mask_path(directory, stem, None)
                if not rgb_path.is_file() or flame_path is None:
                    return None
                frames.append(to_uint8(_read_png(rgb_path)))
                masks.append(emission_mask(_read_png(flame_path, grayscale=True)))
                smoke_path = _mask_path(directory, stem, "smoke")
                ember_path = _mask_path(directory, stem, "ember")
                if smoke_path is None:
                    smoke_masks.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool))
                else:
                    smoke_masks.append(emission_mask(_read_png(smoke_path, grayscale=True)))
                if ember_path is None:
                    ember_masks.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool))
                else:
                    ember_masks.append(ember_mask(_read_png(ember_path, grayscale=True)))
            renders.append(
                _render(
                    viewpoint, scene, weather, frames, masks, cameras[camera_index],
                    smoke_masks, ember_masks,
                )
            )
    return renders


_RENDER_ARRAYS = ("frames", "masks", "smoke_masks", "ember_masks")


def _render(
    viewpoint: str,
    scene: str,
    weather: str,
    frames: list,
    masks: list,
    camera: dict | None,
    smoke_masks: list | None = None,
    ember_masks: list | None = None,
) -> dict:
    blank = [np.zeros(np.asarray(mask).shape[:2], dtype=bool) for mask in masks]
    return {
        "viewpoint": viewpoint,
        "scene": scene,
        "weather": weather,
        "camera": camera,
        "frames": frames,
        "masks": masks,
        "smoke_masks": blank if smoke_masks is None else smoke_masks,
        "ember_masks": blank if ember_masks is None else ember_masks,
    }


def _projected_renders(
    snapshots: list[tuple[np.ndarray, np.ndarray, int]],
    viewpoints: list[str],
    looks: list[tuple[str, str]],
) -> list[dict]:
    """Fallback renders: field projections dressed in 2D. ``snapshots`` are (T, smoke, roll)."""
    renders = []
    for scene, weather in looks:
        for viewpoint in viewpoints:
            frames, masks, smoke_masks = [], [], []
            for temperature, smoke, roll in snapshots:
                mask = project_mask(temperature, viewpoint, roll)
                plate = to_uint8(project_frame(temperature, smoke, viewpoint, roll))
                frames.append(dress_frame(plate, scene, weather, mask))
                masks.append(mask)
                smoke_masks.append(project_smoke_mask(smoke, viewpoint, roll))
            ember_masks = [np.zeros_like(mask, dtype=bool) for mask in masks]
            renders.append(
                _render(viewpoint, scene, weather, frames, masks, None, smoke_masks, ember_masks)
            )
    return renders


def _load_cache(directory: Path) -> dict | None:
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "fields.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        return None
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if meta.get("cache_version") != ENGINE_CACHE_VERSION:
        return None
    with np.load(arrays_path, allow_pickle=False) as data:
        result = dict(meta)
        result.pop("cache_version", None)
        result["temperature"] = np.array(data["temperature"])
        result["smoke"] = np.array(data["smoke"])
        result["flow"] = np.array(data["flow"])
        result["axes"] = {
            "x": np.array(data["axes_x"]),
            "y": np.array(data["axes_y"]),
            "z": np.array(data["axes_z"]),
        }
        frames = np.array(data["render_frames"])
        masks = np.array(data["render_masks"])
        smoke_masks = np.array(data["render_smoke_masks"]) if "render_smoke_masks" in data.files else None
        ember_masks = np.array(data["render_ember_masks"]) if "render_ember_masks" in data.files else None
    for index, render in enumerate(result["renders"]):
        render["frames"] = [frame.copy() for frame in frames[index]]
        render["masks"] = [mask.copy() for mask in masks[index]]
        if smoke_masks is None:
            render["smoke_masks"] = [np.zeros_like(mask, dtype=bool) for mask in render["masks"]]
        else:
            render["smoke_masks"] = [mask.copy() for mask in smoke_masks[index]]
        if ember_masks is None:
            render["ember_masks"] = [np.zeros_like(mask, dtype=bool) for mask in render["masks"]]
        else:
            render["ember_masks"] = [mask.copy() for mask in ember_masks[index]]
    result["engine_frames"] = result["renders"][0]["frames"]
    result["engine_cache"] = "hit"
    return result


def _save_cache(directory: Path, result: dict) -> None:
    np.savez_compressed(
        directory / "fields.npz",
        temperature=np.asarray(result["temperature"]),
        smoke=np.asarray(result["smoke"]),
        flow=np.asarray(result["flow"]),
        axes_x=np.asarray(result["axes"]["x"]),
        axes_y=np.asarray(result["axes"]["y"]),
        axes_z=np.asarray(result["axes"]["z"]),
        render_frames=np.stack([np.stack(render["frames"]) for render in result["renders"]]),
        render_masks=np.stack([np.stack(render["masks"]) for render in result["renders"]]),
        render_smoke_masks=np.stack([np.stack(render["smoke_masks"]) for render in result["renders"]]),
        render_ember_masks=np.stack([np.stack(render["ember_masks"]) for render in result["renders"]]),
    )
    skip = {"temperature", "smoke", "flow", "axes", "engine_frames", "engine_cache", "renders"}
    meta = {key: value for key, value in result.items() if key not in skip}
    meta["renders"] = [
        {key: value for key, value in render.items() if key not in _RENDER_ARRAYS}
        for render in result["renders"]
    ]
    meta["cache_version"] = ENGINE_CACHE_VERSION
    (directory / "manifest.json").write_text(json.dumps(plain(meta), ensure_ascii=False), encoding="utf-8")


def _build(
    spec,
    params: dict,
    fire_engine: str,
    work_dir: Path | None,
    placements: list | None,
    viewpoints: list[str],
    looks: list[tuple[str, str]],
    frame_count: int,
    seed: int | None,
    camera_overrides: dict | None = None,
    gas_override: dict | None = None,
    lights: list | None = None,
    simulation: dict | None = None,
) -> dict:
    from firephantom.physics.placements import prepare
    from firephantom.physics.simulation import estimate_bake_seconds, warmup_frames

    layout = prepare(spec, placements, str(params["fuel_type"]), gas_override, warmup_frames(simulation))
    fields = compute_fields(spec, **params)
    view_args = {"viewpoints": viewpoints, "looks": looks, "frame_count": frame_count, "seed": seed}
    blender_script = render_blender(
        spec, layout=layout, camera_overrides=camera_overrides, lights=lights, simulation=simulation, **params, **view_args
    )
    cameras = camera_poses(spec, viewpoints, camera_overrides)
    blender_path = None
    frames_dir = None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        blender_path = work_dir / f"{spec.chid}_scene.py"
        frames_dir = work_dir / f"{spec.chid}_renders"
        # Blender's cwd is the script directory, so the render path stays relative.
        blender_script = render_blender(
            spec,
            layout=layout,
            render_output=frames_dir.name,
            camera_overrides=camera_overrides,
            lights=lights,
            simulation=simulation,
            **params,
            **view_args,
        )
        blender_path.write_text(blender_script, encoding="utf-8")
    render_count = len(viewpoints) * len(looks) * frame_count
    estimate = estimate_bake_seconds(layout["gas_domain"], simulation, frame_count, render_count, len(looks))
    blender_run = run_blender(blender_path, timeout=_blender_timeout(render_count, estimate, len(looks)))
    tool_runs = [blender_run]
    physical_backend = "parameterized-plume"
    physical_field_source = {
        "backend": physical_backend,
        "reason": blender_run["message"],
    }
    rolled = [(fields["temperature"], fields["smoke"], index) for index in range(frame_count)]
    renders = _projected_renders(rolled, viewpoints, looks)
    engine_frames_source = "parameterized"
    if blender_run.get("success") and frames_dir is not None:
        rendered = _read_blender_renders(frames_dir, viewpoints, looks, frame_count, cameras)
        if rendered is None:
            physical_field_source["reason"] = "Blender completed without every frame and mask; using the parameterized plume"
        else:
            renders = rendered
            engine_frames_source = "blender"
            physical_field_source = {
                "backend": physical_backend,
                "reason": blender_run["message"],
                "frames": frames_dir.name,
            }
    blender_run["physical_field_source"] = physical_field_source
    if frames_dir is not None:
        blender_run["render_output"] = str(frames_dir)
    for render, camera in zip(renders, [cameras[viewpoints.index(item["viewpoint"])] for item in renders]):
        render["camera"] = camera
    return {
        "scenario": spec.name,
        "fire_engine": fire_engine,
        "renders": renders,
        "engine_frames": renders[0]["frames"],
        "engine_frames_source": engine_frames_source,
        "engine_cache": "miss",
        "viewpoints": viewpoints,
        "looks": [list(pair) for pair in looks],
        "frame_count": frame_count,
        "seed": seed,
        "parameters": fields["parameters"],
        "physical_backend": physical_backend,
        "physical_field_source": physical_field_source,
        "geometry": spec_to_dict(spec),
        "temperature": fields["temperature"],
        "smoke": fields["smoke"],
        "flow": fields["flow"],
        "axes": fields["axes"],
        "fuel": fields["fuel"],
        "blender_script": blender_script,
        "tool_runs": tool_runs,
        "blender_path": None if blender_path is None else str(blender_path),
    }


def simulate(
    scenario: str,
    wind_speed: float,
    temperature: float,
    humidity: float,
    fuel_type: str,
    work_dir: Path | None = None,
    fire_engine: str = "blender",
    placements: list | None = None,
    viewpoints: list[str] | tuple[str, ...] | None = None,
    looks: list | tuple | None = None,
    frame_count: int | None = None,
    seed: int | None = None,
    *,
    origin: list | None = None,
    cameras: dict | None = None,
    gas_domain: dict | None = None,
    lights: list | None = None,
    simulation: dict | None = None,
    structures: list | None = None,
    domain: list | None = None,
) -> dict:
    """Bake one fire with Blender and render it for every view and look.

    ``renders`` holds one entry per (environment, weather) x viewpoint, in that
    nesting order, each with ``frame_count`` frames and flame masks. When
    Blender cannot produce them, field projections stand in so the sample is
    not blank. A later call with the same inputs and ``work_dir`` reuses the
    cached renders.
    """
    from firephantom.physics.placements import (
        cameras_outside_solids,
        collapse_cameras,
        collapse_gas,
        collapse_lights,
        layout_environment,
        normalize_camera_map,
        normalize_gas_domain,
        normalize_lights,
        normalize_placements,
        prepared_spec,
        resolved_gas,
    )
    from firephantom.physics.simulation import collapse_simulation, normalize_simulation

    spec, kept_domain, kept_structures, origin_override = prepared_spec(scenario, domain, structures, origin)
    params = physical_parameters(wind_speed, temperature, humidity, fuel_type)
    selected = normalize_fire_engine(fire_engine)
    views, pairs, count = normalize_views(viewpoints, looks, frame_count)
    placed = normalize_placements(spec, placements, str(params["fuel_type"]))
    usable_cameras, _buried = cameras_outside_solids(spec, cameras)
    camera_overrides = collapse_cameras(spec, normalize_camera_map(spec, usable_cameras))
    gas_override = collapse_gas(spec, placed, normalize_gas_domain(spec, gas_domain))
    environment = layout_environment(spec, resolved_gas(spec, placed, gas_override), camera_overrides, views)
    kept_lights = collapse_lights(spec, normalize_lights(spec, lights), environment)
    kept_simulation = collapse_simulation(normalize_simulation(simulation))
    rig = {}
    if origin_override:
        rig["origin"] = origin_override
    if camera_overrides:
        rig["cameras"] = camera_overrides
    if gas_override:
        rig["gas_domain"] = gas_override
    if kept_lights is not None:
        rig["lights"] = kept_lights
    if kept_simulation:
        rig["simulation"] = kept_simulation
    if kept_structures is not None:
        rig["structures"] = kept_structures
    if kept_domain:
        rig["domain"] = kept_domain
    build_args = (spec, params, selected)
    view_args = (placements, views, pairs, count, None if seed is None else int(seed))
    build_tail = {
        "camera_overrides": camera_overrides or None,
        "gas_override": gas_override,
        "lights": kept_lights,
        "simulation": kept_simulation,
    }
    if work_dir is None:
        return _build(*build_args, None, *view_args, **build_tail)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "engine-cache" / _cache_key(
        selected, spec.name, params, placed, views, pairs, count, view_args[-1], rig or None
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (cache_dir / ".lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = _load_cache(cache_dir)
        if cached is not None:
            return cached
        result = _build(*build_args, cache_dir / "work", *view_args, **build_tail)
        _save_cache(cache_dir, result)
        return result
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
