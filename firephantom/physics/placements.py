"""Fuel packages placed in a scenario.

A package is an axis-aligned box plus a fuel name. The mesh the camera sees
can be a preset or an upload. Mantaflow burns the box, not the triangles.
Coordinates in this module are scene meters, Z up, matching Blender.
Three.js is Y up; convert only through the helpers below.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections import Counter
from pathlib import Path

from firephantom.physics.blender_scene import WARMUP_FRAMES, camera_pose, point_in_solid
from firephantom.physics.fields import fuel_properties
from firephantom.physics.scenarios import Obstacle, ScenarioSpec, get_scenario
from firephantom.physics.simulation import (
    GAS_RESOLUTION_RANGE,
    MAX_BAKE_SECONDS,
    NOISE_SCALES,
    SIMULATION_DEFAULTS,
    SIMULATION_RANGES,
)

# Same gap the camera shaft cuts out of inert obstacles.
SHAFT_HALF = 1.0
MAX_PACKAGES = 8
MAX_LIGHTS = 8
MAX_STRUCTURES = 24
MIN_STRUCTURE_M = 0.05
_DOMAIN_MIN_M = 1.0
_DOMAIN_MAX_M = 80.0
_STRUCTURE_ROLES = ("inert", "fuel", "ignition")
# Display names for the preset boxes. The editor uses the same words.
STRUCTURE_LABELS = {
    "wall_west": "西墙",
    "wall_east_south": "东墙南段",
    "wall_east_north": "东墙北段",
    "wall_east_lintel": "东墙门梁",
    "wall_south": "南墙",
    "wall_north": "北墙",
    "ceiling": "天花板",
    "fuel_bed": "燃料床",
    "canopy": "树冠",
    "tank": "罐体",
    "pipe_rack": "管架",
    "pool": "油池",
}
MIN_CELL_M = 0.45
HRRPUA_CAP = 1500.0
# The automatic gas box grows toward nearby fuel up to this edge. A custom box
# may fill the scene; the bake budget in physics/simulation.py limits it.
AUTO_GAS_CAP_M = 4.0
MAX_GLB_BYTES = 8 * 1024 * 1024
MAX_TRIS = 50_000
MAX_BATCH_FILES = 200
MAX_ZIP_BYTES = 80 * 1024 * 1024
FUELS = ("木材", "汽油", "塑料", "植被", "化学品")

# size is meters (x, y, z). Center is the default scene position.
# The nature pack and the extra props are merged from library_manifest.json.
PRESETS: dict[str, dict] = {
    "painted_wooden_sofa": {"label": "沙发", "size": (2.2, 0.9, 0.8), "fuel_type": "木材", "group": "家具", "source": "Poly Haven", "license": "CC0"},
    "WoodenTable_01": {"label": "木桌", "size": (1.2, 0.7, 0.75), "fuel_type": "木材", "group": "家具", "source": "Poly Haven", "license": "CC0"},
    "potted_plant_04": {"label": "盆栽", "size": (0.5, 0.5, 0.5), "fuel_type": "植被", "group": "家具", "source": "Poly Haven", "license": "CC0"},
    "boulder_01": {
        "label": "石块",
        "size": (0.8, 0.8, 0.6),
        "fuel_type": "植被",
        "group": "石头",
        "source": "Poly Haven",
        "license": "CC0",
        "note": "石头按燃料盒燃烧，燃料可以改。",
    },
    "wooden_crate_01": {"label": "木箱", "size": (0.8, 0.8, 0.8), "fuel_type": "木材", "group": "容器", "source": "Poly Haven", "license": "CC0"},
}


def _merge_library_manifest() -> None:
    """Add the committed model library. Files stay under the assets root."""
    path = Path(__file__).with_name("library_manifest.json")
    if not path.is_file():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    for entry in document.get("models") or []:
        asset_id = str(entry.get("id") or "").strip()
        if not asset_id or asset_id in PRESETS:
            continue
        size = entry.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            continue
        PRESETS[asset_id] = {
            "label": str(entry.get("label") or asset_id),
            "size": tuple(float(value) for value in size),
            "fuel_type": str(entry.get("fuel_type") or "木材"),
            "group": str(entry.get("group") or "其他"),
            "source": str(entry.get("source") or ""),
            "license": str(entry.get("license") or "CC0"),
            "note": str(entry.get("note") or ""),
            "triangles": entry.get("triangles"),
        }


_merge_library_manifest()

_TEXTURES = {
    "房间火灾": {"obstacle": "beige_wall_001", "floor": "wood_floor"},
    "林区火灾": {"floor": "forest_floor"},
    "工业区火灾": {"floor": "concrete_floor_02", "tank": "metal_plate"},
}

# Defaults sit outside the 2 m flame shaft and inside the scenario mesh.
_DEFAULTS: dict[str, tuple[dict, ...]] = {
    "房间火灾": (
        {"asset_id": "potted_plant_04", "x": 2.55, "y": 1.2, "z": 0.25, "yaw": 0.0, "fuel_type": "植被", "ignition": False},
        {"asset_id": "painted_wooden_sofa", "x": 3.7, "y": 2.9, "z": 0.4, "yaw": 0.0, "fuel_type": "木材", "ignition": False},
        {"asset_id": "WoodenTable_01", "x": 3.3, "y": 4.2, "z": 0.375, "yaw": 0.0, "fuel_type": "木材", "ignition": False},
    ),
    "林区火灾": (
        {"asset_id": "PineTree_1", "x": 23.2, "y": 20.6, "z": 1.35, "yaw": 0.4, "fuel_type": "植被", "ignition": False},
        {"asset_id": "PineTree_2", "x": 22.6, "y": 23.8, "z": 1.7824, "yaw": -0.5, "fuel_type": "植被", "ignition": False},
        {"asset_id": "Bush_1", "x": 24.8, "y": 18.2, "z": 0.6209, "yaw": 0.2, "fuel_type": "植被", "ignition": False},
        {"asset_id": "Bush_2", "x": 21.4, "y": 17.0, "z": 0.5553, "yaw": 0.6, "fuel_type": "植被", "ignition": False},
        {"asset_id": "Grass", "x": 25.4, "y": 21.6, "z": 0.6964, "yaw": 0.3, "fuel_type": "植被", "ignition": False},
        {"asset_id": "Grass_2", "x": 23.0, "y": 16.2, "z": 0.8548, "yaw": -0.2, "fuel_type": "植被", "ignition": False},
        {"asset_id": "TreeStump", "x": 20.6, "y": 24.4, "z": 0.3108, "yaw": 0.4, "fuel_type": "木材", "ignition": False},
        {"asset_id": "WoodLog", "x": 25.6, "y": 24.2, "z": 0.3742, "yaw": 0.2, "fuel_type": "木材", "ignition": False},
    ),
    "工业区火灾": (
        {"asset_id": "Barrel_01", "x": 15.8, "y": 8.8, "z": 0.44, "yaw": 0.2, "fuel_type": "汽油", "ignition": False},
        {"asset_id": "Barrel_02", "x": 16.9, "y": 8.8, "z": 0.44, "yaw": -0.4, "fuel_type": "塑料", "ignition": False},
        {"asset_id": "barrel_03", "x": 18.05, "y": 8.8, "z": 0.4653, "yaw": 0.15, "fuel_type": "汽油", "ignition": False},
        {"asset_id": "wooden_crate_01", "x": 19.35, "y": 8.8, "z": 0.4, "yaw": 0.5, "fuel_type": "木材", "ignition": False},
        {"asset_id": "Barrel_01", "x": 15.8, "y": 11.5, "z": 0.44, "yaw": 1.1, "fuel_type": "汽油", "ignition": False},
        {"asset_id": "wooden_crate_02", "x": 17.2, "y": 11.5, "z": 0.232, "yaw": 0.1, "fuel_type": "木材", "ignition": False},
        {"asset_id": "small_lpg_tank", "x": 18.4, "y": 11.5, "z": 0.353, "yaw": 0.0, "fuel_type": "化学品", "ignition": False},
        {"asset_id": "plastic_container", "x": 19.7, "y": 11.5, "z": 0.225, "yaw": -0.3, "fuel_type": "塑料", "ignition": False},
    ),
}


def assets_root() -> Path:
    configured = os.environ.get("FIREPHANTOM_ASSETS")
    if configured:
        return Path(configured)
    runtime = Path("/Volumes/T7/FirePhantom-runtime/assets")
    if runtime.parent.is_dir():
        return runtime
    return Path(__file__).resolve().parents[2] / "var" / "assets"


def scene_to_three(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Scene Z-up to Three.js Y-up. ``(x, y, z) -> (x, z, -y)``."""
    return (float(x), float(z), -float(y))


def three_to_scene(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Inverse of :func:`scene_to_three`."""
    return (float(x), -float(z), float(y))


def yaw_to_three(yaw: float) -> float:
    """Scene yaw about +Z equals Three.js ``rotation.y`` for a +X front.

    A scene point ``(cos yaw, sin yaw, 0)`` becomes Three ``(cos yaw, 0, -sin yaw)``,
    which is where ``rotation.y = yaw`` sends local +X.
    """
    return float(yaw)


def three_to_yaw(rotation_y: float) -> float:
    return float(rotation_y)


def preset_path(asset_id: str) -> Path | None:
    library = assets_root() / "library" / f"{asset_id}.glb"
    if library.is_file():
        return library
    directory = assets_root() / "presets" / asset_id
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("*.gltf")) + sorted(directory.glob("*.glb"))
    return matches[0] if matches else None


def _asset_url(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        relative = path.relative_to(assets_root()).as_posix()
    except ValueError:
        return ""
    return f"/assets/{relative}"


def texture_path(name: str) -> Path | None:
    for suffix in (".jpg", ".png"):
        candidate = assets_root() / "textures" / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def upload_path(asset_id: str) -> Path:
    token = asset_id.split(":", 1)[1]
    return assets_root() / "uploads" / f"{token}.glb"


def default_placements(scenario: str) -> list[dict]:
    items = _DEFAULTS.get(scenario, ())
    labels = [PRESETS[item["asset_id"]]["label"] for item in items]
    totals = Counter(labels)
    seen: dict[str, int] = {}
    rows = []
    for item, label in zip(items, labels):
        seen[label] = seen.get(label, 0) + 1
        # The library label already names the variant. Number a name only when
        # the same model is placed more than once, so two oil drums stay distinct.
        name = label if totals[label] == 1 else f"{label} {seen[label]}"
        rows.append({**item, "name": name, "size": list(PRESETS[item["asset_id"]]["size"])})
    return rows


def catalog_extras() -> dict:
    assets = []
    for asset_id, preset in PRESETS.items():
        path = preset_path(asset_id)
        row = {
            "asset_id": asset_id,
            "label": preset["label"],
            "size": list(preset["size"]),
            "fuel_type": preset["fuel_type"],
            "group": preset.get("group") or "其他",
            "source": preset.get("source") or "",
            "license": preset.get("license") or "CC0",
            "note": preset.get("note") or "",
            "builtin": True,
            "url": _asset_url(path),
        }
        if preset.get("triangles"):
            row["triangles"] = int(preset["triangles"])
        assets.append(row)
    shells = {}
    for name in _DEFAULTS:
        spec = get_scenario(name)
        shell = {
            "domain": list(spec.domain),
            "origin": list(spec.origin),
            "obstacles": [
                {
                    "name": item.name,
                    "xb": list(item.xb),
                    "kind": item.role,
                    "center": [(item.xb[0] + item.xb[1]) / 2.0, (item.xb[2] + item.xb[3]) / 2.0, (item.xb[4] + item.xb[5]) / 2.0],
                    "size": [item.xb[1] - item.xb[0], item.xb[3] - item.xb[2], item.xb[5] - item.xb[4]],
                }
                for item in spec.obstacles
            ],
        }
        cameras = {}
        for viewpoint in ("侧视", "俯视", "第一人称"):
            try:
                pose = camera_pose(spec, viewpoint)
            except RuntimeError:
                continue
            cameras[viewpoint] = {
                "location": [float(value) for value in pose["location"]],
                "target": [float(value) for value in pose["target"]],
                "focal_mm": float(pose["focal_mm"]),
                "fallback": bool(pose["fallback"]),
            }
        if cameras:
            shell["cameras"] = cameras
            shell["camera"] = cameras.get("侧视") or next(iter(cameras.values()))
        shells[name] = shell
    return {
        "fuels": list(FUELS),
        "assets": assets,
        "defaults": {name: default_placements(name) for name in _DEFAULTS},
        "shells": shells,
        "max_placements": MAX_PACKAGES,
        "max_lights": MAX_LIGHTS,
        "max_structures": MAX_STRUCTURES,
        "domain_cap_m": AUTO_GAS_CAP_M,
        "gas_limits": {
            "auto_cap_m": AUTO_GAS_CAP_M,
            "min_edge_m": _GAS_MIN_M,
            "resolution": list(GAS_RESOLUTION_RANGE),
            "noise": list(NOISE_SCALES),
            "ranges": {key: list(value) for key, value in SIMULATION_RANGES.items()},
            "defaults": dict(SIMULATION_DEFAULTS),
            "max_bake_seconds": MAX_BAKE_SECONDS,
        },
        "shaft_half_m": SHAFT_HALF,
    }


def _center_size_yaw(item: dict, preset: dict | None) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    if preset is not None:
        size = tuple(float(value) for value in preset["size"])
    else:
        raw = item.get("size")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("上传的模型要带尺寸")
        size = tuple(float(value) for value in raw)
    if any(value <= 0 or not math.isfinite(value) for value in size):
        raise ValueError("燃料包的尺寸必须是正数")
    if min(size) < MIN_CELL_M:
        raise ValueError("燃料包小于一格，细网格点不着它。请换一件更大的，或不要用这么细的模型。")
    center = _center(item)
    if any(not math.isfinite(value) for value in center):
        raise ValueError("燃料包的位置必须是有限数值")
    yaw = float(item.get("yaw") or 0.0)
    if not math.isfinite(yaw):
        raise ValueError("燃料包的朝向必须是有限数值")
    return center, size, yaw


def _center(item: dict) -> tuple[float, float, float]:
    """Client rows use x/y/z. A normalized package already has center."""
    if "x" not in item and "center" in item:
        raw = item.get("center")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("燃料包的位置必须是有限数值")
        return tuple(float(value) for value in raw)
    try:
        return (float(item["x"]), float(item["y"]), float(item["z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("燃料包的位置必须是有限数值") from exc


def _aabb(center: tuple[float, float, float], size: tuple[float, float, float], yaw: float) -> tuple[float, float, float, float, float, float]:
    hx, hy, hz = (value / 2.0 for value in size)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    xs: list[float] = []
    ys: list[float] = []
    for x in (-hx, hx):
        for y in (-hy, hy):
            xs.append(center[0] + cosine * x - sine * y)
            ys.append(center[1] + sine * x + cosine * y)
    return (min(xs), max(xs), min(ys), max(ys), center[2] - hz, center[2] + hz)


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[1] - 1e-6 and b[0] < a[1] - 1e-6 and a[2] < b[3] - 1e-6 and b[2] < a[3] - 1e-6 and a[4] < b[5] - 1e-6 and b[4] < a[5] - 1e-6


def _inside(xb: tuple[float, ...], domain: tuple[float, float, float]) -> bool:
    length, width, height = domain
    return xb[0] >= -1e-4 and xb[1] <= length + 1e-4 and xb[2] >= -1e-4 and xb[3] <= width + 1e-4 and xb[4] >= -1e-4 and xb[5] <= height + 1e-4


def _hrrpua(fuel_type: str, xb: tuple[float, ...]) -> float:
    area = max((xb[1] - xb[0]) * (xb[3] - xb[2]), 1e-3)
    return min(fuel_properties(fuel_type)["hrr_kw"] / area, HRRPUA_CAP)


def normalize_placements(spec: ScenarioSpec, raw: object, default_fuel: str) -> list[dict]:
    """Validate a client placement list. ``None`` means no extra packages."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("placements must be a list")
    if len(raw) > MAX_PACKAGES:
        raise ValueError(f"最多摆 {MAX_PACKAGES} 件")
    shaft = (
        spec.origin[0] - SHAFT_HALF,
        spec.origin[0] + SHAFT_HALF,
        spec.origin[1] - SHAFT_HALF,
        spec.origin[1] + SHAFT_HALF,
        0.0,
        spec.domain[2],
    )
    packages = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("每一件摆放都要是对象")
        asset_id = str(item.get("asset_id") or "").strip()
        preset = PRESETS.get(asset_id)
        upload = asset_id.startswith("upload:") and len(asset_id.split(":", 1)[1]) >= 8
        if preset is None and not upload:
            raise ValueError(f"未知资产 {asset_id!r}")
        if upload and not upload_path(asset_id).is_file():
            raise ValueError("上传的模型已经不在了，请重新上传")
        fuel_type = str(item.get("fuel_type") or default_fuel).strip() or default_fuel
        if len(fuel_type) > 128:
            raise ValueError("fuel_type must be at most 128 characters")
        center, size, yaw = _center_size_yaw(item, preset)
        xb = _aabb(center, size, yaw)
        if not _inside(xb, spec.domain):
            raise ValueError(f"{asset_id} 超出了场景范围")
        if _overlaps(xb, shaft):
            raise ValueError(f"{asset_id} 挡住了起火点上方，火柱需要留空")
        path = upload_path(asset_id) if upload else preset_path(asset_id)
        label = preset["label"] if preset else "上传"
        packages.append(
            {
                "name": f"pkg{index}",
                "label": str(item.get("label") or item.get("name") or f"{label}{index + 1}"),
                "asset_id": asset_id,
                "center": [float(value) for value in center],
                "size": [float(value) for value in size],
                "yaw": yaw,
                "xb": [float(value) for value in xb],
                "fuel_type": fuel_type,
                "ignition": bool(item.get("ignition")),
                "hrr_kw": fuel_properties(fuel_type)["hrr_kw"],
                "hrrpua": _hrrpua(fuel_type, xb),
                "asset_path": "" if path is None else str(path),
            }
        )
    return packages


def _cap_around(lo: float, hi: float, keep_lo: float, keep_hi: float, limit: float) -> tuple[float, float]:
    """Shrink ``[lo, hi]`` to ``limit`` without letting go of ``[keep_lo, keep_hi]``."""
    if hi - lo <= limit:
        return lo, hi
    span = keep_hi - keep_lo
    if span >= limit:
        return keep_lo, keep_lo + limit
    extra = limit - span
    left_need = max(0.0, keep_lo - lo)
    right_need = max(0.0, hi - keep_hi)
    total = left_need + right_need
    if total <= 1e-9:
        return keep_lo, keep_hi
    left = extra * (left_need / total)
    return keep_lo - left, keep_hi + (extra - left)


def gas_domain(spec: ScenarioSpec, packages: list[dict]) -> dict:
    """Box the primary fire, then grow it toward nearby packages, capped at 4 m."""
    ox, oy, oz = spec.origin
    height = 2.25 * (0.92 + 0.08 * float(spec.plume_scale) / 0.85)
    fire = (ox + 0.35 - 1.2, ox + 0.35 + 1.2, oy - 1.2, oy + 1.2, oz, oz + height)
    if not packages:
        return {
            "location": [ox + 0.35, oy, oz + height / 2.0],
            "scale": [1.2, 1.2, height / 2.0],
            "resolution": 40,
        }
    ignitions = [(ox, oy)]
    ignitions.extend((item["center"][0], item["center"][1]) for item in packages if item["ignition"])
    x0, x1, y0, y1, z0, z1 = fire
    for item in packages:
        center = item["center"]
        nearest = min(math.hypot(center[0] - gx, center[1] - gy) for gx, gy in ignitions)
        if nearest > AUTO_GAS_CAP_M and not item["ignition"]:
            continue
        xb = item["xb"]
        x0, x1 = min(x0, xb[0] - 0.15), max(x1, xb[1] + 0.15)
        y0, y1 = min(y0, xb[2] - 0.15), max(y1, xb[3] + 0.15)
        z0, z1 = min(z0, xb[4]), max(z1, max(xb[5] + 0.2, oz + height))
    x0, x1 = _cap_around(x0, x1, fire[0], fire[1], AUTO_GAS_CAP_M)
    y0, y1 = _cap_around(y0, y1, fire[2], fire[3], AUTO_GAS_CAP_M)
    z0, z1 = _cap_around(z0, z1, fire[4], fire[5], AUTO_GAS_CAP_M)
    return {
        "location": [(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0],
        "scale": [(x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0],
        "resolution": 32,
        "bounds": [x0, x1, y0, y1, z0, z1],
    }


def _domain_bounds(domain: dict) -> tuple[float, float, float, float, float, float]:
    bounds = domain.get("bounds")
    if bounds is not None:
        return tuple(bounds)
    location, scale = domain["location"], domain["scale"]
    return (
        location[0] - scale[0],
        location[0] + scale[0],
        location[1] - scale[1],
        location[1] + scale[1],
        location[2] - scale[2],
        location[2] + scale[2],
    )


def _delay(distance: float, warmup: int = WARMUP_FRAMES) -> int:
    # Output PNGs start at the warmup frame. Ignite in the first six after it,
    # so a longer warmup lets the first fire grow without moving the spread.
    return int(min(warmup + 6, max(warmup + 1, round(warmup + distance / 0.45))))


def ignition_fronts(
    package: dict,
    ignitions: list[tuple[float, float]],
    *,
    warmup: int = WARMUP_FRAMES,
) -> list[dict]:
    """Slabs of one fuel box, ordered from the side nearest an ignition.

    An ignition package's first slab reaches full heat on the first rendered
    frame (the warmup frame). Later slabs, and every slab of a package that
    waits for the fire, step three frames apart. Nothing burns during warmup.
    A span of one cell or less stays one slab.
    """
    xb = tuple(float(value) for value in package["xb"])
    center = package["center"]
    gx, gy = min(ignitions, key=lambda point: math.hypot(center[0] - point[0], center[1] - point[1]))
    dx, dy = center[0] - gx, center[1] - gy
    along_x = abs(dx) >= abs(dy)
    lo, hi = (xb[0], xb[1]) if along_x else (xb[2], xb[3])
    extent = hi - lo
    near_is_lo = (dx >= 0) if along_x else (dy >= 0)
    count = 1 if extent <= _SLICE_MIN_M + 1e-9 else min(_MAX_SLICES, max(2, round(extent / _SLICE_PITCH_M)))
    width = extent / count
    fronts = []
    for index in range(count):
        if near_is_lo:
            start = lo + index * width
            end = lo + (index + 1) * width
        else:
            end = hi - index * width
            start = hi - (index + 1) * width
        slab = (start, end, xb[2], xb[3], xb[4], xb[5]) if along_x else (xb[0], xb[1], start, end, xb[4], xb[5])
        if package.get("ignition"):
            full = int(warmup) + index * _SLICE_FRAME_STEP
        else:
            full = int(package["delay"]) + index * _SLICE_FRAME_STEP
        fronts.append(
            {
                "xb": [float(value) for value in slab],
                "center": [(slab[0] + slab[1]) / 2.0, (slab[2] + slab[3]) / 2.0, (slab[4] + slab[5]) / 2.0],
                "scale": [
                    max((slab[1] - slab[0]) / 2.0, 1e-3),
                    max((slab[3] - slab[2]) / 2.0, 1e-3),
                    max((slab[5] - slab[4]) / 2.0, 1e-3),
                ],
                "full_frame": int(full),
            }
        )
    return fronts


def annotate_packages(spec: ScenarioSpec, packages: list[dict], domain: dict, warmup: int = WARMUP_FRAMES) -> list[dict]:
    ox, oy, _oz = spec.origin
    ignitions = [(ox, oy)]
    ignitions.extend((item["center"][0], item["center"][1]) for item in packages if item["ignition"])
    annotated = []
    for item in packages:
        center = item["center"]
        distance = min(math.hypot(center[0] - gx, center[1] - gy) for gx, gy in ignitions)
        covered = _overlaps(tuple(item["xb"]), _domain_bounds(domain))
        noted = {**item, "in_domain": covered, "delay": 1 if item["ignition"] else _delay(distance, warmup)}
        noted["fronts"] = ignition_fronts(noted, ignitions, warmup=warmup) if covered else []
        annotated.append(noted)
    return annotated


def surface_textures(scenario: str) -> dict[str, str]:
    found = {}
    for role, name in _TEXTURES.get(scenario, {}).items():
        path = texture_path(name)
        found[role] = "" if path is None else str(path)
    return found


_VIEWS = ("侧视", "俯视", "第一人称")
_FOCAL_MIN = 12.0
_FOCAL_MAX = 120.0
_GAS_MIN_M = 0.5
_ORIGIN_MARGIN = 0.2
_LIGHT_TYPES = ("POINT", "SUN", "SPOT", "AREA")
_LIGHT_MARGIN = 0.05
# One cell or shorter stays a single inflow. Longer boxes are cut from the
# near side, about one slice per 0.35 m, and at most four.
_SLICE_MIN_M = 0.45
_SLICE_PITCH_M = 0.35
_MAX_SLICES = 4
_SLICE_FRAME_STEP = 3


def _vec3(raw: object, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label}必须是三个数")
    try:
        point = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是三个数") from exc
    if any(not math.isfinite(value) for value in point):
        raise ValueError(f"{label}必须是有限数值")
    return point


def _round_vec(point: tuple[float, float, float]) -> list[float]:
    return [round(float(value), 4) for value in point]


def _near(left: list | tuple, right: list | tuple, tol: float = 1e-3) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= tol for a, b in zip(left, right))


def apply_origin(spec: ScenarioSpec, origin: list | tuple | None) -> ScenarioSpec:
    """Scenario with a moved fire. ``None`` keeps the shell origin."""
    if origin is None:
        return spec
    from dataclasses import replace

    return replace(spec, origin=tuple(float(value) for value in origin))


def normalize_origin(spec: ScenarioSpec, raw: object) -> list[float] | None:
    """A fire point inside the shell, or ``None`` when it is the default."""
    if raw is None:
        return None
    x, y, z = _vec3(raw, "起火点")
    length, width, height = spec.domain
    margin = _ORIGIN_MARGIN
    if not (margin <= x <= length - margin and margin <= y <= width - margin and 0.0 <= z <= height - 0.05):
        raise ValueError("起火点要放在场景里面，并离开四周的边界")
    point = _round_vec((x, y, z))
    if _near(point, spec.origin):
        return None
    return point


def _xb_of(center: tuple[float, float, float], size: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    return (
        center[0] - size[0] / 2.0,
        center[0] + size[0] / 2.0,
        center[1] - size[1] / 2.0,
        center[1] + size[1] / 2.0,
        center[2] - size[2] / 2.0,
        center[2] + size[2] / 2.0,
    )


def preset_structures(spec: ScenarioSpec) -> list[dict]:
    """The preset's boxes in the stored shape. ``None`` on a scene means this list."""
    rows = []
    for item in spec.obstacles:
        x0, x1, y0, y1, z0, z1 = item.xb
        rows.append(
            {
                "id": item.name,
                "name": STRUCTURE_LABELS.get(item.name, item.name),
                "role": item.role,
                "center": _round_vec(((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)),
                "size": _round_vec((x1 - x0, y1 - y0, z1 - z0)),
            }
        )
    return rows


def normalize_domain(spec: ScenarioSpec, raw: object) -> list[float] | None:
    """A custom site size, or ``None`` when it is still the preset's."""
    if raw is None:
        return None
    length, width, height = _vec3(raw, "场地")
    if any(value < _DOMAIN_MIN_M or value > _DOMAIN_MAX_M for value in (length, width, height)):
        raise ValueError(f"场地每边要在 {_DOMAIN_MIN_M:g} 到 {_DOMAIN_MAX_M:g} 米")
    point = _round_vec((length, width, height))
    if _near(point, spec.domain):
        return None
    return point


def _structure_id(raw: object, index: int) -> str:
    token = str(raw or "").strip() or f"s{index + 1}"
    if (
        not token
        or len(token) > 40
        or any(not (ch.isascii() and (ch.isalnum() or ch in "_-")) for ch in token)
    ):
        raise ValueError("结构的编号只能用字母、数字、下划线和短横线")
    return token


def normalize_structures(domain: tuple[float, float, float], raw: object) -> list[dict] | None:
    """Validate boxes. ``None`` keeps the preset's. ``[]`` means the user removed every box."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("结构必须是列表")
    if len(raw) > MAX_STRUCTURES:
        raise ValueError(f"最多 {MAX_STRUCTURES} 块结构")
    length, width, height = (float(value) for value in domain)
    cleaned = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("每一块结构都要是对象")
        token = _structure_id(item.get("id"), index)
        if token in seen:
            raise ValueError(f"结构的编号重复：{token}")
        seen.add(token)
        name = str(item.get("name") or "").strip() or STRUCTURE_LABELS.get(token, token)
        if len(name) > 40:
            raise ValueError("结构的名称最多 40 个字")
        role = str(item.get("role") or "inert").strip()
        if role not in _STRUCTURE_ROLES:
            raise ValueError("结构的角色要是不燃烧、可燃或起火面")
        center = _vec3(item.get("center"), "结构的中心")
        size = _vec3(item.get("size"), "结构的尺寸")
        if any(edge < MIN_STRUCTURE_M for edge in size):
            raise ValueError(f"结构的每边至少 {MIN_STRUCTURE_M:g} m")
        if any(edge > limit + 1e-4 for edge, limit in zip(size, (length, width, height))):
            raise ValueError("结构的边不能比场地更长")
        xb = _xb_of(center, size)
        if (
            xb[0] < -1e-4
            or xb[2] < -1e-4
            or xb[4] < -1e-4
            or xb[1] > length + 1e-4
            or xb[3] > width + 1e-4
            or xb[5] > height + 1e-4
        ):
            raise ValueError(f"{name}超出场地")
        cleaned.append(
            {
                "id": token,
                "name": name,
                "role": role,
                "center": _round_vec(center),
                "size": _round_vec(size),
            }
        )
    return cleaned


def _same_structure(left: dict, right: dict) -> bool:
    return (
        left["id"] == right["id"]
        and left["name"] == right["name"]
        and left["role"] == right["role"]
        and _near(left["center"], right["center"])
        and _near(left["size"], right["size"])
    )


def collapse_structures(base: ScenarioSpec, rows: list[dict] | None) -> list[dict] | None:
    """Drop a list that still matches the preset. ``[]`` stays, meaning no boxes."""
    if rows is None:
        return None
    preset = sorted(preset_structures(base), key=lambda item: item["id"])
    given = sorted(rows, key=lambda item: item["id"])
    if len(given) == len(preset) and all(_same_structure(left, right) for left, right in zip(given, preset)):
        return None
    return rows


def _fit_obstacles(domain: tuple[float, float, float], obstacles: tuple[Obstacle, ...]) -> None:
    length, width, height = domain
    for item in obstacles:
        x0, x1, y0, y1, z0, z1 = item.xb
        if x0 < -1e-4 or y0 < -1e-4 or z0 < -1e-4 or x1 > length + 1e-4 or y1 > width + 1e-4 or z1 > height + 1e-4:
            label = STRUCTURE_LABELS.get(item.name, item.name)
            raise ValueError(f"{label}超出场地")


def spec_with_shell(base: ScenarioSpec, domain: tuple[float, float, float], structures: list[dict] | None) -> ScenarioSpec:
    """Preset coefficients, with the scene's site and boxes when those were edited."""
    from dataclasses import replace

    if structures is None:
        obstacles = base.obstacles
    else:
        obstacles = tuple(
            Obstacle(item["id"], _xb_of(tuple(item["center"]), tuple(item["size"])), item["role"])
            for item in structures
        )
    spec = replace(base, domain=tuple(float(value) for value in domain), obstacles=obstacles)
    _fit_obstacles(spec.domain, spec.obstacles)
    return spec


def prepared_spec(
    scenario: str,
    domain: object = None,
    structures: object = None,
    origin: object = None,
) -> tuple[ScenarioSpec, list[float] | None, list[dict] | None, list[float] | None]:
    """The spec the bake uses, plus the three values worth storing.

    A null domain or structure list means the preset. The returned spec always
    has concrete boxes, including when the stored value collapses back to null.
    """
    base = get_scenario(str(scenario))
    kept_domain = normalize_domain(base, domain)
    live = tuple(kept_domain) if kept_domain else base.domain
    normalized = normalize_structures(live, structures)
    kept_structures = collapse_structures(base, normalized)
    spec = spec_with_shell(base, live, normalized)
    kept_origin = normalize_origin(spec, origin)
    return apply_origin(spec, kept_origin), kept_domain, kept_structures, kept_origin


def layout_environment(spec: ScenarioSpec, gas_box: dict, cameras: dict | None, viewpoints: list | tuple) -> str:
    """室内 when an inert lid covers the rain footprint of this bake, else 室外."""
    from firephantom.physics.blender_scene import camera_poses, environment_for_layout, precipitation_box

    poses = camera_poses(spec, list(viewpoints), cameras or None)
    precip = precipitation_box(spec.domain, gas_box, poses)
    return environment_for_layout(spec.domain, spec.obstacles, precip)


def normalize_camera_map(spec: ScenarioSpec, raw: object) -> dict:
    """Per-viewpoint overrides. An empty map means every camera stays automatic."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("相机必须是对象")
    length, width, height = spec.domain
    cleaned = {}
    for key, pose in raw.items():
        name = str(key)
        if name not in _VIEWS:
            raise ValueError(f"未知视角 {name!r}")
        if not isinstance(pose, dict):
            raise ValueError("相机位姿必须是对象")
        location = _vec3(pose.get("location"), "相机位置")
        target = _vec3(pose.get("target"), "相机目标")
        if not (0.05 <= location[0] <= length - 0.05 and 0.05 <= location[1] <= width - 0.05 and 0.05 <= location[2] <= height - 0.05):
            raise ValueError("相机要放在场景里面")
        if point_in_solid(spec, location):
            raise ValueError("相机在墙或天花板里面")
        if math.dist(location, target) < 0.05:
            raise ValueError("相机离目标太近")
        try:
            focal = float(pose.get("focal_mm"))
        except (TypeError, ValueError) as exc:
            raise ValueError("焦距必须是数值") from exc
        if not math.isfinite(focal) or not _FOCAL_MIN <= focal <= _FOCAL_MAX:
            raise ValueError("焦距要在 12 到 120 毫米之间")
        cleaned[name] = {"location": _round_vec(location), "target": _round_vec(target), "focal_mm": round(focal, 2)}
    return cleaned


def _same_camera(pose: dict, automatic: dict) -> bool:
    return (
        _near(pose["location"], automatic["location"])
        and _near(pose["target"], automatic["target"])
        and abs(float(pose["focal_mm"]) - float(automatic["focal_mm"])) <= 0.05
    )


def cameras_outside_solids(spec: ScenarioSpec, raw: object) -> tuple[object, list[str]]:
    """Drop poses that sit inside a wall or ceiling. Other values pass through.

    Saving still rejects those poses. A scene that already stored one can bake:
    the dropped view uses the searched camera.
    """
    if not isinstance(raw, dict):
        return raw, []
    kept = {}
    buried: list[str] = []
    for key, pose in raw.items():
        location = pose.get("location") if isinstance(pose, dict) else None
        if isinstance(location, (list, tuple)) and len(location) == 3:
            try:
                point = tuple(float(value) for value in location)
            except (TypeError, ValueError):
                kept[key] = pose
                continue
            if point_in_solid(spec, point):
                buried.append(str(key))
                continue
        kept[key] = pose
    return kept, buried


def collapse_cameras(spec: ScenarioSpec, cameras: dict) -> dict:
    """Drop overrides that still match the searched pose."""
    from firephantom.physics.blender_scene import camera_pose

    kept = {}
    for name, pose in cameras.items():
        auto = camera_pose(spec, name)
        if _same_camera(pose, auto):
            continue
        kept[name] = pose
    return kept


def _cleared_gas(spec: ScenarioSpec, box: dict) -> dict:
    from firephantom.physics.blender_scene import clear_of_solids, visible_obstacles

    cleared = clear_of_solids(box, visible_obstacles(spec))
    return {
        "location": _round_vec(tuple(cleared["location"])),
        "scale": _round_vec(tuple(cleared["scale"])),
        "resolution": int(cleared["resolution"]),
    }


def normalize_gas_domain(spec: ScenarioSpec, raw: object) -> dict | None:
    """A Mantaflow box, pulled off floors and full walls. ``None`` stays automatic."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("气体域必须是对象")
    location = _vec3(raw.get("location"), "气体域的位置")
    scale = _vec3(raw.get("scale"), "气体域的尺寸")
    if any(value <= 0 for value in scale):
        raise ValueError("气体域的每边都要是正数")
    if any(value * 2 < _GAS_MIN_M - 1e-6 for value in scale):
        raise ValueError(f"气体域的每边至少 {_GAS_MIN_M:g} m")
    resolution = raw.get("resolution", 32)
    low, high = GAS_RESOLUTION_RANGE
    message = f"气体域分辨率要是 {low} 到 {high} 之间的整数"
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise ValueError(message)
    if isinstance(resolution, float) and not resolution.is_integer():
        raise ValueError(message)
    resolution = int(resolution)
    if not low <= resolution <= high:
        raise ValueError(message)
    length, width, height = spec.domain
    bounds = (
        location[0] - scale[0],
        location[0] + scale[0],
        location[1] - scale[1],
        location[1] + scale[1],
        location[2] - scale[2],
        location[2] + scale[2],
    )
    if bounds[0] < -1e-4 or bounds[1] > length + 1e-4 or bounds[2] < -1e-4 or bounds[3] > width + 1e-4 or bounds[4] < -1e-4 or bounds[5] > height + 1e-4:
        raise ValueError("气体域超出了场景范围")
    return _cleared_gas(spec, {"location": location, "scale": scale, "resolution": resolution})


def resolved_gas(spec: ScenarioSpec, packages: list[dict], gas: dict | None) -> dict:
    """The box the bake uses: the stored one, or the automatic one pulled off solids."""
    return gas if gas is not None else _cleared_gas(spec, gas_domain(spec, packages))


def _same_gas(left: dict, right: dict) -> bool:
    return (
        _near(left["location"], right["location"])
        and _near(left["scale"], right["scale"])
        and int(left["resolution"]) == int(right["resolution"])
    )


def collapse_gas(spec: ScenarioSpec, packages: list[dict], gas: dict | None) -> dict | None:
    if gas is None:
        return None
    automatic = _cleared_gas(spec, gas_domain(spec, packages))
    if _same_gas(gas, automatic):
        return None
    return gas


def _fit_light_point(point: tuple[float, float, float], domain: tuple[float, float, float]) -> list[float]:
    fitted = []
    for value, limit in zip(point, domain):
        if limit <= 2.0 * _LIGHT_MARGIN:
            fitted.append(limit / 2.0)
        else:
            fitted.append(min(max(float(value), _LIGHT_MARGIN), float(limit) - _LIGHT_MARGIN))
    return _round_vec(tuple(fitted))


def _light_number(raw: object, label: str, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数值") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}要在 {low:g} 到 {high:g} 之间")
    return value


def normalize_lights(spec: ScenarioSpec, raw: object) -> list[dict] | None:
    """Validate lamps. ``None`` keeps the automatic fill and sun."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("lights 必须是列表")
    if len(raw) > MAX_LIGHTS:
        raise ValueError(f"最多 {MAX_LIGHTS} 盏灯")
    length, width, height = spec.domain
    cleaned = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("每一盏灯都要是对象")
        token = str(item.get("id") or "").strip() or f"L{index + 1}"
        if (
            not token
            or len(token) > 40
            or any(not (ch.isascii() and (ch.isalnum() or ch in "_-")) for ch in token)
        ):
            raise ValueError("灯的编号只能用字母、数字、下划线和短横线")
        if token in seen:
            raise ValueError(f"灯的编号重复：{token}")
        seen.add(token)
        name = str(item.get("name") or "").strip() or f"灯{index + 1}"
        if len(name) > 80:
            raise ValueError("灯的名称最多 80 个字")
        kind = str(item.get("type") or "").strip().upper()
        if kind not in _LIGHT_TYPES:
            raise ValueError("灯的类型要是点光、太阳、聚光或面光")
        color = item.get("color")
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValueError("灯的颜色要是三个数")
        rgb = tuple(_light_number(channel, "颜色", 0.0, 1.0) for channel in color)
        high = 10.0 if kind == "SUN" else 2000.0
        energy = _light_number(item.get("energy", 0.0), "强度", 0.0, high)
        size = _light_number(item.get("size", 0.5), "尺寸", 0.01, 5.0)
        size_y = _light_number(item.get("size_y", size), "尺寸", 0.01, 5.0)
        spot_size = _light_number(item.get("spot_size", 0.7854), "光斑角度", 0.05, math.pi)
        spot_blend = _light_number(item.get("spot_blend", 0.15), "光斑过渡", 0.0, 1.0)
        location = _vec3(item.get("location"), "灯的位置")
        if not (
            _LIGHT_MARGIN - 1e-4 <= location[0] <= length - _LIGHT_MARGIN + 1e-4
            and _LIGHT_MARGIN - 1e-4 <= location[1] <= width - _LIGHT_MARGIN + 1e-4
            and _LIGHT_MARGIN - 1e-4 <= location[2] <= height - _LIGHT_MARGIN + 1e-4
        ):
            raise ValueError("灯要放在场景里面")
        rotation = _vec3(item.get("rotation") or (0.0, 0.0, 0.0), "灯的朝向")
        cleaned.append(
            {
                "id": token,
                "name": name,
                "type": kind,
                "color": [round(value, 4) for value in rgb],
                "energy": round(energy, 4),
                "size": round(size, 4),
                "size_y": round(size_y, 4),
                "spot_size": round(spot_size, 4),
                "spot_blend": round(spot_blend, 4),
                "location": _round_vec(location),
                "rotation": [round(value, 4) for value in rotation],
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return cleaned


def automatic_lights(spec: ScenarioSpec, environment: str | None = None) -> list[dict]:
    """The fill and sun the bake uses until the scene stores its own lamps.

    ``environment`` is the sky for this layout. A covered top previews the
    indoor clear sun, which is off. Weather still scales a stored sun at bake time.
    """
    from firephantom.physics.blender_scene import fill_light, look_settings, visible_obstacles

    fill = fill_light(spec.domain, spec.origin, visible_obstacles(spec))
    if environment not in ("室内", "室外"):
        environment = layout_environment(spec, resolved_gas(spec, [], None), None, _VIEWS)
    sun = float(look_settings(environment, "晴")["sun"])
    ox, oy, _oz = (float(value) for value in spec.origin)
    height = float(spec.domain[2])
    drafted = [
        {
            "id": "auto-fill",
            "name": "补光",
            "type": "POINT",
            "color": [0.75, 0.8, 0.9],
            "energy": 20.0,
            "size": float(fill["size"]),
            "size_y": 1.0,
            "spot_size": 0.7854,
            "spot_blend": 0.15,
            "location": _fit_light_point(tuple(float(value) for value in fill["location"]), spec.domain),
            "rotation": [0.0, 0.0, 0.0],
            "enabled": True,
        },
        {
            "id": "auto-sun",
            "name": "太阳",
            "type": "SUN",
            "color": [1.0, 0.96, 0.9],
            "energy": sun,
            "size": 0.1,
            "size_y": 1.0,
            "spot_size": 0.7854,
            "spot_blend": 0.15,
            "location": _fit_light_point((ox, oy, height - _LIGHT_MARGIN), spec.domain),
            "rotation": [0.7, 0.2, 0.9],
            "enabled": True,
        },
    ]
    stored = normalize_lights(spec, drafted)
    if stored is None:
        return []
    return stored


def _same_light(left: dict, right: dict) -> bool:
    return (
        left["id"] == right["id"]
        and left["name"] == right["name"]
        and left["type"] == right["type"]
        and left["enabled"] == right["enabled"]
        and _near(left["color"], right["color"])
        and abs(float(left["energy"]) - float(right["energy"])) <= 1e-3
        and abs(float(left["size"]) - float(right["size"])) <= 1e-3
        and abs(float(left["size_y"]) - float(right["size_y"])) <= 1e-3
        and abs(float(left["spot_size"]) - float(right["spot_size"])) <= 1e-3
        and abs(float(left["spot_blend"]) - float(right["spot_blend"])) <= 1e-3
        and _near(left["location"], right["location"])
        and _near(left["rotation"], right["rotation"])
    )


def collapse_lights(spec: ScenarioSpec, lights: list[dict] | None, environment: str | None = None) -> list[dict] | None:
    """Drop a list that still matches the automatic fill and sun. ``[]`` stays."""
    if lights is None:
        return None
    automatic = automatic_lights(spec, environment)
    if len(lights) == len(automatic) and all(_same_light(left, right) for left, right in zip(lights, automatic)):
        return None
    return lights


def _camera_blocked(pose: dict, obstacles: list[dict]) -> bool:
    import numpy as np

    from firephantom.physics.blender_scene import segment_hits_box

    start = np.asarray(pose["location"], dtype=np.float64)
    end = np.asarray(pose["target"], dtype=np.float64)
    return any(segment_hits_box(start, end, tuple(item["xb"])) for item in obstacles)


def layout_preview(
    scenario: str,
    placements: object,
    origin_fuel: str,
    origin: object = None,
    cameras: object = None,
    gas_domain_override: object = None,
    viewpoints: list | tuple | None = None,
    lights: object = None,
    simulation: object = None,
    frame_count: int = 16,
    look_count: int = 1,
    structures: object = None,
    domain: object = None,
) -> dict:
    """Resolved cameras, gas box, lamps, and bake estimate for the editor. Raises when the layout is invalid."""
    from firephantom.physics.simulation import (
        bake_estimate,
        collapse_simulation,
        normalize_simulation,
        resolved_simulation,
    )

    from firephantom.physics.blender_scene import camera_pose, look_settings, pose_from_override, visible_obstacles

    spec, _domain, _structures, origin_value = prepared_spec(scenario, domain, structures, origin)
    packages = normalize_placements(spec, placements, origin_fuel or "木材")
    views = list(viewpoints or _VIEWS)
    for view in views:
        if view not in _VIEWS:
            raise ValueError(f"未知视角 {view!r}")
    usable, buried = cameras_outside_solids(spec, cameras)
    custom = normalize_camera_map(spec, usable)
    obstacles = visible_obstacles(spec)
    resolved_cameras = {}
    automatic_cameras = {}
    buried_views = set(buried)
    for view in views:
        auto = camera_pose(spec, view)
        automatic = {
            "location": _round_vec(tuple(auto["location"])),
            "target": _round_vec(tuple(auto["target"])),
            "focal_mm": round(float(auto["focal_mm"]), 2),
            "fallback": bool(auto["fallback"]),
        }
        automatic_cameras[view] = automatic
        if view in buried_views:
            pose = auto
            overridden = False
            fallback = bool(auto["fallback"])
            inside_solid = True
        elif view in custom:
            pose = pose_from_override(view, custom[view])
            overridden = not _same_camera(custom[view], automatic)
            fallback = False
            inside_solid = False
        else:
            pose = auto
            overridden = False
            fallback = bool(auto["fallback"])
            inside_solid = False
        resolved_cameras[view] = {
            "location": _round_vec(tuple(pose["location"])),
            "target": _round_vec(tuple(pose["target"])),
            "focal_mm": round(float(pose["focal_mm"]), 2),
            "fallback": fallback,
            "overridden": overridden,
            "blocked": _camera_blocked(pose, obstacles),
            "inside_solid": inside_solid,
        }
    automatic_gas = _cleared_gas(spec, gas_domain(spec, packages))
    if gas_domain_override is None:
        resolved_gas = automatic_gas
        gas_overridden = False
    else:
        resolved_gas = normalize_gas_domain(spec, gas_domain_override)
        gas_overridden = not _same_gas(resolved_gas, automatic_gas)

    def publish(box: dict) -> dict:
        return {**box, "bounds": [round(float(value), 4) for value in _domain_bounds(box)]}

    environment = layout_environment(spec, resolved_gas, custom, views)
    stored_lights = collapse_lights(spec, normalize_lights(spec, lights), environment)
    automatic = automatic_lights(spec, environment)
    # The clear-weather world the bake starts from. The editor's rendered
    # shading mode lights the viewport with it, next to the stored lamps.
    look = look_settings(environment, "晴")
    settings = collapse_simulation(normalize_simulation(simulation))
    renders = len(views) * max(1, int(look_count)) * max(1, int(frame_count))
    return {
        "origin": _round_vec(spec.origin),
        "origin_overridden": origin_value is not None,
        "cameras": resolved_cameras,
        "automatic_cameras": automatic_cameras,
        "rejected_cameras": buried,
        "gas_domain": publish(resolved_gas),
        "automatic_gas": publish(automatic_gas),
        "gas_overridden": gas_overridden,
        "lights": automatic if stored_lights is None else stored_lights,
        "automatic_lights": automatic,
        "lights_overridden": stored_lights is not None,
        "environment": environment,
        "world": {"color": _round_vec(tuple(look["world_color"])), "strength": float(look["world_strength"])},
        "simulation": resolved_simulation(settings),
        "simulation_overridden": settings is not None,
        # Each weather is its own bake, so the look count multiplies the solve too.
        "bake_estimate": bake_estimate(resolved_gas, settings, int(frame_count), renders, max(1, int(look_count))),
    }


def prepare(
    spec: ScenarioSpec,
    raw: object,
    default_fuel: str,
    gas_override: dict | None = None,
    warmup: int = WARMUP_FRAMES,
) -> dict:
    packages = normalize_placements(spec, raw, default_fuel)
    domain = gas_domain(spec, packages) if gas_override is None else gas_override
    return {
        "packages": annotate_packages(spec, packages, domain, warmup),
        "gas_domain": {key: value for key, value in domain.items() if key != "bounds"},
        "textures": surface_textures(spec.name),
        "spread_model": "reaction+delay",
    }


def _gltf_json(data: bytes) -> dict:
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("不是 glTF 的 .glb")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 2:
        raise ValueError("只接受 glTF 2 的 .glb")
    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if chunk_length < 0 or offset + chunk_length > len(data):
            break
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk)
    raise ValueError("glb 里没有模型数据")


def glb_stats(data: bytes) -> dict:
    """Triangle count and scene size from the glTF JSON chunk.

    Accessor min/max are in the asset's own space. glTF is Y-up, so the
    scene size is ``(dx, dz, dy)`` from glTF ``(dx, dy, dz)``. Node
    transforms are not applied.
    """
    document = _gltf_json(data)
    accessors = document.get("accessors") or []
    triangles = 0
    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    found = False
    for mesh in document.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            if int(primitive.get("mode", 4)) != 4:
                continue
            attributes = primitive.get("attributes") or {}
            position = attributes.get("POSITION")
            if position is None:
                continue
            if "indices" in primitive:
                triangles += int(accessors[int(primitive["indices"])].get("count", 0)) // 3
            else:
                triangles += int(accessors[int(position)].get("count", 0)) // 3
            accessor = accessors[int(position)]
            minimum, maximum = accessor.get("min"), accessor.get("max")
            if not minimum or not maximum or len(minimum) < 3 or len(maximum) < 3:
                continue
            found = True
            for axis in range(3):
                low[axis] = min(low[axis], float(minimum[axis]))
                high[axis] = max(high[axis], float(maximum[axis]))
    if triangles <= 0 or not found:
        raise ValueError("glb 里没有可用的三角面")
    if triangles > MAX_TRIS:
        raise ValueError(f"模型面数 {triangles} 超过 {MAX_TRIS}，请换一个更简单的模型")
    dx, dy, dz = high[0] - low[0], high[1] - low[1], high[2] - low[2]
    size = (dx, dz, dy)
    if min(size) <= 1e-4:
        raise ValueError("模型有一条边是 0")
    return {"triangles": triangles, "size": size}


def placement_size(raw: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    """Scene size used when a model is placed.

    A positive mesh shorter than one cell is scaled uniformly until its
    shortest edge is 0.45 m. A zero edge still cannot burn.
    """
    size = tuple(float(value) for value in raw)
    if len(size) != 3 or any(value <= 0 or not math.isfinite(value) for value in size):
        raise ValueError("模型有一条边是 0")
    short = min(size)
    if short < MIN_CELL_M:
        factor = MIN_CELL_M / short
        size = tuple(value * factor for value in size)
    return tuple(round(value, 4) for value in size)


def store_upload(data: bytes, label: str | None = None) -> dict:
    """Save one .glb and return the id the placement list uses."""
    if len(data) > MAX_GLB_BYTES:
        raise ValueError("模型不能超过 8 MB")
    stats = glb_stats(data)
    token = hashlib.sha256(data).hexdigest()[:16]
    path = assets_root() / "uploads" / f"{token}.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_bytes(data)
    record = {
        "asset_id": f"upload:{token}",
        "label": (str(label or "").strip() or "上传")[:80],
        "size": list(placement_size(stats["size"])),
        "triangles": int(stats["triangles"]),
        "url": f"/assets/uploads/{token}.glb",
    }
    path.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def list_uploads() -> list[dict]:
    """Every stored upload, newest first, so the editor shelf survives reloads."""
    directory = assets_root() / "uploads"
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.glb"), key=lambda item: item.stat().st_mtime, reverse=True):
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            records.append(json.loads(sidecar.read_text(encoding="utf-8")))
            continue
        try:
            stats = glb_stats(path.read_bytes())
        except (OSError, ValueError):
            continue
        record = {
            "asset_id": f"upload:{path.stem}",
            "label": "上传",
            "size": list(placement_size(stats["size"])),
            "triangles": int(stats["triangles"]),
            "url": f"/assets/uploads/{path.name}",
        }
        sidecar.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        records.append(record)
    return records


def store_upload_many(items: list[tuple[str, bytes]]) -> dict:
    """Save every glb that passes the checks. One failure leaves the others."""
    accepted = []
    rejected = []
    for name, data in items:
        label = Path(str(name)).stem or "上传"
        try:
            accepted.append(store_upload(data, label=label))
        except ValueError as exc:
            rejected.append({"name": Path(str(name)).name or str(name), "reason": str(exc)})
    return {"accepted": accepted, "rejected": rejected}


def delete_upload(token: str) -> dict:
    """Remove one user upload. Built-in library ids are refused."""
    raw = str(token or "").strip()
    if raw.startswith("upload:"):
        raw = raw.split(":", 1)[1]
    if raw in PRESETS or not re.fullmatch(r"[0-9a-f]{16}", raw):
        raise ValueError("内置模型不能删除")
    path = assets_root() / "uploads" / f"{raw}.glb"
    if not path.is_file():
        raise ValueError("没有这个上传")
    path.unlink()
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        sidecar.unlink()
    return {"deleted": f"upload:{raw}"}
