"""Validation and normalization for public and programmatic pipeline calls."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


def finite_float(value: object, name: str) -> float:
    """Convert a value to a finite float and report a domain-friendly error."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def physical_parameters(
    wind_speed: object,
    temperature: object,
    humidity: object,
    fuel_type: object,
) -> dict[str, float | str]:
    """Normalize physical parameters shared by the field model and Blender.

    The limits cover ordinary laboratory, building, and wildland scenarios and
    prevent invalid values from reaching exponential/CFD calculations.
    """
    wind = finite_float(wind_speed, "wind_speed")
    ambient = finite_float(temperature, "temperature")
    moisture = finite_float(humidity, "humidity")
    fuel = str(fuel_type).strip()
    if not fuel:
        raise ValueError("fuel_type must not be empty")
    if len(fuel) > 128:
        raise ValueError("fuel_type must be at most 128 characters")
    if not 0.0 <= wind <= 100.0:
        raise ValueError("wind_speed must be between 0 and 100 m/s")
    if not -80.0 <= ambient <= 100.0:
        raise ValueError("temperature must be between -80 and 100 C")
    if not 0.0 <= moisture <= 100.0:
        raise ValueError("humidity must be between 0 and 100 percent")
    return {
        "wind_speed": wind,
        "temperature": ambient,
        "humidity": moisture,
        "fuel_type": fuel,
    }


def diffusion_steps(value: object) -> int:
    """Normalize Stable Diffusion steps. The local server accepts 1 to 20."""
    if value is None or value == "":
        return 20
    if isinstance(value, bool):
        raise ValueError("steps must be an integer")
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be an integer") from exc
    if not 1 <= steps <= 20:
        raise ValueError("steps must be between 1 and 20")
    return steps


def diffusion_cfg(value: object) -> float:
    """Normalize the classifier-free guidance scale."""
    if value is None or value == "":
        return 7.0
    cfg = finite_float(value, "cfg_scale")
    if not 0.0 <= cfg <= 20.0:
        raise ValueError("cfg_scale must be between 0 and 20")
    return cfg


def optional_seed(value: object) -> int | None:
    """Normalize an optional sample seed. ``None`` means the caller should derive one."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")
    return seed


def normalize_fire_engine(value: object) -> str:
    """The only fire simulator is Blender. An omitted value uses it."""
    if value is None or value == "":
        return "blender"
    engine = str(value).strip().lower()
    if engine != "blender":
        raise ValueError("fire_engine must be blender")
    return engine


def _placements(spec: object, raw: object, fuel_type: object) -> list:
    from firephantom.physics.placements import normalize_placements

    if raw in (None, []):
        return []
    return normalize_placements(spec, raw, str(fuel_type))


def _choice_list(item: Mapping[str, object], plural: str, singular: str, allowed: tuple[str, ...], label: str) -> list[str]:
    raw = item.get(plural)
    if not raw:
        single = item.get(singular)
        raw = [] if single in (None, "") else [single]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    choices: list[str] = []
    for value in raw:
        text = str(value).strip()
        if text not in allowed:
            raise ValueError(f"未知{label} {text!r}，可选：{'、'.join(allowed)}")
        if text not in choices:
            choices.append(text)
    if not choices:
        raise ValueError(f"至少选择一个{label}")
    return choices


def _environment(item: Mapping[str, object], derived: str) -> list[str]:
    """The lid decides 室内/室外. An old ``scene``/``scenes`` must agree with it."""
    from firephantom.synthesis.render import SCENES

    if item.get("scenes") or item.get("scene") not in (None, ""):
        given = _choice_list(item, "scenes", "scene", SCENES, "环境")
        if given != [derived]:
            raise ValueError(f"环境由顶盖决定，这一烘是{derived}，不能选{'、'.join(given)}")
    return [derived]


def _weathers(item: Mapping[str, object]) -> list[str]:
    from firephantom.synthesis.render import WEATHERS

    return _choice_list(item, "weathers", "weather", WEATHERS, "天气")


def reject_generation_diffusion(value: object) -> None:
    """SD is no longer part of generation. It runs on reviewed samples in a dataset."""
    if value in (None, False, ""):
        return
    if value is True or (isinstance(value, Mapping) and value.get("enabled")):
        raise ValueError("生成阶段不再调用 Stable Diffusion，请在数据集里对已审核样本做 SD 增强")
    if not isinstance(value, (Mapping, bool)):
        raise ValueError("diffusion must be an object")


def _positive_count(value: object) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError("count must be an integer")
    try:
        count = int(value if value not in (None, "") else 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("count must be an integer") from exc
    if count < 1 or count > 100:
        raise ValueError("count must be between 1 and 100")
    return count


def _repeat_seed(identity: dict, repeat: int) -> int:
    raw = json.dumps({**identity, "repeat": repeat}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "little")


MAX_JOB_SAMPLES = 20000


def _frame_count(value: object) -> int:
    from firephantom.physics.blender_scene import DEFAULT_FRAME_COUNT, MAX_FRAME_COUNT

    if value in (None, ""):
        return DEFAULT_FRAME_COUNT
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError("frame_count must be an integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_count must be an integer") from exc
    if not 1 <= count <= MAX_FRAME_COUNT:
        raise ValueError(f"frame_count must be between 1 and {MAX_FRAME_COUNT}")
    return count


def _min_flame_area(value: object) -> float:
    from firephantom.quality.filter import DEFAULT_MIN_FLAME_AREA

    if value in (None, ""):
        return DEFAULT_MIN_FLAME_AREA
    area = finite_float(value, "min_flame_area")
    if not 0.0 <= area <= 0.5:
        raise ValueError("min_flame_area must be between 0 and 0.5")
    return area


def _augmentations(raw: object) -> list[str]:
    from firephantom.synthesis.augment import AUGMENTATIONS

    if raw in (None, ""):
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("augmentations must be a list")
    names: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name not in AUGMENTATIONS:
            raise ValueError(f"未知增强 {name!r}，可选：{'、'.join(AUGMENTATIONS)}")
        if name not in names:
            names.append(name)
    return names


def sequence_samples(sequence: Mapping[str, object]) -> int:
    combos = len(sequence["scenes"]) * len(sequence["weathers"]) * len(sequence["viewpoints"])
    return combos * int(sequence["frame_count"]) * (1 + len(sequence["augmentations"]))


def expand_job_items(items: list[dict]) -> list[dict]:
    """Expand workbench rows into one sequence request per bake.

    A sequence is one simulated fire. ``scenes``, ``weathers`` and
    ``viewpoints`` are rendered inside it, ``frame_count`` frames each, so
    their product does not multiply simulations. ``count`` repeats the row as
    new bakes with new seeds (the seed moves the inflow slightly), and wind
    increases by ``wind_step`` once per repeat.
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
        prepared_spec,
        resolved_gas,
    )
    from firephantom.physics.scenarios import SCENARIOS as scenario_specs
    from firephantom.physics.simulation import check_budget, collapse_simulation, normalize_simulation
    from firephantom.synthesis.render import VIEWPOINTS

    if not isinstance(items, list) or not items:
        raise ValueError("a job must contain at least one request")
    expanded: list[dict] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("pipeline request must be a mapping")
        scenario = str(item.get("scenario", "")).strip()
        if scenario not in scenario_specs:
            raise ValueError(f"未知火灾场景 {scenario!r}")
        params = physical_parameters(
            item.get("wind_speed"),
            item.get("temperature"),
            item.get("humidity"),
            item.get("fuel_type"),
        )
        wind_step = finite_float(item.get("wind_step", 0.0), "wind_step")
        if not 0.0 <= wind_step <= 100.0:
            raise ValueError("wind_step must be between 0 and 100 m/s")
        count = _positive_count(item.get("count", 1))
        reject_generation_diffusion(item.get("diffusion"))
        base_seed = optional_seed(item.get("seed"))
        scene_revision = item.get("scene_revision")
        spec, domain, structures, origin = prepared_spec(
            scenario, item.get("domain"), item.get("structures"), item.get("origin")
        )
        placed = _placements(spec, item.get("placements"), params["fuel_type"])
        usable_cameras, _buried = cameras_outside_solids(spec, item.get("cameras"))
        cameras = collapse_cameras(spec, normalize_camera_map(spec, usable_cameras))
        gas = collapse_gas(spec, placed, normalize_gas_domain(spec, item.get("gas_domain")))
        viewpoints = _choice_list(item, "viewpoints", "viewpoint", VIEWPOINTS, "视角")
        environment = layout_environment(spec, resolved_gas(spec, placed, gas), cameras, viewpoints)
        scenes = _environment(item, environment)
        lights = collapse_lights(spec, normalize_lights(spec, item.get("lights")), environment)
        simulation = collapse_simulation(normalize_simulation(item.get("simulation")))
        shared = {
            "scenario": scenario,
            "scene_id": str(item["scene_id"]) if item.get("scene_id") else None,
            "scene_revision": None if scene_revision in (None, "") else int(scene_revision),
            "placements": placed,
            "origin": origin,
            "structures": structures,
            "domain": domain,
            "cameras": cameras,
            "gas_domain": gas,
            "lights": lights,
            "simulation": simulation,
            "temperature": params["temperature"],
            "humidity": params["humidity"],
            "fuel_type": params["fuel_type"],
            "fire_engine": normalize_fire_engine(item.get("fire_engine")),
            "scenes": scenes,
            "weathers": _weathers(item),
            "viewpoints": viewpoints,
            "frame_count": _frame_count(item.get("frame_count")),
            "augmentations": _augmentations(item.get("augmentations")),
            "min_flame_area": _min_flame_area(item.get("min_flame_area")),
        }
        renders = len(shared["viewpoints"]) * len(shared["scenes"]) * len(shared["weathers"]) * shared["frame_count"]
        check_budget(resolved_gas(spec, placed, gas), simulation, shared["frame_count"], renders, len(shared["weathers"]))
        for repeat in range(count):
            wind = min(100.0, max(0.0, float(params["wind_speed"]) + repeat * wind_step))
            identity = {**shared, "wind_speed": wind}
            if base_seed is None:
                seed = _repeat_seed(identity, repeat)
            else:
                seed = (base_seed + repeat) % (2**32)
            expanded.append({**identity, "seed": seed})
    total = sum(sequence_samples(sequence) for sequence in expanded)
    if total > MAX_JOB_SAMPLES:
        raise ValueError(f"a job can contain at most {MAX_JOB_SAMPLES} samples, this one has {total}")
    return expanded
