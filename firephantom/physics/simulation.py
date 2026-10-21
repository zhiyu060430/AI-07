"""Mantaflow and Cycles settings a scene may change, and what they cost.

``None`` means every default, and a stored dict that equals the defaults
collapses back to ``None``, so scenes that never touched these settings keep
their cache keys and sequence ids. The bake budget limits a custom gas box by
estimated Blender time instead of by meters.
"""

from __future__ import annotations

import math

FPS = 12
SIMULATION_DEFAULTS = {
    "warmup_s": 1.0,
    "noise": 2,
    "vorticity": 0.22,
    "burning_rate": 0.6,
    "flame_temp": 3.0,
    "smoke": 1.0,
    "samples": 20,
}
# (low, high) for each number. ``noise`` is one of NOISE_SCALES instead.
SIMULATION_RANGES = {
    "warmup_s": (0.5, 10.0),
    "vorticity": (0.0, 1.0),
    "burning_rate": (0.1, 2.0),
    "flame_temp": (1.0, 5.0),
    "smoke": (0.0, 3.0),
    "samples": (8, 256),
}
NOISE_SCALES = (0, 2, 3)
_LABELS = {
    "warmup_s": "预热时长",
    "noise": "噪声细节",
    "vorticity": "涡度",
    "burning_rate": "燃烧速度",
    "flame_temp": "火焰温度",
    "smoke": "烟量",
    "samples": "渲染采样",
}
GAS_RESOLUTION_RANGE = (16, 128)
MAX_BAKE_SECONDS = 1800.0

# Fitted with scripts/benchmark-gas.py on the Mac mini that runs the bakes
# (six room bakes, 31k-242k cells, 22-113 s; non-negative least squares).
# seconds = BASE + PER_CELL_FRAME * cells * frames * (1 + NOISE_WEIGHT * noise**3)
#         + PER_RENDER * samples / 20 * renders
# 256 px renders at 20 samples barely showed in those runs, so PER_RENDER is a
# conservative guess that still lets more samples raise the estimate.
_BASE_S = 10.8
_PER_CELL_FRAME_S = 2.66e-5
_NOISE_WEIGHT = 0.21
_PER_RENDER_S = 0.5


def warmup_frames(simulation: dict | None) -> int:
    seconds = (simulation or SIMULATION_DEFAULTS)["warmup_s"]
    return max(1, int(round(float(seconds) * FPS)))


def resolved_simulation(simulation: dict | None) -> dict:
    return {**SIMULATION_DEFAULTS, **(simulation or {})}


def normalize_simulation(raw: object) -> dict | None:
    """Fill defaults and check ranges. ``None`` stays ``None``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("模拟设置必须是对象")
    unknown = sorted(set(raw) - set(SIMULATION_DEFAULTS))
    if unknown:
        raise ValueError(f"未知模拟设置 {', '.join(unknown)}")
    settings = dict(SIMULATION_DEFAULTS)
    for key, value in raw.items():
        label = _LABELS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label}必须是数字")
        if key == "noise":
            if float(value) not in NOISE_SCALES:
                raise ValueError(f"{label}只能是 0（关）、2 或 3")
            settings[key] = int(value)
            continue
        low, high = SIMULATION_RANGES[key]
        if not low <= float(value) <= high:
            raise ValueError(f"{label}要在 {low:g} 到 {high:g} 之间")
        settings[key] = int(round(float(value))) if key == "samples" else round(float(value), 4)
    return settings


def collapse_simulation(simulation: dict | None) -> dict | None:
    if simulation is None:
        return None
    if all(abs(float(simulation[key]) - float(value)) <= 1e-6 for key, value in SIMULATION_DEFAULTS.items()):
        return None
    return simulation


def gas_cells(gas: dict) -> int:
    """Mantaflow's base grid: ``resolution`` cells on the longest edge, the rest in proportion."""
    edges = [2.0 * float(value) for value in gas["scale"]]
    resolution = int(gas["resolution"])
    longest = max(edges)
    return math.prod(max(1, round(resolution * edge / longest)) for edge in edges)


def estimate_bake_seconds(
    gas: dict, simulation: dict | None, frame_count: int, render_count: int, bakes: int = 1
) -> float:
    """``bakes`` is the number of weathers: rain and snow change the flame, so each one solves again."""
    settings = resolved_simulation(simulation)
    frames = warmup_frames(settings) + int(frame_count)
    noise = int(settings["noise"])
    sim = _PER_CELL_FRAME_S * gas_cells(gas) * frames * (1.0 + _NOISE_WEIGHT * noise**3) * max(1, int(bakes))
    render = _PER_RENDER_S * float(settings["samples"]) / 20.0 * int(render_count)
    return _BASE_S + sim + render


def bake_estimate(gas: dict, simulation: dict | None, frame_count: int, render_count: int, bakes: int = 1) -> dict:
    seconds = estimate_bake_seconds(gas, simulation, frame_count, render_count, bakes)
    return {
        "cells": gas_cells(gas),
        "seconds": round(seconds, 1),
        "limit": MAX_BAKE_SECONDS,
        "over": seconds > MAX_BAKE_SECONDS,
    }


def check_budget(gas: dict, simulation: dict | None, frame_count: int, render_count: int, bakes: int = 1) -> None:
    estimate = bake_estimate(gas, simulation, frame_count, render_count, bakes)
    if estimate["over"]:
        raise ValueError(
            f"预计烘焙约 {estimate['seconds'] / 60:.0f} 分钟，超过上限 {MAX_BAKE_SECONDS / 60:.0f} 分钟。"
            "缩小气体域、降低分辨率或噪声、缩短预热，或减少天气、帧数和采样"
        )
