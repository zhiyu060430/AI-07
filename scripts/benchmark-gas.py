"""Time Blender gas bakes and fit the cost model in physics/simulation.py.

Run on the machine that bakes, with BLENDER_BIN set:

    python scripts/benchmark-gas.py /tmp/gas-bench

Each row bakes the room scene once, from one camera under one look. The fit
prints the four constants to copy into firephantom/physics/simulation.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from firephantom.physics.adapters import run_blender
from firephantom.physics.blender_scene import render_blender
from firephantom.physics.placements import normalize_gas_domain, prepare
from firephantom.physics.scenarios import get_scenario
from firephantom.physics.simulation import gas_cells, resolved_simulation, warmup_frames
from firephantom.validation import physical_parameters

ROOM_FULL = {"location": [4.0, 3.0, 1.5], "scale": [4.0, 3.0, 1.5]}
# (label, box or None for automatic, resolution, simulation, frame_count)
RUNS = [
    ("auto-32-n0", None, 32, {"noise": 0}, 4),
    ("auto-32-n2", None, 32, {}, 4),
    ("auto-32-n2-f16", None, 32, {}, 16),
    ("room-64-n0", ROOM_FULL, 64, {"noise": 0}, 4),
    ("room-64-n2", ROOM_FULL, 64, {}, 4),
    ("room-96-n0", ROOM_FULL, 96, {"noise": 0}, 4),
]


def bake(work: Path, label: str, box: dict | None, resolution: int, simulation: dict, frame_count: int) -> dict:
    spec = get_scenario("房间火灾")
    params = physical_parameters(1.5, 20, 40, "木材")
    settings = resolved_simulation(simulation)
    gas = None
    if box is not None:
        gas = normalize_gas_domain(spec, {**box, "resolution": resolution})
    layout = prepare(spec, [], params["fuel_type"], gas, warmup_frames(settings))
    if gas is None:
        layout["gas_domain"]["resolution"] = resolution
    folder = work / label
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "scene.py"
    script.write_text(
        render_blender(
            spec,
            layout=layout,
            render_output="renders",
            viewpoints=["侧视"],
            looks=[("室内", "晴")],
            frame_count=frame_count,
            simulation=settings,
            **params,
        ),
        encoding="utf-8",
    )
    started = time.monotonic()
    result = run_blender(script, timeout=3600)
    seconds = time.monotonic() - started
    return {
        "label": label,
        "ok": bool(result["success"]),
        "seconds": round(seconds, 1),
        "cells": gas_cells(layout["gas_domain"]),
        "frames": warmup_frames(settings) + frame_count,
        "noise": int(settings["noise"]),
        "renders": frame_count,
        "samples": int(settings["samples"]),
    }


def fit(rows: list[dict]) -> None:
    good = [row for row in rows if row["ok"]]
    if len(good) < 4:
        print("not enough successful runs to fit")
        return
    # Linear in (BASE, PER_CELL_FRAME, PER_CELL_FRAME * NOISE_WEIGHT, PER_RENDER)
    # once noise is a known cube, so fit that and read NOISE_WEIGHT back out.
    design = np.array(
        [[1.0, r["cells"] * r["frames"], r["cells"] * r["frames"] * r["noise"] ** 3, r["samples"] / 20.0 * r["renders"]] for r in good]
    )
    target = np.array([r["seconds"] for r in good])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    base, per_cell, noisy, per_render = (float(value) for value in coef)
    weight = noisy / per_cell if per_cell > 0 else 0.0
    fitted = {"BASE_S": base, "PER_CELL_FRAME_S": per_cell, "NOISE_WEIGHT": weight, "PER_RENDER_S": per_render}
    print(json.dumps(fitted, indent=2))
    for row, value in zip(good, design @ coef):
        print(f"{row['label']:>16}  measured {row['seconds']:7.1f}s  fitted {value:7.1f}s")


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "gas-bench").resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in RUNS:
        row = bake(work, *run)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (work / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fit(rows)


if __name__ == "__main__":
    main()
