"""Run the sequences of one job and keep what the workbench shows for it.

A job is a list of sequences (one bake each). Their combinations are
numbered globally in order, and each finished combination is one job
result. ``runs/<job_id>/`` keeps the scene plate of each sequence, a preview
per combination and a manifest; samples, videos and fields live in the
dataset root and are addressed by id.
"""

from __future__ import annotations

import json
from pathlib import Path

from firephantom.images import encode_png, to_uint8
from firephantom.physics.plate import render_scenario_plate
from firephantom.pipeline import combo_list, run_sequence, sequence_id_for
from firephantom.serialize import plain

STAGES = ("plate", "engine")


def run_dir(root: Path | str, job_id: str) -> Path:
    return Path(root) / "runs" / job_id


def combo_offsets(sequences: list[dict]) -> list[int]:
    offsets, total = [], 0
    for sequence in sequences:
        offsets.append(total)
        total += len(combo_list(sequence))
    return offsets


def total_combos(sequences: list[dict]) -> int:
    return sum(len(combo_list(sequence)) for sequence in sequences)


def execute(
    root: Path | str,
    job_id: str,
    sequences: list[dict],
    combo_index: int | None,
    stop,
    on_combo,
) -> str:
    """Run every sequence, or only the one combination ``combo_index``.

    Bakes are cached by their inputs, so running again after a change only
    re-bakes when the physics changed. ``on_combo`` receives the global
    combination index and its result; ``stop()`` halts before the next one.
    """
    if not sequences:
        raise ValueError("a job must contain at least one request")
    directory = run_dir(root, job_id)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(directory)
    manifest["sequences"] = [
        {
            "index": index,
            "sequence_id": sequence_id_for(sequence),
            "scenario": sequence["scenario"],
            "scene_id": sequence.get("scene_id"),
            **_previous_sequence(manifest, index),
        }
        for index, sequence in enumerate(sequences)
    ]
    reset = combo_index is None
    manifest["combos"] = _combo_slots(manifest.get("combos") or [], sequences, reset=reset)
    _write_manifest(directory, manifest)
    offsets = combo_offsets(sequences)
    for sequence_index, sequence in enumerate(sequences):
        count = len(combo_list(sequence))
        offset = offsets[sequence_index]
        if combo_index is not None and not offset <= combo_index < offset + count:
            continue
        if stop():
            return "cancelled"
        plate = directory / f"seq{sequence_index}" / "plate.png"
        if reset or not plate.is_file():
            _write_png(plate, render_scenario_plate(sequence["scenario"], sequence.get("placements")))

        def finished(local: int, result: dict, sequence_index: int = sequence_index, offset: int = offset) -> None:
            index = offset + local
            _write_png(directory / "combos" / str(index) / "preview.png", result["preview"])
            current = _load_manifest(directory)
            current["combos"][index].update(_public_combo(result, sequence_index))
            _write_manifest(directory, current)
            on_combo(index, {**result, "index": index, "sequence_index": sequence_index})

        only = None if combo_index is None else {combo_index - offset}
        try:
            outcome = run_sequence(sequence, root, only=only, on_combo=finished, stop=stop)
        except Exception as exc:
            for local in sorted(only or range(count)):
                on_combo(offset + local, {
                    "index": offset + local,
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue
        current = _load_manifest(directory)
        current["sequences"][sequence_index].update({
            "renderer": outcome["renderer"],
            "physical_backend": outcome["physical_backend"],
            "engine_message": _engine_message(outcome),
        })
        _write_manifest(directory, current)
        if outcome["stopped"]:
            return "cancelled"
    return "done"


def _previous_sequence(manifest: dict, index: int) -> dict:
    previous = (manifest.get("sequences") or [])
    if index < len(previous):
        return {key: previous[index].get(key) for key in ("renderer", "physical_backend", "engine_message")}
    return {}


def _engine_message(outcome: dict) -> str:
    if outcome["renderer"] == "blender":
        return ""
    reason = (outcome.get("physical_field_source") or {}).get("reason")
    return f"引擎没有产出画面，用的是物理场投影：{reason}" if reason else "引擎没有产出画面，用的是物理场投影"


def _public_combo(result: dict, sequence_index: int) -> dict:
    return {
        "sequence_index": sequence_index,
        "sequence_id": result["sequence_id"],
        "frames": result["frames"],
        "kept": result["kept"],
        "dropped": result["dropped"],
        "samples": plain(result["samples"]),
        "video": result["video"],
        "camera_fallback": bool((result.get("camera") or {}).get("fallback")),
        "contributors": plain(result["contributors"]),
        "ssim_to_render": result["ssim_to_render"],
    }


def _combo_slots(existing: list[dict], sequences: list[dict], reset: bool) -> list[dict]:
    slots = []
    index = 0
    for sequence_index, sequence in enumerate(sequences):
        for combo in combo_list(sequence):
            previous = existing[index] if index < len(existing) and not reset else {}
            slot = {
                "index": index,
                "sequence_index": sequence_index,
                "scene": combo["scene"],
                "weather": combo["weather"],
                "viewpoint": combo["viewpoint"],
                "key": combo["key"],
            }
            for key in ("sequence_id", "frames", "kept", "dropped", "samples", "video", "camera_fallback", "contributors", "ssim_to_render"):
                if key in previous:
                    slot[key] = previous[key]
            slots.append(slot)
            index += 1
    return slots


def stage_view(root: Path | str, job_id: str) -> dict:
    """Public description of a job's plate, renderer outcome and combinations."""
    directory = run_dir(root, job_id)
    manifest = _load_manifest(directory) if (directory / "manifest.json").is_file() else {}
    sequences = manifest.get("sequences") or []
    combos = []
    for combo in manifest.get("combos") or []:
        sequence_id = combo.get("sequence_id")
        video = combo.get("video")
        combos.append({
            **combo,
            "preview_url": _artifact_url(job_id, f"combos/{combo['index']}/preview.png", directory / "combos" / str(combo["index"]) / "preview.png"),
            "video_url": f"/api/v1/sequences/{sequence_id}/files/{video}" if sequence_id and video else None,
        })
    return {
        "staged": bool(manifest),
        "sequences": [
            {
                **sequence,
                "plate_url": _artifact_url(job_id, f"seq{sequence['index']}/plate.png", directory / f"seq{sequence['index']}" / "plate.png"),
            }
            for sequence in sequences
        ],
        "plate_url": _artifact_url(job_id, "seq0/plate.png", directory / "seq0" / "plate.png"),
        "combos": combos,
    }


def artifact_file(root: Path | str, job_id: str, name: str) -> Path | None:
    directory = run_dir(root, job_id).resolve()
    target = (directory / name).resolve()
    if target != directory and directory not in target.parents:
        raise ValueError("invalid stage artifact")
    if not target.is_file():
        return None
    return target


def _load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        return {"sequences": [], "combos": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(directory: Path, manifest: dict) -> None:
    (directory / "manifest.json").write_text(json.dumps(plain(manifest), ensure_ascii=False), encoding="utf-8")


def _write_png(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(to_uint8(image)))


def _artifact_url(job_id: str, name: str, path: Path) -> str | None:
    if not path.is_file():
        return None
    stamp = int(path.stat().st_mtime)
    return f"/api/v1/jobs/{job_id}/artifacts/{name}?v={stamp}"
