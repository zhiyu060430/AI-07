"""One sequence: bake a fire once, render every view, store every frame as a sample.

A sequence is one simulated fire (scenario, physics, fuel layout, seed). The
renderer draws it for each (environment, weather) and viewpoint, ``frame_count``
frames each. Every frame becomes a sample with an exact flame mask and the
labels derived from it. Photometric augmentations add derived samples next
to the rendered ones. Stable Diffusion is not called here; it restyles
reviewed samples later (``firephantom.synthesis.sd_augment``). Sequence-level files (physical fields, point cloud,
one video per view, solver inputs and outputs) are written once under
``sequences/<sequence_id>/`` instead of being copied into every sample.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from firephantom import __version__
from firephantom.annotation.annotate import annotate
from firephantom.images import encode_mask_png, encode_png, encode_video, to_uint8, video_encoder
from firephantom.physics.engine import simulate
from firephantom.quality.filter import sample_checks
from firephantom.serialize import plain
from firephantom.synthesis.augment import augment_image
from firephantom.synthesis.render import SCENES, VIEWPOINTS, WEATHERS, build_point_cloud
from firephantom.validation import expand_job_items
from firephantom.versioning.samples import SampleCatalog, make_sample_id

logger = logging.getLogger("firephantom")

SCENARIOS = ("房间火灾", "林区火灾", "工业区火灾")
__all__ = ["SCENARIOS", "SCENES", "VIEWPOINTS", "WEATHERS", "combo_list", "run_sequence", "sequence_id_for"]

_BAKE_KEYS = (
    "scenario", "wind_speed", "temperature", "humidity", "fuel_type",
    "fire_engine", "placements", "seed", "scene_id", "scene_revision",
)


def normalize_sequence(request: dict) -> dict:
    """One sequence request. Lists in the request stay lists; ``count`` must be 1."""
    expanded = expand_job_items([request])
    if len(expanded) != 1:
        raise ValueError("run_sequence takes one sequence; submit a job for repeats")
    return expanded[0]


def combo_list(sequence: dict) -> list[dict]:
    """Rendered combinations in the order the simulator returns them."""
    combos = []
    for look_index, (scene, weather) in enumerate(
        (scene, weather) for scene in sequence["scenes"] for weather in sequence["weathers"]
    ):
        for view_index, viewpoint in enumerate(sequence["viewpoints"]):
            combos.append({
                "scene": scene,
                "weather": weather,
                "viewpoint": viewpoint,
                "key": f"v{view_index}_l{look_index}",
            })
    return combos


def sequence_id_for(sequence: dict) -> str:
    identity = {key: sequence.get(key) for key in _BAKE_KEYS}
    # Only a moved origin, camera, gas box, or a stored lamp list changes the bake.
    # Empty cameras stay out. An empty lamp list is "no lamps", distinct from automatic.
    for key in ("origin", "cameras", "gas_domain"):
        value = sequence.get(key)
        if value:
            identity[key] = value
    if sequence.get("lights") is not None:
        identity["lights"] = sequence["lights"]
    if sequence.get("simulation"):
        identity["simulation"] = sequence["simulation"]
    if sequence.get("structures") is not None:
        identity["structures"] = sequence["structures"]
    if sequence.get("domain"):
        identity["domain"] = sequence["domain"]
    identity["generator_version"] = __version__
    raw = json.dumps(plain(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sequence_dir(root: Path | str, sequence_id: str) -> Path:
    return Path(root) / "sequences" / sequence_id


def simulate_sequence(sequence: dict, root: Path | str) -> dict:
    return simulate(
        sequence["scenario"],
        sequence["wind_speed"],
        sequence["temperature"],
        sequence["humidity"],
        sequence["fuel_type"],
        work_dir=Path(root) / "build",
        fire_engine=sequence["fire_engine"],
        placements=sequence.get("placements") or None,
        viewpoints=sequence["viewpoints"],
        looks=[(scene, weather) for scene in sequence["scenes"] for weather in sequence["weathers"]],
        frame_count=sequence["frame_count"],
        seed=sequence["seed"],
        origin=sequence.get("origin"),
        cameras=sequence.get("cameras") or None,
        gas_domain=sequence.get("gas_domain"),
        lights=sequence.get("lights"),
        simulation=sequence.get("simulation"),
        structures=sequence.get("structures"),
        domain=sequence.get("domain"),
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _points(cloud: np.ndarray) -> list[dict]:
    return [
        {"x": float(x), "y": float(y), "z": float(z), "temperature": float(t), "smoke": float(s)}
        for x, y, z, t, s in np.asarray(cloud, dtype=np.float64)
    ]


def _write_sequence_files(directory: Path, sequence: dict, simulated: dict, sequence_id: str) -> None:
    fields = {
        "temperature": simulated["temperature"],
        "smoke": simulated["smoke"],
        "flow": simulated["flow"],
        "axes": simulated["axes"],
        "parameters": simulated["parameters"],
        "fuel": simulated["fuel"],
        "backend": simulated["physical_backend"],
        "field_source": simulated["physical_field_source"],
        "units": {"temperature": "degC", "smoke": "mass_fraction", "flow": "m/s", "axes": "m"},
    }
    _write(directory / "fields.json", json.dumps(plain(fields), ensure_ascii=False).encode("utf-8"))
    # World coordinates: indoor, clear and side view leave the cloud untransformed.
    cloud = build_point_cloud(simulated["temperature"], simulated["smoke"], simulated["axes"], "室内", "晴", "侧视")
    _write(directory / "point_cloud.json", json.dumps(_points(cloud), ensure_ascii=False).encode("utf-8"))
    _write(directory / "blender_scene.py", simulated["blender_script"].encode("utf-8"))
    provenance = {
        "sequence_id": sequence_id,
        "generator_version": __version__,
        "request": sequence,
        "renderer": simulated["engine_frames_source"],
        "physical_backend": simulated["physical_backend"],
        "physical_field_source": simulated["physical_field_source"],
        "geometry": simulated["geometry"],
        "tool_runs": simulated["tool_runs"],
        "cameras": [render["camera"] for render in simulated["renders"]],
    }
    _write(directory / "provenance.json", json.dumps(plain(provenance), ensure_ascii=False).encode("utf-8"))


# Generation never calls SD. These keep the profile, scene params and
# contributors in the shape that older samples were hashed with, so a re-run
# of the same bake upserts the same sample ids.
_NO_DIFFUSION = {"backend": "disabled"}
_SD_CONTRIBUTOR = {
    "name": "stable_diffusion",
    "role": "background_restyle",
    "called": False,
    "success": False,
    "backend": "disabled",
}


def _profile(simulated: dict) -> dict:
    """Stable generator identity. Runtime success flags are not part of it."""
    return {
        "renderer": simulated["engine_frames_source"],
        "physical_backend": simulated["physical_backend"],
        "annotation": "render-mask",
        "video_encoder": video_encoder(),
        "diffusion": dict(_NO_DIFFUSION),
    }


def _save_sample(
    catalog: SampleCatalog,
    *,
    sequence: dict,
    sequence_id: str,
    simulated: dict,
    combo: dict,
    frame_index: int,
    image: np.ndarray,
    mask: np.ndarray,
    smoke_mask: np.ndarray | None,
    ember_mask: np.ndarray | None,
    profile: dict,
    augmentation: str | None,
    derived_from: str | None,
) -> dict:
    scene_params = {
        **{key: sequence.get(key) for key in _BAKE_KEYS if key != "seed"},
        "scene": combo["scene"],
        "weather": combo["weather"],
        "viewpoint": combo["viewpoint"],
        "frame_index": int(frame_index),
        "diffusion": None,
    }
    augmentations = [augmentation] if augmentation else []
    seed = int(sequence["seed"])
    sample_id = make_sample_id(scene_params, seed, __version__, profile, augmentations)
    checks = sample_checks(image, mask, min_flame_area=sequence["min_flame_area"])
    context = {
        "scene": combo["scene"],
        "weather": combo["weather"],
        "viewpoint": combo["viewpoint"],
        "wind_speed": sequence["wind_speed"],
    }
    annotated = annotate(
        image, mask, simulated, context=context, sample_id=sample_id,
        smoke_mask=smoke_mask, ember_mask=ember_mask,
    )
    quality = {
        "checks": checks,
        "ssim_to_render": None,
        "boxes": len(annotated["objects"]),
    }
    provenance = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "derived_from": derived_from,
        "seed": seed,
        "generator_version": __version__,
        "generator_profile": profile,
        "scene_params": scene_params,
        "augmentations": augmentations,
        "camera": next(
            (render["camera"] for render in simulated["renders"]
             if render["viewpoint"] == combo["viewpoint"]),
            None,
        ),
        "prompt": None,
        "contributors": [dict(_SD_CONTRIBUTOR)],
        "quality": quality,
        "sequence_files": f"sequences/{sequence_id}/",
        "video": f"sequences/{sequence_id}/videos/{combo['key']}.mp4",
    }
    files = {
        "image.png": encode_png(image),
        "mask.png": encode_mask_png(mask),
        "annotation.json": json.dumps(plain(annotated["objects"]), ensure_ascii=False).encode("utf-8"),
        "label_studio.json": json.dumps(plain(annotated["label_studio"]), ensure_ascii=False).encode("utf-8"),
        "quality_report.json": json.dumps(plain(quality), ensure_ascii=False).encode("utf-8"),
        "provenance.json": json.dumps(plain(provenance), ensure_ascii=False).encode("utf-8"),
    }
    saved = catalog.save_generation(
        sample_id=sample_id,
        seed=seed,
        generator_version=__version__,
        generator_profile=profile,
        scene_params=scene_params,
        status="generated" if checks["kept"] else "dropped",
        quality=quality,
        filter_decision=checks,
        auto_objects=plain(annotated["objects"]),
        files=files,
        sequence_id=sequence_id,
        derived_from=derived_from,
    )
    return {
        "sample_id": sample_id,
        "frame_index": int(frame_index),
        "augmentation": augmentation,
        "kept": bool(checks["kept"]),
        "reasons": checks["reasons"],
        "status": saved["status"],
        "boxes": len(annotated["objects"]),
    }


def _run_combo(
    catalog: SampleCatalog,
    root: Path,
    *,
    sequence: dict,
    sequence_id: str,
    simulated: dict,
    combo: dict,
    render: dict,
) -> dict:
    frames = [to_uint8(frame) for frame in render["frames"]]
    masks = [np.asarray(mask, dtype=bool) for mask in render["masks"]]
    smoke_masks = [np.asarray(mask, dtype=bool) for mask in render.get("smoke_masks") or []]
    ember_masks = [np.asarray(mask, dtype=bool) for mask in render.get("ember_masks") or []]
    video = sequence_dir(root, sequence_id) / "videos" / f"{combo['key']}.mp4"
    _write(video, encode_video(frames))
    profile = _profile(simulated)
    samples = []
    for frame_index, frame in enumerate(frames):
        common = {
            "sequence": sequence,
            "sequence_id": sequence_id,
            "simulated": simulated,
            "combo": combo,
            "frame_index": frame_index,
            "mask": masks[frame_index],
            "smoke_mask": None if frame_index >= len(smoke_masks) else smoke_masks[frame_index],
            "ember_mask": None if frame_index >= len(ember_masks) else ember_masks[frame_index],
            "profile": profile,
        }
        base = _save_sample(catalog, image=frame, augmentation=None, derived_from=None, **common)
        samples.append(base)
        for name in sequence["augmentations"]:
            variant = augment_image(frame, [name], seed=int(sequence["seed"]) + frame_index)
            samples.append(
                _save_sample(catalog, image=variant, augmentation=name, derived_from=base["sample_id"], **common)
            )
    return {
        **combo,
        "sequence_id": sequence_id,
        "camera": render["camera"],
        "frames": len(frames),
        "samples": samples,
        "kept": sum(item["kept"] for item in samples),
        "dropped": sum(not item["kept"] for item in samples),
        "video": f"videos/{combo['key']}.mp4",
        "preview": frames[0],
        "contributors": [dict(_SD_CONTRIBUTOR)],
        "ssim_to_render": None,
    }


def run_sequence(
    request: dict,
    dataset_root: Path | str,
    *,
    simulated: dict | None = None,
    only: set[int] | None = None,
    on_combo=None,
    stop=None,
) -> dict:
    """Bake (or reuse ``simulated``), then store the samples of every combination.

    ``only`` limits the combinations by index, ``on_combo(index, result)`` is
    called as each one finishes, and ``stop()`` returning true halts before
    the next one.
    """
    sequence = normalize_sequence(request)
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    if simulated is None:
        simulated = simulate_sequence(sequence, root)
    for tool_run in simulated["tool_runs"]:
        logger.info("%s", tool_run["message"])
    sequence_id = sequence_id_for(sequence)
    _write_sequence_files(sequence_dir(root, sequence_id), sequence, simulated, sequence_id)
    catalog = SampleCatalog(root)
    combos = combo_list(sequence)
    results = []
    stopped = False
    for index, (combo, render) in enumerate(zip(combos, simulated["renders"])):
        if only is not None and index not in only:
            continue
        if stop is not None and stop():
            stopped = True
            break
        result = _run_combo(
            catalog,
            root,
            sequence=sequence,
            sequence_id=sequence_id,
            simulated=simulated,
            combo=combo,
            render=render,
        )
        result["index"] = index
        results.append(result)
        if on_combo is not None:
            on_combo(index, result)
    return {
        "sequence_id": sequence_id,
        "sequence": sequence,
        "renderer": simulated["engine_frames_source"],
        "physical_backend": simulated["physical_backend"],
        "physical_field_source": simulated["physical_field_source"],
        "tool_runs": simulated["tool_runs"],
        "combos": results,
        "stopped": stopped,
    }
