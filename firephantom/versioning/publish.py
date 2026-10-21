"""Publish a dataset as a cumulative, immutable DVC version.

A version lists every member, not only what changed. Its members are the
parent's members (the dataset head) that are still in the dataset and still
reviewed, plus reviewed members that no version of this dataset has
contained yet. So rolling the head back to an older version and publishing
again drops what later versions added, while keeping work done since.

Each member keeps the split it got the first time it was published; frames
of one sequence share a split. FID and IS describe the whole version against
the configured real reference set.
"""

from __future__ import annotations

import json
from pathlib import Path

from firephantom.db import connect
from firephantom.pipeline import SCENARIOS, SCENES, WEATHERS
from firephantom.versioning.exporters import SPLITS, export_dataset, image_path
from firephantom.versioning.samples import SampleCatalog, annotation_error_rate
from firephantom.versioning.store import DatasetStore

_FLAME_LABELS = ("[0,0.05)", "[0.05,0.15)", "[0.15,0.35)", "[0.35,1]")
_METRIC_SAMPLE_LIMIT = 256


def version_members(version_id: str) -> dict[str, str | None]:
    with connect() as db:
        rows = db.execute("SELECT sample_id, split FROM version_samples WHERE version_id=?", (version_id,)).fetchall()
    return {row["sample_id"]: row["split"] for row in rows}


def plan_members(root: Path | str, dataset_id: str) -> dict:
    """What publishing this dataset now would contain, relative to its head."""
    with connect() as db:
        row = db.execute("SELECT id, name, head_version_id FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown dataset {dataset_id}")
    dataset = {"dataset_id": row["id"], "name": row["name"], "head_version_id": row["head_version_id"]}
    parent_id = dataset["head_version_id"]
    parent = set(version_members(parent_id)) if parent_id else set()
    with connect() as db:
        reviewed = {
            row["sample_id"]
            for row in db.execute(
                """SELECT samples.sample_id FROM dataset_samples
                   JOIN samples ON samples.sample_id = dataset_samples.sample_id
                   WHERE dataset_samples.dataset_id=? AND samples.status='reviewed'""",
                (dataset_id,),
            ).fetchall()
        }
        ever = {
            row["sample_id"]
            for row in db.execute(
                """SELECT DISTINCT version_samples.sample_id FROM version_samples
                   JOIN versions ON versions.version_id = version_samples.version_id
                   WHERE versions.dataset_id=?""",
                (dataset_id,),
            ).fetchall()
        }
    members = (parent & reviewed) | (reviewed - ever)
    return {
        "dataset": dataset,
        "parent_id": parent_id,
        "members": sorted(members),
        "added": sorted(members - parent),
        "removed": sorted(parent - members),
    }


def publish_dataset(root: Path | str, dataset_id: str) -> dict:
    """Freeze the dataset's members as a new version on top of its head."""
    from firephantom.versioning.datasets import DatasetCatalog

    plan = plan_members(root, dataset_id)
    if not plan["members"]:
        raise ValueError("没有可发布的已审核样本")
    if plan["parent_id"] and not plan["added"] and not plan["removed"]:
        raise ValueError("和当前版本相比没有变化")
    catalog = SampleCatalog(root)
    splits = catalog.assign_splits(plan["members"])
    samples = []
    for sample_id in plan["members"]:
        sample = catalog.get(sample_id)
        if sample is None:
            raise KeyError(f"unknown sample {sample_id}")
        if sample["reviewed_objects"] is None:
            raise ValueError(f"样本 {sample_id} 没有审核标注")
        samples.append({**sample, "split": splits[sample_id]})
    files, metadata = _bundle(root, samples, plan)
    record = DatasetStore(root).add(files, metadata, parent_id=plan["parent_id"], dataset_id=dataset_id)
    with connect() as db:
        db.executemany(
            """INSERT INTO version_samples(version_id, sample_id, split) VALUES(?,?,?)
               ON CONFLICT (version_id, sample_id) DO UPDATE SET split=EXCLUDED.split""",
            [(record["version_id"], sample["sample_id"], sample["split"]) for sample in samples],
        )
    DatasetCatalog(root).set_head(dataset_id, record["version_id"])
    return record


def _bundle(root: Path | str, samples: list[dict], plan: dict) -> tuple[dict, dict]:
    statistics = dataset_statistics(samples)
    error_rate = annotation_error_rate(
        [sample["auto_objects"] for sample in samples],
        [sample["reviewed_objects"] for sample in samples],
    )
    base = Path(root)
    files: dict = {}
    export_items = []
    images = []
    for sample in samples:
        sample_id = sample["sample_id"]
        source = base / "samples" / sample_id / "image.png"
        if not source.is_file():
            raise ValueError(f"样本 {sample_id} 缺少 image.png")
        width, height = _image_size(sample)
        files[image_path(sample)] = source
        images.append(source)
        export_items.append({
            "sample_id": sample_id,
            "split": sample["split"],
            "objects": sample["reviewed_objects"],
            "width": width,
            "height": height,
        })
    files.update(export_dataset(export_items))
    quality = version_quality(images)
    manifest = {
        "parent_version": plan["parent_id"],
        "changes": {"added": plan["added"], "removed": plan["removed"]},
        "samples": [_manifest_entry(sample) for sample in samples],
    }
    files["manifest.json"] = _json_bytes(manifest)
    files["statistics.json"] = _json_bytes(statistics)
    files["quality_report.json"] = _json_bytes(quality)
    metadata = {
        "类型": "dataset",
        "数据集": plan["dataset"]["name"],
        "父版本": plan["parent_id"],
        "变化": {"added": len(plan["added"]), "removed": len(plan["removed"])},
        "样本清单": [sample["sample_id"] for sample in samples],
        "数据统计": statistics,
        "质量报告": quality,
        "标注错误率": round(error_rate, 8),
        "生成器": {"versions": sorted({sample["generator_version"] for sample in samples})},
    }
    return files, metadata


def version_quality(images: list[Path]) -> dict:
    """FID and IS of the version against the real reference set, when one is configured."""
    from firephantom.images import decode_png
    from firephantom.quality.metrics import (
        frechet_inception_distance,
        inception_backend,
        inception_score,
        reference_images_from_env,
    )

    step = max(1, len(images) // _METRIC_SAMPLE_LIMIT)
    chosen = images[::step][:_METRIC_SAMPLE_LIMIT]
    body = {
        "method": "version_set",
        "count": len(images),
        "measured": len(chosen),
        "inception_backend": inception_backend(),
        "fid": None,
        "is": None,
    }
    if len(chosen) < 2:
        body["reason"] = "FID 和 IS 至少需要两张图"
        return body
    pixels = [decode_png(path.read_bytes()) for path in chosen]
    body["is"] = round(float(inception_score(pixels)), 8)
    reference = reference_images_from_env()
    if reference is None:
        body["reason"] = "没有配置 FIRE_REFERENCE_DATASET，FID 不计算"
    else:
        body["fid"] = round(float(frechet_inception_distance(reference, pixels)), 8)
        body["fid_reference_count"] = len(reference)
    return body


def dataset_statistics(samples: list[dict]) -> dict:
    """Counts, flame-size bins, weather/scene ratios and the split for one version."""
    total = len(samples)
    boxes = 0
    bin_counts = [0, 0, 0, 0]
    weather_counts = {name: 0 for name in WEATHERS}
    scene_counts = {name: 0 for name in SCENES}
    scenario_counts = {name: 0 for name in SCENARIOS}
    view_counts: dict[str, int] = {}
    split_counts = {name: 0 for name in SPLITS}
    sequences = set()
    sd_derived = 0
    for sample in samples:
        params = sample["scene_params"]
        if params.get("diffusion"):
            sd_derived += 1
        for counts, key in (
            (weather_counts, params.get("weather")),
            (scene_counts, params.get("scene")),
            (scenario_counts, params.get("scenario")),
            (view_counts, params.get("viewpoint")),
            (split_counts, sample.get("split")),
        ):
            counts[key] = counts.get(key, 0) + 1
        sequences.add(sample.get("sequence_id") or sample["sample_id"])
        for obj in sample["reviewed_objects"] or []:
            boxes += 1
            area = float(obj["image_width"]) * float(obj["image_height"])
            ratio = (float(obj["bbox"][2]) * float(obj["bbox"][3]) / area) if area else 0.0
            bin_counts[_flame_bin(ratio)] += 1
    return {
        "数据量": {"samples": total, "images": total, "boxes": boxes, "sequences": len(sequences)},
        "火焰尺寸分布": {
            "bins": [
                {"label": label, "count": count, "ratio": _ratio(count, boxes)}
                for label, count in zip(_FLAME_LABELS, bin_counts)
            ]
        },
        "天气比例": _share(weather_counts, total),
        "场景比例": _share(scene_counts, total),
        "火灾场景比例": _share(scenario_counts, total),
        "视角比例": _share(view_counts, total),
        "划分": _share(split_counts, total),
        "SD 派生": {"count": sd_derived, "ratio": _ratio(sd_derived, total)},
    }


def _manifest_entry(sample: dict) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "sequence_id": sample.get("sequence_id"),
        "derived_from": sample.get("derived_from"),
        "seed": sample["seed"],
        "generator_version": sample["generator_version"],
        "generator_profile": sample["generator_profile"],
        "scene_params": sample["scene_params"],
        "image": image_path(sample),
        "labels": f"labels/{sample['split']}/{sample['sample_id']}.txt",
        "annotation": sample["reviewed_objects"],
        "sample_store": {
            "mask": f"samples/{sample['sample_id']}/mask.png",
            "provenance": f"samples/{sample['sample_id']}/provenance.json",
            "sequence": f"sequences/{sample.get('sequence_id')}/" if sample.get("sequence_id") else None,
        },
    }


def _image_size(sample: dict) -> tuple[int, int]:
    for group in (sample["reviewed_objects"], sample["auto_objects"]):
        if group:
            return int(group[0]["image_width"]), int(group[0]["image_height"])
    from firephantom.synthesis.render import IMAGE_SIZE

    return IMAGE_SIZE, IMAGE_SIZE


def _flame_bin(ratio: float) -> int:
    if ratio < 0.05:
        return 0
    if ratio < 0.15:
        return 1
    if ratio < 0.35:
        return 2
    return 3


def _share(counts: dict, total: int) -> dict[str, dict]:
    return {name: {"count": count, "ratio": _ratio(count, total)} for name, count in counts.items() if name is not None}


def _ratio(count: int, total: int) -> float:
    return round(count / total, 8) if total else 0.0


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
