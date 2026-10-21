"""Compare two published dataset versions."""

from __future__ import annotations


_DIFF_LIMIT = 500


def member_diff(base: dict[str, str | None], other: dict[str, str | None]) -> dict:
    """Samples added, removed, or moved to another split between two member maps."""
    added = sorted(set(other) - set(base))
    removed = sorted(set(base) - set(other))
    moved = sorted(
        sample_id for sample_id in set(base) & set(other)
        if base[sample_id] and other[sample_id] and base[sample_id] != other[sample_id]
    )
    return {
        "counts": {"added": len(added), "removed": len(removed), "split_changed": len(moved)},
        "added": added[:_DIFF_LIMIT],
        "removed": removed[:_DIFF_LIMIT],
        "split_changed": moved[:_DIFF_LIMIT],
        "truncated": max(len(added), len(removed), len(moved)) > _DIFF_LIMIT,
    }


def compare_versions(
    base: dict,
    other: dict,
    base_detection: dict | None,
    other_detection: dict | None,
    base_members: dict[str, str | None] | None = None,
    other_members: dict[str, str | None] | None = None,
) -> dict:
    """Return volume, flame size, scene mix, annotation error, detection and the sample diff."""
    if base.get("metadata", {}).get("类型") != "dataset" or other.get("metadata", {}).get("类型") != "dataset":
        raise ValueError("只能对比已发布的数据集版本")
    base_stats = base["metadata"]["数据统计"]
    other_stats = other["metadata"]["数据统计"]
    base_error = float(base["metadata"]["标注错误率"])
    other_error = float(other["metadata"]["标注错误率"])
    return {
        "base": _identity(base),
        "other": _identity(other),
        "data_volume": {
            "base": base_stats["数据量"],
            "other": other_stats["数据量"],
            "delta": {
                key: other_stats["数据量"][key] - base_stats["数据量"][key]
                for key in ("samples", "images", "boxes")
            },
        },
        "flame_size": {
            "base": base_stats["火焰尺寸分布"]["bins"],
            "other": other_stats["火焰尺寸分布"]["bins"],
            "delta": _bin_delta(
                base_stats["火焰尺寸分布"]["bins"],
                other_stats["火焰尺寸分布"]["bins"],
            ),
        },
        "scene_weather": {
            "base": {
                "天气比例": base_stats["天气比例"],
                "场景比例": base_stats["场景比例"],
                "火灾场景比例": base_stats["火灾场景比例"],
            },
            "other": {
                "天气比例": other_stats["天气比例"],
                "场景比例": other_stats["场景比例"],
                "火灾场景比例": other_stats["火灾场景比例"],
            },
        },
        "annotation_error_rate": {
            "base": base_error,
            "other": other_error,
            "delta": round(other_error - base_error, 8),
        },
        "detection": {"base": base_detection, "other": other_detection},
        "samples": member_diff(base_members or {}, other_members or {}),
    }


def _identity(record: dict) -> dict:
    return {"version_id": record["version_id"], "版本号": record.get("版本号")}


def _bin_delta(left: list[dict], right: list[dict]) -> list[dict]:
    right_by_label = {item["label"]: item for item in right}
    delta = []
    for item in left:
        match = right_by_label[item["label"]]
        delta.append(
            {
                "label": item["label"],
                "count": match["count"] - item["count"],
                "ratio": round(match["ratio"] - item["ratio"], 8),
            }
        )
    return delta
