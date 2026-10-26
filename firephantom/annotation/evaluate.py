"""Tune detector confidence on validation data and score a held-out test split."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import time

from firephantom.annotation.train import available_memory_bytes, normalize_data_yaml
from firephantom.evaluation import annotation_accuracy

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_name(class_id: int, names: dict[int, str]) -> str:
    return names.get(class_id, str(class_id)).strip().lower()


def _stratum(class_ids: set[int]) -> str:
    if not class_ids:
        return "background"
    if class_ids == {0}:
        return "fire"
    if class_ids == {1}:
        return "smoke"
    if class_ids == {0, 1}:
        return "both"
    return "other"


def sample_strata(items: list[tuple[Path, Path, set[int]]], per_stratum: int, seed: int) -> list[tuple[Path, Path, set[int]]]:
    grouped: dict[str, list[tuple[Path, Path, set[int]]]] = {}
    for item in items:
        grouped.setdefault(_stratum(item[2]), []).append(item)
    required = ("fire", "smoke", "both", "background")
    missing = {name: len(grouped.get(name, [])) for name in required if len(grouped.get(name, [])) < per_stratum}
    if missing:
        raise ValueError(f"not enough images in required strata: {missing}")
    rng = random.Random(seed)
    sampled = []
    for name in required:
        sampled.extend(rng.sample(sorted(grouped[name], key=lambda item: str(item[0])), per_stratum))
    return sampled


def _split_items(images_dir: Path, names: dict[int, str]) -> list[tuple[Path, Path, set[int]]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image split does not exist: {images_dir}")
    labels_dir = images_dir.parent / "labels"
    items = []
    for image in sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES):
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        class_ids: set[int] = set()
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                columns = line.split()
                if len(columns) >= 5:
                    class_ids.add(int(float(columns[0])))
        items.append((image, label, class_ids))
    for item in items:
        unknown = item[2] - names.keys()
        if unknown:
            raise ValueError(f"unknown class IDs {sorted(unknown)} in {item[1]}")
    return items


def _objects_for_labels(label: Path, width: int, height: int, names: dict[int, str]) -> list[dict]:
    if not label.is_file():
        return []
    objects = []
    for line in label.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) < 5:
            continue
        class_id, center_x, center_y, box_width, box_height = map(float, columns[:5])
        box_width *= width
        box_height *= height
        objects.append({
            "category": _class_name(int(class_id), names),
            "bbox": [center_x * width - box_width / 2, center_y * height - box_height / 2, box_width, box_height],
        })
    return objects


def _predict(model, items, names: dict[int, str], *, imgsz: int, batch: int, device: str, min_confidence: float):
    from PIL import Image

    predictions = []
    ground_truth = []
    chunk_size = max(1, min(int(batch), 4))
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        results = model.predict(
            source=[str(image) for image, _label, _class_ids in chunk],
            imgsz=imgsz,
            batch=chunk_size,
            conf=min_confidence,
            device=device,
            verbose=False,
            stream=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"detector returned {len(results)} results for {len(chunk)} images")
        for result, (image, label, _class_ids) in zip(results, chunk):
            with Image.open(image) as source:
                width, height = source.size
            objects = []
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                confidence = boxes.conf.detach().cpu().numpy()
                for coords, class_id, score in zip(xyxy, class_ids, confidence):
                    x1, y1, x2, y2 = map(float, coords)
                    objects.append({
                        "category": _class_name(int(class_id), names),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "confidence": float(score),
                    })
            predictions.append({"objects": objects})
            ground_truth.append({"objects": _objects_for_labels(label, width, height, names)})
        del results
        gc.collect()
        try:
            import torch

            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass
    return predictions, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="YOLO data.yaml path")
    parser.add_argument("weights", type=Path, help="detector checkpoint")
    parser.add_argument("output", type=Path, help="JSON report path")
    parser.add_argument("--samples-per-stratum", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--confidence-values", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None, help="torch device; defaults to mps, cuda, or cpu")
    parser.add_argument("--min-available-gb", type=float, default=3.0)
    parser.add_argument("--allow-low-memory", action="store_true", help="run despite the host memory guard")
    args = parser.parse_args()
    if not args.device:
        from firephantom.runtime import default_torch_device

        args.device = default_torch_device()
    if args.samples_per_stratum < 1:
        parser.error("--samples-per-stratum must be positive")
    available = available_memory_bytes()
    required = int(max(0.0, args.min_available_gb) * 1024**3)
    if available is not None and available < required and not args.allow_low_memory:
        parser.error(
            f"only {available / 1024**3:.2f} GiB appears available; "
            f"need at least {args.min_available_gb:.1f} GiB. Stop other model services or pass --allow-low-memory."
        )

    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized_yaml = normalize_data_yaml(args.dataset, args.output.with_suffix(".data.yaml"))
    import yaml

    config = yaml.safe_load(normalized_yaml.read_text(encoding="utf-8"))
    class_names = config.get("names", {})
    names = {int(key): str(value) for key, value in class_names.items()} if isinstance(class_names, dict) else dict(enumerate(map(str, class_names)))
    if set(names) < {0, 1}:
        raise ValueError("dataset must define fire (class 0) and smoke (class 1)")

    confidence_values = sorted({float(value) for value in args.confidence_values.split(",")})
    if not confidence_values or confidence_values[0] < 0 or confidence_values[-1] > 1:
        parser.error("confidence values must be between 0 and 1")
    minimum_confidence = max(0.001, min(confidence_values))

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    split_results = {}
    chosen_confidence = None
    for split_name, seed_offset in (("val", 0), ("test", 1)):
        items = _split_items(Path(config[split_name]), names)
        sample = sample_strata(items, args.samples_per_stratum, args.seed + seed_offset)
        predictions, ground_truth = _predict(
            model,
            sample,
            names,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            min_confidence=minimum_confidence,
        )
        if split_name == "val":
            metrics_by_confidence = {}
            for threshold in confidence_values:
                filtered = [
                    {"objects": [obj for obj in image["objects"] if obj["confidence"] >= threshold]}
                    for image in predictions
                ]
                metrics_by_confidence[str(threshold)] = annotation_accuracy(filtered, ground_truth, args.iou)
            chosen_confidence = max(
                confidence_values,
                key=lambda threshold: (
                    metrics_by_confidence[str(threshold)]["f1"],
                    metrics_by_confidence[str(threshold)]["precision"],
                    threshold,
                ),
            )
            split_results["validation"] = {
                "samples": len(sample),
                "strata": dict(Counter(_stratum(item[2]) for item in sample)),
                "metrics_by_confidence": metrics_by_confidence,
                "selected_confidence": chosen_confidence,
                "selected_metrics": metrics_by_confidence[str(chosen_confidence)],
            }
        else:
            filtered = [
                {"objects": [obj for obj in image["objects"] if obj["confidence"] >= chosen_confidence]}
                for image in predictions
            ]
            split_results["test"] = {
                "samples": len(sample),
                "strata": dict(Counter(_stratum(item[2]) for item in sample)),
                "confidence": chosen_confidence,
                "metrics": annotation_accuracy(filtered, ground_truth, args.iou),
            }

    report = {
        "model": str(args.weights.resolve()),
        "dataset": str(args.dataset.resolve()),
        "seed": args.seed,
        "iou_threshold": args.iou,
        "confidence_values": confidence_values,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **split_results,
        "scope": "confidence is tuned on validation only; final annotation F1 is measured on an independent test sample",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
