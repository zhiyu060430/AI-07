"""Fine-tune and evaluate a small YOLO detector for fire and smoke labels."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess


def available_memory_bytes() -> int | None:
    """Return an approximate available-memory value on macOS, if observable."""
    if platform.system() != "Darwin":
        return None
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "vm.pagesize"], text=True))
        statistics = subprocess.check_output(["vm_stat"], text=True)
        values = {}
        for line in statistics.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            raw = raw.strip().rstrip(".")
            if raw.isdigit():
                values[key.strip()] = int(raw)
        pages = sum(
            values.get(key, 0)
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        )
        return pages * page_size
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def normalize_data_yaml(dataset_yaml: Path | str, output_yaml: Path | str) -> Path:
    import yaml

    source = Path(dataset_yaml).resolve()
    root = source.parent
    configuration = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError(f"invalid YOLO dataset configuration: {source}")
    for split in ("train", "val", "test"):
        value = configuration.get(split)
        if not isinstance(value, str):
            continue
        raw = Path(value)
        candidates = [raw] if raw.is_absolute() else [root / raw]
        relative = raw
        while relative.parts and relative.parts[0] == "..":
            relative = Path(*relative.parts[1:])
        if relative.parts:
            candidates.append(root / relative)
        candidates.append(root / split / raw.name)
        resolved = next((item.resolve() for item in candidates if item.resolve().exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"YOLO {split} images directory was not found for {value!r} in {source}")
        configuration[split] = str(resolved)
    target = Path(output_yaml)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="YOLO data.yaml path")
    parser.add_argument("output", help="directory for model and evaluation report")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=1.0,
        help="fraction of the independent test split used for evaluation",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument(
        "--min-available-gb",
        type=float,
        default=3.0,
        help="minimum observable macOS available memory before loading the detector",
    )
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="run despite the macOS memory guard; stop manually if swap or memory pressure rises",
    )
    parser.add_argument(
        "--validate-during-training",
        action="store_true",
        help="run the validation split after every epoch; omit for faster local training",
    )
    args = parser.parse_args()

    available = available_memory_bytes()
    required = int(max(0.0, args.min_available_gb) * 1024**3)
    if available is not None and available < required and not args.allow_low_memory:
        parser.error(
            f"only {available / 1024**3:.2f} GiB appears available; "
            f"need at least {args.min_available_gb:.1f} GiB. Stop other MPS services or pass --allow-low-memory."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "ultralytics-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    from ultralytics import YOLO

    fraction = min(1.0, max(0.001, args.fraction))
    test_fraction = min(1.0, max(0.001, args.test_fraction))
    data_path = Path(args.dataset).resolve()
    normalized_data = output / "data.firephantom.yaml"
    normalize_data_yaml(data_path, normalized_data)
    model = YOLO(args.weights)
    trainer_class = None
    if not args.validate_during_training:
        from ultralytics.models.yolo.detect import DetectionTrainer

        class NoEpochValidationTrainer(DetectionTrainer):
            """Keep local training responsive; the explicit test pass still runs below."""

            def validate(self):
                return {}, 0.0

            def final_eval(self):
                return None

        trainer_class = NoEpochValidationTrainer
    trained = model.train(
        trainer=trainer_class,
        data=str(normalized_data),
        epochs=max(1, args.epochs),
        fraction=fraction,
        imgsz=max(128, args.imgsz),
        batch=max(1, args.batch),
        device=args.device,
        workers=0,
        project=str(output),
        name="fire-smoke",
        exist_ok=True,
        plots=False,
        save=True,
        val=args.validate_during_training,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    checkpoint = best if best.is_file() else Path(model.trainer.save_dir) / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training completed without a checkpoint: {checkpoint}")
    model = YOLO(str(checkpoint))
    validation = model.val(
        data=str(normalized_data),
        split="test",
        fraction=test_fraction,
        imgsz=max(128, args.imgsz),
        batch=max(1, args.batch),
        device=args.device,
        workers=0,
        plots=False,
    )
    report = {
        "model": str(checkpoint),
        "classes": validation.names,
        "metrics": {
            "map50": float(validation.box.map50),
            "map50_95": float(validation.box.map),
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
        },
        "dataset": str(Path(args.dataset).resolve()),
        "train_fraction": fraction,
        "test_fraction": test_fraction,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
