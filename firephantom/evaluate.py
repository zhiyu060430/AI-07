"""Command line evaluator for acceptance manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from firephantom.evaluation import annotation_accuracy, detection_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSON list of detector predictions")
    parser.add_argument("ground_truth", type=Path, help="JSON list of ground-truth objects")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    print(json.dumps({
        "annotation": annotation_accuracy(predictions, ground_truth, args.iou),
        "detection": detection_map(predictions, ground_truth, args.iou),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
