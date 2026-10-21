"""Offline acceptance metrics for physics, annotation, and detector outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute IoU for ``[x, y, width, height]`` boxes."""
    x1, y1, w1, h1 = (float(value) for value in left)
    x2, y2, w2, h2 = (float(value) for value in right)
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + max(w1, 0.0), x2 + max(w2, 0.0)), min(y1 + max(h1, 0.0), y2 + max(h2, 0.0))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(w1, 0.0) * max(h1, 0.0) + max(w2, 0.0) * max(h2, 0.0) - intersection
    return intersection / union if union else 0.0


def annotation_accuracy(predictions: Iterable[Mapping], ground_truth: Iterable[Mapping], iou_threshold: float = 0.5) -> dict:
    """Measure precision, recall, and F1 for flame boxes and categories."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    predictions = list(predictions)
    ground_truth = list(ground_truth)
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must contain the same number of images")
    true_positive = false_positive = false_negative = 0
    for predicted_image, truth_image in zip(predictions, ground_truth):
        truth = list(truth_image.get("objects", truth_image.get("boxes", [])))
        predicted = list(predicted_image.get("objects", predicted_image.get("boxes", [])))
        used: set[int] = set()
        for candidate in predicted:
            candidate_box = candidate.get("bbox", candidate) if isinstance(candidate, Mapping) else candidate
            candidate_category = candidate.get("category") if isinstance(candidate, Mapping) else None
            match = None
            for index, target in enumerate(truth):
                if index in used:
                    continue
                target_box = target.get("bbox", target) if isinstance(target, Mapping) else target
                target_category = target.get("category") if isinstance(target, Mapping) else None
                if candidate_category is not None and target_category is not None and candidate_category != target_category:
                    continue
                if box_iou(candidate_box, target_box) >= iou_threshold:
                    match = index
                    break
            if match is None:
                false_positive += 1
            else:
                used.add(match)
                true_positive += 1
        false_negative += len(truth) - len(used)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": float(iou_threshold),
    }


def _average_precision(scores: list[tuple[float, bool]], positives: int) -> float:
    if positives <= 0:
        return 0.0
    scores.sort(key=lambda row: row[0], reverse=True)
    true_positive = false_positive = 0
    precision_recall = []
    for _score, matched in scores:
        true_positive += int(matched)
        false_positive += int(not matched)
        precision_recall.append((true_positive / (true_positive + false_positive), true_positive / positives))
    precision_recall.sort(key=lambda row: row[1])
    previous_recall = 0.0
    running_precision = 0.0
    area = 0.0
    for precision, recall in precision_recall:
        running_precision = max(running_precision, precision)
        area += running_precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return float(area)


def detection_map(predictions: Iterable[Mapping], ground_truth: Iterable[Mapping], iou_threshold: float = 0.5) -> dict:
    """Compute one-to-one detection mAP at an IoU threshold.

    Each image has ``boxes``, ``labels`` and optional ``scores``. Predictions
    are ranked per category, making the result suitable for comparing a
    detector trained with real-only and synthetic-augmented data.
    """
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    predictions = list(predictions)
    ground_truth = list(ground_truth)
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must contain the same number of images")
    ranked: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    positives: dict[str, int] = defaultdict(int)
    for predicted_image, truth_image in zip(predictions, ground_truth):
        truth_boxes = list(truth_image.get("boxes", []))
        truth_labels = list(truth_image.get("labels", ["flame"] * len(truth_boxes)))
        predicted_boxes = list(predicted_image.get("boxes", []))
        predicted_labels = list(predicted_image.get("labels", ["flame"] * len(predicted_boxes)))
        scores = list(predicted_image.get("scores", [1.0] * len(predicted_boxes)))
        if len(truth_labels) != len(truth_boxes):
            raise ValueError("each ground-truth image must have one label per box")
        if len(predicted_labels) != len(predicted_boxes):
            raise ValueError("each prediction image must have one label per box")
        if len(scores) != len(predicted_boxes):
            raise ValueError("each prediction image must have one score per box")
        used: set[int] = set()
        for label in truth_labels:
            positives[str(label)] += 1
        order = np.argsort(-np.asarray(scores, dtype=np.float64))
        for index in order:
            label = str(predicted_labels[int(index)])
            best = 0.0
            best_truth = None
            for truth_index, (truth_box, truth_label) in enumerate(zip(truth_boxes, truth_labels)):
                if truth_index in used or str(truth_label) != label:
                    continue
                overlap = box_iou(predicted_boxes[int(index)], truth_box)
                if overlap > best:
                    best, best_truth = overlap, truth_index
            matched = best_truth is not None and best >= iou_threshold
            if matched:
                used.add(best_truth)
            ranked[label].append((float(scores[int(index)]), matched))
    average_precisions = {
        label: _average_precision(ranked.get(label, []), positives[label])
        for label in sorted(positives)
        if positives[label] > 0
    }
    map_value = float(np.mean(list(average_precisions.values()))) if average_precisions else 0.0
    return {
        "map": map_value,
        "average_precision": average_precisions,
        "classes": sorted(average_precisions),
        "iou_threshold": float(iou_threshold),
        "ground_truth_count": int(sum(positives.values())),
    }


def physics_error(reference: Mapping[str, np.ndarray], predicted: Mapping[str, np.ndarray]) -> dict:
    """Return normalized MAE percentages for physical fields."""
    report = {}
    for name in ("temperature", "smoke", "flow"):
        expected = np.asarray(reference[name], dtype=np.float64)
        actual = np.asarray(predicted[name], dtype=np.float64)
        if expected.shape != actual.shape:
            raise ValueError(f"shape mismatch for {name}: {expected.shape} vs {actual.shape}")
        denominator = float(np.mean(np.abs(expected))) or 1.0
        report[name] = float(np.mean(np.abs(actual - expected)) / denominator * 100.0)
    report["mean_percent"] = float(np.mean(list(report.values())))
    return report
