"""Derive flame boxes, outlines, keypoints and attributes for one frame.

Rendered frames carry an exact flame mask from the renderer, so the labels
come from that mask: one object per connected flame region. A frame without
a mask (an imported image) falls back to a local YOLO model when
``ANNOTATION_MODEL`` is set, and to a color threshold otherwise. ``backend``
says which one produced the labels.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np

from firephantom.annotation.classes import canonical
from firephantom.runtime import default_torch_device
from firephantom.annotation.labelstudio import LABEL_CONFIG, build_task
from firephantom.images import to_uint8

# Flame tongues of one fire flicker apart by a few pixels; join them first.
_JOIN_PIXELS = 5
_MIN_REGION_PIXELS = 12


@lru_cache(maxsize=2)
def _yolo_model(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def _keypoints(xs: np.ndarray, ys: np.ndarray) -> list[dict]:
    bottom = ys.max()
    top = ys.min()
    return [
        {"name": "flame_base", "x": float(xs[ys == bottom].mean()), "y": float(bottom)},
        {"name": "flame_center", "x": float(xs.mean()), "y": float(ys.mean())},
        {"name": "plume_tip", "x": float(xs[ys == top].mean()), "y": float(top)},
    ]


def objects_from_mask(
    mask: np.ndarray,
    category: str = "flame",
    *,
    keypoints: bool = True,
    min_pixels: int = _MIN_REGION_PIXELS,
) -> list[dict]:
    """One object per connected region of the mask, largest first."""
    flame = np.asarray(mask, dtype=np.uint8)
    if int(flame.sum()) == 0:
        return []
    size = 2 * _JOIN_PIXELS + 1
    joined = cv2.dilate(flame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    count, labels = cv2.connectedComponents(joined, connectivity=8)
    objects = []
    for label in range(1, count):
        region = (labels == label) & (flame > 0)
        ys, xs = np.nonzero(region)
        if xs.size < min_pixels:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(outline, 1.0, True).reshape(-1, 2)
        objects.append({
            "category": category,
            "bbox": [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)],
            "area": int(xs.size),
            "polygon": [float(value) for point in polygon for value in point],
            "keypoints": _keypoints(xs, ys) if keypoints else [],
        })
    objects.sort(key=lambda item: item["area"], reverse=True)
    return objects


def _model_objects(image: np.ndarray) -> list[dict] | None:
    configured = os.environ.get("ANNOTATION_MODEL", "").strip()
    if not configured or not Path(configured).is_file():
        return None
    try:
        result = _yolo_model(configured).predict(
            source=image,
            verbose=False,
            device=default_torch_device(),
        )[0]
    except Exception:
        return None
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    objects = []
    for xyxy, class_index, confidence in zip(
        boxes.xyxy.detach().cpu().numpy(),
        boxes.cls.detach().cpu().numpy().astype(int),
        boxes.conf.detach().cpu().numpy(),
    ):
        try:
            category = canonical(result.names.get(class_index, class_index))
        except ValueError:
            continue
        x1, y1, x2, y2 = map(float, xyxy)
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        objects.append({
            "category": category,
            "bbox": [x1, y1, width, height],
            "confidence": float(confidence),
            "keypoints": [
                {"name": "flame_base", "x": x1 + width / 2.0, "y": y2},
                {"name": "flame_center", "x": x1 + width / 2.0, "y": y1 + height / 2.0},
                {"name": "plume_tip", "x": x1 + width / 2.0, "y": y1},
            ],
        })
    return objects


def _color_mask(image: np.ndarray) -> np.ndarray:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    return (red > 150) & (red > green + 10)


def _attributes(context: dict, temperature_max: float, smoke_mean: float) -> list[str]:
    wind = float(context.get("wind_speed", 0.0))
    labels = [
        str(context.get("scene", "")),
        str(context.get("weather", "")),
        "明火" if temperature_max >= 150.0 else "阴燃",
        "浓烟" if smoke_mean >= 0.08 else "淡烟",
        "高风速" if abs(wind) >= 3.0 else "低风速",
    ]
    return [label for label in labels if label]


def annotate(
    image: np.ndarray,
    mask: np.ndarray | None,
    physical: dict,
    context: dict | None = None,
    sample_id: str | None = None,
    smoke_mask: np.ndarray | None = None,
    ember_mask: np.ndarray | None = None,
) -> dict:
    """Labels for one frame, wrapped as a Label Studio task.

    The task's image is the sample file route; the dataset admit step turns
    it into an absolute URL Label Studio can load. Flame comes from the flame
    mask. Smoke and ember masks add those classes and do not get flame keypoints.
    """
    rgb = to_uint8(image)
    height, width = rgb.shape[:2]
    context = dict(context or {})
    if mask is not None:
        objects = objects_from_mask(mask)
        backend = "render-mask"
    else:
        objects = _model_objects(rgb)
        backend = "local-yolo"
        if objects is None:
            objects = objects_from_mask(_color_mask(rgb))
            backend = "color-threshold"
    if smoke_mask is not None:
        objects.extend(objects_from_mask(smoke_mask, "smoke", keypoints=False))
    if ember_mask is not None:
        objects.extend(objects_from_mask(ember_mask, "ember", keypoints=False))
    attributes = _attributes(
        context,
        float(np.max(physical["temperature"])),
        float(np.mean(physical["smoke"])),
    )
    if any(item["category"] == "smoke" for item in objects):
        attributes.append("可见烟羽")
    if any(item["category"] == "ember" for item in objects):
        attributes.append("飞火")
    for obj in objects:
        obj.update({
            "attributes": attributes,
            "image_width": width,
            "image_height": height,
            "image_file": "image.png",
        })
    image_path = f"/api/v1/samples/{sample_id}/files/image.png" if sample_id else "image.png"
    task = build_task(image_path, width, height, objects, sample_id=sample_id, attributes=attributes)
    return {
        "objects": objects,
        "label_studio": task,
        "label_config": LABEL_CONFIG,
        "backend": backend,
    }
