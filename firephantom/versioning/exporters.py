"""COCO, Pascal VOC, and YOLO writers with one fixed class table.

Class ids come from ``firephantom.annotation.classes.CLASSES`` and never
depend on which labels a particular export happens to contain. A dataset
export is laid out for training directly: ``images/<split>/``,
``labels/<split>/`` and a ``data.yaml`` that Ultralytics reads as is.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from firephantom.annotation.classes import CLASSES, canonical, class_id

SPLITS = ("train", "val", "test")


def _coco_categories() -> list[dict]:
    return [
        {"id": index + 1, "name": name, "supercategory": "fire", "keypoints": [], "skeleton": []}
        for index, name in enumerate(CLASSES)
    ]


def _coco_annotation(annotation_id: int, image_id: int, obj: dict) -> dict:
    x, y, box_w, box_h = [float(value) for value in obj["bbox"]]
    keypoints: list[float] = []
    for point in obj.get("keypoints", []):
        keypoints.extend([float(point["x"]), float(point["y"]), 2])
    polygon = [float(value) for value in obj.get("polygon") or []]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_id(obj["category"]) + 1,
        "bbox": [x, y, box_w, box_h],
        "area": float(obj.get("area") or box_w * box_h),
        "segmentation": [polygon] if len(polygon) >= 6 else [],
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": len(obj.get("keypoints", [])),
        "attributes": list(obj.get("attributes", [])),
    }


def export_coco(objects: list[dict]) -> dict:
    if not objects:
        raise ValueError("COCO export requires at least one object")
    width = int(objects[0]["image_width"])
    height = int(objects[0]["image_height"])
    file_name = objects[0].get("image_file", "image.png")
    return {
        "info": {"description": "firephantom flame dataset", "version": "2.0"},
        "images": [{"id": 1, "file_name": file_name, "width": width, "height": height}],
        "annotations": [_coco_annotation(index, 1, obj) for index, obj in enumerate(objects, start=1)],
        "categories": _coco_categories(),
    }


def _voc(file_name: str, width: int, height: int, objects: list[dict]) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "firephantom"
    ET.SubElement(root, "filename").text = file_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for obj in objects:
        x, y, box_w, box_h = [float(v) for v in obj["bbox"]]
        node = ET.SubElement(root, "object")
        ET.SubElement(node, "name").text = canonical(obj["category"])
        ET.SubElement(node, "pose").text = "Unspecified"
        ET.SubElement(node, "truncated").text = "1" if x <= 0 or y <= 0 or x + box_w >= width or y + box_h >= height else "0"
        ET.SubElement(node, "difficult").text = "0"
        box = ET.SubElement(node, "bndbox")
        ET.SubElement(box, "xmin").text = repr(x)
        ET.SubElement(box, "ymin").text = repr(y)
        ET.SubElement(box, "xmax").text = repr(x + box_w)
        ET.SubElement(box, "ymax").text = repr(y + box_h)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def export_pascal_voc(objects: list[dict]) -> str:
    if not objects:
        raise ValueError("Pascal VOC export requires at least one object")
    return _voc(
        objects[0].get("image_file", "image.png"),
        int(objects[0]["image_width"]),
        int(objects[0]["image_height"]),
        objects,
    )


def _yolo_lines(objects: list[dict], width: float, height: float) -> str:
    lines = []
    for obj in objects:
        x, y, box_w, box_h = [float(v) for v in obj["bbox"]]
        x_center = (x + box_w / 2.0) / width
        y_center = (y + box_h / 2.0) / height
        lines.append(
            f"{class_id(obj['category'])} {x_center:.8f} {y_center:.8f} {box_w / width:.8f} {box_h / height:.8f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def export_yolo(objects: list[dict]) -> dict:
    if not objects:
        raise ValueError("YOLO export requires at least one object")
    return {
        "labels": _yolo_lines(objects, float(objects[0]["image_width"]), float(objects[0]["image_height"])),
        "classes": "\n".join(CLASSES) + "\n",
    }


def export_all(objects: list[dict]) -> dict:
    return {
        "coco": export_coco(objects),
        "pascal_voc": export_pascal_voc(objects),
        "yolo": export_yolo(objects),
    }


def data_yaml() -> str:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    return (
        "# FirePhantom dataset version. Paths are relative to this file.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )


def image_path(sample: dict) -> str:
    return f"images/{sample['split']}/{sample['sample_id']}.png"


def export_dataset(samples: list[dict]) -> dict[str, bytes]:
    """Annotation files for a split dataset. Images are written by the caller.

    Each sample provides ``sample_id``, ``split``, ``width``, ``height`` and
    ``objects``; an image may have no boxes. YOLO labels sit next to
    ``images/<split>/`` as ``labels/<split>/``; COCO has one file per split.
    """
    files: dict[str, bytes] = {"data.yaml": data_yaml().encode("utf-8")}
    files["labels/classes.txt"] = ("\n".join(CLASSES) + "\n").encode("utf-8")
    for split in SPLITS:
        members = [sample for sample in samples if sample["split"] == split]
        images, annotations = [], []
        annotation_id = 1
        for image_id, sample in enumerate(members, start=1):
            width, height = int(sample["width"]), int(sample["height"])
            name = image_path(sample)
            images.append({"id": image_id, "file_name": name, "width": width, "height": height, "sample_id": sample["sample_id"]})
            for obj in sample["objects"]:
                annotations.append(_coco_annotation(annotation_id, image_id, obj))
                annotation_id += 1
            files[f"labels/{split}/{sample['sample_id']}.txt"] = _yolo_lines(sample["objects"], width, height).encode("utf-8")
            files[f"annotations/voc/{sample['sample_id']}.xml"] = _voc(name, width, height, sample["objects"]).encode("utf-8")
        coco = {
            "info": {"description": f"firephantom flame dataset ({split})", "version": "2.0"},
            "images": images,
            "annotations": annotations,
            "categories": _coco_categories(),
        }
        files[f"annotations/coco_{split}.json"] = json.dumps(coco, ensure_ascii=False, indent=2).encode("utf-8")
    return files
