"""FID, IS, SSIM, mask annotations with fixed class ids, and the per-frame checks."""

import xml.etree.ElementTree as ET

import numpy as np

from firephantom.annotation.annotate import annotate
from firephantom.physics.engine import simulate
from firephantom.annotation.annotate import objects_from_mask
from firephantom.annotation.classes import CLASSES
from firephantom.quality.filter import sample_checks
from firephantom.quality.metrics import (
    frechet_inception_distance,
    inception_backend,
    inception_score,
    ssim,
)
from firephantom.versioning.exporters import export_all

def _solid(color, size=48):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    return image


def test_backend_reports_the_missing_inception_weights():
    message = inception_backend()
    assert "inception weights were not loaded" in message


def test_ssim_and_fid_separate_identical_from_dissimilar_images():
    red = _solid((255, 0, 0))
    green = _solid((0, 255, 0))
    blue = _solid((0, 0, 255))
    identical_ssim = ssim(red, red.copy())
    red_blue = ssim(red, blue)
    red_green = ssim(red, green)
    assert identical_ssim > 0.99
    assert red_blue < identical_ssim
    assert red_green < identical_ssim
    assert red_blue != red_green

    identical_fid = frechet_inception_distance([red, red, red, red], [red.copy(), red, red, red])
    red_blue_fid = frechet_inception_distance([red, red], [blue, blue])
    red_green_fid = frechet_inception_distance([red, red], [green, green])
    assert identical_fid < 1e-6
    assert red_blue_fid > identical_fid
    assert red_green_fid > identical_fid
    assert red_blue_fid != red_green_fid


def test_inception_score_is_higher_for_peaked_than_uniform():
    peaked = [
        _solid((255, 0, 0)),
        _solid((0, 255, 0)),
        _solid((0, 0, 255)),
        _solid((255, 255, 0)),
    ]
    uniform = [_solid((128, 128, 128)) for _ in range(4)]
    peaked_score = inception_score(peaked)
    uniform_score = inception_score(uniform)
    assert np.isfinite(peaked_score)
    assert np.isfinite(uniform_score)
    assert peaked_score > uniform_score


def test_checks_drop_tiny_flames_and_flag_edge_cuts():
    image = _solid((40, 40, 40), size=64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:30] = True
    kept = sample_checks(image, mask)
    assert kept["kept"] is True and kept["reasons"] == []
    assert kept["flame_area"] == round(200 / 4096, 6)
    assert kept["flame_cut_by_edge"] is False
    tiny = np.zeros_like(mask)
    tiny[5, 5] = True
    dropped = sample_checks(image, tiny)
    assert dropped["kept"] is False and dropped["reasons"] == ["flame_too_small"]
    edge = np.zeros_like(mask)
    edge[0:30, 10:20] = True
    assert sample_checks(image, edge)["flame_cut_by_edge"] is True
    assert sample_checks(image, edge)["kept"] is True
    white = _solid((255, 255, 255), size=64)
    assert "overexposed" in sample_checks(white, mask)["reasons"]
    black = _solid((0, 0, 0), size=64)
    assert "underexposed" in sample_checks(black, mask)["reasons"]


def test_one_object_per_flame_region_with_outline_and_keypoints():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 5:15] = True
    mask[40:60, 40:44] = True
    mask[2:4, 60:61] = True  # too small to label on its own, too far to join
    objects = objects_from_mask(mask)
    assert [obj["bbox"] for obj in objects] == [[5.0, 10.0, 10.0, 20.0], [40.0, 40.0, 4.0, 20.0]]
    first = objects[0]
    assert len(first["polygon"]) >= 8
    names = {point["name"]: point for point in first["keypoints"]}
    assert names["flame_base"]["y"] == 29 and names["plume_tip"]["y"] == 10
    close = np.zeros((64, 64), dtype=bool)
    close[10:30, 5:15] = True
    close[10:30, 18:28] = True
    assert len(objects_from_mask(close)) == 1


def test_mask_annotation_label_studio_and_three_exports_describe_the_same_object():
    physical = simulate("林区火灾", wind_speed=0.4, temperature=18.0, humidity=30.0, fuel_type="植被")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[18:52, 14:46] = True
    image[mask] = (240, 30, 10)
    annotated = annotate(
        image,
        mask,
        physical,
        context={"scene": "室外", "weather": "晴", "wind_speed": 0.4},
        sample_id="abc",
    )
    assert annotated["backend"] == "render-mask"
    task = annotated["label_studio"]
    assert task["data"]["image"] == "/api/v1/samples/abc/files/image.png"
    assert task["data"]["sample_id"] == "abc"
    result = task["predictions"][0]["result"]
    kinds = {item["type"] for item in result}
    assert {"rectanglelabels", "keypointlabels", "choices"} <= kinds
    assert any(item["value"].get("rectanglelabels") == ["flame"] for item in result)

    smoke = np.zeros((64, 64), dtype=bool)
    smoke[4:20, 40:58] = True
    embers = np.zeros((64, 64), dtype=bool)
    embers[48:56, 8:16] = True
    both = annotate(image, mask, physical, context={"scene": "室外", "weather": "晴", "wind_speed": 0.4}, smoke_mask=smoke, ember_mask=embers)
    kinds = [(item["category"], bool(item["keypoints"])) for item in both["objects"]]
    assert kinds == [("flame", True), ("smoke", False), ("ember", False)]
    assert "可见烟羽" in both["objects"][0]["attributes"]
    assert "飞火" in both["objects"][0]["attributes"]
    labels = [item["value"].get("rectanglelabels") for item in both["label_studio"]["predictions"][0]["result"] if item["type"] == "rectanglelabels"]
    assert labels == [["flame"], ["smoke"], ["ember"]]

    obj = annotated["objects"][0]
    assert obj["bbox"] == [14.0, 18.0, 32.0, 34.0]
    assert obj["category"] == "flame"
    exported = export_all(annotated["objects"])
    coco = exported["coco"]["annotations"][0]
    assert coco["bbox"] == obj["bbox"]
    assert coco["category_id"] == 1
    assert [item["name"] for item in exported["coco"]["categories"]] == list(CLASSES)
    assert coco["segmentation"] and len(coco["segmentation"][0]) >= 8
    assert coco["num_keypoints"] == 3

    root = ET.fromstring(exported["pascal_voc"])
    box = root.find("./object/bndbox")
    assert root.findtext("./object/name") == "flame"
    assert float(box.findtext("xmin")) == 14.0
    assert float(box.findtext("ymax")) == 52.0

    class_id, x_center, y_center, norm_w, norm_h = (float(part) for part in exported["yolo"]["labels"].split())
    assert int(class_id) == 0
    assert exported["yolo"]["classes"].split() == list(CLASSES)
    assert abs((x_center - norm_w / 2.0) * 64 - 14.0) < 1e-4
    assert abs(norm_h * 64 - 34.0) < 1e-4


def test_smoke_keeps_its_class_id_even_alone():
    smoke = [{"category": "smoke", "bbox": [1, 1, 4, 4], "image_width": 16, "image_height": 16}]
    assert export_all(smoke)["yolo"]["labels"].startswith("1 ")
    assert export_all([{**smoke[0], "category": "fire"}])["yolo"]["labels"].startswith("0 ")
