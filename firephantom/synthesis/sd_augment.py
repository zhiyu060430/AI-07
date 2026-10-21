"""Stable Diffusion restyles the background of reviewed samples.

Generation never calls SD. Once a person has reviewed a frame, one masked
img2img call per variant repaints everything outside a dilated flame mask.
The pixels inside that guard are copied back from the rendered frame, so the
reviewed boxes still hold and the derived sample is stored as reviewed with
the same boxes. It keeps the source's sequence, so it lands in the same
train/val/test split and cannot leak across splits.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import cv2
import numpy as np
from PIL import Image

from firephantom import __version__
from firephantom.images import encode_mask_png, encode_png, to_uint8
from firephantom.quality.metrics import ssim
from firephantom.serialize import plain
from firephantom.synthesis.render import IMAGE_SIZE
from firephantom.synthesis.stable_diffusion import IMG2IMG_STRENGTH, call_stable_diffusion

DEFAULT_OPTIONS = {"variants": 1, "styles": None, "strength": IMG2IMG_STRENGTH, "steps": 20, "cfg_scale": 7.0}
MAX_VARIANTS = 4
# Below this the background has collapsed into something else entirely.
MIN_SSIM = 0.35
# Pixels around the flame that diffusion may not touch, so glow is not restyled.
_GUARD_PIXELS = 9
AUGMENTATION_PREFIX = "SD:"

# Backgrounds that fit each environment. Fire words are left out on purpose:
# asking for flames made the model paint extra fire, which then had no label.
STYLE_BANK: dict[str, dict[str, tuple[str, str]]] = {
    "室外": {
        "dusk": ("黄昏", "outdoor at dusk, low warm sky, long shadows"),
        "overcast": ("阴天", "outdoor under an overcast grey sky, flat diffuse light"),
        "night": ("夜晚路灯", "outdoor at night, sodium street lamps, dark sky"),
        "yard": ("厂区路面", "industrial yard, asphalt ground, metal fences and containers"),
        "grass": ("干草地", "dry grassland, scattered shrubs and trees"),
    },
    "室内": {
        "warehouse": ("仓库", "warehouse interior, concrete floor, steel shelving"),
        "office": ("办公室", "office interior, desks, chairs, ceiling panels"),
        "kitchen": ("厨房", "kitchen interior, tiled walls, cabinets"),
        "corridor": ("走廊", "long building corridor, doors along painted walls"),
    },
}
_WEATHER_EN = {"晴": "clear weather", "雨": "rainy weather, wet reflective surfaces", "雪": "snowy weather, snow on the ground"}
_VIEW_EN = {"侧视": "eye-level side view", "俯视": "high top-down view", "第一人称": "first-person view"}


def styles_for(environment: str) -> list[dict]:
    return [{"key": key, "label": label} for key, (label, _prompt) in STYLE_BANK.get(environment, {}).items()]


def build_prompt(environment: str, weather: str, viewpoint: str, style: str) -> str:
    """English prompt for the background only."""
    _label, scenery = STYLE_BANK[environment][style]
    parts = ["sharp photograph, realistic surroundings", scenery]
    if environment == "室外" and weather in _WEATHER_EN:
        parts.append(_WEATHER_EN[weather])
    if viewpoint in _VIEW_EN:
        parts.append(_VIEW_EN[viewpoint])
    parts.append("high detail")
    return ", ".join(parts)


def normalize_options(raw: object) -> dict:
    from firephantom.validation import diffusion_cfg, diffusion_steps, finite_float

    if raw in (None, ""):
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("SD 增强参数必须是对象")
    variants = raw.get("variants", 1)
    if isinstance(variants, bool) or not isinstance(variants, int) or not 1 <= variants <= MAX_VARIANTS:
        raise ValueError(f"每个样本的变体数要在 1 到 {MAX_VARIANTS} 之间")
    strength = finite_float(raw.get("strength", IMG2IMG_STRENGTH), "strength")
    if not 0.0 < strength <= 0.6:
        raise ValueError("strength must be between 0 and 0.6")
    styles = raw.get("styles")
    if styles is not None:
        if not isinstance(styles, (list, tuple)) or not styles:
            raise ValueError("styles must be a non-empty list")
        known = {key for bank in STYLE_BANK.values() for key in bank}
        unknown = [str(item) for item in styles if str(item) not in known]
        if unknown:
            raise ValueError(f"未知背景风格 {'、'.join(unknown)}")
        styles = list(dict.fromkeys(str(item) for item in styles))
    return {
        "variants": int(variants),
        "styles": styles,
        "strength": round(strength, 4),
        "steps": diffusion_steps(raw.get("steps", 20)),
        "cfg_scale": diffusion_cfg(raw.get("cfg_scale", 7.0)),
    }


def flame_guard(mask: np.ndarray) -> np.ndarray:
    """The flame, grown by a few pixels."""
    flame = np.asarray(mask, dtype=np.uint8)
    size = 2 * _GUARD_PIXELS + 1
    return cv2.dilate(flame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0


def _fit(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rgb = to_uint8(image)
    if rgb.shape[:2] == shape:
        return rgb
    shrink = rgb.shape[0] > shape[0] or rgb.shape[1] > shape[1]
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=interpolation)


def composite(frame: np.ndarray, background: np.ndarray, guard: np.ndarray) -> np.ndarray:
    """Rendered pixels inside the guard, diffused background outside, feathered outward."""
    alpha = cv2.GaussianBlur(guard.astype(np.float32), (0, 0), 2.0)
    alpha = np.maximum(alpha, guard.astype(np.float32))[..., None]
    out = alpha * frame.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)
    return to_uint8(out)


def restyle(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str,
    strength: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> dict:
    """One masked img2img call. ``image`` is None when SD failed or the result is unusable."""
    frame = to_uint8(image)
    guard = flame_guard(mask)
    wait = float(os.environ.get("SD_TIMEOUT_SECONDS", "180")) if timeout is None else float(timeout)
    result = call_stable_diffusion(
        prompt,
        endpoint=endpoint,
        timeout=wait,
        seed=seed,
        init_image=frame,
        strength=strength,
        steps=steps,
        cfg_scale=cfg_scale,
        mask=~guard,
    )
    contributor = {key: value for key, value in result.items() if key != "image"}
    contributor["role"] = "background_restyle"
    if not result["success"]:
        return {"image": None, "ssim": None, "reason": result["error"], "contributor": contributor}
    out = composite(frame, _fit(result["image"], frame.shape[:2]), guard)
    if not np.array_equal(out[guard], frame[guard]):
        return {"image": None, "ssim": None, "reason": "flame pixels changed", "contributor": contributor}
    structural = float(np.round(ssim(frame, out), 8))
    if structural < MIN_SSIM:
        return {"image": None, "ssim": structural, "reason": f"ssim {structural:.3f} < {MIN_SSIM}", "contributor": contributor}
    return {"image": out, "ssim": structural, "reason": None, "contributor": contributor}


def _seed(source_id: str, style: str, variant: int) -> int:
    raw = f"{source_id}:{style}:{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def plan_variants(source: dict, options: dict) -> list[dict]:
    """(style, seed) for each variant of one source. Styles cycle when there are more variants."""
    params = source["scene_params"]
    environment = params.get("scene") or "室外"
    bank = STYLE_BANK.get(environment, STYLE_BANK["室外"])
    chosen = [key for key in (options["styles"] or bank) if key in bank] or list(bank)
    plans = []
    for variant in range(options["variants"]):
        style = chosen[variant % len(chosen)]
        plans.append({
            "variant": variant,
            "style": style,
            "seed": _seed(source["sample_id"], style, variant),
            "prompt": build_prompt(environment if environment in STYLE_BANK else "室外", params.get("weather", "晴"), params.get("viewpoint", ""), style),
        })
    return plans


def derived_identity(source: dict, plan: dict, options: dict) -> tuple[str, dict, dict]:
    """Sample id, scene params and profile of one SD child. The same inputs give the same id."""
    from firephantom.versioning.samples import make_sample_id

    diffusion = {
        "source": source["sample_id"],
        "style": plan["style"],
        "variant": plan["variant"],
        "strength": options["strength"],
        "steps": options["steps"],
        "cfg_scale": options["cfg_scale"],
    }
    scene_params = {**source["scene_params"], "diffusion": diffusion}
    profile = {**source["generator_profile"], "diffusion": {"backend": "img2img"}}
    augmentations = [AUGMENTATION_PREFIX + plan["style"]]
    sample_id = make_sample_id(scene_params, plan["seed"], __version__, profile, augmentations)
    return sample_id, scene_params, profile


def is_sd_derived(record: dict) -> bool:
    return bool((record.get("scene_params") or {}).get("diffusion"))


def eligible_sources(root, dataset_id: str, sample_ids: list[str] | None = None) -> list[dict]:
    """Reviewed members of the dataset that are not themselves SD children."""
    from firephantom.db import connect
    from firephantom.versioning.samples import SampleCatalog

    catalog = SampleCatalog(root)
    with connect() as db:
        rows = db.execute(
            """SELECT samples.sample_id FROM dataset_samples
               JOIN samples ON samples.sample_id = dataset_samples.sample_id
               WHERE dataset_samples.dataset_id=? AND samples.status='reviewed'
               ORDER BY samples.sample_id""",
            (dataset_id,),
        ).fetchall()
    members = [row["sample_id"] for row in rows]
    records = [catalog.get(sample_id) for sample_id in members]
    eligible = {record["sample_id"]: record for record in records if record and not is_sd_derived(record)}
    if sample_ids is None:
        return list(eligible.values())
    missing = [sample_id for sample_id in sample_ids if sample_id not in eligible]
    if missing:
        raise ValueError(f"只能对数据集里已审核、且不是 SD 派生的样本做增强：{missing[0]}")
    return [eligible[sample_id] for sample_id in dict.fromkeys(sample_ids)]


def _decode(payload: bytes, grayscale: bool = False) -> np.ndarray:
    image = Image.open(io.BytesIO(payload))
    return np.asarray(image.convert("L" if grayscale else "RGB"))


def augment_source(
    root,
    dataset_id: str,
    source: dict,
    options: dict,
    *,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> list[dict]:
    """Every planned variant of one reviewed source. Existing children are left as they are."""
    from firephantom.versioning.datasets import DatasetCatalog
    from firephantom.versioning.samples import SampleCatalog, rebase_objects

    catalog = SampleCatalog(root)
    image = _decode(catalog.read_file(source["sample_id"], "image.png"))
    mask = _decode(catalog.read_file(source["sample_id"], "mask.png"), grayscale=True) > 0
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    datasets = DatasetCatalog(root)
    outcomes = []
    for plan in plan_variants(source, options):
        sample_id, scene_params, profile = derived_identity(source, plan, options)
        existing = catalog.get(sample_id)
        if existing is not None:
            datasets.attach_derived(dataset_id, sample_id)
            outcomes.append({"sample_id": sample_id, "style": plan["style"], "state": "exists"})
            continue
        restyled = restyle(
            image,
            mask,
            prompt=plan["prompt"],
            strength=options["strength"],
            steps=options["steps"],
            cfg_scale=options["cfg_scale"],
            seed=plan["seed"],
            endpoint=endpoint,
            timeout=timeout,
        )
        if restyled["image"] is None:
            outcomes.append({"sample_id": None, "style": plan["style"], "state": "failed", "reason": restyled["reason"]})
            continue
        quality = {**(source.get("quality") or {}), "ssim_to_render": restyled["ssim"]}
        provenance = {
            "sample_id": sample_id,
            "sequence_id": source.get("sequence_id"),
            "derived_from": source["sample_id"],
            "seed": plan["seed"],
            "generator_version": __version__,
            "generator_profile": profile,
            "scene_params": scene_params,
            "augmentations": [AUGMENTATION_PREFIX + plan["style"]],
            "prompt": plan["prompt"],
            "contributors": [restyled["contributor"]],
            "quality": quality,
            "boxes": "inherited from the reviewed source; flame pixels are unchanged",
        }
        boxes = rebase_objects(source["reviewed_objects"] or [], sample_id)
        files = {
            "image.png": encode_png(restyled["image"]),
            "mask.png": encode_mask_png(mask),
            "annotation.json": json.dumps(plain(boxes), ensure_ascii=False).encode("utf-8"),
            "quality_report.json": json.dumps(plain(quality), ensure_ascii=False).encode("utf-8"),
            "provenance.json": json.dumps(plain(provenance), ensure_ascii=False).encode("utf-8"),
        }
        catalog.save_sd_derived(
            source,
            sample_id=sample_id,
            seed=plan["seed"],
            generator_version=__version__,
            generator_profile=profile,
            scene_params=scene_params,
            quality=quality,
            files=files,
        )
        datasets.attach_derived(dataset_id, sample_id)
        outcomes.append({"sample_id": sample_id, "style": plan["style"], "state": "created", "ssim": restyled["ssim"]})
    return outcomes


__all__ = [
    "IMAGE_SIZE",
    "STYLE_BANK",
    "augment_source",
    "build_prompt",
    "composite",
    "eligible_sources",
    "flame_guard",
    "normalize_options",
    "plan_variants",
    "restyle",
    "styles_for",
]
