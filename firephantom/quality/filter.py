"""Keep or drop one rendered frame by checks that mean something per image.

FID and IS describe a set of images against a reference set, so they are
computed when a dataset version is published, not per sample. Here a frame
is dropped when its flame is too small to label or the picture is
unusable; a flame cut by the image edge is only flagged.
"""

from __future__ import annotations

import numpy as np

from firephantom.images import to_uint8

DEFAULT_MIN_FLAME_AREA = 0.002


def sample_checks(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    min_flame_area: float = DEFAULT_MIN_FLAME_AREA,
    max_dark: float = 0.97,
    max_blown: float = 0.4,
) -> dict:
    """Decide keep/drop for one frame and report the measured values.

    ``max_dark`` and ``max_blown`` are the largest allowed shares of
    near-black and saturated pixels outside the flame.
    """
    rgb = to_uint8(image)
    flame = np.asarray(mask, dtype=bool)
    if flame.shape != rgb.shape[:2]:
        raise ValueError("mask and image sizes differ")
    area = float(flame.mean())
    border = np.zeros_like(flame)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    edge_pixels = int((flame & border).sum())
    background = rgb[~flame]
    if background.size:
        luminance = background.astype(np.float32).mean(axis=1) / 255.0
        dark = float((luminance < 0.01).mean())
        blown = float((background.min(axis=1) >= 250).mean())
    else:
        dark = blown = 0.0
    reasons = []
    if area < float(min_flame_area):
        reasons.append("flame_too_small")
    if dark > float(max_dark):
        reasons.append("underexposed")
    if blown > float(max_blown):
        reasons.append("overexposed")
    kept = not reasons
    return {
        "action": "keep" if kept else "drop",
        "kept": kept,
        "reasons": reasons,
        "flame_area": round(area, 6),
        "flame_cut_by_edge": edge_pixels > 0,
        "edge_pixels": edge_pixels,
        "dark_background": round(dark, 6),
        "blown_background": round(blown, 6),
        "thresholds": {
            "min_flame_area": float(min_flame_area),
            "max_dark": float(max_dark),
            "max_blown": float(max_blown),
        },
    }
