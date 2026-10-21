"""Photometric augmentations: 噪声, 模糊, 光照变化.

Each one makes a derived sample next to the rendered one; it never replaces
it. They change pixels only, so the flame mask and the boxes from it still
hold. Viewpoint changes come from real cameras in the renderer, so there is
no geometric warp here.
"""

from __future__ import annotations

import numpy as np
import cv2

from firephantom.images import to_uint8

AUGMENTATIONS = ("噪声", "模糊", "光照变化")


def augment_image(image: np.ndarray, names: list[str], seed: int | None = None) -> np.ndarray:
    out = to_uint8(image)
    noise_seed = 20261014 if seed is None else int(seed)
    for name in names:
        if name == "噪声":
            rng = np.random.default_rng(noise_seed)
            noise = rng.normal(0.0, 18.0, size=out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif name == "模糊":
            out = cv2.GaussianBlur(out, (15, 15), 2.4)
        elif name == "光照变化":
            out = np.clip(out.astype(np.float32) * 0.55 + 40.0, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"未知增强 {name!r}，可选：{'、'.join(AUGMENTATIONS)}")
    return np.ascontiguousarray(out)
