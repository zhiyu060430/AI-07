"""Stable Diffusion HTTP adapter.

Diffusion is optional and off by default. When enabled, synthesis sends one
masked img2img call per view: only pixels outside the flame mask may change.
A missing endpoint is recorded and returned; the caller then keeps the
rendered frames.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os

import httpx
import numpy as np
from PIL import Image

from firephantom.images import encode_png, to_uint8

DEFAULT_ENDPOINT = "http://127.0.0.1:7861/sdapi/v1/txt2img"
IMG2IMG_STRENGTH = 0.2
NEGATIVE_PROMPT = (
    "blurry, out of focus, soft focus, abstract, gradient, color fog, "
    "smooth blob, low resolution, watermark, text, logo, cartoon"
)


def endpoint_from_env() -> str:
    return os.environ.get("SD_API_URL", DEFAULT_ENDPOINT)


def _img2img_endpoint(target: str) -> str:
    trimmed = target.rstrip("/")
    if trimmed.endswith("/txt2img"):
        return trimmed[: -len("txt2img")] + "img2img"
    return target


def stable_error(exc: Exception) -> str:
    text = str(exc).splitlines()[0][:300] if str(exc) else ""
    return f"{type(exc).__name__}: {text}"


def _decode_response(response: httpx.Response) -> np.ndarray:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return to_uint8(np.asarray(image))
    payload = response.json()
    encoded = None
    if isinstance(payload, dict):
        if payload.get("images"):
            encoded = payload["images"][0]
        elif payload.get("image"):
            encoded = payload["image"]
    if not encoded:
        raise ValueError("stable diffusion response did not contain an image")
    if "," in encoded[:64]:
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return to_uint8(np.asarray(image))


def call_stable_diffusion(
    prompt: str,
    endpoint: str | None = None,
    timeout: float = 180.0,
    seed: int | None = None,
    steps: int = 20,
    cfg_scale: float = 7.0,
    init_image: np.ndarray | None = None,
    strength: float = IMG2IMG_STRENGTH,
    mask: np.ndarray | None = None,
) -> dict:
    """POST one generation request. Failure is a structured result, not an exception.

    An ``init_image`` selects img2img. ``mask`` (true where the image may
    change) is sent in the AUTOMATIC1111 inpaint form, so the flame stays the
    simulator's. Without an init image, the call stays txt2img.
    """
    target = endpoint or endpoint_from_env()
    if init_image is not None:
        target = _img2img_endpoint(target)
    resolved_seed = (
        int(seed)
        if seed is not None
        else int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "little")
    )
    body = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": int(steps),
        "width": 256,
        "height": 256,
        "cfg_scale": float(cfg_scale),
        "seed": resolved_seed,
    }
    if init_image is not None:
        body["init_images"] = [base64.b64encode(encode_png(init_image)).decode("ascii")]
        body["strength"] = float(strength)
        body["denoising_strength"] = float(strength)
    if init_image is not None and mask is not None:
        white = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
        body["mask"] = base64.b64encode(encode_png(np.stack([white] * 3, axis=-1))).decode("ascii")
        body["mask_blur"] = 4
        body["inpainting_fill"] = 1
        body["inpaint_full_res"] = False
    role = "scene_synthesis" if init_image is not None else "generative_prior"
    try:
        # Keep local calls direct; remote compatible services can opt into auth and a proxy.
        token = os.environ.get("SD_API_TOKEN", "").strip()
        proxy = os.environ.get("SD_HTTP_PROXY", "").strip() or None
        headers = {"Authorization": f"Bearer {token}"} if token else None
        with httpx.Client(trust_env=False, timeout=timeout, proxy=proxy) as client:
            response = client.post(target, json=body, headers=headers)
        response.raise_for_status()
        image = _decode_response(response)
        return {
            "name": "stable_diffusion",
            "role": role,
            "called": True,
            "success": True,
            "endpoint": target,
            "error": None,
            "image": image,
        }
    except Exception as exc:
        return {
            "name": "stable_diffusion",
            "role": role,
            "called": True,
            "success": False,
            "endpoint": target,
            "error": stable_error(exc),
            "image": None,
        }
