"""Image quality scores.

FID follows Heusel et al. (Fréchet distance between Gaussian embeddings).
IS follows Salimans et al. (exp of the mean KL between p(y|x) and p(y)).
SSIM follows Wang et al. 2004 (windowed luminance, contrast, and structure).

Inception / torchvision weights are loaded only from the explicit local path in
``INCEPTION_WEIGHTS``. When no checkpoint is configured, the same functions
fall back to a documented deterministic embedding and a 4-class classifier.
FID is a distance (lower is closer); it is not converted to a percentage.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import normalize

_EMBED_DIM = 48


def inception_backend() -> str:
    """Report the actual metric backend, without downloading model weights."""
    configured = os.environ.get("INCEPTION_WEIGHTS", "").strip()
    if not configured:
        return "inception weights were not loaded: set INCEPTION_WEIGHTS; using deterministic embedding and classifier"
    model = _inception_model(configured)
    if model is None:
        return f"inception weights were not loaded: invalid checkpoint {Path(configured).name}; using deterministic embedding and classifier"
    return f"torchvision-inception: loaded {Path(configured).name}"


@lru_cache(maxsize=2)
def _inception_model(path: str):
    """Load an explicitly supplied torchvision checkpoint, never from network."""
    if not Path(path).is_file():
        return None
    try:
        import torch
        from torchvision.models import inception_v3

        model = inception_v3(weights=None, aux_logits=True, init_weights=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return None
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, torch
    except Exception:
        return None


def _inception_forward(images) -> tuple[np.ndarray, np.ndarray] | None:
    """Return pooled Inception features and class probabilities when configured."""
    configured = os.environ.get("INCEPTION_WEIGHTS", "").strip()
    if not configured:
        return None
    loaded = _inception_model(configured)
    if loaded is None:
        return None
    model, torch = loaded
    pooled: list = []
    hook = model.avgpool.register_forward_hook(lambda _module, _inputs, output: pooled.append(output.detach()))
    try:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        feature_batches = []
        probability_batches = []
        for start in range(0, len(images), 32):
            tensors = []
            for image in images[start : start + 32]:
                rgb = _rgb_uint8(image)
                resized = cv2.resize(rgb, (299, 299), interpolation=cv2.INTER_AREA)
                tensors.append(torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0)
            batch = torch.stack(tensors)
            pooled.clear()
            with torch.inference_mode():
                logits = model((batch - mean) / std)
            feature_batches.append(pooled[-1].flatten(1).cpu().numpy().astype(np.float64))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
        return np.concatenate(feature_batches), np.concatenate(probability_batches)
    finally:
        hook.remove()


def _as_batch(images) -> list[np.ndarray]:
    if isinstance(images, np.ndarray) and images.ndim == 3:
        return [images]
    batch = list(images)
    if not batch:
        raise ValueError("at least one image is required")
    return batch


def _rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float64)
        if float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


@lru_cache(maxsize=1)
def _projection() -> np.ndarray:
    rng = np.random.default_rng(20261014)
    return rng.normal(0.0, 1.0 / np.sqrt(768.0), size=(768, _EMBED_DIM))


def embed_images(images) -> np.ndarray:
    """Embed a batch with the deterministic projection.

    A configured local Inception checkpoint is preferred; the projection is the
    deterministic fallback used when no valid checkpoint is available.
    """
    batch = _as_batch(images)
    inception = _inception_forward(batch)
    if inception is not None:
        return inception[0]
    rows = []
    matrix = _projection()
    for image in batch:
        small = cv2.resize(_rgb_uint8(image), (16, 16), interpolation=cv2.INTER_AREA)
        flat = small.astype(np.float64).reshape(-1) / 255.0
        rows.append(flat @ matrix)
    embedded = np.stack(rows, axis=0)
    return normalize(embedded, axis=1)


def _gaussian_stats(embedded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = embedded.mean(axis=0)
    if embedded.shape[0] < 2:
        covariance = np.zeros((embedded.shape[1], embedded.shape[1]), dtype=np.float64)
    else:
        covariance = np.cov(embedded, rowvar=False)
        covariance = np.atleast_2d(covariance)
    return mean, covariance


def _frechet(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)
    value = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    if not np.isfinite(value):
        raise ValueError("FID was not finite")
    if -1e-6 < value < 0.0:
        return 0.0
    return value


def frechet_inception_distance(real_images, generated_images) -> float:
    """Fréchet distance between the embeddings of two image sets."""
    configured = os.environ.get("FIRE_REFERENCE_DATASET", "").strip()
    if configured and real_images is reference_images_from_env():
        limit = max(1, int(os.environ.get("FIRE_REFERENCE_LIMIT", "512")))
        weights = os.environ.get("INCEPTION_WEIGHTS", "").strip()
        mu1, sigma1 = _cached_reference_stats(configured, limit, weights)
        real = None
    else:
        real = embed_images(real_images)
    generated = embed_images(generated_images)
    if real is not None:
        mu1, sigma1 = _gaussian_stats(real)
    mu2, sigma2 = _gaussian_stats(generated)
    if np.allclose(mu1, mu2) and np.allclose(sigma1, sigma2):
        return 0.0
    return _frechet(mu1, sigma1, mu2, sigma2)


def _logits(image: np.ndarray) -> np.ndarray:
    """Deterministic 4-class head used when Inception weights are absent.

    Classes are flame (red), vegetation (green), water (blue), and smoke
    (yellow). A mid-gray image has equal logits, so its softmax is uniform.
    """
    array = _rgb_uint8(image).astype(np.float64)
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    mean_r = float(red.mean() / 255.0)
    mean_g = float(green.mean() / 255.0)
    mean_b = float(blue.mean() / 255.0)
    hot = float(np.mean((red > 200) & (green < 80) & (blue < 80)))
    vegetation = float(np.mean((green > 200) & (red < 80) & (blue < 80)))
    water = float(np.mean((blue > 200) & (red < 80) & (green < 80)))
    smoke = float(np.mean((red > 180) & (green > 180) & (blue < 80)))
    return np.array(
        [
            hot * 10.0 + max(mean_r - mean_g - mean_b, 0.0) * 5.0,
            vegetation * 10.0 + max(mean_g - mean_r - mean_b, 0.0) * 5.0,
            water * 10.0 + max(mean_b - mean_r - mean_g, 0.0) * 5.0,
            smoke * 10.0 + max(min(mean_r, mean_g) - mean_b - 0.2, 0.0) * 5.0,
        ],
        dtype=np.float64,
    )


def class_probabilities(images) -> np.ndarray:
    batch = _as_batch(images)
    inception = _inception_forward(batch)
    if inception is not None:
        return inception[1]
    rows = []
    for image in batch:
        logits = _logits(image)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        rows.append(exp / exp.sum())
    return np.stack(rows, axis=0)


def inception_score(images) -> float:
    """exp(E[KL(p(y|x) || p(y))]). Finite, and higher when conditionals are peaked and diverse."""
    probabilities = class_probabilities(images)
    eps = 1e-12
    marginal = np.clip(probabilities.mean(axis=0), eps, 1.0)
    marginal = marginal / marginal.sum()
    conditional = np.clip(probabilities, eps, 1.0)
    conditional = conditional / conditional.sum(axis=1, keepdims=True)
    kl = np.sum(conditional * (np.log(conditional) - np.log(marginal)), axis=1)
    score = float(np.exp(np.mean(kl)))
    if not np.isfinite(score):
        raise ValueError("IS was not finite")
    return score


def ssim(image_a, image_b) -> float:
    """Mean windowed SSIM on luminance. Identical images score 1."""
    gray_a = _luminance(image_a)
    gray_b = _luminance(image_b)
    if gray_a.shape != gray_b.shape:
        raise ValueError(f"SSIM shape mismatch {gray_a.shape} vs {gray_b.shape}")
    data_range = 255.0 if max(float(gray_a.max()), float(gray_b.max())) > 1.0 else 1.0
    if data_range == 1.0 and gray_a.dtype != np.float64:
        gray_a = gray_a.astype(np.float64)
        gray_b = gray_b.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = 11 if min(gray_a.shape) >= 11 else (min(gray_a.shape) // 2 * 2 - 1)
    kernel = max(kernel, 1)
    if kernel % 2 == 0:
        kernel -= 1
    sigma = 1.5 if kernel >= 5 else 0.5
    mu_a = cv2.GaussianBlur(gray_a, (kernel, kernel), sigma)
    mu_b = cv2.GaussianBlur(gray_b, (kernel, kernel), sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(gray_a * gray_a, (kernel, kernel), sigma) - mu_a2
    sigma_b2 = cv2.GaussianBlur(gray_b * gray_b, (kernel, kernel), sigma) - mu_b2
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (kernel, kernel), sigma) - mu_ab
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    score = float(np.mean(numerator / denominator))
    if not np.isfinite(score):
        raise ValueError("SSIM was not finite")
    return score


@lru_cache(maxsize=2)
def _reference_images(configured: str, limit: int) -> list[np.ndarray]:
    root = Path(configured)
    if not root.exists():
        raise ValueError(f"FIRE_REFERENCE_DATASET does not exist: {root}")
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
    else:
        paths = []
        for line in root.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                path = Path(entry)
                paths.append(path if path.is_absolute() else root.parent / path)
    if not paths:
        raise ValueError(f"FIRE_REFERENCE_DATASET contains no image files: {root}")
    if len(paths) > limit:
        indices = np.random.default_rng(20261014).choice(len(paths), size=limit, replace=False)
        paths = [paths[index] for index in sorted(indices)]
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not images:
        raise ValueError(f"FIRE_REFERENCE_DATASET images could not be decoded: {root}")
    return images


@lru_cache(maxsize=2)
def _cached_reference_stats(configured: str, limit: int, weights: str) -> tuple[np.ndarray, np.ndarray]:
    images = _reference_images(configured, limit)
    if weights and os.environ.get("INCEPTION_WEIGHTS", "").strip() != weights:
        raise RuntimeError("INCEPTION_WEIGHTS changed while reference statistics were being computed")
    return _gaussian_stats(embed_images(images))


def reference_images_from_env() -> list[np.ndarray] | None:
    """Load a deterministic, bounded sample from the configured local fire set."""
    configured = os.environ.get("FIRE_REFERENCE_DATASET", "").strip()
    if not configured:
        return None
    limit = int(os.environ.get("FIRE_REFERENCE_LIMIT", "512"))
    if limit < 1:
        raise ValueError("FIRE_REFERENCE_LIMIT must be at least 1")
    return _reference_images(configured, limit)


def _luminance(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float64)
    else:
        rgb = array[..., :3].astype(np.float64)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.ascontiguousarray(gray)
