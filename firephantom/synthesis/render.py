"""Project physical fields to an image, a frame sequence, and a point cloud."""

from __future__ import annotations

import numpy as np
import cv2

from firephantom.images import to_uint8

IMAGE_SIZE = 256

SCENES = ("室内", "室外")
WEATHERS = ("晴", "雨", "雪")
VIEWPOINTS = ("俯视", "侧视", "第一人称")


def _check(value: str, options: tuple[str, ...], label: str) -> None:
    if value not in options:
        raise ValueError(f"未知{label} {value!r}，可选：{'、'.join(options)}")


def _colormap(temperature: np.ndarray, smoke: np.ndarray) -> np.ndarray:
    """Map Celsius and smoke to a dark field with an orange-yellow flame core.

    The scale is fixed. Per-image min-max stretching turns a mild plume into a
    full-frame color wash, which no longer reads as fire.
    """
    temp = temperature.astype(np.float64)
    smoke_field = smoke.astype(np.float64)
    # 50°C stays dark. Around 550°C is a full yellow core. Ambient air is below the floor.
    heat = np.clip((temp - 50.0) / 500.0, 0.0, 1.0)
    heat = np.power(heat, 0.65)
    smoke_scale = float(np.percentile(smoke_field, 99))
    smoke_n = np.clip(smoke_field / (smoke_scale + 1e-6), 0.0, 1.0)
    red = np.clip(0.02 + heat * 1.7, 0.0, 1.0)
    green = np.clip((heat - 0.25) * 1.55, 0.0, 1.0)
    blue = np.clip((heat - 0.78) * 2.8, 0.0, 1.0) * 0.35
    veil = smoke_n * (1.0 - heat) * 0.42
    red = np.clip(red * (1.0 - 0.22 * smoke_n) + veil, 0.0, 1.0)
    green = np.clip(green * (1.0 - 0.22 * smoke_n) + veil, 0.0, 1.0)
    blue = np.clip(blue + veil * 1.05, 0.0, 1.0)
    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def _resize(field: np.ndarray) -> np.ndarray:
    return cv2.resize(field.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)


# The colormap turns yellow-orange above this. Cooler gas reads as smoke or glow.
FLAME_MASK_CELSIUS = 175.0


def _projected(field: np.ndarray, viewpoint: str, frame_index: int, scale: float = 1.0) -> np.ndarray:
    """Max projection of one time step. Viewpoint selects the axis."""
    _check(viewpoint, VIEWPOINTS, "视角")
    rolled = np.roll(field, shift=-int(frame_index), axis=2) * scale
    # Max keeps the hot core. A mean projection smears it into a soft blob.
    if viewpoint == "俯视":
        axis = 2
    elif viewpoint == "侧视":
        axis = 1
    else:
        axis = 0
    flat = rolled.max(axis=axis)
    # Array rows are the first remaining axis. Turn world +z (or +y in plan view) into image up.
    return _resize(np.flipud(np.transpose(flat)))


def project_frame(temperature: np.ndarray, smoke: np.ndarray, viewpoint: str, frame_index: int) -> np.ndarray:
    """Camera projection of one time step. Viewpoint selects the projection."""
    pulse = 1.0 + 0.05 * int(frame_index)
    temp_2d = _projected(temperature, viewpoint, frame_index, pulse)
    smoke_2d = _projected(smoke, viewpoint, frame_index)
    return _colormap(temp_2d, smoke_2d)


def project_mask(temperature: np.ndarray, viewpoint: str, frame_index: int) -> np.ndarray:
    """Flame pixels of ``project_frame`` taken from temperature, not color."""
    pulse = 1.0 + 0.05 * int(frame_index)
    return _projected(temperature, viewpoint, frame_index, pulse) >= FLAME_MASK_CELSIUS


def project_smoke_mask(smoke: np.ndarray, viewpoint: str, frame_index: int) -> np.ndarray:
    """Smoke pixels of one projected field. The cool plume, not the hot core."""
    flat = _projected(smoke, viewpoint, frame_index)
    peak = float(np.percentile(flat, 99))
    if peak <= 1e-8:
        return np.zeros(flat.shape, dtype=bool)
    return flat >= 0.22 * peak


def apply_scene(rgb: np.ndarray, scene: str) -> np.ndarray:
    _check(scene, SCENES, "场景")
    out = np.array(rgb, dtype=np.float64, copy=True)
    # 5/64 and 18/64 are the proportions chosen for the original 64px frame.
    border = max(1, int(round(out.shape[0] * 5 / 64)))
    sky_rows = max(1, int(round(out.shape[0] * 18 / 64)))
    if scene == "室内":
        out[:border, :, :] *= 0.15
        out[-border:, :, :] *= 0.15
        out[:, :border, :] *= 0.15
        out[:, -border:, :] *= 0.15
        out += np.array([0.04, 0.01, 0.0])
    else:
        sky = np.linspace(0.62, 0.18, sky_rows, dtype=np.float64)[:, None]
        out[:sky_rows, :, 2] = np.maximum(out[:sky_rows, :, 2], sky)
        out[:sky_rows, :, 0] *= 0.72
        out[:sky_rows, :, 1] *= 0.84
    return np.clip(out, 0.0, 1.0)


def apply_weather(rgb: np.ndarray, weather: str) -> np.ndarray:
    _check(weather, WEATHERS, "天气")
    out = np.array(rgb, dtype=np.float64, copy=True)
    if weather == "晴":
        out = np.clip(out * np.array([1.08, 1.02, 0.92]), 0.0, 1.0)
    elif weather == "雨":
        out *= 0.68
        streaks = np.zeros_like(out)
        streaks[:, 2::7, 0] = 0.05
        streaks[:, 2::7, 1] = 0.12
        streaks[:, 2::7, 2] = 0.28
        out = np.clip(out + streaks, 0.0, 1.0)
    else:
        out = np.clip(out * 0.86 + np.array([0.10, 0.11, 0.13]), 0.0, 1.0)
        out[3::6, 2::9, :] = 1.0
    return out


def flame_mask(frame: np.ndarray) -> np.ndarray:
    """Pixels that read as flame: orange, yellow, or a white-hot core.

    Neutral gray and pure white stay outside the mask, so a bare floor is not
    treated as fire.
    """
    rgb = to_uint8(frame).astype(np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    return (red > 150) & (red > blue + 35) & (green > blue) & (red + green > 240)


def dress_frame(frame: np.ndarray, scene: str, weather: str, mask: np.ndarray | None = None) -> np.ndarray:
    """2D environment and weather for the parameterized fallback.

    Blender renders these in the scene. Flame pixels stay as they are.
    """
    original = to_uint8(frame)
    rgb = original.astype(np.float64) / 255.0
    dressed = apply_weather(apply_scene(rgb, scene), weather)
    out = to_uint8(dressed)
    keep = flame_mask(original) if mask is None else np.asarray(mask, dtype=bool)
    out[keep] = original[keep]
    return out


def _view_matrix(viewpoint: str) -> np.ndarray:
    if viewpoint == "俯视":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if viewpoint == "侧视":
        return np.eye(3)
    theta = np.deg2rad(28.0)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_point_cloud(
    temperature: np.ndarray,
    smoke: np.ndarray,
    axes: dict,
    scene: str,
    weather: str,
    viewpoint: str,
) -> np.ndarray:
    """Return Nx5 rows: x, y, z, temperature, smoke. Weather may append particles."""
    _check(scene, SCENES, "场景")
    _check(weather, WEATHERS, "天气")
    _check(viewpoint, VIEWPOINTS, "视角")
    threshold = float(np.quantile(temperature, 0.82))
    indices = np.argwhere(temperature >= threshold)
    if indices.size == 0:
        indices = np.array([np.unravel_index(int(np.argmax(temperature)), temperature.shape)])
    xs = np.asarray(axes["x"], dtype=np.float64)
    ys = np.asarray(axes["y"], dtype=np.float64)
    zs = np.asarray(axes["z"], dtype=np.float64)
    rows = []
    for i, j, k in indices:
        rows.append(
            [
                xs[int(i)],
                ys[int(j)],
                zs[int(k)],
                float(temperature[int(i), int(j), int(k)]),
                float(smoke[int(i), int(j), int(k)]),
            ]
        )
    cloud = np.asarray(rows, dtype=np.float64)
    if scene == "室外":
        cloud[:, :3] = cloud[:, :3] * 1.2 + np.array([0.5, 0.0, 0.8])
    cloud[:, :3] = cloud[:, :3] @ _view_matrix(viewpoint).T
    if weather == "雨":
        rain = np.stack(
            [
                np.linspace(0.2, 6.0, 12),
                np.full(12, 0.4),
                np.linspace(2.0, 8.0, 12),
                np.full(12, -1.0),
                np.full(12, 0.0),
            ],
            axis=1,
        )
        cloud = np.vstack([cloud, rain])
    elif weather == "雪":
        snow = np.stack(
            [
                np.linspace(0.3, 5.0, 10),
                np.linspace(0.2, 4.0, 10),
                np.full(10, 7.5),
                np.full(10, -2.0),
                np.full(10, 0.02),
            ],
            axis=1,
        )
        cloud = np.vstack([cloud, snow])
    order = np.lexsort((cloud[:, 0], cloud[:, 1], cloud[:, 2]))
    return cloud[order]
