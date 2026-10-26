"""Deterministic RGB image conversions used by synthesis and export."""

from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2

from firephantom.paths import find_ffmpeg


def to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float64)
    peak = float(np.nanmax(array)) if array.size else 0.0
    if peak <= 1.0:
        array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    rgb = to_uint8(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("encode_png expects an HxWx3 image")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(
        ".png",
        bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
    )
    if not ok:
        raise RuntimeError("failed to encode png")
    return buffer.tobytes()


def encode_mask_png(mask: np.ndarray) -> bytes:
    """One-channel PNG: 255 where the mask is set."""
    ok, buffer = cv2.imencode(".png", np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("failed to encode png")
    return buffer.tobytes()


def decode_png(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("invalid png bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ffmpeg_executable() -> str | None:
    return find_ffmpeg()


def video_encoder() -> str:
    """Describe the encoder that new dataset versions will contain."""
    if _ffmpeg_executable():
        return f"h264:{os.environ.get('FIREPHANTOM_VIDEO_CODEC', 'libx264')}:yuv420p"
    return "opencv:mp4v"


@lru_cache(maxsize=8)
def browser_compatible_video(payload: bytes) -> bytes:
    """Upgrade legacy MP4 bytes to H.264 at the HTTP boundary.

    Versions created before ffmpeg support contain OpenCV ``mp4v`` streams.
    They remain immutable on disk, while browser responses and downloads get
    the same H.264 stream used by newly generated versions.
    """
    if not payload or b"avc1" in payload:
        return payload
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return payload
    input_name = output_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as source:
            source.write(payload)
            input_name = source.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as target:
            output_name = target.name
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_name,
            "-an",
            "-c:v",
            os.environ.get("FIREPHANTOM_VIDEO_CODEC", "libx264"),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_name,
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=60)
        converted = Path(output_name).read_bytes()
        return converted or payload
    except (OSError, subprocess.SubprocessError):
        return payload
    finally:
        for filename in (input_name, output_name):
            if filename:
                try:
                    os.unlink(filename)
                except FileNotFoundError:
                    pass


def encode_video(frames: list[np.ndarray], fps: float = 8.0) -> bytes:
    """Encode RGB frames as a browser-compatible MP4 byte stream.

    Prefer H.264 through ffmpeg because OpenCV's ``mp4v`` output is decoded
    incorrectly by some Safari/Chromium combinations on Apple Silicon. Keep
    the OpenCV encoder as a portable fallback for minimal containers.
    """
    if not frames:
        raise ValueError("encode_video expects at least one frame")
    normalized = [to_uint8(frame) for frame in frames]
    height, width = normalized[0].shape[:2]
    if any(frame.shape != normalized[0].shape for frame in normalized):
        raise ValueError("all video frames must have the same shape")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        filename = handle.name
    try:
        ffmpeg = _ffmpeg_executable()
        if ffmpeg:
            raw = b"".join(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)[:, :, ::-1].tobytes()
                for frame in normalized
            )
            # Feed RGB24 explicitly so the encoder cannot infer a swapped
            # channel order from OpenCV's BGR convention.
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(float(fps)),
                "-i",
                "-",
                "-an",
                "-c:v",
                os.environ.get("FIREPHANTOM_VIDEO_CODEC", "libx264"),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                filename,
            ]
            try:
                subprocess.run(
                    command,
                    input=raw,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=60,
                )
                payload = open(filename, "rb").read()
                if payload:
                    return payload
            except (OSError, subprocess.SubprocessError):
                pass
        writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError("MP4 video codec is unavailable")
        try:
            for frame in normalized:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return open(filename, "rb").read()
    finally:
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
