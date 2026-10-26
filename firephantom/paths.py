"""Local data, model, and tool locations.

``FIREPHANTOM_RUNTIME`` overrides the root. The default is the repository
``data`` directory on macOS, Linux, and Windows.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    configured = os.environ.get("FIREPHANTOM_RUNTIME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return repo_root() / "data"


def dataset_root() -> Path:
    configured = os.environ.get("DATASET_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_root() / "dataset"


def postgres_password_file() -> Path:
    configured = os.environ.get("FIREPHANTOM_POSTGRES_PASSWORD_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_root() / "postgres.password"


def session_secret_file() -> Path:
    configured = os.environ.get("FIREPHANTOM_SESSION_SECRET_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_root() / "firephantom-session.secret"


def sd_model_path() -> Path:
    configured = os.environ.get("SD_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_root() / "models" / "tiny-sd"


def assets_root() -> Path:
    configured = os.environ.get("FIREPHANTOM_ASSETS", "").strip()
    if configured:
        return Path(configured).expanduser()
    return runtime_root() / "assets"


def _executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if platform.system() == "Windows":
        return True
    return os.access(path, os.X_OK)


def _unique(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def blender_candidates() -> list[Path]:
    """Platform Blender binaries, preferring a copy inside the runtime tools directory."""
    root = runtime_root() / "tools"
    system = platform.system()
    candidates: list[Path] = []
    if system == "Darwin":
        candidates.extend(
            [
                root / "Blender.app" / "Contents" / "MacOS" / "Blender",
                Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                Path.home() / "Applications" / "Blender.app" / "Contents" / "MacOS" / "Blender",
            ]
        )
    elif system == "Windows":
        candidates.append(root / "blender.exe")
        candidates.append(root / "blender" / "blender.exe")
        candidates.extend(sorted(root.glob("blender*/blender.exe"), reverse=True))
        program_dirs = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        for base in program_dirs:
            foundation = Path(base) / "Blender Foundation"
            if foundation.is_dir():
                candidates.extend(sorted(foundation.glob("Blender*/blender.exe"), reverse=True))
    else:
        candidates.extend(
            [
                root / "blender",
                root / "blender" / "blender",
                Path("/usr/bin/blender"),
                Path("/usr/local/bin/blender"),
                Path("/snap/bin/blender"),
            ]
        )
    return _unique(candidates)


def find_blender() -> str | None:
    configured = os.environ.get("BLENDER_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if _executable(path) or path.is_file():
            return str(path)
    for candidate in blender_candidates():
        if _executable(candidate):
            return str(candidate)
    return shutil.which("blender")


def find_label_studio() -> str | None:
    configured = os.environ.get("LABEL_STUDIO_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
    root = runtime_root() / "label-studio-env"
    for candidate in (
        root / "bin" / "label-studio",
        root / "Scripts" / "label-studio.exe",
        root / "Scripts" / "label-studio",
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("label-studio")


def ffmpeg_candidates() -> list[Path]:
    """ffmpeg binaries, preferring a copy inside the runtime tools directory."""
    root = runtime_root() / "tools"
    system = platform.system()
    candidates = [
        root / "ffmpeg",
        root / "ffmpeg.exe",
        root / "ffmpeg" / "bin" / "ffmpeg",
        root / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    if system == "Darwin":
        candidates.extend(
            [
                Path("/opt/homebrew/bin/ffmpeg"),
                Path("/usr/local/bin/ffmpeg"),
            ]
        )
    elif system != "Windows":
        candidates.extend(
            [
                Path("/usr/bin/ffmpeg"),
                Path("/usr/local/bin/ffmpeg"),
            ]
        )
    return candidates


def find_ffmpeg() -> str | None:
    configured = os.environ.get("FFMPEG_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if _executable(path) or path.is_file():
            return str(path)
    tools = runtime_root() / "tools"
    ordered = list(ffmpeg_candidates())
    for candidate in ordered:
        if _is_under(candidate, tools) and _executable(candidate):
            return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in ordered:
        if _executable(candidate):
            return str(candidate)
    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True
