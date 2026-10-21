"""Runtime probes used by the API and local model services.

The project is commonly run on an Apple Silicon machine with unified memory.
These probes are intentionally best-effort: a missing OS probe never prevents
the application from starting, but a known-low memory condition is surfaced so
callers can avoid loading another large model.
"""

from __future__ import annotations

import os
import math
import platform
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None


def available_memory_bytes() -> int | None:
    """Return a conservative available-memory estimate on macOS."""
    if platform.system() != "Darwin":
        return None
    page_size_raw = _command_output(["sysctl", "-n", "vm.pagesize"])
    vm_stat = _command_output(["vm_stat"])
    if not page_size_raw or not vm_stat:
        return None
    try:
        page_size = int(page_size_raw.strip())
        values: dict[str, int] = {}
        for line in vm_stat.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            raw = raw.strip().rstrip(".")
            if raw.isdigit():
                values[key.strip()] = int(raw)
        pages = sum(
            values.get(key, 0)
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        )
        return pages * page_size
    except (TypeError, ValueError):
        return None


def total_memory_bytes() -> int | None:
    raw = _command_output(["sysctl", "-n", "hw.memsize"]) if platform.system() == "Darwin" else None
    try:
        return int(raw.strip()) if raw else None
    except (TypeError, ValueError):
        return None


def swap_status() -> dict[str, int] | None:
    """Return macOS swap totals in bytes when the platform exposes them."""
    raw = _command_output(["sysctl", "-n", "vm.swapusage"]) if platform.system() == "Darwin" else None
    if not raw:
        return None
    import re
    parsed: dict[str, int] = {}
    for match in re.finditer(r"(total|used|free)\s*=\s*([0-9.]+)([KMG]?)", raw):
        number, unit = float(match.group(2)), match.group(3)
        parsed[match.group(1)] = int(number * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit])
    return parsed or None


def memory_status(min_available_gb: float | None = None) -> dict:
    """Describe memory headroom and whether a heavy model may be loaded."""
    raw_threshold = min_available_gb if min_available_gb is not None else os.environ.get("FIREPHANTOM_MIN_AVAILABLE_GB", "3")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        threshold = 3.0
    if not math.isfinite(threshold) or threshold < 0:
        threshold = 3.0
    available = available_memory_bytes()
    total = total_memory_bytes()
    threshold_bytes = max(0, int(threshold * 1024**3))
    if available is None:
        state = "unknown"
    elif available < threshold_bytes:
        state = "critical"
    elif available < threshold_bytes * 2:
        state = "warning"
    else:
        state = "ok"
    return {
        "platform": platform.system(),
        "total_bytes": total,
        "available_bytes": available,
        "total_gib": None if total is None else round(total / 1024**3, 2),
        "available_gib": None if available is None else round(available / 1024**3, 2),
        "minimum_available_gib": threshold,
        "state": state,
        "swap": swap_status(),
    }


def _executable_status(configured: str | None, fallback: str) -> dict:
    value = (configured or "").strip()
    path = Path(value) if value else (Path(shutil.which(fallback)) if shutil.which(fallback) else None)
    return {
        "configured": bool(value),
        "path": None if path is None else str(path),
        "available": bool(path and path.is_file() and os.access(path, os.X_OK)),
    }


def _service_status(endpoint: str | None, health_path: str) -> dict:
    target = (endpoint or "").strip()
    if not target:
        return {"configured": False, "reachable": False, "url": None}
    parsed = urlsplit(target)
    base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    url = f"{base}{health_path}"
    try:
        request = Request(url, method="GET")
        # Health checks are local service probes; never send them through a
        # user's download proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=0.6) as response:
            return {"configured": True, "reachable": 200 <= response.status < 500, "url": target}
    except Exception:
        return {"configured": True, "reachable": False, "url": target}


def system_status() -> dict:
    """Return resource headroom and configured integration availability."""
    memory = memory_status()
    integrations = {
        "blender": _executable_status(os.environ.get("BLENDER_BIN"), "blender"),
        "stable_diffusion": _service_status(os.environ.get("SD_API_URL"), "/health"),
        "label_studio": _service_status(os.environ.get("LABEL_STUDIO_URL"), "/api/health"),
    }
    return {
        "status": "degraded" if memory["state"] in {"critical", "warning", "unknown"} else "ok",
        "memory": memory,
        "integrations": integrations,
    }


def safe_job_workers(requested: int) -> int:
    """Choose a worker limit that respects the local unified-memory budget."""
    try:
        workers = max(1, min(int(requested), 8))
    except (TypeError, ValueError):
        workers = 1
    if platform.system() != "Darwin":
        return workers
    total = total_memory_bytes()
    allow = os.environ.get("FIREPHANTOM_ALLOW_CONCURRENT_MPS", "").lower() in {"1", "true", "yes"}
    if total is not None and total <= 16 * 1024**3 and not allow:
        return 1
    return workers
