"""Thin process adapter for the Blender executable.

A missing binary is reported and does not invent solver output. Field values
come from the parameter-dependent plume model, not from this adapter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def _missing(tool: str) -> dict:
    return {
        "tool": tool,
        "executed": False,
        "success": False,
        "returncode": None,
        "message": f"{tool} executable was not run: not found on PATH",
        "stdout": "",
        "stderr": "",
        "outputs": {},
    }


def run_blender(script_file: Path | None, timeout: float = 60.0) -> dict:
    exe = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if exe is None:
        return _missing("blender")
    if script_file is None:
        return {
            "tool": "blender",
            "executed": False,
            "success": False,
            "returncode": None,
            "message": "blender executable was not run: scene script was not provided",
            "stdout": "",
            "stderr": "",
            "outputs": {},
        }
    script_path = Path(script_file)
    # Kept on every run so physics/simulation.py can be refitted from real bakes.
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [exe, "--background", "--python", script_path.name],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "tool": "blender", "executed": True, "success": False, "returncode": None,
            "message": f"blender executable timed out after {timeout:g}s",
            "stdout": str(exc.stdout or "")[-12000:], "stderr": str(exc.stderr or "")[-12000:],
            "outputs": {}, "elapsed_seconds": round(time.monotonic() - started, 1),
        }
    except OSError as exc:
        return {
            "tool": "blender", "executed": False, "success": False, "returncode": None,
            "message": f"blender executable could not be started: {exc}", "stdout": "", "stderr": str(exc),
            "outputs": {},
        }
    return {
        "tool": "blender",
        "executed": True,
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "message": "blender executable completed successfully" if proc.returncode == 0 else f"blender executable exited with status {proc.returncode}",
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
        "outputs": {},
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
