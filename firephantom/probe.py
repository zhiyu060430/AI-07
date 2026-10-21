"""Probe optional external tools and print a stable log."""

from __future__ import annotations

from firephantom.annotation.labelstudio import push_task
from firephantom.physics.adapters import run_blender
from firephantom.quality.metrics import inception_backend
from firephantom.synthesis.stable_diffusion import call_stable_diffusion


def collect_report() -> str:
    lines = []
    lines.append(run_blender(None)["message"])
    diffusion = call_stable_diffusion("probe flame image", timeout=2.0)
    if diffusion["success"]:
        lines.append(f"stable diffusion endpoint {diffusion['endpoint']} call succeeded")
    else:
        lines.append(
            f"stable diffusion endpoint {diffusion['endpoint']} call failed: {diffusion['error']}"
        )
    studio = push_task({"data": {"probe": True}}, timeout=1.5)
    if studio["success"]:
        lines.append(f"label studio endpoint {studio['endpoint']} call succeeded")
    else:
        lines.append(f"label studio endpoint {studio['endpoint']} call failed: {studio['error']}")
    lines.append(inception_backend())
    return "\n".join(lines) + "\n"


def main() -> None:
    print(collect_report(), end="")


if __name__ == "__main__":
    main()
