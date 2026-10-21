"""Still picture of a scenario's obstacles and fuel bed, without a flame."""

from __future__ import annotations

import numpy as np

from firephantom.physics.scenarios import get_scenario


def render_scenario_plate(scenario: str, placements: list | None = None) -> np.ndarray:
    """Return a 256×256 side view of the scenario geometry.

    Horizontal is the scenario length and vertical is height. Fuel is drawn
    apart from walls so the base scene can be checked before any engine runs.
    """
    spec = get_scenario(scenario)
    length, _width, height = spec.domain
    image = np.full((256, 256, 3), (32, 36, 34), dtype=np.uint8)
    ground = int(np.clip(255 - 0.0, 0, 255))
    image[max(0, ground - 2):256, :] = (54, 58, 52)
    for obstacle in spec.obstacles:
        x0, x1, _y0, _y1, z0, z1 = obstacle.xb
        col0 = int(np.clip(x0 / length * 255, 0, 255))
        col1 = int(np.clip(np.ceil(x1 / length * 255), 0, 255))
        row0 = int(np.clip(255 - z1 / max(height, 0.1) * 255, 0, 255))
        row1 = int(np.clip(np.ceil(255 - z0 / max(height, 0.1) * 255), 0, 255))
        if col1 <= col0:
            col1 = min(255, col0 + 2)
        if row1 <= row0:
            row1 = min(255, row0 + 2)
        color = (176, 112, 62) if "fuel" in obstacle.name else (96, 114, 106)
        image[row0:row1, col0:col1] = color
    for package in placements or []:
        xb = package.get("xb")
        if not xb or len(xb) != 6:
            continue
        x0, x1, _y0, _y1, z0, z1 = xb
        col0 = int(np.clip(x0 / length * 255, 0, 255))
        col1 = int(np.clip(np.ceil(x1 / length * 255), 0, 255))
        row0 = int(np.clip(255 - z1 / max(height, 0.1) * 255, 0, 255))
        row1 = int(np.clip(np.ceil(255 - z0 / max(height, 0.1) * 255), 0, 255))
        if col1 <= col0:
            col1 = min(255, col0 + 2)
        if row1 <= row0:
            row1 = min(255, row0 + 2)
        color = (186, 78, 42) if package.get("ignition") else (168, 132, 74)
        image[row0:row1, col0:col1] = color
    return image
