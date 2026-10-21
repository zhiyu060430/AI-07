"""Parameter-dependent plume for temperature, smoke, and flow.

Blender renders the pictures. These grids feed the point cloud and the
fallback frames when Blender produces none. Exported metadata marks the
backend as parameterized-plume.
"""

from __future__ import annotations

import hashlib

import numpy as np

from firephantom.physics.scenarios import GRID_SHAPE, ScenarioSpec

_FUEL_TABLE: dict[str, tuple[float, float]] = {
    "木材": (900.0, 0.018),
    "wood": (900.0, 0.018),
    "汽油": (2800.0, 0.11),
    "gasoline": (2800.0, 0.11),
    "塑料": (1900.0, 0.13),
    "plastic": (1900.0, 0.13),
    "植被": (650.0, 0.025),
    "vegetation": (650.0, 0.025),
    "化学品": (3400.0, 0.09),
    "chemical": (3400.0, 0.09),
}


def fuel_properties(fuel_type: str) -> dict:
    """Map a fuel name to heat-release rate (kW) and soot yield."""
    if fuel_type in _FUEL_TABLE:
        heat_release_kw, soot_yield = _FUEL_TABLE[fuel_type]
    else:
        digest = hashlib.sha256(fuel_type.encode("utf-8")).digest()
        heat_release_kw = 500.0 + digest[0] / 255.0 * 3000.0
        soot_yield = 0.01 + digest[1] / 255.0 * 0.2
    return {
        "name": fuel_type,
        "hrr_kw": float(heat_release_kw),
        "soot_yield": float(soot_yield),
    }


def compute_fields(
    spec: ScenarioSpec,
    wind_speed: float,
    temperature: float,
    humidity: float,
    fuel_type: str,
) -> dict:
    """Return temperature (C), smoke, and velocity (m/s) on the scenario grid.

    Every supplied parameter enters all three fields: wind advects the plume
    and sets the streamwise velocity, ambient temperature shifts the thermal
    field and the buoyancy density, humidity cools the plume and suppresses
    smoke, and the fuel sets the heat-release rate and soot yield.
    """
    wind = float(wind_speed)
    ambient_c = float(temperature)
    relative_humidity = float(humidity)
    fuel = fuel_properties(fuel_type)
    heat = fuel["hrr_kw"]
    soot = fuel["soot_yield"]

    nx, ny, nz = GRID_SHAPE
    length, width, height = spec.domain
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, width, ny)
    zs = np.linspace(0.0, height, nz)
    coord_x, coord_y, coord_z = np.meshgrid(xs, ys, zs, indexing="ij")

    origin_x, origin_y, origin_z = spec.origin
    # Keep the virtual origin off the singular z=0 of a point-plume formula.
    height_above = np.maximum(coord_z - origin_z, 0.0) + 0.8
    wind_shift = wind * (height_above / height) * length * 0.18
    center_x = origin_x + wind_shift
    center_y = origin_y + 0.05 * wind * np.sin(height_above / height)
    radius = (0.35 + 0.14 * height_above) * spec.plume_scale
    radial = (coord_x - center_x) ** 2 + (coord_y - center_y) ** 2
    gaussian = np.exp(-radial / (2.0 * np.maximum(radius, 0.2) ** 2))

    # Milder than the unbounded near-field Heskestad limit, still monotonic in Q.
    centerline_rise = 18.0 * (heat ** 0.42) / (height_above ** 1.05)
    humidity_cooling = 1.0 - 0.0022 * relative_humidity
    rise = centerline_rise * gaussian * humidity_cooling * spec.confinement
    temperature_field = ambient_c + rise

    ambient_k = ambient_c + 273.15
    density_ratio = 293.15 / np.maximum(ambient_k, 200.0)
    humidity_smoke = np.maximum(1.0 - 0.0035 * relative_humidity, 0.05)
    dilution = 1.0 + 0.12 * np.abs(wind) * (height_above / height)
    smoke_field = (
        soot
        * (heat ** 0.5)
        * gaussian
        * humidity_smoke
        * density_ratio
        * spec.smoke_scale
        / dilution
    )
    smoke_field = np.maximum(smoke_field, 0.0)

    gravity = 9.81
    thermal_k = temperature_field + 273.15
    buoyancy = np.sqrt(
        np.maximum(2.0 * gravity * height_above * np.maximum(rise, 0.0) / thermal_k, 0.0)
    )
    entrainment = 0.28 * buoyancy / np.maximum(radius, 0.2)
    velocity_x = wind + entrainment * (coord_x - center_x)
    velocity_y = 0.15 * wind + entrainment * (coord_y - center_y)
    velocity_z = buoyancy * spec.buoyancy_scale
    flow_field = np.stack([velocity_x, velocity_y, velocity_z], axis=-1)

    return {
        "temperature": temperature_field,
        "smoke": smoke_field,
        "flow": flow_field,
        "axes": {"x": xs, "y": ys, "z": zs},
        "fuel": fuel,
        "parameters": {
            "wind_speed": wind,
            "temperature": ambient_c,
            "humidity": relative_humidity,
            "fuel_type": fuel_type,
        },
    }
