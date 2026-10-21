"""Base fire scenarios shared by the field model and the Blender script."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    name: str
    # xmin, xmax, ymin, ymax, zmin, zmax in meters
    xb: tuple[float, float, float, float, float, float]
    # inert blocks rain and does not burn. fuel and ignition are visual roles;
    # the fire point stays a separate inflow.
    role: str = "inert"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    chid: str
    domain: tuple[float, float, float]
    origin: tuple[float, float, float]
    obstacles: tuple[Obstacle, ...]
    confinement: float
    smoke_scale: float
    buoyancy_scale: float
    plume_scale: float
    open_boundary: tuple[float, float, float, float, float, float] | None = None


def _room() -> ScenarioSpec:
    length, width, height = 8.0, 6.0, 3.0
    thickness = 0.2
    obstacles = (
        Obstacle("wall_west", (0.0, thickness, 0.0, width, 0.0, height)),
        Obstacle("wall_east_south", (length - thickness, length, 0.0, 2.0, 0.0, height)),
        Obstacle("wall_east_north", (length - thickness, length, 4.0, width, 0.0, height)),
        Obstacle("wall_east_lintel", (length - thickness, length, 2.0, 4.0, 2.2, height)),
        Obstacle("wall_south", (0.0, length, 0.0, thickness, 0.0, height)),
        Obstacle("wall_north", (0.0, length, width - thickness, width, 0.0, height)),
        Obstacle("ceiling", (0.0, length, 0.0, width, height - thickness, height)),
        Obstacle("fuel_bed", (0.6, 1.8, 0.6, 1.8, 0.0, 0.25), "ignition"),
    )
    return ScenarioSpec(
        name="房间火灾",
        chid="room_fire",
        domain=(length, width, height),
        origin=(1.2, 1.2, 0.25),
        obstacles=obstacles,
        confinement=1.45,
        smoke_scale=1.5,
        buoyancy_scale=0.8,
        plume_scale=0.85,
        open_boundary=(length, length, 2.0, 4.0, 0.0, 2.2),
    )


def _forest() -> ScenarioSpec:
    length, width, height = 40.0, 40.0, 20.0
    # No canopy and no fuel slab. Rain reaches the ground, and the trees are placed models.
    return ScenarioSpec(
        name="林区火灾",
        chid="forest_fire",
        domain=(length, width, height),
        origin=(20.0, 20.0, 0.4),
        obstacles=(),
        confinement=0.72,
        smoke_scale=2.4,
        buoyancy_scale=1.15,
        plume_scale=1.6,
    )


def _industrial() -> ScenarioSpec:
    length, width, height = 30.0, 20.0, 12.0
    obstacles = (
        Obstacle("tank", (8.0, 14.0, 6.0, 12.0, 0.0, 6.0)),
        Obstacle("pipe_rack", (15.0, 24.0, 4.0, 7.0, 0.0, 4.5)),
        Obstacle("pool", (7.5, 14.5, 5.5, 12.5, 0.0, 0.15), "ignition"),
    )
    return ScenarioSpec(
        name="工业区火灾",
        chid="industrial_fire",
        domain=(length, width, height),
        origin=(11.0, 9.0, 0.15),
        obstacles=obstacles,
        confinement=1.05,
        smoke_scale=1.8,
        buoyancy_scale=1.0,
        plume_scale=1.25,
    )


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.name: spec for spec in (_room(), _forest(), _industrial())
}

GRID_SHAPE = (16, 16, 8)


def get_scenario(name: str) -> ScenarioSpec:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        known = "、".join(SCENARIOS)
        raise ValueError(f"未知场景 {name!r}，可选：{known}") from exc


def spec_to_dict(spec: ScenarioSpec) -> dict:
    return {
        "name": spec.name,
        "chid": spec.chid,
        "domain": list(spec.domain),
        "origin": list(spec.origin),
        "obstacles": [{"name": item.name, "xb": list(item.xb), "role": item.role} for item in spec.obstacles],
        "confinement": spec.confinement,
        "smoke_scale": spec.smoke_scale,
        "buoyancy_scale": spec.buoyancy_scale,
        "plume_scale": spec.plume_scale,
        "open_boundary": None if spec.open_boundary is None else list(spec.open_boundary),
    }
