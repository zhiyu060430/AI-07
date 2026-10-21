"""Saved scenes: a scenario shell, its origin fuel and a fuel layout, reusable by jobs.

Every content change is a new revision, and old revisions are kept, so a job
records the exact layout it rendered (``scene_id`` + ``scene_revision``) even
after the scene is edited again. Renaming does not make a revision.
Thumbnails are PNGs the editor uploads from its viewport.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from firephantom.db import connect, ensure_schema

MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PLACEMENT_KEYS = ("asset_id", "name", "label", "x", "y", "z", "yaw", "fuel_type", "ignition", "size")
_MISSING = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remembered_size(row: dict) -> list[float] | None:
    """Editor rows omit size for catalog models. Preset sizes fill that gap."""
    size = row.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 3:
        return [float(value) for value in size]
    from firephantom.physics.placements import PRESETS

    preset = PRESETS.get(str(row.get("asset_id") or ""))
    if preset is None:
        return None
    return [float(value) for value in preset["size"]]


def _with_sizes(placements: list) -> list[dict]:
    filled = []
    for row in placements:
        item = dict(row)
        size = _remembered_size(item)
        if size is not None:
            item["size"] = size
        filled.append(item)
    return filled


def _clean_placements(spec, placements: object, origin_fuel: str) -> list[dict]:
    """Validate a layout and keep it in the editor's own form.

    The editor does not send a size for catalog models. The stored row still
    keeps one, so reopening the scene can draw the box without the catalog.
    """
    from firephantom.physics.placements import normalize_placements

    rows = list(placements or [])
    packages = normalize_placements(spec, rows, origin_fuel)
    cleaned = []
    for row, package in zip(rows, packages, strict=True):
        kept = {key: row[key] for key in _PLACEMENT_KEYS if key in row}
        kept["size"] = [float(value) for value in package["size"]]
        cleaned.append(kept)
    return cleaned


def _clean_content(
    scenario: str,
    placements: object,
    origin_fuel: str,
    origin: object,
    cameras: object,
    gas_domain: object,
    lights: object,
    simulation: object = None,
    structures: object = None,
    domain: object = None,
):
    """Editor rows plus rig. Automatic cameras, gas boxes, lamps, and default simulation settings are stored empty.

    Null structures and domain keep the preset. An explicit structure list, including
    an empty one, is the user's boxes.
    """
    from firephantom.physics.placements import (
        collapse_cameras,
        collapse_gas,
        collapse_lights,
        layout_environment,
        normalize_camera_map,
        normalize_gas_domain,
        normalize_lights,
        normalize_placements,
        prepared_spec,
        resolved_gas,
    )
    from firephantom.physics.simulation import collapse_simulation, normalize_simulation

    spec, kept_domain, kept_structures, kept_origin = prepared_spec(scenario, domain, structures, origin)
    cleaned = _clean_placements(spec, placements, origin_fuel)
    packages = normalize_placements(spec, cleaned, origin_fuel)
    kept_cameras = collapse_cameras(spec, normalize_camera_map(spec, cameras))
    kept_gas = collapse_gas(spec, packages, normalize_gas_domain(spec, gas_domain))
    environment = layout_environment(spec, resolved_gas(spec, packages, kept_gas), kept_cameras, ("侧视", "俯视", "第一人称"))
    kept_lights = collapse_lights(spec, normalize_lights(spec, lights), environment)
    kept_simulation = collapse_simulation(normalize_simulation(simulation))
    return cleaned, kept_origin, kept_cameras, kept_gas, kept_lights, kept_simulation, kept_structures, kept_domain


def _load_origin(value: object):
    if not value:
        return None
    return json.loads(value)


def _load_cameras(value: object) -> dict:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _load_gas(value: object):
    if not value:
        return None
    return json.loads(value)


def _load_structures(value: object):
    if not value:
        return None
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else None


def _load_domain(value: object):
    if not value:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, list) or len(loaded) != 3:
        return None
    return [float(item) for item in loaded]


def _load_lights(value: object):
    if value is None or value == "":
        return None
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else None


def _dump(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _check_scenario(scenario: str) -> str:
    from firephantom.physics.scenarios import SCENARIOS

    name = str(scenario or "").strip()
    if name not in SCENARIOS:
        raise ValueError(f"未知火灾场景 {name!r}")
    return name


def _check_name(name: object) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("场景名称不能为空")
    if len(text) > 120:
        raise ValueError("场景名称最多 120 个字")
    return text


def _check_fuel(fuel: object) -> str:
    text = str(fuel or "").strip()
    if not text:
        raise ValueError("起火点燃料不能为空")
    if len(text) > 128:
        raise ValueError("fuel_type must be at most 128 characters")
    return text


class SceneStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        ensure_schema()

    @contextmanager
    def _db(self):
        with connect() as connection:
            yield connection

    def thumbnail_path(self, scene_id: str) -> Path:
        return self.root / "scenes" / f"{scene_id}.png"

    def _public(self, row) -> dict:
        stamp = row["updated_at"]
        return {
            "scene_id": row["id"],
            "name": row["name"],
            "scenario": row["scenario"],
            "origin_fuel": row["origin_fuel"],
            "placements": _with_sizes(json.loads(row["placements"])),
            "origin": _load_origin(row["origin"]),
            "cameras": _load_cameras(row["cameras"]),
            "gas_domain": _load_gas(row["gas_domain"]),
            "lights": _load_lights(row["lights"]),
            "simulation": _load_gas(row["simulation"]),
            "structures": _load_structures(row["structures"]),
            "domain": _load_domain(row["domain"]),
            "revision": int(row["revision"]),
            "thumbnail_url": f"/api/v1/scenes/{row['id']}/thumbnail?v={stamp}" if row["has_thumbnail"] else None,
            "created_by": row["created_by"],
            "created_at": row["created_at"],
            "updated_at": stamp,
        }

    def list_all(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM scenes ORDER BY updated_at DESC").fetchall()
        return [self._public(row) for row in rows]

    def get(self, scene_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM scenes WHERE id=?", (scene_id,)).fetchone()
        return self._public(row) if row else None

    def revision(self, scene_id: str, revision: int | None = None) -> dict:
        """One stored revision (the latest when ``revision`` is None)."""
        with self._db() as db:
            if revision is None:
                row = db.execute(
                    "SELECT * FROM scene_revisions WHERE scene_id=? ORDER BY revision DESC LIMIT 1",
                    (scene_id,),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM scene_revisions WHERE scene_id=? AND revision=?",
                    (scene_id, int(revision)),
                ).fetchone()
        if row is None:
            raise KeyError(f"unknown scene {scene_id}" if revision is None else f"unknown scene revision {scene_id}@{revision}")
        return {
            "scene_id": row["scene_id"],
            "revision": int(row["revision"]),
            "scenario": row["scenario"],
            "origin_fuel": row["origin_fuel"],
            "placements": _with_sizes(json.loads(row["placements"])),
            "origin": _load_origin(row["origin"]),
            "cameras": _load_cameras(row["cameras"]),
            "gas_domain": _load_gas(row["gas_domain"]),
            "lights": _load_lights(row["lights"]),
            "simulation": _load_gas(row["simulation"]),
            "structures": _load_structures(row["structures"]),
            "domain": _load_domain(row["domain"]),
        }

    def _write_revision(self, db, scene_id: str, revision: int, scenario: str, fuel: str, placements: list, origin, cameras: dict, gas_domain, lights, simulation, structures, domain, stamp: str) -> None:
        db.execute(
            """INSERT INTO scene_revisions(scene_id, revision, scenario, origin_fuel, placements, origin, cameras, gas_domain, lights,
                                           simulation, structures, domain, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scene_id,
                revision,
                scenario,
                fuel,
                json.dumps(placements, ensure_ascii=False),
                _dump(origin),
                _dump(cameras or {}),
                _dump(gas_domain),
                _dump(lights),
                _dump(simulation),
                _dump(structures),
                _dump(domain),
                stamp,
            ),
        )

    def create(
        self,
        name: str,
        scenario: str,
        origin_fuel: str,
        placements: list | None,
        created_by: str | None = None,
        *,
        origin: object = None,
        cameras: object = None,
        gas_domain: object = None,
        lights: object = None,
        simulation: object = None,
        structures: object = None,
        domain: object = None,
    ) -> dict:
        scenario = _check_scenario(scenario)
        fuel = _check_fuel(origin_fuel)
        cleaned, kept_origin, kept_cameras, kept_gas, kept_lights, kept_simulation, kept_structures, kept_domain = _clean_content(
            scenario, placements, fuel, origin, cameras, gas_domain, lights, simulation, structures, domain
        )
        scene_id = uuid.uuid4().hex
        stamp = _now()
        with self._db() as db:
            db.execute(
                """INSERT INTO scenes(id, name, scenario, origin_fuel, placements, origin, cameras, gas_domain, lights,
                                      simulation, structures, domain, revision, has_thumbnail, created_by, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1,0,?,?,?)""",
                (
                    scene_id,
                    _check_name(name),
                    scenario,
                    fuel,
                    json.dumps(cleaned, ensure_ascii=False),
                    _dump(kept_origin),
                    _dump(kept_cameras),
                    _dump(kept_gas),
                    _dump(kept_lights),
                    _dump(kept_simulation),
                    _dump(kept_structures),
                    _dump(kept_domain),
                    created_by,
                    stamp,
                    stamp,
                ),
            )
            self._write_revision(
                db, scene_id, 1, scenario, fuel, cleaned, kept_origin, kept_cameras, kept_gas, kept_lights, kept_simulation,
                kept_structures, kept_domain, stamp,
            )
        return self.get(scene_id)

    def update(
        self,
        scene_id: str,
        *,
        name: str | None = None,
        scenario: str | None = None,
        origin_fuel: str | None = None,
        placements: list | None = None,
        base_revision: int | None = None,
        origin: object = _MISSING,
        cameras: object = _MISSING,
        gas_domain: object = _MISSING,
        lights: object = _MISSING,
        simulation: object = _MISSING,
        structures: object = _MISSING,
        domain: object = _MISSING,
    ) -> dict:
        """Change a scene. A content change bumps the revision.

        ``base_revision`` is the revision the editor started from; a stale
        one is refused so two editors do not overwrite each other.
        Omitted rig fields stay as they are. An explicit null origin or gas
        box returns that part to the automatic value.
        """
        current = self.get(scene_id)
        if current is None:
            raise KeyError(f"unknown scene {scene_id}")
        if base_revision is not None and int(base_revision) != current["revision"]:
            raise ValueError(f"场景已被改到第 {current['revision']} 版，请先重新载入")
        next_scenario = _check_scenario(scenario) if scenario is not None else current["scenario"]
        next_fuel = _check_fuel(origin_fuel) if origin_fuel is not None else current["origin_fuel"]
        next_placements = current["placements"] if placements is None else placements
        next_origin = current["origin"] if origin is _MISSING else origin
        next_cameras = current["cameras"] if cameras is _MISSING else cameras
        next_gas = current["gas_domain"] if gas_domain is _MISSING else gas_domain
        next_lights = current["lights"] if lights is _MISSING else lights
        next_simulation = current["simulation"] if simulation is _MISSING else simulation
        next_structures = current["structures"] if structures is _MISSING else structures
        next_domain = current["domain"] if domain is _MISSING else domain
        cleaned, kept_origin, kept_cameras, kept_gas, kept_lights, kept_simulation, kept_structures, kept_domain = _clean_content(
            next_scenario, next_placements, next_fuel, next_origin, next_cameras, next_gas, next_lights, next_simulation,
            next_structures, next_domain,
        )
        changed = (
            next_scenario != current["scenario"]
            or next_fuel != current["origin_fuel"]
            or cleaned != current["placements"]
            or kept_origin != current["origin"]
            or kept_cameras != current["cameras"]
            or kept_gas != current["gas_domain"]
            or kept_lights != current["lights"]
            or kept_simulation != current["simulation"]
            or kept_structures != current["structures"]
            or kept_domain != current["domain"]
        )
        revision = current["revision"] + 1 if changed else current["revision"]
        stamp = _now()
        with self._db() as db:
            db.execute(
                """UPDATE scenes SET name=?, scenario=?, origin_fuel=?, placements=?, origin=?, cameras=?,
                                      gas_domain=?, lights=?, simulation=?, structures=?, domain=?, revision=?, updated_at=?
                   WHERE id=?""",
                (
                    _check_name(name) if name is not None else current["name"],
                    next_scenario,
                    next_fuel,
                    json.dumps(cleaned, ensure_ascii=False),
                    _dump(kept_origin),
                    _dump(kept_cameras),
                    _dump(kept_gas),
                    _dump(kept_lights),
                    _dump(kept_simulation),
                    _dump(kept_structures),
                    _dump(kept_domain),
                    revision,
                    stamp,
                    scene_id,
                ),
            )
            if changed:
                self._write_revision(
                    db,
                    scene_id,
                    revision,
                    next_scenario,
                    next_fuel,
                    cleaned,
                    kept_origin,
                    kept_cameras,
                    kept_gas,
                    kept_lights,
                    kept_simulation,
                    kept_structures,
                    kept_domain,
                    stamp,
                )
        return self.get(scene_id)

    def duplicate(self, scene_id: str, name: str | None = None, created_by: str | None = None) -> dict:
        source = self.get(scene_id)
        if source is None:
            raise KeyError(f"unknown scene {scene_id}")
        copy = self.create(
            name or f"{source['name']} 副本",
            source["scenario"],
            source["origin_fuel"],
            source["placements"],
            created_by=created_by,
            origin=source["origin"],
            cameras=source["cameras"],
            gas_domain=source["gas_domain"],
            lights=source["lights"],
            simulation=source["simulation"],
            structures=source["structures"],
            domain=source["domain"],
        )
        thumbnail = self.thumbnail_path(scene_id)
        if thumbnail.is_file():
            self.save_thumbnail(copy["scene_id"], thumbnail.read_bytes())
            copy = self.get(copy["scene_id"])
        return copy

    def delete(self, scene_id: str) -> None:
        """Remove the scene from the library. Its revisions stay for job provenance."""
        with self._db() as db:
            deleted = db.execute("DELETE FROM scenes WHERE id=?", (scene_id,)).rowcount
        if not deleted:
            raise KeyError(f"unknown scene {scene_id}")
        self.thumbnail_path(scene_id).unlink(missing_ok=True)

    def save_thumbnail(self, scene_id: str, data: bytes) -> dict:
        if self.get(scene_id) is None:
            raise KeyError(f"unknown scene {scene_id}")
        if len(data) > MAX_THUMBNAIL_BYTES:
            raise ValueError("缩略图不能超过 2 MB")
        if not data.startswith(_PNG_MAGIC):
            raise ValueError("缩略图必须是 PNG")
        path = self.thumbnail_path(scene_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        with self._db() as db:
            db.execute("UPDATE scenes SET has_thumbnail=1, updated_at=? WHERE id=?", (_now(), scene_id))
        return self.get(scene_id)


def resolve_scene_items(root: Path | str, items: list[dict]) -> list[dict]:
    """Fill scenario, origin fuel and layout of rows that name a saved scene.

    The row keeps ``scene_id`` and gets the exact ``scene_revision`` used, so
    later edits to the scene do not change what this job rendered.
    """
    store: SceneStore | None = None
    resolved = []
    for item in items:
        if not isinstance(item, dict) or not item.get("scene_id"):
            resolved.append(item)
            continue
        store = store or SceneStore(root)
        revision = store.revision(str(item["scene_id"]), item.get("scene_revision"))
        resolved.append({
            **item,
            "scenario": revision["scenario"],
            "fuel_type": revision["origin_fuel"],
            "placements": revision["placements"],
            "origin": revision["origin"],
            "cameras": revision["cameras"],
            "gas_domain": revision["gas_domain"],
            "lights": revision["lights"],
            "simulation": revision["simulation"],
            "structures": revision["structures"],
            "domain": revision["domain"],
            "scene_revision": revision["revision"],
        })
    return resolved
