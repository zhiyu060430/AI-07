"""Fuel packages, spread decks, and the scene-to-Three.js frame."""

import json
import math
import struct

import pytest

from firephantom.physics.blender_scene import (
    SOLID_CLEARANCE,
    clear_of_solids,
    fill_light,
    render_blender,
    visible_obstacles,
)
from firephantom.physics.engine import ENGINE_CACHE_VERSION
from firephantom.physics.fields import fuel_properties
from firephantom.physics.placements import (
    automatic_lights,
    collapse_lights,
    default_placements,
    gas_domain,
    glb_stats,
    ignition_fronts,
    normalize_lights,
    normalize_placements,
    prepare,
    scene_to_three,
    store_upload,
    three_to_scene,
    three_to_yaw,
    yaw_to_three,
)
from firephantom.physics.scenarios import get_scenario
from firephantom.validation import expand_job_items


def _payload(script: str) -> dict:
    start = script.index('r"""') + 4
    end = script.index('"""', start)
    return json.loads(script[start:end])


def make_glb(minimum, maximum, triangles=1) -> bytes:
    document = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"componentType": 5126, "count": 3, "type": "VEC3", "min": list(minimum), "max": list(maximum)},
            {"componentType": 5123, "count": triangles * 3, "type": "SCALAR"},
        ],
    }
    payload = json.dumps(document).encode("utf-8")
    while len(payload) % 4:
        payload += b" "
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_scene_axes_round_trip_and_yaw_matches_three_rotation():
    assert three_to_scene(*scene_to_three(3.0, 4.0, 5.0)) == pytest.approx((3.0, 4.0, 5.0))
    assert scene_to_three(*three_to_scene(1.5, 2.5, -3.5)) == pytest.approx((1.5, 2.5, -3.5))
    yaw = math.pi / 2
    assert yaw_to_three(yaw) == pytest.approx(yaw)
    assert three_to_yaw(yaw_to_three(0.3)) == pytest.approx(0.3)
    # Scene +X rotated by π/2 about +Z lands on +Y, which is Three.js (0, 0, -1).
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotated = (cosine * 1.0 - sine * 0.0, sine * 1.0 + cosine * 0.0, 0.0)
    assert rotated[0] == pytest.approx(0.0)
    assert rotated[1] == pytest.approx(1.0)
    assert scene_to_three(*rotated) == pytest.approx((0.0, 0.0, -1.0))


def test_defaults_fit_and_replay_without_client_coordinates():
    room = default_placements("房间火灾")
    assert [item["asset_id"] for item in room] == ["potted_plant_04", "painted_wooden_sofa", "WoodenTable_01"]
    forest = default_placements("林区火灾")
    forest_ids = {item["asset_id"] for item in forest}
    assert len(forest) == 8
    assert "boulder_01" not in forest_ids
    assert {"PineTree_1", "PineTree_2", "Bush_1", "Bush_2", "TreeStump", "WoodLog"} <= forest_ids
    assert {item["name"] for item in forest} >= {"松树 1", "松树 2", "灌木 1", "树桩", "倒木"}
    assert "松树 11" not in {item["name"] for item in forest}
    assert all(item.name != "canopy" for item in get_scenario("林区火灾").obstacles)
    industrial = default_placements("工业区火灾")
    assert len(industrial) == 8
    assert "Barrel_01" in {item["asset_id"] for item in industrial}
    assert {item["name"] for item in industrial} >= {"油桶 1", "油桶 2", "大木箱", "液化气罐"}
    for name in ("房间火灾", "林区火灾", "工业区火灾"):
        spec = get_scenario(name)
        raw = default_placements(name)
        once = normalize_placements(spec, raw, "木材")
        twice = normalize_placements(spec, once, "木材")
        assert once
        assert [item["xb"] for item in twice] == [item["xb"] for item in once]
        layout = prepare(spec, once, "木材")
        # The automatic gas box stays inside 4 m of the fire. Forest props stand
        # in a grove beside that box, and the industrial barrels sit outside the
        # tank, so only the room defaults are all inside the bake volume.
        if name == "房间火灾":
            assert all(item["in_domain"] for item in layout["packages"])
        assert layout["gas_domain"]["resolution"] == 32


def test_too_many_packages_shaft_and_subcell_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREPHANTOM_ASSETS", str(tmp_path))
    spec = get_scenario("房间火灾")
    crowd = [
        {"asset_id": "potted_plant_04", "x": 4.2, "y": 3.0, "z": 0.25, "yaw": 0.0, "fuel_type": "植被"}
        for _ in range(9)
    ]
    with pytest.raises(ValueError, match="最多摆"):
        normalize_placements(spec, crowd, "木材")
    with pytest.raises(ValueError, match="挡住了起火点"):
        normalize_placements(
            spec,
            [{"asset_id": "potted_plant_04", "x": 1.2, "y": 1.2, "z": 0.25, "fuel_type": "植被"}],
            "木材",
        )
    with pytest.raises(ValueError, match="未知资产"):
        normalize_placements(spec, [{"asset_id": "missing", "x": 4.0, "y": 3.0, "z": 0.4}], "木材")
    token = "abcdef1234567890"
    upload = tmp_path / "uploads"
    upload.mkdir()
    (upload / f"{token}.glb").write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="小于一格"):
        normalize_placements(
            spec,
            [{"asset_id": f"upload:{token}", "x": 4.0, "y": 3.0, "z": 0.2, "size": [0.2, 0.6, 0.6], "fuel_type": "木材"}],
            "木材",
        )


def test_fill_light_stays_inside_the_domain_and_off_the_solids():
    for name in ("房间火灾", "林区火灾", "工业区火灾"):
        spec = get_scenario(name)
        obstacles = visible_obstacles(spec)
        light = fill_light(spec.domain, spec.origin, obstacles)
        length, width, height = spec.domain
        x, y, z = light["location"]
        assert light["energy"] == 20.0
        assert light["size"] == 1.0
        assert 0.35 <= x <= length - 0.35
        assert 0.35 <= y <= width - 0.35
        assert 0.2 < z < height
        for item in obstacles:
            x0, x1, y0, y1, z0, z1 = item["xb"]
            assert not (x0 < x < x1 and y0 < y < y1 and z0 < z < z1)
    script = render_blender(get_scenario("房间火灾"), 1.5, 28.0, 45.0, "木材")
    assert 'volume.inputs["Temperature Attribute"].default_value = ""' in script
    assert 'type="POINT"' in script
    assert "shadow_soft_size" in script
    room = _payload(script)["fill_light"]["location"]
    assert room[1] > 0.5


def test_empty_placements_keep_the_old_gas_domain_and_cache_generation():
    spec = get_scenario("房间火灾")
    ox, oy, oz = spec.origin
    height = 2.25 * (0.92 + 0.08 * float(spec.plume_scale) / 0.85)
    domain = gas_domain(spec, [])
    assert domain["resolution"] == 40
    assert domain["location"] == pytest.approx([ox + 0.35, oy, oz + height / 2.0])
    assert domain["scale"] == pytest.approx([1.2, 1.2, height / 2.0])
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材")
    payload = _payload(script)
    assert payload["gas_domain"]["resolution"] == 40
    # Only the face in wall_south moves, to the wall's inner face plus the gap.
    assert payload["gas_domain"] == pytest.approx(clear_of_solids(domain, visible_obstacles(spec)))
    assert _bounds(payload["gas_domain"])[2] == pytest.approx(0.2 + SOLID_CLEARANCE)
    assert payload["packages"] == []
    assert ENGINE_CACHE_VERSION == "engine-renders-v15"


def _bounds(box: dict) -> list[float]:
    return [
        value
        for axis in range(3)
        for value in (box["location"][axis] - box["scale"][axis], box["location"][axis] + box["scale"][axis])
    ]


def test_gas_box_clears_the_floor_and_walls():
    # A face on the floor or in a wall left hard squares of flame glow there.
    box = {"location": [2.35, 2.0, 1.25], "scale": [2.0, 2.0, 1.25], "resolution": 32}
    wall_south = {"name": "wall_south", "xb": [0.0, 8.0, 0.0, 0.2, 0.0, 3.0]}
    # A ceiling with the shaft cut does not cover the top face, so it stays.
    ceiling = {"name": "ceiling_1", "xb": [2.2, 8.0, 0.0, 6.0, 2.8, 3.0]}
    cleared = clear_of_solids(box, [wall_south, ceiling])
    assert _bounds(cleared) == pytest.approx(
        [0.35, 4.35, 0.2 + SOLID_CLEARANCE, 4.0, SOLID_CLEARANCE, 2.5]
    )
    assert cleared["resolution"] == 32
    clear = {"location": [2.0, 2.0, 1.5], "scale": [1.0, 1.0, 1.25]}
    assert clear_of_solids(clear, [wall_south]) == clear
    spec = get_scenario("房间火灾")
    plant = {"asset_id": "potted_plant_04", "x": 2.55, "y": 1.2, "z": 0.25, "yaw": 0.0, "fuel_type": "植被"}
    payload = _payload(render_blender(spec, 1.5, 28.0, 45.0, "木材", placements=[plant]))
    x0, x1, y0, y1, z0, z1 = _bounds(payload["gas_domain"])
    assert z0 >= SOLID_CLEARANCE - 1e-9
    for item in payload["obstacles"]:
        a0, a1, b0, b1, c0, c1 = item["xb"]
        overlaps = a0 < x1 and x0 < a1 and b0 < y1 and y0 < b1 and c0 < z1 and z0 < c1
        assert not overlaps or not (a0 <= x0 and a1 >= x1 and c0 <= z0 and c1 >= z1), item["name"]


def test_room_package_is_an_ignitable_obstacle_and_the_shell_stays_inert():
    spec = get_scenario("房间火灾")
    plant = {"asset_id": "potted_plant_04", "x": 2.55, "y": 1.2, "z": 0.25, "yaw": 0.0, "fuel_type": "植被"}
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", placements=[plant])
    payload = _payload(script)
    obstacle_names = {item["name"] for item in payload["obstacles"]}
    assert any(name == "wall_south" or name.startswith("wall_south") for name in obstacle_names)
    assert any(name == "ceiling" or name.startswith("ceiling") for name in obstacle_names)
    # The flame shaft removes the fuel bed from the solids. It is not a second fuel package.
    assert not any(name == "fuel_bed" or name.startswith("fuel_bed") for name in obstacle_names)
    assert all(not str(item["name"]).startswith(("wall_", "ceiling", "fuel_bed")) for item in payload["packages"])
    assert payload["gas_domain"]["resolution"] == 32
    assert payload["spread_model"] == "reaction+delay"
    assert payload["packages"][0]["in_domain"] is True
    assert "frame_change_pre" in script
    assert "keyframe_insert" in script
    assert 'flow.name = "fuel_"' in script
    # The fire box, short of the 0.2 m shell walls it would run into.
    shell = 0.2 + SOLID_CLEARANCE
    fire = (
        max(spec.origin[0] + 0.35 - 1.2, shell),
        spec.origin[0] + 0.35 + 1.2,
        max(spec.origin[1] - 1.2, shell),
        spec.origin[1] + 1.2,
    )
    location, scale = payload["gas_domain"]["location"], payload["gas_domain"]["scale"]
    assert location[0] - scale[0] <= fire[0] + 1e-6
    assert location[0] + scale[0] >= fire[1] - 1e-6
    assert location[1] - scale[1] <= fire[2] + 1e-6
    assert location[1] + scale[1] >= fire[3] - 1e-6


def test_extra_ignition_is_its_own_inflow_and_the_origin_hrr_stays():
    spec = get_scenario("房间火灾")
    crate = {
        "asset_id": "wooden_crate_01",
        "x": 3.2,
        "y": 1.2,
        "z": 0.4,
        "yaw": 0.0,
        "fuel_type": "木材",
        "ignition": True,
    }
    alone = _payload(render_blender(spec, 1.5, 28.0, 45.0, "木材"))
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", placements=[crate])
    payload = _payload(script)
    assert payload["hrr_kw"] == alone["hrr_kw"] == pytest.approx(fuel_properties("木材")["hrr_kw"])
    package = payload["packages"][0]
    assert package["ignition"] is True
    assert package["fronts"][0]["full_frame"] == 12
    assert 'flow.name = "fire_inflow"' in script
    assert 'flow.name = "fuel_"' in script


def test_forest_without_placements_stays_a_local_gas_domain():
    spec = get_scenario("林区火灾")
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材")
    payload = _payload(script)
    scale = payload["gas_domain"]["scale"]
    assert scale[0] * 2 < spec.domain[0] * 0.5
    assert scale[1] * 2 < spec.domain[1] * 0.5
    assert payload["packages"] == []
    assert spec.obstacles == ()
    assert payload["obstacles"] == []
    assert "0.2 + soot" not in script
    assert "settings.density = 0.15" not in script
    assert "settings.density = 0.0" in script
    assert "_key_origin(flow, int(PARAMS[\"warmup\"]))" in script
    assert "_place_fuels(bpy, flow)" in script


def test_glb_upload_uses_scene_axes_and_scales_a_short_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREPHANTOM_ASSETS", str(tmp_path))
    data = make_glb((0.0, 0.0, 0.0), (1.0, 0.6, 0.8))
    stats = glb_stats(data)
    assert stats["triangles"] == 1
    assert stats["size"] == pytest.approx((1.0, 0.8, 0.6))
    stored = store_upload(data)
    assert stored["asset_id"].startswith("upload:")
    assert stored["size"] == pytest.approx([1.0, 0.8, 0.6])
    assert (tmp_path / "uploads" / f"{stored['asset_id'].split(':', 1)[1]}.glb").is_file()
    # glTF Y is the short axis, so the scene height is 0.2 m. Placement scales it up.
    thin = make_glb((0.0, 0.0, 0.0), (1.0, 0.2, 0.8))
    assert glb_stats(thin)["size"] == pytest.approx((1.0, 0.8, 0.2))
    scaled = store_upload(thin, label="thin")
    assert scaled["size"] == pytest.approx([2.25, 1.8, 0.45])
    assert min(scaled["size"]) == pytest.approx(0.45)
    with pytest.raises(ValueError, match="一条边是 0"):
        glb_stats(make_glb((0.0, 0.0, 0.0), (1.0, 0.0, 0.8)))
    with pytest.raises(ValueError, match="超过"):
        glb_stats(make_glb((0.0, 0.0, 0.0), (1.0, 0.6, 0.8), triangles=50_001))


def test_job_replay_accepts_the_stored_package_shape():
    raw = default_placements("房间火灾")
    items = expand_job_items([{
        "scenario": "房间火灾",
        "wind_speed": 1.5,
        "temperature": 28.0,
        "humidity": 45.0,
        "fuel_type": "木材",
        "scene": "室内",
        "weather": "晴",
        "viewpoint": "侧视",
        "placements": raw,
    }])
    assert len(items) == 1
    assert items[0]["placements"][0]["center"]
    again = expand_job_items(items)
    assert again[0]["placements"][0]["xb"] == items[0]["placements"][0]["xb"]


def test_catalog_lists_fuels_and_upload_route(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREPHANTOM_ASSETS", str(tmp_path / "assets"))
    from fastapi.testclient import TestClient

    from firephantom.api import create_app

    with TestClient(create_app(tmp_path / "data")) as client:
        catalog = client.get("/api/v1/catalog")
        assert catalog.status_code == 200
        body = catalog.json()
        assert "木材" in body["fuels"]
        assert {item["asset_id"] for item in body["assets"]} >= {"painted_wooden_sofa", "boulder_01"}
        assert body["defaults"]["房间火灾"]
        assert body["shells"]["工业区火灾"]["obstacles"]
        for name in ("房间火灾", "林区火灾", "工业区火灾"):
            camera = body["shells"][name]["camera"]
            assert len(camera["location"]) == 3
            assert len(camera["target"]) == 3
            assert camera["focal_mm"] > 0
            assert set(body["shells"][name]["cameras"]) == {"侧视", "俯视", "第一人称"}
        assert client.get("/api/v1/assets/uploads").json() == {"uploads": []}
        uploaded = client.post(
            "/api/v1/assets/uploads",
            files={"file": ("chair.glb", make_glb((0, 0, 0), (1.2, 0.7, 0.9)), "model/gltf-binary")},
        )
        assert uploaded.status_code == 200, uploaded.text
        payload = uploaded.json()
        assert payload["triangles"] == 1
        assert payload["url"].startswith("/assets/uploads/")
        listed = client.get("/api/v1/assets/uploads").json()["uploads"]
        assert [(item["asset_id"], item["label"]) for item in listed] == [(payload["asset_id"], "chair")]
        rejected = client.post(
            "/api/v1/assets/uploads",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert rejected.status_code == 400
        kept = make_glb((0, 0, 0), (1.2, 0.8, 0.9))
        other = make_glb((0, 0, 0), (0.9, 0.7, 1.1))
        huge = make_glb((0, 0, 0), (1.0, 0.6, 0.8), triangles=50_001)
        batch = client.post(
            "/api/v1/assets/uploads/batch",
            files=[
                ("files", ("one.glb", kept, "model/gltf-binary")),
                ("files", ("two.glb", other, "model/gltf-binary")),
                ("files", ("heavy.glb", huge, "model/gltf-binary")),
                ("files", ("notes.txt", b"nope", "text/plain")),
            ],
        )
        assert batch.status_code == 200, batch.text
        body = batch.json()
        assert len(body["accepted"]) == 2
        assert {item["label"] for item in body["accepted"]} == {"one", "two"}
        reasons = {item["name"]: item["reason"] for item in body["rejected"]}
        assert "heavy.glb" in reasons
        assert "超过" in reasons["heavy.glb"]
        assert reasons["notes.txt"] == "只接受 .glb"
        listed = {item["asset_id"] for item in client.get("/api/v1/assets/uploads").json()["uploads"]}
        assert {item["asset_id"] for item in body["accepted"]} <= listed
        builtin = client.delete("/api/v1/assets/uploads/PineTree_1")
        assert builtin.status_code == 400
        assert "不能删除" in builtin.json()["detail"]
        import io
        import zipfile

        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w") as archive:
            archive.writestr("nested/extra.glb", kept)
            archive.writestr("readme.txt", b"skip me")
        zipped = client.post(
            "/api/v1/assets/uploads/batch",
            files=[("files", ("pack.zip", packed.getvalue(), "application/zip"))],
        )
        assert zipped.status_code == 200, zipped.text
        assert [item["label"] for item in zipped.json()["accepted"]] == ["extra"]
        assert zipped.json()["rejected"] == []
        token = body["accepted"][0]["asset_id"].split(":", 1)[1]
        removed = client.delete(f"/api/v1/assets/uploads/{token}")
        assert removed.status_code == 200
        assert token not in {item["asset_id"].split(":", 1)[1] for item in client.get("/api/v1/assets/uploads").json()["uploads"]}


def test_overrides_are_checked_and_automatic_values_collapse():
    from firephantom.physics.blender_scene import camera_pose
    from firephantom.physics.placements import (
        collapse_cameras,
        collapse_gas,
        layout_preview,
        normalize_camera_map,
        normalize_gas_domain,
        normalize_origin,
    )
    from firephantom.physics.scenarios import get_scenario

    spec = get_scenario("房间火灾")
    with pytest.raises(ValueError, match="起火点"):
        normalize_origin(spec, [0.0, 1.2, 0.25])
    with pytest.raises(ValueError, match="焦距"):
        normalize_camera_map(spec, {"侧视": {"location": [3, 2, 1.2], "target": [1.2, 1.2, 1], "focal_mm": 200}})
    # Past the 4 m automatic cap is fine now; the bake budget limits it instead.
    wide = normalize_gas_domain(spec, {"location": [4, 3, 1.5], "scale": [2.1, 1, 1], "resolution": 128})
    assert wide["resolution"] == 128
    with pytest.raises(ValueError, match="超出了场景范围"):
        normalize_gas_domain(spec, {"location": [4, 3, 1.5], "scale": [4.5, 1, 1], "resolution": 32})
    with pytest.raises(ValueError, match="16 到 128"):
        normalize_gas_domain(spec, {"location": [4, 3, 1.5], "scale": [1, 1, 1], "resolution": 129})
    auto = camera_pose(spec, "侧视")
    assert collapse_cameras(spec, normalize_camera_map(spec, {"侧视": auto})) == {}
    assert collapse_gas(spec, [], normalize_gas_domain(spec, gas_domain(spec, []))) is None
    preview = layout_preview(
        "房间火灾",
        [],
        "木材",
        None,
        {"侧视": {"location": [4.0, 1.0, 1.2], "target": [0.1, 1.0, 1.2], "focal_mm": 50}},
        {"location": [2.5, 2.5, 1.2], "scale": [1.0, 1.0, 1.0], "resolution": 24},
        ["侧视"],
    )
    assert preview["cameras"]["侧视"]["overridden"] is True
    assert preview["cameras"]["侧视"]["blocked"] is True
    assert preview["cameras"]["侧视"]["inside_solid"] is False
    assert preview["rejected_cameras"] == []
    assert preview["gas_overridden"] is True
    assert preview["gas_domain"]["resolution"] == 24
    assert len(preview["gas_domain"]["bounds"]) == 6
    assert preview["world"]["strength"] == pytest.approx(0.04)
    assert len(preview["world"]["color"]) == 3


def test_a_camera_inside_the_ceiling_is_rejected_and_the_bake_uses_the_search():
    from firephantom.physics.blender_scene import camera_pose, camera_poses, point_in_solid
    from firephantom.physics.placements import layout_preview, normalize_camera_map
    from firephantom.physics.scenarios import get_scenario
    from firephantom.validation import expand_job_items

    spec = get_scenario("房间火灾")
    buried = {"俯视": {"location": [7.0979, 5.2121, 2.95], "target": [1.2, 1.2, 0.7675], "focal_mm": 25.99}}
    with pytest.raises(ValueError, match="相机在墙或天花板里面"):
        normalize_camera_map(spec, buried)
    poses = camera_poses(spec, ["俯视"], buried)
    auto = camera_pose(spec, "俯视")
    assert poses[0]["location"] == pytest.approx(auto["location"])
    assert not point_in_solid(spec, poses[0]["location"])
    # In the air, with the view ray crossing the west wall. Still the user's camera.
    occluded = {"侧视": {"location": [4.0, 1.0, 1.2], "target": [0.1, 1.0, 1.2], "focal_mm": 50}}
    assert normalize_camera_map(spec, occluded)["侧视"]["location"] == pytest.approx([4.0, 1.0, 1.2])
    assert camera_poses(spec, ["侧视"], occluded)[0]["location"] == pytest.approx([4.0, 1.0, 1.2])
    preview = layout_preview("房间火灾", [], "木材", None, buried, None, ["俯视"])
    assert preview["rejected_cameras"] == ["俯视"]
    assert preview["cameras"]["俯视"]["inside_solid"] is True
    assert preview["cameras"]["俯视"]["overridden"] is False
    assert preview["cameras"]["俯视"]["location"] == pytest.approx(preview["automatic_cameras"]["俯视"]["location"])
    blocked = layout_preview("房间火灾", [], "木材", None, occluded, None, ["侧视"])
    assert blocked["cameras"]["侧视"]["blocked"] is True
    assert blocked["cameras"]["侧视"]["overridden"] is True
    assert blocked["cameras"]["侧视"]["inside_solid"] is False
    expanded = expand_job_items([{
        "scenario": "房间火灾",
        "wind_speed": 1.2,
        "temperature": 22.0,
        "humidity": 35.0,
        "fuel_type": "木材",
        "weathers": ["晴"],
        "viewpoints": ["俯视"],
        "cameras": buried,
    }])
    assert expanded[0]["cameras"] == {}


def test_automatic_rig_does_not_change_bake_identity():
    from firephantom.physics.engine import _cache_key
    from firephantom.pipeline import sequence_id_for
    from firephantom.validation import physical_parameters

    params = physical_parameters(1.5, 20, 40, "木材")
    base = _cache_key("blender", "房间火灾", params, [], ["侧视"], [("室内", "晴")], 8, None)
    same = _cache_key(
        "blender",
        "房间火灾",
        params,
        [],
        ["侧视"],
        [("室内", "晴")],
        8,
        None,
        {"origin": None, "cameras": {}, "gas_domain": None},
    )
    assert base == same
    moved = _cache_key(
        "blender",
        "房间火灾",
        params,
        [],
        ["侧视"],
        [("室内", "晴")],
        8,
        None,
        {"cameras": {"侧视": {"location": [3, 1, 1], "target": [1.2, 1.2, 1], "focal_mm": 50}}},
    )
    assert moved != base
    identity = {"scenario": "房间火灾", "placements": [], "seed": None}
    assert sequence_id_for(identity) == sequence_id_for({**identity, "origin": None, "cameras": {}, "gas_domain": None})
    assert sequence_id_for(identity) == sequence_id_for({**identity, "lights": None})
    assert sequence_id_for(identity) != sequence_id_for({**identity, "origin": [2.0, 2.0, 0.3]})
    assert sequence_id_for(identity) != sequence_id_for({**identity, "lights": []})
    lamp = {"id": "L1", "name": "台灯", "type": "POINT", "energy": 200}
    assert sequence_id_for(identity) != sequence_id_for({**identity, "lights": [lamp]})
    bare = _cache_key("blender", "房间火灾", params, [], ["侧视"], [("室内", "晴")], 8, None, {"lights": []})
    assert bare != base


def test_fuel_catches_from_the_near_side_instead_of_all_at_once():
    spec = get_scenario("房间火灾")
    sofa = {"asset_id": "painted_wooden_sofa", "x": 3.7, "y": 2.9, "z": 0.4, "yaw": 0.0, "fuel_type": "木材"}
    layout = prepare(spec, [sofa], "木材")
    fronts = layout["packages"][0]["fronts"]
    assert len(fronts) == 4
    assert fronts[0]["xb"][0] < fronts[-1]["xb"][0]
    assert [item["full_frame"] for item in fronts] == [fronts[0]["full_frame"] + 3 * index for index in range(4)]
    small = {
        "xb": [1.0, 1.4, 1.0, 1.4, 0.0, 0.4],
        "center": [1.2, 1.2, 0.2],
        "ignition": False,
        "delay": 14,
    }
    assert len(ignition_fronts(small, [(spec.origin[0], spec.origin[1])])) == 1
    crate = {
        "asset_id": "wooden_crate_01",
        "x": 3.2,
        "y": 1.2,
        "z": 0.4,
        "yaw": 0.0,
        "fuel_type": "木材",
        "ignition": True,
    }
    burning = prepare(spec, [crate], "木材")["packages"][0]["fronts"]
    assert burning[0]["full_frame"] == 12
    assert burning[1]["full_frame"] == 15
    assert burning[0]["xb"][0] < burning[1]["xb"][0]
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", placements=[sofa])
    assert 'flow.name = "fuel_"' in script
    assert "for index, front in enumerate" in script
    assert len(_payload(script)["packages"][0]["fronts"]) == 4


def _lamp(**patch):
    light = {
        "id": "L1",
        "name": "台灯",
        "type": "POINT",
        "color": [1.0, 0.9, 0.7],
        "energy": 200.0,
        "size": 0.4,
        "size_y": 1.0,
        "spot_size": 0.8,
        "spot_blend": 0.15,
        "location": [2.0, 2.0, 1.5],
        "rotation": [0.0, 0.0, 0.0],
        "enabled": True,
    }
    light.update(patch)
    return light


def test_custom_lights_replace_the_automatic_fill_and_sun():
    spec = get_scenario("房间火灾")
    automatic = automatic_lights(spec)
    assert [item["id"] for item in automatic] == ["auto-fill", "auto-sun"]
    assert automatic[0]["energy"] == 20
    assert automatic[1]["energy"] == 0
    assert collapse_lights(spec, normalize_lights(spec, automatic)) is None
    assert collapse_lights(spec, []) == []
    with pytest.raises(ValueError, match="场景里面"):
        normalize_lights(spec, [_lamp(location=[0.0, 2.0, 1.0])])
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", lights=[_lamp()])
    payload = _payload(script)
    assert payload["custom_lights"] is True
    assert payload["fill_light"] is None
    assert payload["lights"][0]["energy"] == 200
    assert 'type="POINT"' in script
    assert "fill_light" not in script.split("if PARAMS.get(\"custom_lights\"):", 1)[-1].split("else:", 1)[0]
    outdoor = automatic_lights(get_scenario("林区火灾"))
    assert outdoor[1]["type"] == "SUN" and outdoor[1]["energy"] == pytest.approx(2.2)


def test_simulation_settings_check_ranges_and_collapse_to_defaults():
    from firephantom.physics.simulation import (
        SIMULATION_DEFAULTS,
        collapse_simulation,
        gas_cells,
        normalize_simulation,
        warmup_frames,
    )

    assert normalize_simulation(None) is None
    assert collapse_simulation(normalize_simulation({})) is None
    assert collapse_simulation(normalize_simulation(dict(SIMULATION_DEFAULTS))) is None
    custom = normalize_simulation({"noise": 0, "warmup_s": 2.5})
    assert collapse_simulation(custom) == {**SIMULATION_DEFAULTS, "noise": 0, "warmup_s": 2.5}
    assert warmup_frames(custom) == 30
    for bad, message in (
        ({"noise": 1}, "噪声"),
        ({"warmup_s": 20}, "预热"),
        ({"samples": True}, "采样"),
        ({"wind": 1}, "未知"),
    ):
        with pytest.raises(ValueError, match=message):
            normalize_simulation(bad)
    assert gas_cells({"scale": [4.0, 3.0, 1.5], "resolution": 64}) == 64 * 48 * 24


def test_ignition_delay_follows_the_warmup():
    from firephantom.physics.placements import _delay

    assert [_delay(distance) for distance in (0.0, 1.0, 5.0)] == [13, 14, 18]
    assert [_delay(distance, 36) for distance in (0.0, 1.0, 5.0)] == [37, 38, 42]


def test_preview_reports_the_bake_estimate():
    from firephantom.physics.placements import layout_preview

    small = layout_preview("房间火灾", [], "木材")
    assert small["simulation_overridden"] is False
    assert small["bake_estimate"]["over"] is False
    full = layout_preview(
        "房间火灾",
        [],
        "木材",
        gas_domain_override={"location": [4.0, 3.0, 1.5], "scale": [3.9, 2.9, 1.45], "resolution": 128},
        simulation={"noise": 3, "warmup_s": 10},
        frame_count=48,
        look_count=6,
    )
    assert full["simulation"]["noise"] == 3
    assert full["bake_estimate"]["cells"] > small["bake_estimate"]["cells"]
    assert full["bake_estimate"]["over"] is True
