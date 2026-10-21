"""Editable structure boxes, and weather that follows the lid instead of the preset name."""

import ast

import pytest

from firephantom.physics.blender_scene import environment_for_layout, render_blender
from firephantom.physics.placements import prepared_spec, preset_structures
from firephantom.physics.scenarios import get_scenario
from firephantom.scenes import SceneStore, resolve_scene_items
from firephantom.validation import expand_job_items

BASE = {
    "wind_speed": 1.5,
    "temperature": 28.0,
    "humidity": 45.0,
    "fuel_type": "木材",
    "viewpoints": ["侧视"],
}


def _room(**extra):
    return {"scenario": "房间火灾", **BASE, **extra}


def _payload(script: str) -> dict:
    import json

    start = script.index("PARAMS = json.loads(r\"\"\"") + len("PARAMS = json.loads(r\"\"\"")
    end = script.index("\"\"\")", start)
    return json.loads(script[start:end])


def test_omitted_structures_match_the_preset_boxes():
    for name in ("房间火灾", "林区火灾", "工业区火灾"):
        spec, domain, structures, _origin = prepared_spec(name)
        base = get_scenario(name)
        assert domain is None and structures is None
        assert [item.name for item in spec.obstacles] == [item.name for item in base.obstacles]
        assert [(item.xb, item.role) for item in spec.obstacles] == [(item.xb, item.role) for item in base.obstacles]


def test_a_room_can_take_rain_while_its_ceiling_stays():
    expanded = expand_job_items([_room(weathers=["雨", "晴"])])
    assert expanded[0]["scenes"] == ["室内"]
    assert expanded[0]["weathers"] == ["雨", "晴"]
    assert expanded[0]["structures"] is None
    with pytest.raises(ValueError, match="顶盖"):
        expand_job_items([_room(scene="室外", weather="晴")])


def test_deleting_the_ceiling_opens_the_sky_and_reaches_the_bake():
    boxes = [item for item in preset_structures(get_scenario("房间火灾")) if item["id"] != "ceiling"]
    expanded = expand_job_items([_room(structures=boxes, weathers=["雨"], scene="室外")])
    assert expanded[0]["scenes"] == ["室外"]
    assert expanded[0]["structures"]
    assert all(item["id"] != "ceiling" for item in expanded[0]["structures"])
    spec, _domain, structures, _origin = prepared_spec("房间火灾", None, boxes)
    assert structures
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", looks=[("室外", "雨")])
    ast.parse(script)
    payload = _payload(script)
    assert all(not str(item["name"]).startswith("ceiling") for item in payload["obstacles"])
    assert payload["looks"][0]["ground_covered"] is False
    assert "cover_boxes" in payload
    assert "_cull_precip" in script
    assert 'modifier_add(type="COLLISION")' in script


def test_a_full_lid_keeps_the_indoor_sky_and_a_dry_ground():
    spec = get_scenario("房间火灾")
    script = render_blender(spec, 1.5, 28.0, 45.0, "木材", looks=[("室内", "雨")])
    payload = _payload(script)
    assert payload["looks"][0]["scene"] == "室内"
    assert payload["looks"][0]["ground_covered"] is True
    assert payload["looks"][0]["precipitation"] == "rain"
    assert payload["looks"][0]["physics"]["temperature_scale"] < 1
    assert any(box[5] == pytest.approx(3.0) for box in payload["cover_boxes"])
    assert environment_for_layout(spec.domain, spec.obstacles, payload["precip_box"]) == "室内"


def test_applying_a_preset_clears_custom_boxes(tmp_path):
    store = SceneStore(tmp_path)
    scene = store.create("客厅", "房间火灾", "木材", [])
    assert scene["structures"] is None and scene["domain"] is None
    boxes = [item for item in preset_structures(get_scenario("房间火灾")) if item["id"] != "ceiling"]
    edited = store.update(scene["scene_id"], structures=boxes, domain=[9, 6, 3.4], base_revision=1)
    assert edited["revision"] == 2
    assert all(item["id"] != "ceiling" for item in edited["structures"])
    assert edited["domain"] == [9, 6, 3.4]
    assert store.revision(scene["scene_id"], 2)["structures"] == edited["structures"]
    reset = store.update(scene["scene_id"], scenario="林区火灾", structures=None, domain=None, placements=[], base_revision=2)
    assert reset["scenario"] == "林区火灾"
    assert reset["structures"] is None and reset["domain"] is None
    (resolved,) = resolve_scene_items(tmp_path, [{"scene_id": scene["scene_id"], "wind_speed": 1}])
    assert resolved["structures"] is None and resolved["scenario"] == "林区火灾"


def test_a_box_outside_the_site_is_rejected():
    with pytest.raises(ValueError, match="超出场地"):
        expand_job_items([_room(structures=[{
            "id": "s1",
            "name": "墙",
            "role": "inert",
            "center": [9, 3, 1.5],
            "size": [0.2, 2, 2],
        }])])
