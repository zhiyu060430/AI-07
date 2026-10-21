"""Parameter-dependent physical fields and Blender scripts."""

import ast
import json

import numpy as np
import pytest

from firephantom.physics.engine import simulate

SCENARIOS = ("房间火灾", "林区火灾", "工业区火灾")
BASE = {
    "wind_speed": 1.2,
    "temperature": 22.0,
    "humidity": 35.0,
    "fuel_type": "木材",
}


def _params(text: str) -> dict:
    for line in text.splitlines():
        if "PARAMS_JSON=" in line:
            return json.loads(line.split("PARAMS_JSON=", 1)[1])
    raise AssertionError("emitted artifact has no PARAMS_JSON")


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("wind_speed", 7.5),
        ("temperature", -5.0),
        ("humidity", 95.0),
        ("fuel_type", "汽油"),
    ],
)
def test_each_parameter_moves_every_field(scenario, key, value):
    base = simulate(scenario, **BASE)
    changed = dict(BASE)
    changed[key] = value
    other = simulate(scenario, **changed)
    for field in ("temperature", "smoke", "flow"):
        assert base[field].size > 0
        assert float(base[field].std()) > 0
        assert not np.allclose(base[field], other[field])


def test_unknown_fuel_is_not_a_fixed_placeholder():
    wood = simulate("工业区火灾", **BASE)
    other = simulate("工业区火灾", **{**BASE, "fuel_type": "未知燃料-A"})
    another = simulate("工业区火灾", **{**BASE, "fuel_type": "未知燃料-B"})
    assert not np.allclose(wood["smoke"], other["smoke"])
    assert not np.allclose(other["temperature"], another["temperature"])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_blender_script_names_the_scenario_and_parameters(scenario, tmp_path):
    result = simulate(scenario, work_dir=tmp_path, **BASE)
    params = _params(result["blender_script"])
    assert params["scenario"] == scenario
    assert params["wind_speed"] == BASE["wind_speed"]
    assert params["temperature"] == BASE["temperature"]
    assert params["humidity"] == BASE["humidity"]
    assert params["fuel_type"] == BASE["fuel_type"]
    if scenario == "林区火灾":
        assert params["obstacles"] == []
    else:
        assert params["obstacles"]
    ast.parse(result["blender_script"])
    assert not list(tmp_path.rglob("*.fds"))
    assert list(tmp_path.rglob("*_scene.py"))
    by_tool = {item["tool"]: item for item in result["tool_runs"]}
    assert set(by_tool) == {"blender"}
    assert "blender executable was not run" in by_tool["blender"]["message"]
    assert by_tool["blender"]["executed"] is False
    assert result["physical_backend"] == "parameterized-plume"
    assert result["fire_engine"] == "blender"
    assert len(result["engine_frames"]) == 16
    assert not np.array_equal(result["engine_frames"][0], result["engine_frames"][3])


def test_missing_executables_do_not_freeze_the_fields(tmp_path):
    calm = simulate("房间火灾", work_dir=tmp_path / "calm", **BASE)
    windy = simulate("房间火灾", work_dir=tmp_path / "windy", **{**BASE, "wind_speed": 9.0})
    assert "was not run" in next(item["message"] for item in calm["tool_runs"] if item["tool"] == "blender")
    assert not np.allclose(calm["flow"], windy["flow"])
    assert calm["blender_script"] != windy["blender_script"]


def test_simulate_starts_only_blender(monkeypatch, tmp_path):
    import firephantom.physics.engine as physics_engine

    calls = []
    monkeypatch.setattr(
        physics_engine,
        "run_blender",
        lambda *_args, **_kwargs: calls.append("blender") or {"success": False, "message": "called", "outputs": {}},
    )

    simulate("房间火灾", work_dir=tmp_path / "blender", fire_engine="blender", **BASE)
    assert calls == ["blender"]
    with pytest.raises(ValueError, match="fire_engine must be blender"):
        simulate("房间火灾", work_dir=tmp_path / "rejected", fire_engine="fds", **BASE)
    assert calls == ["blender"]


def test_blender_adapter_uses_script_name_inside_relative_workdir(monkeypatch, tmp_path):
    import firephantom.physics.adapters as adapters

    script = tmp_path / "scene.py"
    script.write_text("# scene", encoding="utf-8")
    calls = {}

    monkeypatch.setattr(adapters.shutil, "which", lambda _name: "/usr/bin/blender")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["cwd"] = kwargs["cwd"]
        return Completed()

    monkeypatch.setattr(adapters.subprocess, "run", fake_run)
    result = adapters.run_blender(script)
    assert result["success"] is True
    assert calls["command"][-1] == "scene.py"
    assert calls["cwd"] == script.parent


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        simulate("海上火灾", **BASE)
