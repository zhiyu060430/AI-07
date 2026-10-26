"""Windows and Linux compatibility that does not need PostgreSQL."""

import os
import signal
import sys
import threading
import time
import types

import pytest

from firephantom.locking import exclusive_lock


def test_engine_module_does_not_import_fcntl():
    import firephantom.physics.engine as engine

    assert not hasattr(engine, "fcntl")


def test_windows_blender_and_ffmpeg_candidates(monkeypatch, tmp_path):
    import firephantom.paths as paths

    monkeypatch.setenv("FIREPHANTOM_RUNTIME", str(tmp_path))
    monkeypatch.setattr(paths.platform, "system", lambda: "Windows")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "blender.exe").write_bytes(b"")
    portable = tools / "blender"
    portable.mkdir()
    (portable / "blender.exe").write_bytes(b"")
    versioned = tools / "blender-4.5-windows-x64"
    versioned.mkdir()
    (versioned / "blender.exe").write_bytes(b"")
    program = tmp_path / "pf" / "Blender Foundation" / "Blender 4.5"
    program.mkdir(parents=True)
    (program / "blender.exe").write_bytes(b"")
    (tmp_path / "pf86").mkdir()
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "pf86"))

    found = {path.resolve() for path in paths.blender_candidates()}
    assert (tools / "blender.exe").resolve() in found
    assert (portable / "blender.exe").resolve() in found
    assert (versioned / "blender.exe").resolve() in found
    assert (program / "blender.exe").resolve() in found

    (tools / "ffmpeg.exe").write_bytes(b"")
    nested = tools / "ffmpeg" / "bin"
    nested.mkdir(parents=True)
    (nested / "ffmpeg.exe").write_bytes(b"")
    ffmpeg = [path.resolve() for path in paths.ffmpeg_candidates()]
    assert (tools / "ffmpeg.exe").resolve() in ffmpeg
    assert (nested / "ffmpeg.exe").resolve() in ffmpeg
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr(paths.shutil, "which", lambda _name: None)
    assert paths.find_ffmpeg() == str(tools / "ffmpeg.exe")


def test_linux_tool_candidates_include_system_paths(monkeypatch, tmp_path):
    import firephantom.paths as paths

    monkeypatch.setenv("FIREPHANTOM_RUNTIME", str(tmp_path))
    monkeypatch.setattr(paths.platform, "system", lambda: "Linux")
    blender = [str(path) for path in paths.blender_candidates()]
    assert str(tmp_path / "tools" / "blender") in blender
    assert "/usr/bin/blender" in blender
    assert "/snap/bin/blender" in blender
    ffmpeg = [str(path) for path in paths.ffmpeg_candidates()]
    assert str(tmp_path / "tools" / "ffmpeg") in ffmpeg
    assert "/usr/bin/ffmpeg" in ffmpeg


def test_macos_tool_candidates_keep_app_and_homebrew_paths(monkeypatch, tmp_path):
    import firephantom.paths as paths

    monkeypatch.setenv("FIREPHANTOM_RUNTIME", str(tmp_path))
    monkeypatch.setattr(paths.platform, "system", lambda: "Darwin")
    blender = [str(path) for path in paths.blender_candidates()]
    assert str(tmp_path / "tools" / "Blender.app" / "Contents" / "MacOS" / "Blender") in blender
    assert "/Applications/Blender.app/Contents/MacOS/Blender" in blender
    ffmpeg = [str(path) for path in paths.ffmpeg_candidates()]
    assert "/opt/homebrew/bin/ffmpeg" in ffmpeg


def test_exclusive_lock_is_exclusive_and_released(tmp_path):
    path = tmp_path / "cache" / ".lock"
    release = threading.Event()
    entered = threading.Event()
    acquired = threading.Event()

    def hold():
        with exclusive_lock(path):
            entered.set()
            assert release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert entered.wait(5)

    def wait_for_lock():
        with exclusive_lock(path):
            acquired.set()

    waiter = threading.Thread(target=wait_for_lock)
    waiter.start()
    time.sleep(0.2)
    assert not acquired.is_set()
    release.set()
    holder.join(5)
    waiter.join(5)
    assert not holder.is_alive()
    assert not waiter.is_alive()
    assert acquired.is_set()
    with exclusive_lock(path):
        pass


def test_windows_lock_uses_nonblocking_retry(monkeypatch, tmp_path):
    import firephantom.locking as locking

    calls = []
    fake = types.ModuleType("msvcrt")
    fake.LK_NBLCK = 11
    fake.LK_UNLCK = 0
    fake.LK_LOCK = 1
    attempts = {"n": 0}

    def locking_fn(_fd, mode, _nbytes):
        calls.append(mode)
        if mode == fake.LK_NBLCK:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("busy")

    fake.locking = locking_fn
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(locking.sys, "platform", "win32")
    monkeypatch.setattr(locking.time, "sleep", lambda _seconds: None)
    def explode(*_args):
        raise AssertionError("os.kill")

    monkeypatch.setattr(locking.os, "kill", explode)

    with locking.exclusive_lock(tmp_path / ".lock"):
        pass

    assert fake.LK_NBLCK in calls
    assert fake.LK_LOCK not in calls
    assert calls[-1] == fake.LK_UNLCK


def test_external_processes_use_utf8(monkeypatch, tmp_path):
    import firephantom.physics.adapters as adapters
    import firephantom.versioning.store as store

    script = tmp_path / "scene.py"
    script.write_text("# scene", encoding="utf-8")
    captured = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(_command, **kwargs):
        captured.append(kwargs)
        return Completed()

    monkeypatch.setattr(adapters.shutil, "which", lambda _name: "/usr/bin/blender")
    monkeypatch.setattr(adapters.subprocess, "run", fake_run)
    assert adapters.run_blender(script)["success"] is True

    monkeypatch.setattr(store.subprocess, "run", fake_run)
    dataset_store = object.__new__(store.DatasetStore)
    dataset_store.root = tmp_path
    dataset_store._dvc("status")

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["text"] is True


def test_private_secrets_are_created_once(tmp_path):
    import firephantom.local as local

    first = local.prepare_secrets(tmp_path)
    second = local.prepare_secrets(tmp_path)
    assert first == second
    assert len(first["label_studio_token"]) == 40
    assert len(first["session_secret"]) == 64
    if os.name != "nt":
        for name in (
            "label-studio.token",
            "label-studio.password",
            "postgres.password",
            "firephantom-session.secret",
        ):
            assert (tmp_path / name).stat().st_mode & 0o777 == 0o600
    (tmp_path / "label-studio.token").write_text("short\n", encoding="ascii")
    refreshed = local.prepare_secrets(tmp_path)
    assert len(refreshed["label_studio_token"]) == 40
    assert refreshed["label_studio_token"] != "short"
    assert refreshed["postgres_password"] == first["postgres_password"]


def test_python_candidates_prefer_configured_then_conda(monkeypatch, tmp_path):
    import firephantom.local as local

    monkeypatch.setenv("FIREPHANTOM_PYTHON", "/custom/python")
    runtime = tmp_path / "runtime"
    candidates = local.python_candidates(runtime)
    conda = runtime / "conda-envs" / "firephantom"
    assert candidates[0] == local.Path("/custom/python")
    assert candidates[1:5] == [
        conda / "bin" / "python",
        conda / "bin" / "python3",
        conda / "python.exe",
        conda / "Scripts" / "python.exe",
    ]
    assert candidates[-1] == local.Path(sys.executable)


def test_interpreter_reexecs_into_conda(monkeypatch, tmp_path):
    import firephantom.local as local

    runtime = tmp_path / "runtime"
    conda = runtime / "conda-envs" / "firephantom" / "bin" / "python"
    conda.parent.mkdir(parents=True)
    conda.write_bytes(b"")
    monkeypatch.delenv("FIREPHANTOM_PYTHON", raising=False)
    monkeypatch.delenv("FIREPHANTOM_LOCAL_REEXEC", raising=False)
    monkeypatch.setattr(local, "python_is_supported", lambda path: path == conda)
    calls = {}

    def fake_exec(executable, argv, env):
        calls["executable"] = executable
        calls["argv"] = argv
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(local.os, "execve", fake_exec)
    with pytest.raises(SystemExit):
        local.ensure_interpreter(runtime)
    assert calls["executable"] == str(conda)
    assert calls["argv"][1:3] == ["-m", "firephantom.local"]
    assert calls["env"]["FIREPHANTOM_LOCAL_REEXEC"] == "1"
    assert calls["env"]["FIREPHANTOM_RUNTIME"] == str(runtime)


def test_windows_pid_check_does_not_use_signal_zero(monkeypatch):
    import firephantom.local as local

    monkeypatch.setattr(local.os, "name", "nt")
    monkeypatch.setattr(local, "_windows_pid_exists", lambda pid: pid == 7)

    def explode(*_args):
        raise AssertionError("os.kill")

    monkeypatch.setattr(local.os, "kill", explode)
    assert local.pid_exists(7) is True
    assert local.pid_exists(0) is False


def test_stop_terminates_recorded_pids(monkeypatch, tmp_path):
    import firephantom.local as local

    (tmp_path / "api.pid").write_text("42\n", encoding="ascii")
    (tmp_path / "diffusion.pid").write_text("not-a-pid\n", encoding="ascii")
    alive = {42}
    killed = []

    monkeypatch.setattr(local, "pid_exists", lambda pid: pid in alive)

    def kill(pid, sig):
        killed.append((pid, sig))
        alive.discard(pid)

    monkeypatch.setattr(local.os, "kill", kill)
    local.stop_services(tmp_path)
    assert killed == [(42, signal.SIGTERM)]
    assert not (tmp_path / "api.pid").exists()
    assert not (tmp_path / "diffusion.pid").exists()


def test_windows_npm_bat_goes_through_cmd(monkeypatch):
    import firephantom.local as local

    monkeypatch.setattr(local.os, "name", "nt")
    command = local.npm_command(r"C:\Program Files\nodejs\npm.cmd", ["ci"])
    assert command == ["cmd.exe", "/c", r"C:\Program Files\nodejs\npm.cmd", "ci"]


def test_annotation_device_defaults(monkeypatch):
    import firephantom.local as local

    monkeypatch.delenv("ANNOTATION_DEVICE", raising=False)
    monkeypatch.setattr(local.sys, "platform", "darwin")
    assert local.annotation_device() == "mps"
    monkeypatch.setattr(local.sys, "platform", "linux")
    monkeypatch.setattr(local.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    assert local.annotation_device() == "cuda"
    monkeypatch.setattr(local.shutil, "which", lambda _name: None)
    assert local.annotation_device() == "cpu"
    monkeypatch.setenv("ANNOTATION_DEVICE", "cuda")
    monkeypatch.setattr(local.sys, "platform", "darwin")
    assert local.annotation_device() == "cuda"
