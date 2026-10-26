"""Start and stop the local FirePhantom services on macOS, Linux, and Windows.

``python -m firephantom.local start`` replaces the platform-specific process
supervisor that used to live in the shell scripts. The shell and PowerShell
wrappers only choose a Python interpreter and forward to this module.
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from firephantom.paths import find_blender, find_label_studio, repo_root


_REEXEC = "FIREPHANTOM_LOCAL_REEXEC"
_SERVICE_NAMES = ("api", "label-studio", "diffusion")


def resolve_runtime() -> Path:
    configured = os.environ.get("FIREPHANTOM_RUNTIME", "").strip()
    if not configured:
        return repo_root() / "data"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path


def python_candidates(runtime: Path) -> list[Path]:
    """Interpreter search order used by the old start script, plus Windows conda layouts."""
    home = Path.home()
    conda = runtime / "conda-envs" / "firephantom"
    candidates: list[Path] = []
    configured = os.environ.get("FIREPHANTOM_PYTHON", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            conda / "bin" / "python",
            conda / "bin" / "python3",
            conda / "python.exe",
            conda / "Scripts" / "python.exe",
            Path("/opt/anaconda3/envs/firephantom/bin/python"),
            home / "miniconda3" / "envs" / "firephantom" / "bin" / "python",
            home / "anaconda3" / "envs" / "firephantom" / "bin" / "python",
            home / "miniconda3" / "envs" / "firephantom" / "python.exe",
            home / "anaconda3" / "envs" / "firephantom" / "python.exe",
            home / "miniconda3" / "envs" / "firephantom" / "Scripts" / "python.exe",
            home / "anaconda3" / "envs" / "firephantom" / "Scripts" / "python.exe",
            Path("/usr/bin/python3"),
        ]
    )
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.append(Path(sys.executable))
    return candidates


def python_is_supported(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.resolve() == Path(sys.executable).resolve():
            return sys.version_info >= (3, 11)
    except OSError:
        return False
    try:
        proc = subprocess.run(
            [str(path), "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def select_python(runtime: Path) -> Path | None:
    for candidate in python_candidates(runtime):
        if python_is_supported(candidate):
            return candidate
    return None


def ensure_interpreter(runtime: Path) -> None:
    """Re-exec once into the project interpreter when it is not the current process."""
    if os.environ.get(_REEXEC) == "1":
        if sys.version_info < (3, 11):
            _missing_python(runtime)
        return
    chosen = select_python(runtime)
    if chosen is None:
        _missing_python(runtime)
    try:
        same = chosen.resolve() == Path(sys.executable).resolve()
    except OSError:
        same = False
    if same:
        os.environ[_REEXEC] = "1"
        return
    env = os.environ.copy()
    env[_REEXEC] = "1"
    env["FIREPHANTOM_RUNTIME"] = str(runtime)
    root = str(repo_root())
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not current else root + os.pathsep + current
    os.execve(str(chosen), [str(chosen), "-m", "firephantom.local", *sys.argv[1:]], env)


def _missing_python(runtime: Path) -> None:
    print(
        "Python 3.11+ was not found. Set FIREPHANTOM_PYTHON or create "
        f"{runtime / 'conda-envs' / 'firephantom'}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _read_secret(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, (value + "\n").encode("ascii"))
    finally:
        os.close(descriptor)
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def prepare_secrets(runtime: Path) -> dict[str, str]:
    """Create the local secret files with ``secrets`` instead of ``openssl``."""
    token_path = runtime / "label-studio.token"
    password_path = runtime / "label-studio.password"
    postgres_path = runtime / "postgres.password"
    session_path = runtime / "firephantom-session.secret"
    token = _read_secret(token_path)
    if token is None or len(token) != 40:
        token = secrets.token_hex(20)
        _write_private(token_path, token)
    password = _read_secret(password_path)
    if not password:
        password = secrets.token_hex(12)
        _write_private(password_path, password)
    postgres = _read_secret(postgres_path)
    if not postgres:
        postgres = secrets.token_hex(16)
        _write_private(postgres_path, postgres)
    session = _read_secret(session_path)
    if not session:
        session = secrets.token_hex(32)
        _write_private(session_path, session)
    return {
        "label_studio_token": token,
        "label_studio_password": password,
        "postgres_password": postgres,
        "session_secret": session,
    }


def annotation_device() -> str:
    configured = os.environ.get("ANNOTATION_DEVICE", "").strip()
    if configured:
        return configured
    if sys.platform == "darwin":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def apply_environment(runtime: Path, secret_values: dict[str, str]) -> None:
    password = secret_values["postgres_password"]
    os.environ["FIREPHANTOM_RUNTIME"] = str(runtime)
    os.environ.setdefault("DATASET_ROOT", str(runtime / "dataset"))
    os.environ.setdefault("SD_MODEL_PATH", str(runtime / "models" / "tiny-sd"))
    os.environ.setdefault("SD_API_URL", "http://127.0.0.1:7861/sdapi/v1/txt2img")
    os.environ.setdefault("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    os.environ.setdefault("INCEPTION_WEIGHTS", str(runtime / "models" / "inception_v3_google-0cc3c7bd.pth"))
    os.environ.setdefault("FIRE_REFERENCE_DATASET", str(runtime / "reference" / "valid" / "images"))
    os.environ.setdefault("FIRE_REFERENCE_LIMIT", "128")
    os.environ["DVC_NO_ANALYTICS"] = "1"
    os.environ["YOLO_CONFIG_DIR"] = str(runtime / "config" / "ultralytics")
    os.environ.setdefault("LABEL_STUDIO_PROJECT_ID", "1")
    os.environ.setdefault("LABEL_STUDIO_EMAIL", "firephantom@localhost")
    os.environ["LABEL_STUDIO_TOKEN"] = secret_values["label_studio_token"]
    os.environ["POSTGRES_PASSWORD"] = password
    os.environ["POSTGRES_HOST"] = "127.0.0.1"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_USER"] = "firephantom"
    os.environ["POSTGRES_DB"] = "firephantom"
    os.environ["DJANGO_DB"] = "postgresql"
    os.environ["POSTGRE_HOST"] = "127.0.0.1"
    os.environ["POSTGRE_PORT"] = "5432"
    os.environ["POSTGRE_USER"] = "firephantom"
    os.environ["POSTGRE_PASSWORD"] = password
    os.environ["POSTGRE_NAME"] = "labelstudio"
    os.environ["DISABLE_SIGNUP_WITHOUT_LINK"] = "true"
    os.environ["FIREPHANTOM_DATABASE_URL"] = (
        f"postgresql://firephantom:{quote(password, safe='')}@127.0.0.1:5432/firephantom"
    )
    os.environ["FIREPHANTOM_SESSION_SECRET_FILE"] = str(runtime / "firephantom-session.secret")
    os.environ["ANNOTATION_DEVICE"] = annotation_device()
    weights = runtime / "models" / "fire_smoke_yolo.pt"
    if weights.is_file():
        os.environ["ANNOTATION_MODEL"] = str(weights)


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), 1):
            return True
    except OSError:
        return False


def http_healthy(port: int) -> bool:
    request = Request(f"http://127.0.0.1:{port}/health", method="GET")
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=1) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def command_args(executable: str, args: list[str]) -> list[str]:
    """Run ``*.cmd`` / ``*.bat`` through ``cmd.exe`` so Windows can start npm."""
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        return ["cmd.exe", "/c", executable, *args]
    return [executable, *args]


def resolve_npm() -> str | None:
    configured = os.environ.get("NPM_BIN", "").strip()
    if configured:
        return configured
    found = shutil.which("npm")
    if found:
        return found
    for candidate in ("/opt/homebrew/opt/node@22/bin/npm", "/opt/homebrew/bin/npm", "/usr/local/bin/npm"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def npm_command(npm: str, args: list[str]) -> list[str]:
    return command_args(npm, args)


def build_frontend() -> None:
    index = repo_root() / "web" / "dist" / "index.html"
    if index.is_file():
        return
    npm = resolve_npm()
    if not npm:
        print(
            "Frontend build is missing and npm is unavailable; run cd web && npm ci && npm run build first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Frontend build is missing; building web/dist with npm.")
    env = os.environ.copy()
    env["PATH"] = str(Path(npm).resolve().parent) + os.pathsep + env.get("PATH", "")
    for args in (["ci"], ["run", "build"]):
        subprocess.run(npm_command(npm, args), cwd=repo_root() / "web", env=env, check=True)


def _import_psycopg():
    try:
        import psycopg
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "psycopg[binary]==3.3.6"],
            check=False,
        )
        import psycopg
    return psycopg


def ensure_postgres(password: str) -> None:
    psycopg = _import_psycopg()
    if not port_open(5432):
        docker = shutil.which("docker")
        if not docker:
            print(
                "PostgreSQL is not listening on 127.0.0.1:5432 and docker is not available.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("Starting PostgreSQL with docker compose.")
        env = os.environ.copy()
        env["POSTGRES_PASSWORD"] = password
        result = subprocess.run(
            command_args(docker, ["compose", "up", "-d", "postgres"]),
            cwd=repo_root(),
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    ready = False
    for _ in range(60):
        try:
            psycopg.connect(
                host="127.0.0.1",
                user="firephantom",
                password=password,
                dbname="firephantom",
                connect_timeout=2,
            ).close()
        except Exception:
            time.sleep(1)
            continue
        ready = True
        break
    if not ready:
        print("PostgreSQL did not become ready. Check docker compose logs postgres.", file=sys.stderr)
        raise SystemExit(1)
    connection = psycopg.connect(
        host="127.0.0.1",
        user="firephantom",
        password=password,
        dbname="postgres",
        autocommit=True,
        connect_timeout=5,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname='labelstudio'")
            if cursor.fetchone() is None:
                cursor.execute("CREATE DATABASE labelstudio")
    finally:
        connection.close()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_exists(pid)
    return _posix_pid_exists(pid)


def _posix_pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_pid_exists(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def terminate_pid(pid: int) -> None:
    """Stop one process. Windows ``SIGTERM`` calls TerminateProcess; never signal 0 there."""
    if not pid_exists(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(50):
        if not pid_exists(pid):
            return
        time.sleep(0.1)


def stop_pids(pids: list[int]) -> None:
    for pid in pids:
        terminate_pid(pid)


def start_service(
    runtime: Path,
    name: str,
    port: int,
    command: list[str],
    extra_env: dict[str, str] | None = None,
) -> int | None:
    if http_healthy(port):
        print(f"{name} is already responding on port {port}")
        return None
    log_path = runtime / "logs" / f"{name}.log"
    log_handle = log_path.open("ab")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    kwargs: dict = {
        "args": command_args(command[0], command[1:]),
        "cwd": repo_root(),
        "env": env,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(**kwargs)
    finally:
        log_handle.close()
    (runtime / f"{name}.pid").write_text(f"{proc.pid}\n", encoding="ascii")
    print(f"Started {name} (pid {proc.pid}); log: {log_path}")
    return proc.pid


def launch_services(runtime: Path, secret_values: dict[str, str]) -> list[int]:
    started: list[int] = []
    model = Path(os.environ["SD_MODEL_PATH"])
    if (model / "model_index.json").is_file():
        pid = start_service(
            runtime,
            "diffusion",
            7861,
            [sys.executable, "-m", "firephantom.synthesis.diffusion_server"],
        )
        if pid is not None:
            started.append(pid)
    else:
        print("Tiny-SD weights are incomplete; Stable Diffusion will be unavailable.")
    label_bin = find_label_studio()
    if label_bin and Path(label_bin).is_file():
        pid = start_service(
            runtime,
            "label-studio",
            8080,
            [
                label_bin,
                "start",
                "FirePhantom",
                "--init",
                "--no-browser",
                "--enable-legacy-api-token",
                "--label-config",
                str(repo_root() / "firephantom" / "annotation" / "label_config.xml"),
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--data-dir",
                str(runtime / "label-studio"),
            ],
            {
                "LABEL_STUDIO_BASE_DATA_DIR": str(runtime / "label-studio"),
                "USERNAME": os.environ.get("LABEL_STUDIO_EMAIL", "firephantom@localhost"),
                "PASSWORD": secret_values["label_studio_password"],
                "USER_TOKEN": secret_values["label_studio_token"],
            },
        )
        if pid is not None:
            started.append(pid)
        print("Label Studio uses the PostgreSQL labelstudio database. People sign in at http://127.0.0.1:8765/.")
    else:
        print("Label Studio is not installed; task calls will be recorded as unavailable.")
    pid = start_service(
        runtime,
        "api",
        8765,
        [sys.executable, "-m", "uvicorn", "firephantom.api:app", "--host", "127.0.0.1", "--port", "8765"],
    )
    if pid is not None:
        started.append(pid)
    return started


def wait_for_api(runtime: Path) -> bool:
    for _ in range(60):
        if http_healthy(8765):
            print("FirePhantom is ready: http://127.0.0.1:8765/")
            return True
        time.sleep(1)
    print(f"FirePhantom API did not become ready. Check {runtime}/logs/api.log", file=sys.stderr)
    return False


def supervise(runtime: Path, pids: list[int]) -> int:
    if not pids:
        print("All local services were already running; nothing to supervise in this process.")
        return 0
    print("Local services are supervised by this process. Press Ctrl-C to stop services started here.")
    while True:
        for pid in pids:
            if not pid_exists(pid):
                print(
                    f"A local service exited unexpectedly (pid {pid}). Check {runtime}/logs.",
                    file=sys.stderr,
                )
                return 1
        time.sleep(2)


def start(runtime: Path) -> int:
    for relative in ("logs", "dataset", "label-studio", "models", "tools", "assets", "config/ultralytics"):
        (runtime / relative).mkdir(parents=True, exist_ok=True)
    secret_values = prepare_secrets(runtime)
    apply_environment(runtime, secret_values)
    blender = find_blender()
    if blender:
        os.environ["BLENDER_BIN"] = blender
        print(f"Using Blender: {blender}")
    else:
        os.environ.pop("BLENDER_BIN", None)
        print("Blender was not found; simulations will use the parametric fallback. Set BLENDER_BIN to override.")
    foreground = os.environ.get("FIREPHANTOM_FOREGROUND", "1") == "1"
    started: list[int] = []

    def _term(_signum, _frame):
        raise SystemExit(143)

    try:
        if foreground:
            signal.signal(signal.SIGTERM, _term)
        ensure_postgres(secret_values["postgres_password"])
        build_frontend()
        started.extend(launch_services(runtime, secret_values))
        if not wait_for_api(runtime):
            return 1
        if not foreground:
            hint = "scripts/stop-local.ps1" if os.name == "nt" else "scripts/stop-local.sh"
            print(f"Local services are running in the background. Use {hint} to stop services started here.")
            return 0
        return supervise(runtime, started)
    except KeyboardInterrupt:
        return 130
    finally:
        if foreground:
            stop_pids(started)


def stop_services(runtime: Path) -> None:
    for name in _SERVICE_NAMES:
        pid_file = runtime / f"{name}.pid"
        if not pid_file.is_file():
            continue
        raw = pid_file.read_text(encoding="utf-8").strip()
        try:
            pid = int(raw)
        except ValueError:
            pid = 0
        if pid_exists(pid):
            terminate_pid(pid)
            print(f"Stopped {name} (pid {pid})")
        pid_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = "start"
    if args and args[0] in {"start", "stop", "-h", "--help"}:
        command = args.pop(0)
    if command in {"-h", "--help"}:
        print("usage: python -m firephantom.local [start|stop]")
        return 0
    runtime = resolve_runtime()
    os.environ["FIREPHANTOM_RUNTIME"] = str(runtime)
    if command == "stop":
        stop_services(runtime)
        return 0
    ensure_interpreter(runtime)
    return start(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
