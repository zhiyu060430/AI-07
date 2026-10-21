#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="/Volumes/T7/FirePhantom-runtime"
PYTHON="$RUNTIME/conda-envs/firephantom/bin/python"
if [[ ! -d "/Volumes/T7" || ! -d "$RUNTIME" ]]; then
  printf 'T7 runtime is not mounted; refusing to write models or dataset files to the internal disk: %s\n' "$RUNTIME" >&2
  exit 1
fi
mkdir -p "$RUNTIME/logs" "$RUNTIME/data/dataset" "$RUNTIME/data/label-studio"
mkdir -p "$RUNTIME/config/ultralytics"

if [[ ! -x "$PYTHON" ]]; then
  printf 'FirePhantom Python environment is missing: %s\n' "$PYTHON" >&2
  exit 1
fi

export DATASET_ROOT="$RUNTIME/data/dataset"
export BLENDER_BIN="$RUNTIME/tools/Blender.app/Contents/MacOS/Blender"
export SD_MODEL_PATH="$RUNTIME/models/tiny-sd"
export SD_API_URL="${SD_API_URL:-http://127.0.0.1:7861/sdapi/v1/txt2img}"
export LABEL_STUDIO_URL="${LABEL_STUDIO_URL:-http://127.0.0.1:8080/api/tasks}"
export INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-$RUNTIME/models/inception_v3_google-0cc3c7bd.pth}"
export FIRE_REFERENCE_DATASET="${FIRE_REFERENCE_DATASET:-$RUNTIME/data/FASDD_CV/valid/images}"
export FIRE_REFERENCE_LIMIT="${FIRE_REFERENCE_LIMIT:-128}"
export DVC_NO_ANALYTICS=1
export YOLO_CONFIG_DIR="$RUNTIME/config/ultralytics"
export LABEL_STUDIO_PROJECT_ID="${LABEL_STUDIO_PROJECT_ID:-1}"
# Label Studio 1.23 authenticates against its email field.  Keep the local
# account address explicit so the startup log is directly usable for login.
export LABEL_STUDIO_EMAIL="${LABEL_STUDIO_EMAIL:-firephantom@localhost}"
started_pids=()
if [[ -z "${LABEL_STUDIO_TOKEN:-}" && -f "$RUNTIME/label-studio.token" ]]; then
  export LABEL_STUDIO_TOKEN="$(<"$RUNTIME/label-studio.token")"
fi

# Label Studio stores this value in authtoken_token.key, which is varchar(40).
if [[ -z "${LABEL_STUDIO_TOKEN:-}" || ${#LABEL_STUDIO_TOKEN} -ne 40 ]]; then
  umask 077
  openssl rand -hex 20 >"$RUNTIME/label-studio.token"
  export LABEL_STUDIO_TOKEN="$(<"$RUNTIME/label-studio.token")"
fi
if [[ ! -f "$RUNTIME/label-studio.password" ]]; then
  umask 077
  openssl rand -hex 12 >"$RUNTIME/label-studio.password"
fi
if [[ ! -f "$RUNTIME/postgres.password" ]]; then
  umask 077
  openssl rand -hex 16 >"$RUNTIME/postgres.password"
fi
if [[ ! -f "$RUNTIME/firephantom-session.secret" ]]; then
  umask 077
  openssl rand -hex 32 >"$RUNTIME/firephantom-session.secret"
fi
export POSTGRES_PASSWORD="$(<"$RUNTIME/postgres.password")"
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_USER=firephantom
export POSTGRES_DB=firephantom
export DJANGO_DB=postgresql
export POSTGRE_HOST="$POSTGRES_HOST"
export POSTGRE_PORT="$POSTGRES_PORT"
export POSTGRE_USER="$POSTGRES_USER"
export POSTGRE_PASSWORD="$POSTGRES_PASSWORD"
export POSTGRE_NAME=labelstudio
export DISABLE_SIGNUP_WITHOUT_LINK=true
export FIREPHANTOM_DATABASE_URL="postgresql://firephantom:${POSTGRES_PASSWORD}@127.0.0.1:5432/firephantom"
export FIREPHANTOM_SESSION_SECRET_FILE="$RUNTIME/firephantom-session.secret"

if [[ -f "$RUNTIME/models/fire_smoke_yolo.pt" ]]; then
  export ANNOTATION_MODEL="$RUNTIME/models/fire_smoke_yolo.pt"
  export ANNOTATION_DEVICE="${ANNOTATION_DEVICE:-mps}"
fi

start_service() {
  local name="$1"
  local port="$2"
  shift 2
  if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    printf '%s is already responding on port %s\n' "$name" "$port"
    return
  fi
  nohup "$@" >"$RUNTIME/logs/$name.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"$RUNTIME/$name.pid"
  started_pids+=("$!")
  printf 'Started %s (pid %s); log: %s\n' "$name" "$!" "$RUNTIME/logs/$name.log"
}

stop_started_services() {
  trap - EXIT INT TERM
  if ((${#started_pids[@]} == 0)); then
    return
  fi
  for pid in "${started_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${started_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

postgres_is_listening() {
  "$PYTHON" - <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", 5432), 1):
        pass
except OSError:
    sys.exit(1)
PY
}

ensure_postgres() {
  if ! "$PYTHON" -c "import psycopg" >/dev/null 2>&1; then
    "$PYTHON" -m pip install -q 'psycopg[binary]==3.3.6'
  fi
  if ! postgres_is_listening; then
    if ! command -v docker >/dev/null 2>&1; then
      printf 'PostgreSQL is not listening on 127.0.0.1:5432 and docker is not available.\n' >&2
      exit 1
    fi
    printf 'Starting PostgreSQL with docker compose.\n'
    (
      cd "$ROOT"
      POSTGRES_PASSWORD="$POSTGRES_PASSWORD" docker compose up -d postgres
    )
  fi
  local ready=0
  for _ in {1..60}; do
    if "$PYTHON" - <<PY
import psycopg, sys
try:
    psycopg.connect(host="127.0.0.1", user="firephantom", password="${POSTGRES_PASSWORD}", dbname="firephantom", connect_timeout=2).close()
except Exception:
    sys.exit(1)
PY
    then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    printf 'PostgreSQL did not become ready. Check docker compose logs postgres.\n' >&2
    exit 1
  fi
  "$PYTHON" - <<PY
import psycopg
connection = psycopg.connect(
    host="127.0.0.1",
    user="firephantom",
    password="${POSTGRES_PASSWORD}",
    dbname="postgres",
    autocommit=True,
    connect_timeout=5,
)
with connection.cursor() as cursor:
    cursor.execute("SELECT 1 FROM pg_database WHERE datname='labelstudio'")
    if cursor.fetchone() is None:
        cursor.execute("CREATE DATABASE labelstudio")
connection.close()
PY
}

trap 'exit 130' INT
trap 'exit 143' TERM

cd "$ROOT"
ensure_postgres
if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  NPM_BIN="${NPM_BIN:-$(command -v npm || true)}"
  if [[ -z "$NPM_BIN" ]]; then
    for candidate in /opt/homebrew/opt/node@22/bin/npm /opt/homebrew/bin/npm /usr/local/bin/npm; do
      if [[ -x "$candidate" ]]; then
        NPM_BIN="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$NPM_BIN" ]]; then
    export PATH="$(dirname "$NPM_BIN"):$PATH"
    printf 'Frontend build is missing; building web/dist with npm.\n'
    (cd "$ROOT/web" && "$NPM_BIN" ci && "$NPM_BIN" run build)
  else
    printf 'Frontend build is missing and npm is unavailable; run cd web && npm ci && npm run build first.\n' >&2
    exit 1
  fi
fi
if [[ -f "$SD_MODEL_PATH/model_index.json" ]]; then
  start_service diffusion 7861 env SD_MODEL_PATH="$SD_MODEL_PATH" "$PYTHON" -m firephantom.synthesis.diffusion_server
else
  printf 'Tiny-SD weights are incomplete; Stable Diffusion will be unavailable.\n'
fi

if [[ -x "$RUNTIME/label-studio-env/bin/label-studio" ]]; then
  start_service label-studio 8080 env LABEL_STUDIO_BASE_DATA_DIR="$RUNTIME/data/label-studio" DJANGO_DB=postgresql POSTGRE_HOST=127.0.0.1 POSTGRE_PORT=5432 POSTGRE_USER=firephantom POSTGRE_PASSWORD="$POSTGRES_PASSWORD" POSTGRE_NAME=labelstudio DISABLE_SIGNUP_WITHOUT_LINK=true USERNAME="$LABEL_STUDIO_EMAIL" PASSWORD="$(<"$RUNTIME/label-studio.password")" USER_TOKEN="$LABEL_STUDIO_TOKEN" "$RUNTIME/label-studio-env/bin/label-studio" start FirePhantom --init --no-browser --enable-legacy-api-token --label-config "$ROOT/firephantom/annotation/label_config.xml" --host 127.0.0.1 --port 8080 --data-dir "$RUNTIME/data/label-studio"
  printf 'Label Studio uses the PostgreSQL labelstudio database. People sign in at http://127.0.0.1:8765/.\n'
else
  printf 'Label Studio is not installed; task calls will be recorded as unavailable.\n'
fi

start_service api 8765 env DATASET_ROOT="$DATASET_ROOT" BLENDER_BIN="$BLENDER_BIN" SD_API_URL="$SD_API_URL" LABEL_STUDIO_URL="$LABEL_STUDIO_URL" CORS_ALLOW_ORIGINS="${CORS_ALLOW_ORIGINS:-}" LABEL_STUDIO_TOKEN="$LABEL_STUDIO_TOKEN" LABEL_STUDIO_PROJECT_ID="$LABEL_STUDIO_PROJECT_ID" INCEPTION_WEIGHTS="$INCEPTION_WEIGHTS" FIRE_REFERENCE_DATASET="$FIRE_REFERENCE_DATASET" FIRE_REFERENCE_LIMIT="$FIRE_REFERENCE_LIMIT" ANNOTATION_MODEL="${ANNOTATION_MODEL:-}" ANNOTATION_DEVICE="${ANNOTATION_DEVICE:-mps}" "$PYTHON" -m uvicorn firephantom.api:app --host 127.0.0.1 --port 8765

ready=0
for _ in {1..60}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8765/health >/dev/null 2>&1; then
    printf 'FirePhantom is ready: http://127.0.0.1:8765/\n'
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  printf 'FirePhantom API did not become ready. Check %s/logs/api.log\n' "$RUNTIME" >&2
  exit 1
fi

if [[ "${FIREPHANTOM_FOREGROUND:-1}" == "1" ]]; then
  if ((${#started_pids[@]} == 0)); then
    printf 'All local services were already running; nothing to supervise in this process.\n'
    exit 0
  fi
  trap stop_started_services EXIT
  printf 'Local services are supervised by this process. Press Ctrl-C to stop services started here.\n'
  while true; do
    for pid in "${started_pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        printf 'A local service exited unexpectedly (pid %s). Check %s/logs.\n' "$pid" "$RUNTIME" >&2
        exit 1
      fi
    done
    sleep 2
  done
fi

printf 'Local services are running in the background. Use scripts/stop-local.sh to stop services started here.\n'
