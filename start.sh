#!/bin/sh
# POSIX launcher: no host Python, Node, uv or npm.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
action=start
project=${COMPOSE_PROJECT_NAME:-ai07-portable}
backup_name=
transfer_path=
install_docker=0
rebuild=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action) action=$2;shift 2;;
        --project) project=$2;shift 2;;
        --gpu) export AI07_GPU=1;shift;;
        --rebuild) rebuild=1;shift;;
        --port) export AI07_WEB_PORT=$2;shift 2;;
        --api-port) export AI07_API_PORT=$2;shift 2;;
        --label-port) export AI07_LS_PORT=$2;shift 2;;
        --backup) backup_name=$2;shift 2;;
        --path) transfer_path=$2;shift 2;;
        --install-docker) install_docker=1;shift;;
        *) echo "Unknown argument: $1" >&2;exit 2;;
    esac
done
case "$project" in ''|*[!a-z0-9_-]*) echo 'Invalid project name' >&2;exit 2;;esac
if ! command -v docker >/dev/null 2>&1; then
    if [ "$install_docker" != 1 ]; then echo 'Install Docker or run: sh start.sh --install-docker' >&2;exit 1;fi
    sh scripts/install-docker.sh
fi
if ! docker info >/dev/null 2>&1; then
    echo 'Docker is not accessible. Start the engine; on Linux run this script with sudo or configure Docker group access.' >&2
    exit 1
fi
export COMPOSE_PROJECT_NAME=$project
python_image=python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
retry_download() {
    attempt=1
    while ! "$@";do
        if [ "$attempt" -ge 4 ];then
            echo 'Docker download/build failed. Check Docker Desktop/Engine registry and proxy connectivity, then rerun. Existing data volumes are preserved.' >&2
            return 1
        fi
        echo "Download/build failed; retrying ($attempt/4)..." >&2
        sleep "$((attempt*2))"
        attempt=$((attempt+1))
    done
}
ensure_image() {
    if ! docker image inspect "$1" >/dev/null 2>&1;then retry_download docker pull "$1";fi
}
ensure_image "$python_image"
load_settings() {
    settings=$(docker run --rm -i \
        --mount "type=volume,source=${project}_credentials,target=/state" \
        --mount "type=volume,source=${project}_platform-state,target=/platform" \
        --mount "type=volume,source=${project}_backups,target=/backups" \
        -e AI07_WEB_PORT -e AI07_API_PORT -e AI07_LS_PORT -e AI07_GPU \
        -e AI07_BIND_ADDRESS -e AI07_ALLOWED_ORIGINS -e AI07_LABEL_STUDIO_PUBLIC_URL -e AI07_SECURE_COOKIE -e AI07_REQUIRE_LOGIN \
        "$python_image" python - < scripts/portable-init.py)
    while IFS='=' read -r key value; do
        case "$key" in AI07_*) export "$key=$value";;esac
    done <<EOF
$settings
EOF
}
load_settings
compose() {
    if [ "$AI07_GPU" = 1 ]; then
        AI07_WORKER_BASE_IMAGE="${project}-gpu:local" docker compose --project-name "$project" -f compose.portable.yaml -f compose.portable.gpu.yaml "$@"
    else
        AI07_WORKER_BASE_IMAGE="${project}-api:local" docker compose --project-name "$project" -f compose.portable.yaml "$@"
    fi
}
start_services() { compose up -d --wait --no-build --pull never platform-web platform-api worker annotation-platform; }
runtime_ready() {
    for image in api web worker;do
        docker image inspect "${project}-${image}:local" >/dev/null 2>&1 || return 1
    done
    probe="import os,importlib.util;from pathlib import Path;gpu=os.getenv('AI07_CHECK_GPU')=='1';cuda=bool(__import__('torch').version.cuda) if importlib.util.find_spec('torch') else False;paths=['/app/artifacts/scene-assets/library/wooden_crate_01.glb'];paths+=['/app/models/sd15/model_index.json','/app/models/sd15/unet/diffusion_pytorch_model.fp16.safetensors','/app/models/yolo11n.pt','/app/artifacts/torch-cache/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'] if gpu else [];print('ready' if cuda==gpu and all(Path(p).is_file() for p in paths) else 'missing')"
    result=$(docker run --rm --pull never --entrypoint python -e "AI07_CHECK_GPU=$AI07_GPU" \
        --mount "type=volume,source=${project}_models,target=/app/models,readonly" \
        --mount "type=volume,source=${project}_artifacts,target=/app/artifacts,readonly" "${project}-worker:local" -c "$probe") || return 1
    [ "$result" = ready ]
}
case "$action" in
    start|setup)
        ensure_image postgres:16-alpine
        ensure_image heartexlabs/label-studio:1.22.0
        prepare=0
        if [ "$rebuild" = 1 ] || [ "$action" = setup ];then prepare=1
        elif ! runtime_ready;then prepare=1;fi
        if [ "$prepare" = 1 ];then
            retry_download compose --profile setup build assets platform-api
            if ! docker image inspect "${project}-engines:fds6.11.1-blender4.5.3" >/dev/null 2>&1;then retry_download compose --profile setup build engines;fi
            if [ "$AI07_GPU" = 1 ]; then
                retry_download compose --profile gpu build gpu
                for asset in sd15 yolo fid;do compose --profile setup run --rm --no-deps --pull never assets --only "$asset";done
            fi
            retry_download compose build worker platform-web
            compose run --rm --no-deps --pull never --entrypoint blender worker --background --python /app/tools/create-scene-models.py
        else
            echo 'Using prepared local images and data. Skipping registry access and image builds; use --rebuild after changing source.'
        fi
        if [ "$action" = start ]; then
            start_services
            echo "Workbench: http://localhost:$AI07_WEB_PORT  API: http://localhost:$AI07_API_PORT/docs"
            echo 'New installation: choose your administrator username and password. Existing installations keep their accounts. Data lives in Docker named volumes.'
        fi;;
    stop) compose stop;;
    status) compose ps;;
    list-backups) compose --profile tools run --rm --no-deps backup list;;
    backup|restore)
        if [ "$action" = restore ] && [ -z "$backup_name" ]; then echo 'Restore requires --backup name.tar; it replaces target stack data.' >&2;exit 2;fi
        compose stop platform-web platform-api worker annotation-platform
        completed=0
        trap 'if [ "$completed" = 1 ] || [ "$action" = backup ];then load_settings;start_services;fi' EXIT
        compose up -d --wait platform-db
        if [ -n "$backup_name" ]; then
            compose --profile tools run --rm --no-deps backup "$action" --name "$backup_name"
        else
            compose --profile tools run --rm --no-deps backup "$action"
        fi
        completed=1;;
    export|import)
        case "$backup_name" in ''|*[!a-zA-Z0-9_.-]*) echo 'Invalid backup name' >&2;exit 2;;esac
        [ -n "$transfer_path" ] || { echo 'Specify --path and --backup' >&2;exit 2; }
        transfer_id=$(docker create --mount "type=volume,source=${project}_backups,target=/backups" "$python_image" python -c pass)
        trap 'docker rm "$transfer_id" >/dev/null' EXIT
        if [ "$action" = export ]; then
            mkdir -p "$transfer_path"
            docker cp "$transfer_id:/backups/$backup_name" "$transfer_path/$backup_name"
            docker cp "$transfer_id:/backups/$backup_name.sha256" "$transfer_path/$backup_name.sha256"
        else
            docker cp "$transfer_path" "$transfer_id:/backups/$backup_name"
            docker cp "$transfer_path.sha256" "$transfer_id:/backups/$backup_name.sha256"
        fi;;
    *) echo "Unknown action: $action" >&2;exit 2;;
esac
