#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${FIREPHANTOM_RUNTIME:-$ROOT/data}"
case "$RUNTIME" in
  /*|[A-Za-z]:/*) ;;
  *) RUNTIME="$ROOT/$RUNTIME" ;;
esac

for name in api label-studio diffusion; do
  file="$RUNTIME/$name.pid"
  if [[ -f "$file" ]]; then
    pid="$(<"$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      printf 'Stopped %s (pid %s)\n' "$name" "$pid"
    fi
    rm -f "$file"
  fi
done
