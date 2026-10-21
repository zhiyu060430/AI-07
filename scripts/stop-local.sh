#!/usr/bin/env bash
set -euo pipefail

RUNTIME="/Volumes/T7/FirePhantom-runtime"
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
