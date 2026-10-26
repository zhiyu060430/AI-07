#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${FIREPHANTOM_PYTHON:-}" ]]; then
  exec "$FIREPHANTOM_PYTHON" -m firephantom.local start "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m firephantom.local start "$@"
fi
exec python -m firephantom.local start "$@"
