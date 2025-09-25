#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-api}"
if [[ $# -gt 0 ]]; then
  shift
fi

start_api() {
  exec uvicorn src.app.api:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    "$@"
}

case "$cmd" in
  api)
    start_api "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
