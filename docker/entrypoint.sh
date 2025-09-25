#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-api}"
if [[ $# -gt 0 ]]; then
  shift
fi

start_api() {
  exec uvicorn src.app.api:app --host 0.0.0.0 --port "${API_PORT:-8000}" "$@"
}

start_ui() {
  exec streamlit run src/app/app.py \
    --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
    --server.address="0.0.0.0" \
    "$@"
}

start_all() {
  uvicorn src.app.api:app --host 0.0.0.0 --port "${API_PORT:-8000}" "$@" &
  api_pid=$!

  streamlit run src/app/app.py \
    --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
    --server.address="0.0.0.0" &
  ui_pid=$!

  term() {
    kill "$api_pid" "$ui_pid" 2>/dev/null || true
    wait "$api_pid" "$ui_pid" 2>/dev/null || true
  }
  trap term SIGINT SIGTERM

  wait -n "$api_pid" "$ui_pid"
  exit_code=$?
  term
  exit "$exit_code"
}

case "$cmd" in
  api)
    start_api "$@"
    ;;
  ui)
    start_ui "$@"
    ;;
  all)
    start_all "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac

