#!/usr/bin/env bash
# Start the Streamlit app headless on a spare port, wait for /_stcore/health, then stop it.
set -euo pipefail
PORT="${SMOKE_PORT:-8599}"
LOG="$(mktemp -t risklab_smoke.XXXXXX)"
uv run streamlit run app/Home.py --server.headless true --server.port "$PORT" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do
  if curl -fsS "http://localhost:${PORT}/_stcore/health" 2>/dev/null | grep -q ok; then
    echo "app smoke: health ok on port ${PORT}"
    exit 0
  fi
  sleep 1
done
echo "app smoke: FAILED (no health response); log follows" >&2
cat "$LOG" >&2
exit 1
