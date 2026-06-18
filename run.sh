#!/usr/bin/env bash
# Start the Agent Response Predictor backend + TUI.
# Usage:  ./run.sh
# Set PY env var to point at a specific python binary, e.g.:
#   PY=/path/to/venv/bin/python3 ./run.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
PORT=8888
LOG=/tmp/agent_predict_api.log

cd "$DIR"

# Is the backend already up?
if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "Backend already running on :${PORT}"
else
  echo "Starting backend on :${PORT} (log: ${LOG}) ..."
  nohup "$PY" -m uvicorn api:app --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
  disown
  # wait for health
  for i in $(seq 1 30); do
    if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      echo "Backend healthy."
      break
    fi
    sleep 1
  done
fi

echo "Launching TUI ..."
exec "$PY" tui.py
