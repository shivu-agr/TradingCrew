#!/usr/bin/env bash
# Launch the TradingCrew web UI.
#
# Usage:
#   ./run_web.sh              # listens on 0.0.0.0:8001
#   PORT=9001 ./run_web.sh    # custom port
#
# The shell script reuses the workspace .venv at ../.venv (the one shared
# with the rest of the crewai/ workspace); fall back to a local .venv if
# you create one inside this folder.

set -euo pipefail

cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif [[ -x ../.venv/bin/python ]]; then
  PY=../.venv/bin/python
else
  echo "error: no .venv found at ./.venv or ../.venv. Create one and install requirements.txt first." >&2
  exit 1
fi

if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Installing web dependencies (fastapi, uvicorn, websockets, python-multipart)…"
  "$PY" -m pip install fastapi 'uvicorn[standard]' websockets python-multipart
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

export TRADINGCREW_WEB_HOST="$HOST"
export TRADINGCREW_WEB_PORT="$PORT"

echo "TradingCrew UI -> http://${HOST}:${PORT}"
exec "$PY" -m uvicorn web.backend.app:app --host "$HOST" --port "$PORT"
