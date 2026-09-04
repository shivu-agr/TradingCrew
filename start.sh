#!/usr/bin/env bash
# Start the TradingCrew web UI in the background.
#
# - Runs ./run_web.sh (which sets up the CA bundle + Xvfb and execs uvicorn)
#   under nohup so it survives the shell/SSH session.
# - Writes all server output to a timestamped log under logs/server/ and
#   keeps logs/server/current.log pointing at the active file (for analysis).
# - Records the server PID in .server.pid so ./stop.sh can stop it cleanly.
#
# Usage:
#   ./start.sh                 # 0.0.0.0:8002  (v3; leave v2 on 8001)
#   PORT=9001 ./start.sh       # custom port
#   HOST=127.0.0.1 ./start.sh  # custom host
#
# Idempotent: refuses to start a second instance if one is already running.

set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
PIDFILE=".server.pid"
LOG_DIR="logs/server"

# Already running? (tracked PID alive)
if [[ -f "$PIDFILE" ]]; then
  existing="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$existing" ]] && kill -0 "$existing" 2>/dev/null; then
    echo "TradingCrew UI already running (pid $existing). Use ./stop.sh first to restart."
    exit 0
  fi
  # Stale pidfile — clean it up.
  rm -f "$PIDFILE"
fi

# Warn (don't clobber) if something else is already bound to the port.
if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
  echo "warning: a process is already listening on port $PORT but no $PIDFILE is tracked." >&2
  echo "         stop it manually or set PORT=<other> before starting." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/server-$TS.log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/current.log"

echo "Starting TradingCrew UI on http://${HOST}:${PORT}"
echo "Logs: $LOG_FILE (symlinked as $LOG_DIR/current.log)"

HOST="$HOST" PORT="$PORT" nohup ./run_web.sh >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"

# Wait for uvicorn to report readiness (or the process to die).
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: server exited during startup. Last log lines:" >&2
    tail -n 30 "$LOG_FILE" >&2 || true
    rm -f "$PIDFILE"
    exit 1
  fi
  if grep -qE "Uvicorn running on|Application startup complete" "$LOG_FILE" 2>/dev/null; then
    echo "TradingCrew UI is up (pid $SERVER_PID) -> http://${HOST}:${PORT}"
    exit 0
  fi
  sleep 1
done

echo "warning: server did not report readiness within 60s; it may still be starting." >&2
echo "         check $LOG_FILE (pid $SERVER_PID)." >&2
exit 0
