#!/usr/bin/env bash
# Stop the background TradingCrew web UI started by ./start.sh.
#
# Sends SIGTERM to the tracked PID (uvicorn shuts down gracefully), waits,
# then escalates to SIGKILL if it hasn't exited.  Leaves the shared Xvfb
# display running (it's reused across cron + web).
#
# Usage:
#   ./stop.sh

set -euo pipefail

cd "$(dirname "$0")"

PIDFILE=".server.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "No $PIDFILE found — server not running (or was not started via ./start.sh)."
  exit 0
fi

PID="$(cat "$PIDFILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
  echo "Empty $PIDFILE — nothing to stop."
  rm -f "$PIDFILE"
  exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID is not running — clearing stale $PIDFILE."
  rm -f "$PIDFILE"
  exit 0
fi

echo "Stopping TradingCrew UI (pid $PID)…"
kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Stopped."
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep 0.5
done

echo "Did not exit after SIGTERM; sending SIGKILL…"
kill -KILL "$PID" 2>/dev/null || true
sleep 1
if kill -0 "$PID" 2>/dev/null; then
  echo "error: failed to kill pid $PID." >&2
  exit 1
fi
echo "Stopped (forced)."
rm -f "$PIDFILE"
