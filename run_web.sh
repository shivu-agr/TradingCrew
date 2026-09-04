#!/usr/bin/env bash
# Launch the TradingCrew web UI.
#
# Usage:
#   ./run_web.sh              # listens on 0.0.0.0:8002 (v3; v2 stays on 8001)
#   PORT=9001 ./run_web.sh    # custom port
#
# The shell script auto-discovers a virtualenv, preferring a local one
# inside this folder (./.venv or ./venv) and falling back to the shared
# workspace venv one level up (../.venv or ../venv).

set -euo pipefail

cd "$(dirname "$0")"

PY=""
for candidate in .venv/bin/python venv/bin/python ../.venv/bin/python ../venv/bin/python; do
  if [[ -x "$candidate" ]]; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "error: no virtualenv found at ./.venv, ./venv, ../.venv or ../venv. Create one and install requirements.txt first." >&2
  exit 1
fi

if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Installing web dependencies (fastapi, uvicorn, websockets, python-multipart)…"
  "$PY" -m pip install fastapi 'uvicorn[standard]' websockets python-multipart
fi

# Point every HTTP client (yfinance/curl_cffi, requests, urllib) at a CA
# bundle that TLS actually validates against.  Framework Python builds ship
# without OS CA certs, so out of the box TLS to Yahoo/screener.in fails with
# "unable to get local issuer certificate" and the OHLCV/fundamentals data
# path comes back empty.  On macOS behind a TLS-inspecting corporate proxy
# (e.g. Cisco Secure Access) the re-signed certs also need the corporate root,
# which lives in the system keychain but not in certifi.  Build a combined
# bundle = certifi + macOS keychain roots so both cases resolve.
CERTIFI_BUNDLE="$("$PY" -c "import certifi; print(certifi.where())" 2>/dev/null || true)"
if [[ -n "$CERTIFI_BUNDLE" ]]; then
  CA_BUNDLE="$CERTIFI_BUNDLE"
  if [[ "$(uname)" == "Darwin" ]] && command -v security >/dev/null 2>&1; then
    COMBINED_CA="${TRADINGCREW_CACHE_DIR:-$HOME/.trading_crew}/ca-bundle.pem"
    mkdir -p "$(dirname "$COMBINED_CA")"
    {
      cat "$CERTIFI_BUNDLE"
      security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain 2>/dev/null
      security find-certificate -a -p /Library/Keychains/System.keychain 2>/dev/null
      security find-certificate -a -p "$HOME/Library/Keychains/login.keychain-db" 2>/dev/null
    } > "$COMBINED_CA"
    CA_BUNDLE="$COMBINED_CA"
  fi
  export CURL_CA_BUNDLE="$CA_BUNDLE"
  export SSL_CERT_FILE="$CA_BUNDLE"
  export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
fi

# The google-headless websearch backend runs a *headful* Chromium (Google
# CAPTCHAs true-headless on datacenter IPs), so on Linux it needs a virtual X
# display via Xvfb.  Start one on GOOGLE_HEADLESS_DISPLAY when the backend is
# enabled and that display isn't already up.  Idempotent: reuse an existing
# server (the /tmp/.X11-unix/X<n> socket).  .env is authoritative for the app,
# so mirror the two knobs we need from it here (shell env still wins).
if [[ "$(uname)" == "Linux" ]] && [[ -f .env ]]; then
  gh_enabled="$(grep -E '^[[:space:]]*TRADINGCREW_GOOGLE_HEADLESS_ENABLED=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
  gh_display_env="$(grep -E '^[[:space:]]*GOOGLE_HEADLESS_DISPLAY=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
  GH_DISPLAY="${GOOGLE_HEADLESS_DISPLAY:-${gh_display_env:-:99}}"
  case "$gh_enabled" in
    1|true|True|TRUE|yes|on)
      if command -v Xvfb >/dev/null 2>&1; then
        dnum="${GH_DISPLAY#:}"
        if [[ -e "/tmp/.X11-unix/X${dnum}" ]]; then
          echo "google-headless: reusing X display ${GH_DISPLAY} (already running)."
        else
          echo "google-headless: starting Xvfb on ${GH_DISPLAY}…"
          nohup Xvfb "$GH_DISPLAY" -screen 0 1366x800x24 -nolisten tcp \
            >"/tmp/xvfb${dnum}.log" 2>&1 &
          for _ in 1 2 3 4 5 6 7 8 9 10; do
            [[ -e "/tmp/.X11-unix/X${dnum}" ]] && break; sleep 0.3
          done
          [[ -e "/tmp/.X11-unix/X${dnum}" ]] \
            && echo "google-headless: Xvfb up on ${GH_DISPLAY}." \
            || echo "warning: Xvfb did not come up on ${GH_DISPLAY} (see /tmp/xvfb${dnum}.log)." >&2
        fi
      else
        echo "warning: TRADINGCREW_GOOGLE_HEADLESS_ENABLED set but Xvfb not installed; google search will fall back to searxng/yahoo." >&2
      fi
      ;;
  esac
fi

# --- Memory hygiene for the long-running batch process ------------------
# Everything (FastAPI + the Kafka workflow consumers + up to WORKERS crews)
# runs in THIS one Python process, and each crew spins up its own CrewAI
# thread pool.  glibc gives every thread its own malloc arena (up to
# 8×ncpu of them), so a multi-hour 200-name batch fragments the heap and RSS
# climbs monotonically — freed Python objects never hand pages back to the OS
# — until the VM OOMs.  Capping arenas + lowering the trim/mmap thresholds
# keeps the resident set bounded (the app also calls malloc_trim() between
# runs; see scale_runner).  Shell env still wins, so these are overridable.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

export TRADINGCREW_WEB_HOST="$HOST"
export TRADINGCREW_WEB_PORT="$PORT"

echo "TradingCrew UI -> http://${HOST}:${PORT}"
exec "$PY" -m uvicorn web.backend.app:app --host "$HOST" --port "$PORT"
