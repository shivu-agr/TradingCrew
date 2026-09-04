#!/usr/bin/env bash
# =============================================================================
# scale_run.sh - daily forward-collection over the 200-stock universe.
# =============================================================================
#
# Runs the crew LIVE over universe/all200.txt (100 IN + 100 US), records one
# forward episode per name, resolves whatever horizons have elapsed, and writes
# a post-run analysis report.  Designed to be driven by cron once per day and
# to finish well within 8h via parallel workers.
#
#   - RESUMABLE: re-running the same UTC day skips names already collected
#     (trading_crew.agentic.scale keys a per-day state file).
#   - SINGLE-FLIGHT: an flock on <cache>/forward/collect.lock (shared with the
#     UI job in web/backend/scale_runner.py) prevents cron and the dashboard
#     from collecting at the same time and racing the episode store.
#   - LOGGED: JSONL progress + crew logs land under logs/scale/<UTC-date>/ for
#     scripts/scale_report.py to analyse.
#
# Env overrides:
#   WORKERS       parallel crew runs (default 6).  Tune to the hosted LLM's
#                 throughput; the run is resumable so a partial day is fine.
#   UNIVERSE      universe file (default universe/all200.txt).
#   SKIP_RESOLVE  set to 1 to skip the resolve sweep.
#   SKIP_REPORT   set to 1 to skip the post-run report.
#   REQUIRE_SEARXNG  set to 1 to make an unreachable SearXNG a hard failure
#                 (default: warn + continue, since the search chain falls back).
#
# Usage:
#   scripts/scale_run.sh
#   WORKERS=8 UNIVERSE=universe/in100.txt scripts/scale_run.sh
# =============================================================================

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
cd "$repo_root"

# --- Python interpreter (mirror backfill.sh resolution) ----------------------
if [[ -x "$repo_root/.venv/bin/python" ]]; then
    PY="$repo_root/.venv/bin/python"
elif [[ -x "$repo_root/../.venv/bin/python" ]]; then
    PY="$repo_root/../.venv/bin/python"
else
    PY="${PYTHON:-python}"
fi

WORKERS="${WORKERS:-6}"
UNIVERSE="${UNIVERSE:-universe/all200.txt}"
DAY="$(date -u +%F)"
CACHE_DIR="${TRADINGCREW_CACHE_DIR:-$HOME/.trading_crew}"
LOG_DIR="$repo_root/logs/scale/$DAY"
LOCK_FILE="$CACHE_DIR/forward/collect.lock"

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') [scale_run] $*" | tee -a "$LOG_DIR/run.log" >&2; }

if [[ ! -f "$UNIVERSE" ]]; then
    log "FATAL: universe file not found: $UNIVERSE"
    exit 2
fi

# --- Single-flight: acquire the shared collect lock (non-blocking) -----------
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "another collect run holds $LOCK_FILE; exiting without starting a second run"
    exit 0
fi

# --- Read the few env vars the preflight needs (python loads .env itself) -----
read_env() { grep -E "^$1=" "$repo_root/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true; }
SEARXNG_URL="${SEARXNG_URL:-$(read_env SEARXNG_URL)}"
LLM_BASE="$(read_env VLLM_LLM_BASE_URL)"
LLM_KEY="$(read_env VLLM_LLM_API_KEY)"

# --- Preflight: SearXNG (search-at-scale) + LLM (mandatory) -------------------
if [[ -n "$SEARXNG_URL" ]] && curl -sf --max-time 8 "${SEARXNG_URL%/}/search?q=ping&format=json" >/dev/null 2>&1; then
    log "preflight: SearXNG OK ($SEARXNG_URL)"
else
    if [[ "${REQUIRE_SEARXNG:-0}" == "1" ]]; then
        log "FATAL: SearXNG unreachable at '${SEARXNG_URL:-<unset>}' and REQUIRE_SEARXNG=1"
        exit 3
    fi
    log "WARN: SearXNG unreachable at '${SEARXNG_URL:-<unset>}' - search will fall back to slower/rate-limited backends"
fi

if [[ -z "$LLM_BASE" ]]; then
    log "FATAL: VLLM_LLM_BASE_URL not set in .env"
    exit 3
fi
if curl -sf --max-time 12 "${LLM_BASE%/}/models" -H "Authorization: Bearer ${LLM_KEY:-dummy}" >/dev/null 2>&1; then
    log "preflight: LLM OK ($LLM_BASE)"
else
    log "FATAL: LLM endpoint unreachable at $LLM_BASE (check .env lines 10-14 / network)"
    exit 3
fi

n_names="$($PY -c "
import sys
c=0
for line in open('$UNIVERSE'):
    line=line.split('#',1)[0].strip()
    if line: c+=1
print(c)
")"
log "starting collect: universe=$UNIVERSE names=$n_names workers=$WORKERS day=$DAY log_dir=$LOG_DIR"

# --- Collect: stdout=JSONL progress, stderr=crew logging ---------------------
start_ts=$(date +%s)
set +e
"$PY" -m trading_crew.agentic.scale collect \
    --universe "$UNIVERSE" \
    --workers "$WORKERS" \
    > >(tee -a "$LOG_DIR/collect.jsonl") \
    2> >(tee -a "$LOG_DIR/run.log" >&2)
collect_rc=$?
set -e
log "collect finished rc=$collect_rc elapsed=$(( $(date +%s) - start_ts ))s"

# --- Resolve elapsed horizons (cheap; safe to run daily) ---------------------
if [[ "${SKIP_RESOLVE:-0}" != "1" ]]; then
    log "resolving elapsed horizons (--reflect)"
    set +e
    "$PY" -m trading_crew.agentic.scale resolve --reflect \
        > >(tee -a "$LOG_DIR/resolve.jsonl") \
        2> >(tee -a "$LOG_DIR/run.log" >&2)
    log "resolve finished rc=$?"
    set -e
fi

# --- Post-run analysis report ------------------------------------------------
if [[ "${SKIP_REPORT:-0}" != "1" ]]; then
    log "writing post-run report"
    "$PY" scripts/scale_report.py --day "$DAY" --universe "$UNIVERSE" --log-dir "$LOG_DIR" \
        | tee "$LOG_DIR/report.md" || log "WARN: report generation failed (rc=$?)"
fi

log "done (report: $LOG_DIR/report.md)"
exit "$collect_rc"
