#!/usr/bin/env bash
# =============================================================================
# backfill.sh — offline point-in-time decision-label backfill
# =============================================================================
#
# Re-runs the crew "as of" historical decision dates so the desk builds a
# corpus of labeled decisions NOW instead of waiting months for live ones to
# resolve.  Every decision runs inside a strict point-in-time context (only
# information public on the decision date is used), is stamped
# source="backfill", and is resolved immediately (the horizon is in the past).
#
# The run is RESUMABLE: Ctrl-C is safe, and re-running the same args skips the
# cells already completed.  Requests are rate-limited (screener throttle +
# --cell-gap-seconds) and the crew uses the fast VLLM_LLM_* endpoint from .env.
#
# Usage
# -----
#   scripts/backfill.sh --tickers RELIANCE.NS,TCS.NS --start 2023-01-01 --end 2024-06-01
#   scripts/backfill.sh --tickers TCS.NS --start 2023-01-01 --end 2024-01-01 \
#                       --spacing-days 30 --horizon-days 90 --cell-gap-seconds 8 --reflect
#   scripts/backfill.sh --help
# =============================================================================

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
cd "$repo_root"

if [[ -x "$repo_root/../.venv/bin/python" ]]; then
    PY="$repo_root/../.venv/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    PY="$repo_root/.venv/bin/python"
else
    PY="${PYTHON:-python}"
fi

exec "$PY" -m trading_crew.agentic.backfill "$@"
