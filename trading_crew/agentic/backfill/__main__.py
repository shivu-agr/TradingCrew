"""CLI: python -m trading_crew.agentic.backfill --tickers RELIANCE.NS,TCS.NS ...

Runs (or resumes) an offline point-in-time decision-label backfill.  Safe to
Ctrl-C and re-run with the same args — completed cells are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import List, Optional

from .harness import BackfillConfig, job_id_for, run_backfill


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline PIT decision-label backfill.")
    p.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. RELIANCE.NS,TCS.NS")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--spacing-days", type=int, default=30, help="Days between decision dates (default 30)")
    p.add_argument("--horizon-days", type=int, default=90, help="Outcome horizon per decision (default 90)")
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)
    p.add_argument("--cell-gap-seconds", type=float, default=5.0, help="Polite gap between cells")
    p.add_argument("--no-resolve", action="store_true", help="Skip the outcome-resolution pass")
    p.add_argument("--reflect", action="store_true", help="Generate LLM reflections during resolution")
    p.add_argument("--include-unresolvable", action="store_true",
                   help="Also emit cells whose outcome window hasn't elapsed yet (recorded PENDING)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = BackfillConfig(
        tickers=[t.strip() for t in args.tickers.split(",") if t.strip()],
        start_date=args.start,
        end_date=args.end,
        spacing_days=args.spacing_days,
        horizon_days=args.horizon_days,
        debate_rounds=args.debate_rounds,
        risk_rounds=args.risk_rounds,
        cell_gap_seconds=args.cell_gap_seconds,
        resolve=not args.no_resolve,
        reflect=args.reflect,
        only_resolvable=not args.include_unresolvable,
    )
    print(f"Backfill job {job_id_for(config)} — resumable; Ctrl-C is safe.\n")

    stop = threading.Event()

    def _log(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False))

    try:
        summary = run_backfill(config, should_stop=stop.is_set, on_log=_log)
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted — progress saved; re-run the same command to resume.")
        return 130
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
