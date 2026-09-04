"""CLI: python -m trading_crew.agentic.validate --tickers TCS.NS [...]

One command that exercises every deterministic part of the stack *together*
on a cohort of logged decisions and prints a health report:

    memory (episodes)  ->  M6 backtest (walk-forward / CPCV)
                       ->  M2/M5 PnL attribution
                       ->  L3 PSO expectancy + fusion (guardrailed signal)
                       ->  L4 RL policy availability probe

This is the "all parts working together" smoke test.  It does NOT call the
LLM (that's the live crew / backfill); it validates that logged decisions
flow cleanly through sizing, execution, backtest, attribution and fusion.

Exit code is 0 when at least one ticker produced a measurable backtest
(>=1 filled trade), 1 otherwise — so CI / scripts can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from trading_crew.agentic.evaluation import (
    EvalReport,
    default_cache_dir,
    evaluate_ticker,
    latest_fused_signal,
)


def _discover_tickers(cache_dir: str) -> List[str]:
    """List every symbol that has at least one logged episode."""
    from trading_crew.agentic.memory import EpisodicMemory

    store = Path(cache_dir) / "memory" / "episodes.jsonl"
    if not store.exists():
        return []
    seen = []
    for e in EpisodicMemory(store).all_episodes():
        s = e.symbol.upper()
        if s not in seen:
            seen.append(s)
    return seen


def _rl_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "  n/a"
    if pct:
        return f"{v * 100:+6.2f}%"
    return f"{v:+7.3f}"


def _print_report(report: EvalReport, fused: Optional[dict]) -> None:
    print(f"\n=== {report.ticker} ===")
    if report.error and report.n_proposals == 0:
        print(f"  episodes={report.n_episodes}  -> {report.error}")
        return
    print(
        f"  episodes={report.n_episodes}  proposals={report.n_proposals}  "
        f"sides={report.sides}"
    )
    if report.error:
        print(f"  backtest: {report.error}")
    else:
        m = report.metrics or {}
        print(
            f"  backtest[{report.validation}]: folds={report.n_folds}  "
            f"filled={report.n_filled}  statuses={report.trade_status_counts}"
        )
        print(
            f"    return={_fmt(m.get('total_return_pct'), pct=True)}  "
            f"sharpe={_fmt(m.get('sharpe'))}  maxdd={_fmt(m.get('max_drawdown'), pct=True)}  "
            f"win_rate={_fmt(m.get('win_rate'))}  DSR={_fmt(m.get('deflated_sharpe'))}"
        )
        if report.attribution:
            a = report.attribution
            print(
                f"    attribution: decision={_fmt(a.get('total_decision_pnl'))}  "
                f"sizing={_fmt(a.get('total_sizing_pnl'))}  cost={_fmt(a.get('total_cost_pnl'))}  "
                f"hit_rate={_fmt(a.get('hit_rate'))}"
            )
            if a.get("diagnosis"):
                print(f"    diagnosis: {a['diagnosis']}")
    if fused:
        f = fused["fused"]
        print(
            f"  fusion: side={f['side']}  score={_fmt(f['score'])}  "
            f"conf={_fmt(f['confidence'])}  weight={_fmt(f['target_weight'])}  "
            f"(policy={'yes' if fused['policy_available'] else 'no'}, "
            f"expectancy={'yes' if fused['expectancy_available'] else 'no'})"
        )
        for g in f["guardrails"]:
            print(f"    guardrail: {g}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end pipeline validation harness.")
    p.add_argument("--tickers", default="", help="Comma-separated; blank = every ticker in the store")
    p.add_argument("--cache-dir", default="", help="Episode store root (default $TRADINGCREW_CACHE_DIR or ~/.trading_crew)")
    p.add_argument("--workflow-version", default="", help="Cohort filter — only this stamped workflow_version")
    p.add_argument("--source", default="", help="Cohort filter — 'live' | 'backfill'")
    p.add_argument("--validation", default="walk_forward", choices=["walk_forward", "cpcv"])
    p.add_argument("--train-size", type=int, default=3)
    p.add_argument("--embargo-size", type=int, default=1)
    p.add_argument("--test-size", type=int, default=1)
    p.add_argument("--cpcv-groups", type=int, default=6)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--cost-model", default="standard")
    p.add_argument("--lookback-years", type=int, default=5)
    p.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary at the end")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cache_dir = args.cache_dir or default_cache_dir()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = _discover_tickers(cache_dir)

    print("Pipeline validation harness")
    print(f"  cache_dir = {cache_dir}")
    print(f"  cohort    = workflow_version={args.workflow_version or '*'} source={args.source or '*'}")
    print(f"  RL/torch  = {'available' if _rl_available() else 'NOT installed (RL inference skipped)'}")
    if not tickers:
        print("  No episodes found — run an analysis or a backfill first.")
        return 1

    reports = []
    any_measurable = False
    for ticker in tickers:
        report = evaluate_ticker(
            ticker,
            cache_dir=cache_dir,
            workflow_version=args.workflow_version or None,
            source=args.source or None,
            lookback_years=args.lookback_years,
            validation=args.validation,
            train_size=args.train_size,
            embargo_size=args.embargo_size,
            test_size=args.test_size,
            cpcv_groups=args.cpcv_groups,
            cpcv_k=args.cpcv_k,
            cost_model=args.cost_model,
        )
        fused = None
        if report.n_proposals:
            try:
                fused = latest_fused_signal(
                    ticker,
                    cache_dir=cache_dir,
                    workflow_version=args.workflow_version or None,
                    source=args.source or None,
                    lookback_years=args.lookback_years,
                    cost_model=args.cost_model,
                )
            except Exception:  # noqa: BLE001
                fused = None
        _print_report(report, fused)
        any_measurable = any_measurable or report.n_filled > 0
        reports.append({"report": report.to_dict(), "fused": fused})

    measurable = sum(1 for r in reports if r["report"]["n_filled"] > 0)
    print(
        f"\nSummary: {len(tickers)} ticker(s), "
        f"{measurable} produced a measurable backtest (>=1 filled trade)."
    )
    if args.json:
        print("\nJSON:")
        print(json.dumps(reports, indent=2))
    return 0 if any_measurable else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
