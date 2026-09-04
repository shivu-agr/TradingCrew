"""A/B harness for the PM direct-signal injections (and decision-desk planning).

Why
---
We keep adding *direct* signals to the Portfolio Manager (the deterministic
guidance ledger, the market regime, and now the per-desk bias board) and
turned on ReAct-style planning for the decision desks.  The open question the
operator asked is blunt: **do these earn their wall-clock cost, or is it "a lot
of time for little to no addition"?**  This harness answers it empirically.

What it does
------------
For each ticker it runs the *real* workflow in-process (same ``AnalysisRunner``
the web/Kafka paths use) under two arms — a BASELINE and a TREATMENT that
differ by exactly one feature flag — and records, per run:

* wall-clock seconds,
* the decision (action / size / conviction),
* which analyst LENSES actually survived into the PM's final rationale
  (reusing ``scale_report``'s classifier — the whole point of the bias board
  is to raise this coverage), and
* whether the decision *changed* between arms.

It then prints a side-by-side table.  Because the bias board adds **no** extra
LLM/tool/network call (it is a pure parse of TLDRs we already have), the time
delta should be ~noise; the payoff shows up as higher non-yfinance lens
coverage in the rationale.

Usage
-----
::

    PYTHONPATH=. .venv/bin/python scripts/ab_pm_signals.py \\
        --tickers AAPL RELIANCE.NS --repeats 1 --feature board

    # compare planning on/off instead (this is the heavy, time-costly one):
    PYTHONPATH=. .venv/bin/python scripts/ab_pm_signals.py \\
        --tickers AAPL --feature planning

Runs are sequential (the CrewAI event bus is a process singleton), so a run of
N tickers * 2 arms * R repeats takes N*2*R full workflows.  Time is not a
concern for a batch job, but keep the set small for an interactive check.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

# The feature under test, expressed as {env_var: (baseline_value, treatment_value)}.
_FEATURES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "board": {"TRADINGCREW_PM_DESK_BIAS_BOARD": ("0", "1")},
    "planning": {"TRADINGCREW_DECISION_PLANNING": ("0", "1")},
    # Deterministic price-structure card (vol-state / regime-age / DIP_BUY)
    # into the PM prompt.  One OHLCV read + a BOCPD pass (~1s), no LLM — so the
    # time delta should be ~noise; the payoff is size discipline in high-vol /
    # fresh-break names and a flagged mean-reversion entry on oversold dips.
    "price": {"TRADINGCREW_PM_PRICE_STRUCTURE": ("0", "1")},
    # "all" is filled in main() from every single-feature map.
}


def _lens_tools():
    """Import the shared lens classifier from scale_report (same repo)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scale_report import _lenses_in_rationale, _NON_YFINANCE_LENSES  # type: ignore

    return _lenses_in_rationale, _NON_YFINANCE_LENSES


def _extract(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the decision + rationale out of the emitted event stream."""
    action = size = conviction = None
    rationale = ""
    board_seen = False
    for ev in events:
        t = ev.get("type")
        if t == "final_decision":
            d = ev.get("decision") or {}
            action = d.get("action")
            size = d.get("size_pct_of_book")
            conviction = d.get("conviction_score", d.get("confidence"))
        elif t == "action_proposal":
            prop = ev.get("proposal") or {}
            rationale = prop.get("rationale") or rationale
            if conviction is None:
                conviction = prop.get("conviction_score")
            if action is None:
                action = prop.get("action")
        elif t == "node_completed" and ev.get("role") == "Portfolio Manager":
            rationale = rationale or (ev.get("output") or "")
    return {
        "action": action,
        "size_pct": size,
        "conviction": conviction,
        "rationale": rationale,
        "board_seen": board_seen,
    }


async def _run_and_collect(ticker: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from web.backend.runner import AnalysisRunner

    loop = asyncio.get_running_loop()
    runner = AnalysisRunner(loop)
    await runner.run(ticker, cfg)
    q = runner.events  # property, not a method
    events: List[Dict[str, Any]] = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _one_run(ticker: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    lenses_in_rationale, non_yf = _lens_tools()
    t0 = time.perf_counter()
    events = asyncio.run(_run_and_collect(ticker, cfg))
    elapsed = time.perf_counter() - t0
    info = _extract(events)
    lenses = lenses_in_rationale(info["rationale"])
    info["elapsed_s"] = elapsed
    info["lenses"] = lenses
    info["non_yf_lenses"] = sorted(lenses & non_yf)
    info["error"] = any(e.get("type") == "error" for e in events)
    return info


def _apply_env(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    prev: Dict[str, Optional[str]] = {}
    for k, v in overrides.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = v
    return prev


def _restore_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _arm_env(feature_map: Dict[str, Tuple[str, str]], arm: int) -> Dict[str, str]:
    return {k: pair[arm] for k, pair in feature_map.items()}


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--feature", choices=list(_FEATURES) + ["all"], default="board")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)
    p.add_argument("--memory", action="store_true", default=False)
    p.add_argument("--trade-date", default=None)
    args = p.parse_args()

    if args.feature == "all":
        feature_map: Dict[str, Tuple[str, str]] = {}
        for m in _FEATURES.values():
            feature_map.update(m)
    else:
        feature_map = _FEATURES[args.feature]

    base_cfg = {
        "trade_date": args.trade_date or date.today().isoformat(),
        "asset_class": "stock",
        "book": "paper",
        "memory": args.memory,
        "max_debate_rounds": args.debate_rounds,
        "max_risk_rounds": args.risk_rounds,
        "tools_enabled": None,
        "quality_review": False,
    }

    # results[arm_name] = list of per-run dicts
    results: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "treatment": []}
    rows: List[str] = []
    print(f"A/B feature={args.feature} flags={feature_map} "
          f"tickers={args.tickers} repeats={args.repeats}\n")

    for ticker in args.tickers:
        for rep in range(args.repeats):
            per_arm: Dict[str, Dict[str, Any]] = {}
            for arm_idx, arm_name in ((0, "baseline"), (1, "treatment")):
                overrides = _arm_env(feature_map, arm_idx)
                prev = _apply_env(overrides)
                try:
                    print(f"[{ticker} r{rep} {arm_name}] {overrides} …", flush=True)
                    info = _one_run(ticker, dict(base_cfg))
                finally:
                    _restore_env(prev)
                results[arm_name].append(info)
                per_arm[arm_name] = info
                print(f"    -> action={info['action']} size={info['size_pct']} "
                      f"conv={info['conviction']} time={info['elapsed_s']:.1f}s "
                      f"non_yf_lenses={info['non_yf_lenses']} err={info['error']}")
            b, t = per_arm["baseline"], per_arm["treatment"]
            changed = (b["action"] != t["action"])
            rows.append(
                f"{ticker:14} r{rep}  "
                f"act {str(b['action']):>10} -> {str(t['action']):<10}"
                f"{'  *FLIP*' if changed else '':8}  "
                f"time {b['elapsed_s']:6.1f}s -> {t['elapsed_s']:6.1f}s  "
                f"non_yf_lenses {len(b['non_yf_lenses'])} -> {len(t['non_yf_lenses'])}"
            )

    def _agg(arm: str) -> Dict[str, Any]:
        rs = [r for r in results[arm] if not r["error"]]
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "mean_time": statistics.mean(r["elapsed_s"] for r in rs),
            "mean_non_yf": statistics.mean(len(r["non_yf_lenses"]) for r in rs),
            "mean_lenses": statistics.mean(len(r["lenses"]) for r in rs),
        }

    ab = _agg("baseline")
    at = _agg("treatment")

    print("\n" + "=" * 78)
    print("PER-RUN")
    for r in rows:
        print("  " + r)
    print("\nAGGREGATE (errors excluded)")
    print(f"  {'metric':28} {'baseline':>12} {'treatment':>12} {'delta':>12}")
    if ab.get("n") and at.get("n"):
        for key, label in (
            ("mean_time", "mean wall-clock (s)"),
            ("mean_non_yf", "mean non-yfinance lenses"),
            ("mean_lenses", "mean total lenses"),
        ):
            bv, tv = ab[key], at[key]
            print(f"  {label:28} {bv:12.2f} {tv:12.2f} {tv - bv:+12.2f}")
        print(f"\n  runs: baseline={ab['n']} treatment={at['n']}")
        dt = at["mean_time"] - ab["mean_time"]
        dl = at["mean_non_yf"] - ab["mean_non_yf"]
        print(f"\n  VERDICT: time {dt:+.1f}s/run, non-yfinance lens coverage "
              f"{dl:+.2f}/run.")
    else:
        print("  (insufficient successful runs to aggregate)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
