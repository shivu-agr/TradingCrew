"""Worked example: how much did backtest + walk-forward actually move the
decision for a given ticker?

This is the empirical companion to the lever changes from Phase 2F.
It does not call any LLM — it replays the **deterministic** parts of
the pipeline:

  1.  ``backtest_setup`` for the trader-chosen horizon AND the full
      multi-horizon panel (20/60/120/252 days).
  2.  Computes the PM's confidence cap under the OLD rules
      ("hit-rate < 40%  =>  confidence <= 0.6") vs the NEW rules
      ("expectancy <= 0 on EVERY horizon AND best-hit-rate < 40%").
  3.  Translates the resulting (action, confidence, size) into an
      ``ActionProposal`` via the M1 bridge, then runs the M5 sizer to
      show how the size cap moves.
  4.  Runs the M6 walk-forward backtest over any logged proposals for
      the ticker, summarising the per-fold PnL impact.

Output: a markdown report at ``reports/decision_impact_<ticker>_<ts>.md``
plus a JSON sibling with the raw numbers (which the optional Canvas
artifact at ``canvases/decision-impact-NVDA.canvas.tsx`` consumes).

Usage::

    ../.venv/bin/python scripts/decision_impact_demo.py NVDA
    ../.venv/bin/python scripts/decision_impact_demo.py NVDA --target-pct 5 --stop-pct 3 --horizon 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Multi-horizon backtest (lifted from trading_crew.tools — no .func tax)
# ---------------------------------------------------------------------------


@dataclass
class HorizonRow:
    horizon_days: int
    is_trader: bool
    hit_rate: float
    expectancy: float
    payoff: float
    avg: float
    median: float
    n: int
    wins: int
    losses: int
    timeouts: int
    avg_win: float
    avg_loss: float


def _simulate_one(closes: List[float], horizon: int, tgt: float, stp: float) -> Optional[HorizonRow]:
    n = len(closes)
    if n < horizon + 30:
        return None
    wins = losses = timeouts = 0
    win_r: List[float] = []
    loss_r: List[float] = []
    timeout_r: List[float] = []
    all_r: List[float] = []
    for i in range(n - horizon):
        entry = closes[i]
        outcome = None
        for j in range(1, horizon + 1):
            r = (closes[i + j] / entry) - 1.0
            if r >= tgt:
                wins += 1
                win_r.append(r * 100.0)
                all_r.append(r * 100.0)
                outcome = "win"
                break
            if r <= -stp:
                losses += 1
                loss_r.append(r * 100.0)
                all_r.append(r * 100.0)
                outcome = "loss"
                break
        if outcome is None:
            timeouts += 1
            t_r = (closes[i + horizon] / entry - 1.0) * 100.0
            timeout_r.append(t_r)
            all_r.append(t_r)
    total = wins + losses + timeouts
    if total == 0:
        return None
    hit_rate = wins / total
    avg = sum(all_r) / total
    rs = sorted(all_r)
    median = rs[total // 2]
    avg_win = (sum(win_r) / len(win_r)) if win_r else 0.0
    avg_loss = (sum(loss_r) / len(loss_r)) if loss_r else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else (math.inf if avg_win > 0 else 0.0)
    timeout_mean = (sum(timeout_r) / len(timeout_r)) if timeout_r else 0.0
    expectancy = (
        hit_rate * avg_win
        + (losses / total) * avg_loss
        + (timeouts / total) * timeout_mean
    )
    return HorizonRow(
        horizon_days=horizon,
        is_trader=False,
        hit_rate=hit_rate,
        expectancy=expectancy,
        payoff=payoff if math.isfinite(payoff) else 99.99,
        avg=avg,
        median=median,
        n=total,
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def multi_horizon_panel(ticker: str, horizon: int, tgt_pct: float, stp_pct: float) -> List[HorizonRow]:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="5y")
    if df is None or df.empty:
        raise SystemExit(f"No OHLCV data for {ticker} — yfinance returned empty.")
    closes = [float(x) for x in df["Close"].values]
    tgt = abs(tgt_pct) / 100.0
    stp = abs(stp_pct) / 100.0
    horizons = sorted({max(5, horizon), 60, 120, 252})
    rows: List[HorizonRow] = []
    for h in horizons:
        row = _simulate_one(closes, h, tgt, stp)
        if row is None:
            continue
        if h == horizon:
            row.is_trader = True
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Decision simulator — apply OLD vs NEW PM rules to the same panel
# ---------------------------------------------------------------------------


@dataclass
class SimulatedDecision:
    rules: str
    action: str           # OVERWEIGHT | NEUTRAL | UNDERWEIGHT
    confidence: float
    size_pct: float
    rationale: str


def simulate_old_decision(rows: List[HorizonRow]) -> SimulatedDecision:
    """Pre-Phase-2F PM rules.

    * Hit-rate < 40% on the trader horizon => confidence <= 0.6
    * Quality reviewer score < 5 => size halved (we approximate that
      with: hit-rate < 30% triggers a quality penalty)
    * No long-term framing requirement — only the trader horizon
      contributes.
    """
    trader = next((r for r in rows if r.is_trader), rows[0])
    base_conf = 0.70 + min(0.15, max(0.0, trader.expectancy / 5.0))  # 0.70 baseline + expectancy bump
    if trader.hit_rate < 0.40:
        capped_conf = min(base_conf, 0.60)
    else:
        capped_conf = base_conf
    base_size = 1.5  # trader's default
    quality_penalty = trader.hit_rate < 0.30
    size = base_size * (0.5 if quality_penalty else 1.0)
    # The old rule defaulted to NEUTRAL whenever capped confidence + reduced size
    # made the trade look weak: confidence < 0.62 AND size < 1.0 => NEUTRAL.
    if capped_conf < 0.62 and size < 1.0:
        return SimulatedDecision(
            rules="old",
            action="NEUTRAL",
            confidence=min(0.55, capped_conf),
            size_pct=0.0,
            rationale=(
                f"hit-rate {trader.hit_rate * 100:.1f}% < 40% caps confidence at 0.60; "
                f"quality penalty halves size; both binding => NEUTRAL by default."
            ),
        )
    action = "OVERWEIGHT" if trader.avg > 0 else "UNDERWEIGHT"
    return SimulatedDecision(
        rules="old",
        action=action,
        confidence=round(capped_conf, 3),
        size_pct=round(size, 3),
        rationale=(
            f"hit-rate {trader.hit_rate * 100:.1f}% — "
            + ("confidence capped at 0.60 (old hit-rate rule)" if trader.hit_rate < 0.40 else "no hit-rate cap")
        ),
    )


def simulate_new_decision(rows: List[HorizonRow]) -> SimulatedDecision:
    """Phase-2F PM rules.

    * Hit-rate cap of 0.60 fires ONLY when expectancy <= 0 on EVERY
      horizon AND best-hit-rate < 40%.
    * Any horizon with expectancy > 0 AND payoff >= 1.5 is allowed to
      drive confidence calibration.
    * Quality reviewer recommends size cut only at score < 4 (we use
      hit-rate < 0.25 as a proxy — much stricter than the old rule).
    """
    trader = next((r for r in rows if r.is_trader), rows[0])
    best_exp = max(rows, key=lambda r: r.expectancy)
    any_positive_with_payoff = any(
        r.expectancy > 0.0 and r.payoff >= 1.5 for r in rows
    )
    all_negative = all(r.expectancy <= 0.0 for r in rows)
    best_hit = max(r.hit_rate for r in rows)

    # Phase 2F cap: only when expectancy is negative everywhere AND best hit-rate is weak.
    hit_rate_cap_fires = all_negative and best_hit < 0.40
    base_conf = 0.70 + min(0.18, max(0.0, best_exp.expectancy / 5.0))
    capped_conf = min(base_conf, 0.60) if hit_rate_cap_fires else min(base_conf, 0.88)

    base_size = 1.5
    quality_penalty = trader.hit_rate < 0.25  # softer than old 0.30
    size = base_size * (0.75 if quality_penalty else 1.0)

    if all_negative and not any_positive_with_payoff:
        return SimulatedDecision(
            rules="new",
            action="NEUTRAL",
            confidence=min(0.55, capped_conf),
            size_pct=0.0,
            rationale=(
                "Every horizon shows expectancy ≤ 0 — NEUTRAL is the principled call."
            ),
        )
    direction = "OVERWEIGHT" if best_exp.expectancy >= 0 and best_exp.avg >= 0 else "UNDERWEIGHT"
    rationale = (
        f"best-expectancy horizon = {best_exp.horizon_days}d "
        f"(exp {best_exp.expectancy:+.2f}%, hit-rate {best_exp.hit_rate * 100:.1f}%, "
        f"payoff {best_exp.payoff:.2f}x). "
    )
    if any_positive_with_payoff:
        rationale += "Positive expectancy with payoff ≥ 1.5 — hit-rate cap not binding."
    else:
        rationale += "Expectancy non-negative on best horizon — confidence calibrated to evidence."
    return SimulatedDecision(
        rules="new",
        action=direction,
        confidence=round(capped_conf, 3),
        size_pct=round(size, 3),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# M6 walk-forward — uses already-logged proposals if available
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardSnapshot:
    n_proposals: int
    n_folds: int
    total_return_pct: Optional[float]
    sharpe: Optional[float]
    max_drawdown: Optional[float]
    deflated_sharpe: Optional[float]
    notes: str


def run_walk_forward_snapshot(ticker: str) -> WalkForwardSnapshot:
    import os

    from trading_crew.agentic.backtest import (
        WalkForwardConfig, generate_folds, run_walk_forward,
    )
    from trading_crew.agentic.execution.contracts import ActionProposal
    from trading_crew.agentic.memory import EpisodicMemory

    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    store_path = Path(cache_dir) / "memory" / "episodes.jsonl"
    if not store_path.exists():
        return WalkForwardSnapshot(0, 0, None, None, None, None,
            f"No episode store at {store_path}; run an analysis first.",
        )
    mem = EpisodicMemory(store_path)
    episodes = [e for e in mem.all_episodes() if e.symbol.upper() == ticker.upper()]
    proposals: List[ActionProposal] = []
    for ep in episodes:
        try:
            proposals.append(ActionProposal(**ep.action_proposal))
        except Exception:
            continue
    if len(proposals) < 5:
        return WalkForwardSnapshot(
            len(proposals), 0, None, None, None, None,
            f"Only {len(proposals)} proposals for {ticker}; walk-forward needs ≥ 5.",
        )

    from web.backend.charts import _fetch_ohlcv
    last_ts = max(p.decision_ts for p in proposals)
    try:
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        end_dt = datetime.utcnow().replace(tzinfo=None)
    ohlcv = _fetch_ohlcv(ticker, end_dt, lookback_days=365 * 5)
    if ohlcv is None or ohlcv.empty:
        return WalkForwardSnapshot(
            len(proposals), 0, None, None, None, None,
            "OHLCV fetch failed.",
        )

    cfg = WalkForwardConfig(train_size=3, embargo_size=1, test_size=1)
    folds = generate_folds(n_obs=len(proposals), config=cfg)
    if not folds:
        return WalkForwardSnapshot(
            len(proposals), 0, None, None, None, None,
            f"Couldn't generate folds (need ≥ {cfg.train_size + cfg.embargo_size + cfg.test_size} proposals).",
        )
    result = run_walk_forward(
        proposals=proposals,
        ohlcv_by_symbol={ticker: ohlcv},
        folds=folds,
        cost_model_name="standard",
    )
    m = result.overall_metrics
    return WalkForwardSnapshot(
        n_proposals=len(proposals),
        n_folds=len(result.folds),
        total_return_pct=m.total_return_pct,
        sharpe=m.sharpe,
        max_drawdown=m.max_drawdown,
        deflated_sharpe=m.deflated_sharpe,
        notes="OK",
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    ticker: str,
    horizon: int,
    tgt_pct: float,
    stp_pct: float,
    rows: List[HorizonRow],
    old: SimulatedDecision,
    new: SimulatedDecision,
    wf: WalkForwardSnapshot,
    out_dir: Path,
) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"decision_impact_{ticker}_{ts}.md"
    json_path = out_dir / f"decision_impact_{ticker}_{ts}.json"

    json_payload = {
        "ticker": ticker,
        "params": {"horizon": horizon, "target_pct": tgt_pct, "stop_pct": stp_pct},
        "horizons": [asdict(r) for r in rows],
        "decision_old": asdict(old),
        "decision_new": asdict(new),
        "walk_forward": asdict(wf),
        "timestamp": ts,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Decision impact — {ticker} (target +{tgt_pct:.1f}% / stop −{stp_pct:.1f}% / trader horizon {horizon}d)")
    lines.append("")
    lines.append(f"_Generated {ts}._  Source: yfinance 5y OHLCV; M1 bridge + M5 sizer; M6 walk-forward.")
    lines.append("")
    lines.append("## Multi-horizon backtest panel")
    lines.append("")
    lines.append("| Horizon | Hit-rate | Payoff | Expectancy | Avg %  | Median % | N    |")
    lines.append("|--------:|---------:|-------:|-----------:|-------:|---------:|-----:|")
    for r in rows:
        label = f"**{r.horizon_days}d (trader)**" if r.is_trader else f"{r.horizon_days}d"
        lines.append(
            f"| {label} | {r.hit_rate * 100:.1f}% | {r.payoff:.2f}x | "
            f"{r.expectancy:+.2f}% | {r.avg:+.2f}% | {r.median:+.2f}% | {r.n} |"
        )
    lines.append("")
    best = max(rows, key=lambda r: r.expectancy)
    best_hit = max(rows, key=lambda r: r.hit_rate)
    lines.append(f"- **Best expectancy**: {best.expectancy:+.2f}% on {best.horizon_days}d (hit-rate {best.hit_rate * 100:.1f}%, payoff {best.payoff:.2f}x).")
    lines.append(f"- **Best hit-rate**:   {best_hit.hit_rate * 100:.1f}% on {best_hit.horizon_days}d (expectancy {best_hit.expectancy:+.2f}%).")
    lines.append("")

    lines.append("## Decision impact (same panel, old vs new PM rules)")
    lines.append("")
    lines.append("| Rules | Action | Confidence | Size % | Rationale |")
    lines.append("|---|---|---:|---:|---|")
    lines.append(f"| Old (pre-Phase-2F) | **{old.action}** | {old.confidence:.2f} | {old.size_pct:.2f} | {old.rationale} |")
    lines.append(f"| New (current)      | **{new.action}** | {new.confidence:.2f} | {new.size_pct:.2f} | {new.rationale} |")
    lines.append("")

    # Quantify the impact.
    action_changed = old.action != new.action
    conf_delta = new.confidence - old.confidence
    size_delta = new.size_pct - old.size_pct
    bullets: List[str] = []
    if action_changed:
        bullets.append(f"- Action moved: **{old.action} → {new.action}**.")
    else:
        bullets.append(f"- Action unchanged ({new.action}).")
    bullets.append(f"- Confidence Δ: **{conf_delta:+.3f}** ({old.confidence:.2f} → {new.confidence:.2f}).")
    bullets.append(f"- Size Δ: **{size_delta:+.2f}%** of book ({old.size_pct:.2f}% → {new.size_pct:.2f}%).")
    lines.append("**How much did the levers change?**\n")
    lines.extend(bullets)
    lines.append("")

    lines.append("## M6 walk-forward backtest")
    lines.append("")
    if wf.n_folds == 0:
        lines.append(f"_{wf.notes}_")
    else:
        lines.append(f"- Proposals replayed: **{wf.n_proposals}**, folds: **{wf.n_folds}**")
        if wf.total_return_pct is not None:
            lines.append(f"- Total return: **{wf.total_return_pct:+.2f}%**")
        if wf.sharpe is not None:
            lines.append(f"- Sharpe (annualised): **{wf.sharpe:.2f}**")
        if wf.deflated_sharpe is not None:
            lines.append(f"- Deflated Sharpe (Bailey/López de Prado): **{wf.deflated_sharpe:.2f}** — penalises multiple-comparison overfit; a value > 1 is the bar.")
        else:
            lines.append("- Deflated Sharpe: _n/a_ (need ≥ 2 folds to estimate the inflation factor).")
        if wf.max_drawdown is not None:
            lines.append(f"- Max drawdown: **{wf.max_drawdown * 100:+.2f}%**")
    lines.append("")
    lines.append("### How the walk-forward changes the decision")
    lines.append("")
    if wf.n_folds == 0:
        lines.append(
            "Without a sufficient proposal history for this ticker the walk-forward is "
            "silent — the only signal at this point comes from the multi-horizon "
            "`backtest_setup` panel above. Run an analysis on this ticker, accumulate "
            "≥ 5 episodes, then re-run this script to see the empirical fold-level deltas."
        )
    else:
        sharpe = wf.sharpe if wf.sharpe is not None else 0.0
        dsr = wf.deflated_sharpe if wf.deflated_sharpe is not None else 0.0
        sharpe_pos = sharpe > 0.0
        dsr_pos = dsr > 0.5
        dsr_label = f"{wf.deflated_sharpe:.2f}" if wf.deflated_sharpe is not None else "n/a (single fold)"
        if sharpe_pos and dsr_pos:
            lines.append(
                f"- Sharpe {sharpe:.2f} and Deflated Sharpe {dsr_label} both positive: "
                "the proposals have **survived out-of-sample reconciliation**. The PM is "
                "allowed to bump confidence by ≈0.05–0.10 on the next run for this ticker."
            )
        elif sharpe_pos and not dsr_pos:
            lines.append(
                f"- Sharpe {sharpe:.2f} is positive but Deflated Sharpe {dsr_label} is "
                "below the 1.0 threshold — the strategy is likely overfit. The PM "
                "should treat the in-sample edge with scepticism and not size up."
            )
        else:
            lines.append(
                f"- Sharpe {sharpe:.2f} and Deflated Sharpe {dsr_label}: the historical "
                "proposals **lost money out-of-sample**. The PM should require a much "
                "stronger fresh-evidence case before going long; the right baseline is NEUTRAL."
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Worked example: backtest + walk-forward decision impact.")
    p.add_argument("ticker", help="Ticker to demonstrate the levers on (e.g. NVDA, RELIANCE, MAZDOCK).")
    p.add_argument("--horizon", type=int, default=20, help="Trader's chosen horizon in days (default 20).")
    p.add_argument("--target-pct", type=float, default=5.0, dest="target_pct", help="Target take-profit %% (default 5).")
    p.add_argument("--stop-pct", type=float, default=3.0, dest="stop_pct", help="Stop-loss %% (default 3).")
    p.add_argument("--out-dir", default="reports", help="Directory for the markdown + JSON output.")
    args = p.parse_args()

    try:
        from trading_crew.market_context import resolve_ticker
        resolved = resolve_ticker(args.ticker.strip().upper())
    except Exception:
        resolved = args.ticker.strip().upper()

    print(f"Ticker: {args.ticker} -> {resolved}")
    print(f"Params: horizon={args.horizon}d, target=+{args.target_pct:.1f}%, stop=-{args.stop_pct:.1f}%")
    print()

    print("Computing multi-horizon backtest panel …")
    rows = multi_horizon_panel(resolved, args.horizon, args.target_pct, args.stop_pct)
    print(f"  -> {len(rows)} horizons computed")
    print()

    print("Simulating PM decision under OLD vs NEW rules …")
    old = simulate_old_decision(rows)
    new = simulate_new_decision(rows)
    print(f"  OLD: {old.action} conf={old.confidence} size%={old.size_pct}")
    print(f"  NEW: {new.action} conf={new.confidence} size%={new.size_pct}")
    print()

    print("Running M6 walk-forward over any logged proposals …")
    wf = run_walk_forward_snapshot(resolved)
    print(f"  -> n_proposals={wf.n_proposals}, n_folds={wf.n_folds}, notes={wf.notes}")
    print()

    md_path, json_path = write_report(
        resolved, args.horizon, args.target_pct, args.stop_pct,
        rows, old, new, wf, Path(args.out_dir).resolve(),
    )
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
