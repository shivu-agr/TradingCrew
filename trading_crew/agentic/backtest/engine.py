"""Walk-forward backtest engine.

The engine consumes a list of pre-recorded ``ActionProposal`` decisions
(one per (ticker, decision_ts)) and replays them through the M2
execution pipeline + M5 risk gates against a fresh ``PortfolioState``.
The output is an equity curve, a per-trade log, and ``BacktestMetrics``.

We *do not* call the LLM during a backtest.  Calling the agents on
historical data is expensive and non-deterministic; instead the user is
expected to log proposals during live / paper runs and use the
backtester to evaluate parameter changes (cost params, risk gate
thresholds, sizing config) over the same set of proposals.  This is the
"replay-and-perturb" loop the paper §8.4 recommends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from trading_crew.agentic.execution.contracts import ActionProposal
from trading_crew.agentic.execution.cost import get_cost_model
from trading_crew.agentic.execution.simulator import (
    Bar,
    ExecutionSimulator,
    FillStatus,
    proposal_to_order,
)
from trading_crew.agentic.portfolio.state import PortfolioState
from trading_crew.agentic.risk import (
    GateConfig,
    SizingConfig,
    compute_size,
    run_risk_gates,
)

from .metrics import BacktestMetrics, compute_metrics
from .walk_forward import Fold

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One executed (or rejected) decision in the backtest log."""

    fold_id: int
    ts: str
    symbol: str
    side: str
    intent_weight: float
    sized_weight: float
    fill_qty: float
    fill_price: float
    fees: float
    status: str
    rejection_reason: str = ""


@dataclass
class FoldResult:
    """Per-fold backtest result for the UI."""

    fold: Fold
    equity_curve: List[float]
    timestamps: List[str]
    metrics: BacktestMetrics
    trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Top-level result aggregated across folds."""

    folds: List[FoldResult]
    overall_metrics: BacktestMetrics
    combined_equity: List[float]
    combined_timestamps: List[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _bar_from_row(date: pd.Timestamp, row: pd.Series, adv: float) -> Bar:
    return Bar(
        ts=date.isoformat(),
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
        adv=float(adv),
    )


def _normalise_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Sort + tag dates so we can index by trading day deterministically.

    Dates are normalised to *naive* UTC midnights so they compare cleanly
    against the proposal's ``decision_ts`` (which we also strip to naive
    UTC before comparing).  Keeping everything in UTC avoids the silent
    miscomparison the paper §8.1 warns about — a tz mismatch can shift a
    "decision at close" into the wrong trading day.
    """
    df = ohlcv.copy()
    date_col = "Date" if "Date" in df.columns else "date"
    dts = pd.to_datetime(df[date_col])
    if getattr(dts.dt, "tz", None) is not None:
        dts = dts.dt.tz_convert("UTC").dt.tz_localize(None)
    df[date_col] = dts.dt.normalize()
    return df.sort_values(date_col).reset_index(drop=True)


def _to_naive_utc(ts: str) -> pd.Timestamp:
    """Parse ``ts`` to a tz-naive UTC Timestamp normalised to midnight."""
    t = pd.to_datetime(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def run_backtest(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    cost_model_name: str = "standard",
    sizing_config: Optional[SizingConfig] = None,
    gate_config: Optional[GateConfig] = None,
    fold: Optional[Fold] = None,
) -> FoldResult:
    """Replay ``proposals`` through the M2/M5 pipeline against a fresh portfolio.

    Equity curve is recorded as NAV after every proposal (whether it
    executed or not), so the time index matches the proposal sequence.

    ``ohlcv_by_symbol`` must contain at least one bar after every
    proposal's ``decision_ts`` for that symbol — otherwise that proposal
    is logged as ``NO_DATA``.

    The ``fold`` argument is only used to tag the trades and the
    returned ``FoldResult``; it does not slice the proposals (the caller
    is responsible for filtering proposals to the fold's test window).
    """
    sizing_config = sizing_config or SizingConfig()
    gate_config = gate_config or GateConfig()
    cost_model = get_cost_model(cost_model_name)
    sim = ExecutionSimulator(cost_model=cost_model)

    state = PortfolioState(
        portfolio_id="backtest",
        base_currency="USD",
        starting_cash=starting_cash,
        cash=starting_cash,
        peak_nav=starting_cash,
    )

    equity_curve: List[float] = [starting_cash]
    timestamps: List[str] = []
    trades: List[TradeRecord] = []

    # Pre-normalise OHLCV frames once.
    frames = {sym: _normalise_ohlcv(df) for sym, df in ohlcv_by_symbol.items()}

    for proposal in sorted(proposals, key=lambda p: p.decision_ts):
        symbol = proposal.symbol
        df = frames.get(symbol)
        if df is None or df.empty:
            trades.append(
                TradeRecord(
                    fold_id=fold.fold_id if fold else -1,
                    ts=proposal.decision_ts, symbol=symbol,
                    side=proposal.side.value,
                    intent_weight=proposal.target_weight,
                    sized_weight=0.0, fill_qty=0.0, fill_price=0.0,
                    fees=0.0, status="NO_DATA",
                    rejection_reason=f"No OHLCV for {symbol}",
                )
            )
            equity_curve.append(state.nav)
            timestamps.append(proposal.decision_ts)
            continue

        decision_date = _to_naive_utc(proposal.decision_ts)
        date_col = "Date" if "Date" in df.columns else "date"
        future = df[df[date_col] > decision_date]

        # Mark-to-market on the decision day's close (or last close <= decision_date).
        past_or_present = df[df[date_col] <= decision_date]
        if not past_or_present.empty:
            ref_close = float(past_or_present.iloc[-1]["Close"])
            state.mark_to_market({symbol: ref_close}, ts=proposal.decision_ts)
        else:
            ref_close = float(future.iloc[0]["Close"]) if not future.empty else 0.0

        if future.empty:
            trades.append(
                TradeRecord(
                    fold_id=fold.fold_id if fold else -1,
                    ts=proposal.decision_ts, symbol=symbol,
                    side=proposal.side.value,
                    intent_weight=proposal.target_weight,
                    sized_weight=0.0, fill_qty=0.0, fill_price=0.0,
                    fees=0.0, status="NO_FUTURE_BAR",
                    rejection_reason="No bar after decision_ts",
                )
            )
            equity_curve.append(state.nav)
            timestamps.append(proposal.decision_ts)
            continue

        next_row = future.iloc[0]
        next_date = pd.to_datetime(next_row[date_col])
        adv = float(df["Volume"].tail(20).mean())
        bar = _bar_from_row(next_date, next_row, adv)

        # ---- Sizing ----------------------------------------------------
        # Estimate realised vol from prior 63 bars
        prior_closes = past_or_present["Close"].astype(float).tolist()
        if len(prior_closes) >= 30:
            import math
            log_rets = [
                math.log(prior_closes[i] / prior_closes[i - 1])
                for i in range(1, len(prior_closes))
                if prior_closes[i - 1] > 0
            ][-63:]
            mu = sum(log_rets) / len(log_rets) if log_rets else 0.0
            var = sum((r - mu) ** 2 for r in log_rets) / max(1, len(log_rets) - 1)
            realised_vol = math.sqrt(var) * math.sqrt(252)
            sorted_rets = sorted(log_rets)
            tail = sorted_rets[: max(1, int(len(sorted_rets) * 0.05))]
            cvar_1d = abs(sum(tail) / len(tail)) if tail else 0.02
        else:
            realised_vol = 0.20
            cvar_1d = 0.02

        sizing = compute_size(
            proposal,
            realised_vol_annualised=realised_vol,
            cvar_one_day=cvar_1d,
            config=sizing_config,
        )

        if abs(sizing.final_weight) < 1e-6:
            trades.append(
                TradeRecord(
                    fold_id=fold.fold_id if fold else -1,
                    ts=proposal.decision_ts, symbol=symbol,
                    side=proposal.side.value,
                    intent_weight=proposal.target_weight,
                    sized_weight=0.0, fill_qty=0.0, fill_price=0.0,
                    fees=0.0, status="SIZED_TO_ZERO",
                    rejection_reason=f"binding={sizing.binding_constraint}",
                )
            )
            equity_curve.append(state.nav)
            timestamps.append(proposal.decision_ts)
            continue

        sized_proposal = proposal.model_copy(update={"target_weight": sizing.final_weight})

        # ---- Risk gates ------------------------------------------------
        est_fees = abs(sizing.final_weight * state.nav) * 0.0001
        gate = run_risk_gates(
            sized_proposal, state,
            cvar_one_day=cvar_1d,
            reference_price=ref_close,
            est_fees=est_fees,
            last_bar_ts=bar.ts,
            config=gate_config,
        )
        if not gate.passed:
            trades.append(
                TradeRecord(
                    fold_id=fold.fold_id if fold else -1,
                    ts=proposal.decision_ts, symbol=symbol,
                    side=proposal.side.value,
                    intent_weight=proposal.target_weight,
                    sized_weight=sizing.final_weight,
                    fill_qty=0.0, fill_price=0.0, fees=0.0,
                    status="RISK_REJECTED",
                    rejection_reason="; ".join(f"{n}: {r}" for n, r in gate.failures),
                )
            )
            equity_curve.append(state.nav)
            timestamps.append(proposal.decision_ts)
            continue

        # ---- Order + fill ----------------------------------------------
        order = proposal_to_order(sized_proposal, state, ref_close, risk_mult=1.0)
        if order is None:
            trades.append(
                TradeRecord(
                    fold_id=fold.fold_id if fold else -1,
                    ts=proposal.decision_ts, symbol=symbol,
                    side=proposal.side.value,
                    intent_weight=proposal.target_weight,
                    sized_weight=sizing.final_weight,
                    fill_qty=0.0, fill_price=0.0, fees=0.0,
                    status="NO_DELTA",
                    rejection_reason="Sized weight matches current position",
                )
            )
            equity_curve.append(state.nav)
            timestamps.append(proposal.decision_ts)
            continue

        fill = sim.execute(order, bar, state)
        fees_total = sum(fill.cost_breakdown.values()) if fill.cost_breakdown else 0.0
        trades.append(
            TradeRecord(
                fold_id=fold.fold_id if fold else -1,
                ts=proposal.decision_ts, symbol=symbol,
                side=proposal.side.value,
                intent_weight=proposal.target_weight,
                sized_weight=sizing.final_weight,
                fill_qty=fill.qty_filled,
                fill_price=fill.avg_price or 0.0,
                fees=fees_total,
                status=fill.status.value,
                rejection_reason=fill.reason if fill.status != FillStatus.FILLED else "",
            )
        )

        # Mark-to-market on the fill bar's close so NAV reflects the new position
        state.mark_to_market({symbol: bar.close}, ts=bar.ts)
        equity_curve.append(state.nav)
        timestamps.append(proposal.decision_ts)

    metrics = compute_metrics(equity_curve, periods_per_year=252)
    return FoldResult(
        fold=fold if fold else Fold(0, 0, 0, 0, 0, 0, len(proposals)),
        equity_curve=equity_curve,
        timestamps=timestamps,
        metrics=metrics,
        trades=trades,
    )


def run_walk_forward(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    folds: Sequence[Fold],
    starting_cash: float = 100_000.0,
    **kwargs,
) -> BacktestResult:
    """Run a fold-by-fold walk-forward backtest.

    Proposals are assigned to folds by ``decision_ts``: the i-th proposal
    in chronological order goes into the fold whose ``test_indices``
    contains ``i``.  This makes the contract explicit — the proposals
    list is treated as an ordered timeline, and the fold indices are
    positions in that timeline.

    Each fold runs against a *fresh* portfolio so per-fold metrics
    aren't contaminated by previous folds; the ``combined_equity`` is a
    concatenation of fold equity curves (re-based to start where the
    previous fold ended) so the user can see the cumulative effect.
    """
    sorted_proposals = sorted(proposals, key=lambda p: p.decision_ts)

    fold_results: List[FoldResult] = []
    combined: List[float] = [starting_cash]
    combined_ts: List[str] = []
    last_nav = starting_cash

    for fold in folds:
        fold_proposals = [
            sorted_proposals[i]
            for i in fold.test_indices
            if i < len(sorted_proposals)
        ]
        if not fold_proposals:
            continue
        result = run_backtest(
            proposals=fold_proposals,
            ohlcv_by_symbol=ohlcv_by_symbol,
            starting_cash=last_nav,
            fold=fold,
            **kwargs,
        )
        fold_results.append(result)
        combined.extend(result.equity_curve[1:])  # drop the duplicate starting point
        combined_ts.extend(result.timestamps)
        last_nav = result.equity_curve[-1]

    overall = compute_metrics(combined, periods_per_year=252, n_trials=max(1, len(folds)))
    return BacktestResult(
        folds=fold_results,
        overall_metrics=overall,
        combined_equity=combined,
        combined_timestamps=combined_ts,
    )
