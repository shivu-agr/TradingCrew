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
from trading_crew.agentic.portfolio.allocator import (
    AllocationMethod,
    AllocatorConfig,
    allocate,
)
from trading_crew.agentic.portfolio.state import PortfolioState
from trading_crew.agentic.risk import (
    GateConfig,
    SizingConfig,
    compute_size,
    run_risk_gates,
)

from .metrics import BacktestMetrics, compute_metrics
from .walk_forward import CPCVFold, Fold

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
    benchmark_curve: List[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Top-level result aggregated across folds."""

    folds: List[FoldResult]
    overall_metrics: BacktestMetrics
    combined_equity: List[float]
    combined_timestamps: List[str]
    combined_benchmark: List[float] = field(default_factory=list)


@dataclass
class CPCVPathResult:
    """One combinatorial test path's result (a union of held-out groups)."""

    fold_id: int
    equity_curve: List[float]
    timestamps: List[str]
    metrics: BacktestMetrics
    trades: List[TradeRecord] = field(default_factory=list)
    benchmark_curve: List[float] = field(default_factory=list)


@dataclass
class PortfolioBacktestResult:
    """Multi-asset (portfolio) backtest result.

    Unlike the single-name replay, decisions across symbols are combined
    at each rebalance date by the M7 allocator (HRP / MV / ERC) into one
    coherent weight vector, then executed against a *single shared*
    ``PortfolioState`` — so correlated bets can't quietly blow out gross
    exposure (paper §9.1).
    """

    equity_curve: List[float]
    timestamps: List[str]
    metrics: BacktestMetrics
    trades: List[TradeRecord] = field(default_factory=list)
    weights_over_time: List[Dict[str, float]] = field(default_factory=list)
    benchmark_curve: List[float] = field(default_factory=list)
    method_used: str = ""


@dataclass
class CPCVResult:
    """Combinatorial Purged Cross-Validation result for a single config.

    Unlike walk-forward (one contiguous OOS path), CPCV produces
    ``C(n_groups, k_test)`` distinct out-of-sample paths.  The headline
    output is therefore a *distribution* of OOS Sharpes, which is far
    harder to overfit than a single number (paper §8.1 / López de Prado
    AFML ch. 12).
    """

    paths: List[CPCVPathResult]
    oos_sharpes: List[float]
    mean_oos_sharpe: float
    median_oos_sharpe: float
    sharpe_std: float
    aggregate_metrics: BacktestMetrics
    combined_equity: List[float]
    combined_timestamps: List[str]
    combined_benchmark: List[float] = field(default_factory=list)


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


def _pick_benchmark_symbol(proposals: Sequence[ActionProposal]) -> Optional[str]:
    """The symbol that appears in the most proposals (ties: first seen).

    Used as the buy-and-hold reference so the equity curve can be scored
    for *alpha* — did the strategy beat simply holding the underlying?
    """
    counts: Dict[str, int] = {}
    order: List[str] = []
    for p in proposals:
        if p.symbol not in counts:
            order.append(p.symbol)
        counts[p.symbol] = counts.get(p.symbol, 0) + 1
    if not order:
        return None
    return max(order, key=lambda s: counts[s])


def _benchmark_curve(
    timestamps: Sequence[str],
    frames: Dict[str, pd.DataFrame],
    benchmark_symbol: Optional[str],
    starting_cash: float,
) -> List[float]:
    """Buy-and-hold equity curve for ``benchmark_symbol`` aligned to
    ``timestamps`` (the same time index the strategy equity curve uses).

    Each point is ``starting_cash * close_t / close_0`` where ``close_t``
    is the last close at or before the proposal timestamp.  Missing data
    carries the previous value forward (no invented prices).
    """
    curve: List[float] = [starting_cash]
    df = frames.get(benchmark_symbol) if benchmark_symbol else None
    if df is None or df.empty:
        curve.extend([starting_cash] * len(timestamps))
        return curve
    date_col = "Date" if "Date" in df.columns else "date"
    base: Optional[float] = None
    last = starting_cash
    for ts in timestamps:
        d = _to_naive_utc(ts)
        past = df[df[date_col] <= d]
        if not past.empty:
            close = float(past.iloc[-1]["Close"])
            if base is None and close > 0:
                base = close
            if base and base > 0:
                last = starting_cash * close / base
        curve.append(last)
    return curve


def run_backtest(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    cost_model_name: str = "standard",
    sizing_config: Optional[SizingConfig] = None,
    gate_config: Optional[GateConfig] = None,
    fold: Optional[Fold] = None,
    bootstrap: bool = False,
    allow_fractional_shares: bool = True,
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
        order = proposal_to_order(
            sized_proposal, state, ref_close, risk_mult=1.0,
            allow_fractional=allow_fractional_shares,
        )
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

    benchmark_symbol = _pick_benchmark_symbol(list(proposals))
    benchmark = _benchmark_curve(timestamps, frames, benchmark_symbol, starting_cash)
    metrics = compute_metrics(
        equity_curve,
        periods_per_year=252,
        benchmark_equity=benchmark,
        bootstrap=bootstrap,
    )
    return FoldResult(
        fold=fold if fold else Fold(0, 0, 0, 0, 0, 0, len(proposals)),
        equity_curve=equity_curve,
        timestamps=timestamps,
        metrics=metrics,
        trades=trades,
        benchmark_curve=benchmark,
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
    combined_bench: List[float] = [starting_cash]
    last_nav = starting_cash
    last_bench = starting_cash

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
        # Chain the benchmark the same way — re-base each fold's benchmark
        # so it starts where the previous fold's benchmark ended.
        if result.benchmark_curve:
            bench_base = result.benchmark_curve[0] or last_bench
            for b in result.benchmark_curve[1:]:
                combined_bench.append(last_bench * (b / bench_base) if bench_base else last_bench)
            last_bench = combined_bench[-1]

    overall = compute_metrics(
        combined,
        periods_per_year=252,
        n_trials=max(1, len(folds)),
        benchmark_equity=combined_bench if len(combined_bench) == len(combined) else None,
        bootstrap=True,
    )
    return BacktestResult(
        folds=fold_results,
        overall_metrics=overall,
        combined_equity=combined,
        combined_timestamps=combined_ts,
        combined_benchmark=combined_bench,
    )


def run_cpcv(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    cpcv_folds: Sequence[CPCVFold],
    starting_cash: float = 100_000.0,
    **kwargs,
) -> CPCVResult:
    """Run a Combinatorial Purged Cross-Validation backtest.

    Proposals are treated as an ordered timeline (sorted by
    ``decision_ts``); each :class:`CPCVFold`'s ``test_indices`` selects
    the positions that form that path's out-of-sample block.  Every path
    runs against a *fresh* portfolio (they are independent held-out
    samples, not a chained live account), so the OOS Sharpe of each path
    is directly comparable.

    Returns the full distribution of OOS Sharpes plus an aggregate metric
    bundle computed over the concatenation of every path's returns.  The
    caller (grid / swarm search) can feed the per-config OOS return
    series into :func:`probability_of_backtest_overfitting`.
    """
    sorted_proposals = sorted(proposals, key=lambda p: p.decision_ts)

    paths: List[CPCVPathResult] = []
    oos_sharpes: List[float] = []
    combined: List[float] = [starting_cash]
    combined_ts: List[str] = []
    combined_bench: List[float] = [starting_cash]
    last_nav = starting_cash
    last_bench = starting_cash

    for fold in cpcv_folds:
        idxs = [i for i in fold.test_indices if i < len(sorted_proposals)]
        fold_props = [sorted_proposals[i] for i in idxs]
        if not fold_props:
            continue
        res = run_backtest(
            proposals=fold_props,
            ohlcv_by_symbol=ohlcv_by_symbol,
            starting_cash=starting_cash,
            fold=None,
            **kwargs,
        )
        paths.append(CPCVPathResult(
            fold_id=fold.fold_id,
            equity_curve=res.equity_curve,
            timestamps=res.timestamps,
            metrics=res.metrics,
            trades=res.trades,
            benchmark_curve=res.benchmark_curve,
        ))
        oos_sharpes.append(res.metrics.sharpe)
        combined.extend(res.equity_curve[1:])
        combined_ts.extend(res.timestamps)
        if res.benchmark_curve:
            bench_base = res.benchmark_curve[0] or last_bench
            for b in res.benchmark_curve[1:]:
                combined_bench.append(last_bench * (b / bench_base) if bench_base else last_bench)
            last_bench = combined_bench[-1]
        last_nav = combined[-1]

    n = len(oos_sharpes)
    mean_s = sum(oos_sharpes) / n if n else 0.0
    if n:
        srt = sorted(oos_sharpes)
        median_s = srt[n // 2] if n % 2 == 1 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
        var_s = sum((s - mean_s) ** 2 for s in oos_sharpes) / max(1, n - 1)
        std_s = var_s ** 0.5
    else:
        median_s = 0.0
        std_s = 0.0

    aggregate = compute_metrics(
        combined,
        periods_per_year=252,
        n_trials=max(1, len(cpcv_folds)),
        benchmark_equity=combined_bench if len(combined_bench) == len(combined) else None,
        bootstrap=True,
    )
    return CPCVResult(
        paths=paths,
        oos_sharpes=oos_sharpes,
        mean_oos_sharpe=mean_s,
        median_oos_sharpe=median_s,
        sharpe_std=std_s,
        aggregate_metrics=aggregate,
        combined_equity=combined,
        combined_timestamps=combined_ts,
        combined_benchmark=combined_bench,
    )


# ---------------------------------------------------------------------------
# Multi-asset portfolio backtest (M7 allocator inside the replay)
# ---------------------------------------------------------------------------


def _log_returns(closes: Sequence[float], lookback: int = 252) -> List[float]:
    """Trailing log returns of ``closes`` (last ``lookback`` observations)."""
    import math

    tail = list(closes)[-(lookback + 1):]
    out: List[float] = []
    for i in range(1, len(tail)):
        if tail[i - 1] > 0 and tail[i] > 0:
            out.append(math.log(tail[i] / tail[i - 1]))
    return out


def _estimate_vol_cvar(prior_closes: Sequence[float]) -> tuple[float, float]:
    """Annualised realised vol + 1-day CVaR(95%) from prior closes.

    Mirrors the estimator used in :func:`run_backtest` so the portfolio
    path sizes / gates positions on the same footing as the single-name
    path.  Falls back to (0.20, 0.02) when there is too little history.
    """
    import math

    log_rets = _log_returns(prior_closes, lookback=63)
    if len(log_rets) < 20:
        return 0.20, 0.02
    mu = sum(log_rets) / len(log_rets)
    var = sum((r - mu) ** 2 for r in log_rets) / max(1, len(log_rets) - 1)
    realised_vol = math.sqrt(var) * math.sqrt(252)
    srt = sorted(log_rets)
    tail = srt[: max(1, int(len(srt) * 0.05))]
    cvar_1d = abs(sum(tail) / len(tail)) if tail else 0.02
    return realised_vol, cvar_1d


def run_portfolio_backtest(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    starting_cash: float = 100_000.0,
    cost_model_name: str = "standard",
    gate_config: Optional[GateConfig] = None,
    allocator_config: Optional[AllocatorConfig] = None,
    allow_fractional_shares: bool = True,
) -> PortfolioBacktestResult:
    """Replay ``proposals`` as a *coherent multi-asset book*.

    At every rebalance date (a distinct ``decision_ts``) all proposals
    dated that day are handed to the M7 :func:`allocate` function, which
    produces one weight vector respecting the gross-risk budget and
    per-name caps.  Each name's allocated weight is then executed through
    the M5 risk gates + M2 simulator against a *single shared*
    ``PortfolioState``.

    The allocator performs the portfolio-level risk budgeting (HRP / MV /
    ERC + conviction scaling), so the single-name Kelly sizer is *not*
    re-applied here — doing so would double-count the risk budget.  The
    benchmark is an equal-weight buy-and-hold of every symbol.
    """
    gate_config = gate_config or GateConfig()
    allocator_config = allocator_config or AllocatorConfig()
    cost_model = get_cost_model(cost_model_name)
    sim = ExecutionSimulator(cost_model=cost_model)

    state = PortfolioState(
        portfolio_id="portfolio_backtest",
        base_currency="USD",
        starting_cash=starting_cash,
        cash=starting_cash,
        peak_nav=starting_cash,
    )

    frames = {sym: _normalise_ohlcv(df) for sym, df in ohlcv_by_symbol.items()}

    def _date_col(df: pd.DataFrame) -> str:
        return "Date" if "Date" in df.columns else "date"

    def _last_close_on_or_before(sym: str, d: pd.Timestamp) -> Optional[float]:
        df = frames.get(sym)
        if df is None or df.empty:
            return None
        col = _date_col(df)
        past = df[df[col] <= d]
        if past.empty:
            return None
        return float(past.iloc[-1]["Close"])

    # Group proposals by rebalance date (latest proposal per symbol wins).
    from collections import OrderedDict

    by_date: "OrderedDict[str, Dict[str, ActionProposal]]" = OrderedDict()
    for p in sorted(proposals, key=lambda x: x.decision_ts):
        by_date.setdefault(p.decision_ts, {})[p.symbol] = p

    equity_curve: List[float] = [starting_cash]
    timestamps: List[str] = []
    trades: List[TradeRecord] = []
    weights_over_time: List[Dict[str, float]] = []
    method_used = allocator_config.method.value

    for decision_ts, sym_props in by_date.items():
        d = _to_naive_utc(decision_ts)

        # Mark the whole book to the latest close on/before this date.
        price_map: Dict[str, float] = {}
        for sym in state.positions:
            c = _last_close_on_or_before(sym, d)
            if c is not None:
                price_map[sym] = c
        if price_map:
            state.mark_to_market(price_map, ts=decision_ts)

        day_proposals = list(sym_props.values())
        returns_by_symbol: Dict[str, List[float]] = {}
        for sym in sym_props:
            df = frames.get(sym)
            if df is None or df.empty:
                returns_by_symbol[sym] = []
                continue
            col = _date_col(df)
            closes = df[df[col] <= d]["Close"].astype(float).tolist()
            returns_by_symbol[sym] = _log_returns(closes, lookback=252)

        allocation = allocate(day_proposals, returns_by_symbol, allocator_config)
        method_used = allocation.method_used.value
        weights_over_time.append(dict(allocation.weights))

        for sym, target_w in allocation.weights.items():
            proposal = sym_props[sym]
            df = frames.get(sym)
            if df is None or df.empty:
                continue
            col = _date_col(df)
            ref_close = _last_close_on_or_before(sym, d)
            future = df[df[col] > d]
            if ref_close is None or future.empty:
                continue
            next_row = future.iloc[0]
            next_date = pd.to_datetime(next_row[col])
            adv = float(df["Volume"].tail(20).mean())
            bar = _bar_from_row(next_date, next_row, adv)

            if abs(target_w) < 1e-9 and sym not in state.positions:
                continue

            sized_proposal = proposal.model_copy(update={"target_weight": target_w})
            _, cvar_1d = _estimate_vol_cvar(
                df[df[col] <= d]["Close"].astype(float).tolist()
            )
            est_fees = abs(target_w * state.nav) * 0.0001
            gate = run_risk_gates(
                sized_proposal, state,
                cvar_one_day=cvar_1d,
                reference_price=ref_close,
                est_fees=est_fees,
                last_bar_ts=bar.ts,
                config=gate_config,
            )
            if not gate.passed:
                trades.append(TradeRecord(
                    fold_id=-1, ts=decision_ts, symbol=sym,
                    side=proposal.side.value, intent_weight=proposal.target_weight,
                    sized_weight=target_w, fill_qty=0.0, fill_price=0.0, fees=0.0,
                    status="RISK_REJECTED",
                    rejection_reason="; ".join(f"{nm}: {r}" for nm, r in gate.failures),
                ))
                continue
            order = proposal_to_order(
                sized_proposal, state, ref_close, risk_mult=1.0,
                allow_fractional=allow_fractional_shares,
            )
            if order is None:
                continue
            fill = sim.execute(order, bar, state)
            fees_total = sum(fill.cost_breakdown.values()) if fill.cost_breakdown else 0.0
            trades.append(TradeRecord(
                fold_id=-1, ts=decision_ts, symbol=sym,
                side=proposal.side.value, intent_weight=proposal.target_weight,
                sized_weight=target_w, fill_qty=fill.qty_filled,
                fill_price=fill.avg_price or 0.0, fees=fees_total,
                status=fill.status.value,
                rejection_reason=fill.reason if fill.status != FillStatus.FILLED else "",
            ))

        equity_curve.append(state.nav)
        timestamps.append(decision_ts)

    benchmark = _equal_weight_benchmark(timestamps, frames, list(frames.keys()), starting_cash)
    metrics = compute_metrics(
        equity_curve,
        periods_per_year=252,
        benchmark_equity=benchmark,
        bootstrap=True,
    )
    return PortfolioBacktestResult(
        equity_curve=equity_curve,
        timestamps=timestamps,
        metrics=metrics,
        trades=trades,
        weights_over_time=weights_over_time,
        benchmark_curve=benchmark,
        method_used=method_used,
    )


def _equal_weight_benchmark(
    timestamps: Sequence[str],
    frames: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    starting_cash: float,
) -> List[float]:
    """Equal-weight buy-and-hold of ``symbols`` aligned to ``timestamps``.

    Value at each date is ``starting_cash`` times the average of each
    symbol's ``close_t / close_0``.  Symbols with no price yet are simply
    omitted from that date's average (never invented).
    """
    curve: List[float] = [starting_cash]
    bases: Dict[str, float] = {}
    last = starting_cash
    for ts in timestamps:
        d = _to_naive_utc(ts)
        ratios: List[float] = []
        for sym in symbols:
            df = frames.get(sym)
            if df is None or df.empty:
                continue
            col = "Date" if "Date" in df.columns else "date"
            past = df[df[col] <= d]
            if past.empty:
                continue
            close = float(past.iloc[-1]["Close"])
            if sym not in bases and close > 0:
                bases[sym] = close
            base = bases.get(sym)
            if base and base > 0:
                ratios.append(close / base)
        if ratios:
            last = starting_cash * (sum(ratios) / len(ratios))
        curve.append(last)
    return curve
