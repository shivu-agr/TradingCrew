"""Multi-asset portfolio backtest tests (run_portfolio_backtest + M7 allocator)."""

from __future__ import annotations

import random

import pandas as pd

from trading_crew.agentic.backtest import run_portfolio_backtest
from trading_crew.agentic.portfolio.allocator import (
    AllocationMethod,
    AllocatorConfig,
)
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


def _make_proposal(symbol: str, ts: str, weight: float = 0.08) -> ActionProposal:
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol=symbol, decision_ts=ts, side=side, target_weight=weight,
        horizon_days=10, conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.05 if weight != 0 else 0.0,
        rationale="Portfolio test proposal with enough text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _make_ohlcv(seed: int, drift: float, n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(drift, 0.012)))
    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low": [c * 0.995 for c in closes],
        "Close": closes,
        "Volume": [2_000_000] * n,
    })


def _frames():
    return {
        "AAA": _make_ohlcv(1, 0.0006),
        "BBB": _make_ohlcv(2, 0.0004),
        "CCC": _make_ohlcv(3, 0.0005),
    }


def _proposals(frames):
    dates = frames["AAA"]["Date"]
    props = []
    for i in range(6):
        ts = dates.iloc[100 + i * 30].strftime("%Y-%m-%dT16:00:00+00:00")
        for sym in frames:
            props.append(_make_proposal(sym, ts, 0.08))
    return props


def test_portfolio_backtest_runs_end_to_end():
    frames = _frames()
    props = _proposals(frames)
    res = run_portfolio_backtest(
        proposals=props, ohlcv_by_symbol=frames,
        allocator_config=AllocatorConfig(method=AllocationMethod.HRP),
    )
    # One equity point per rebalance date (+ starting point).
    n_dates = len({p.decision_ts for p in props})
    assert len(res.equity_curve) == n_dates + 1
    assert len(res.weights_over_time) == n_dates
    assert res.metrics.n_periods >= 1


def test_portfolio_backtest_respects_gross_budget():
    frames = _frames()
    props = _proposals(frames)
    res = run_portfolio_backtest(
        proposals=props, ohlcv_by_symbol=frames,
        allocator_config=AllocatorConfig(
            method=AllocationMethod.EQUAL_RISK, gross_budget=0.6,
        ),
    )
    for weights in res.weights_over_time:
        gross = sum(abs(w) for w in weights.values())
        assert gross <= 0.6 + 1e-6


def test_portfolio_backtest_is_deterministic():
    frames = _frames()
    props = _proposals(frames)
    kw = dict(proposals=props, ohlcv_by_symbol=frames,
              allocator_config=AllocatorConfig(method=AllocationMethod.HRP))
    r1 = run_portfolio_backtest(**kw)
    r2 = run_portfolio_backtest(**kw)
    assert r1.equity_curve == r2.equity_curve
    assert r1.weights_over_time == r2.weights_over_time


def test_portfolio_backtest_has_equal_weight_benchmark():
    frames = _frames()
    props = _proposals(frames)
    res = run_portfolio_backtest(proposals=props, ohlcv_by_symbol=frames)
    assert len(res.benchmark_curve) == len(res.equity_curve)
    # benchmark starts at the same starting cash.
    assert res.benchmark_curve[0] == res.equity_curve[0]
