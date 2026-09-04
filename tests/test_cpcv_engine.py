"""CPCV backtest engine tests (run_cpcv over CPCVFold.test_ranges)."""

from __future__ import annotations

import random

import pandas as pd

from trading_crew.agentic.backtest import (
    CPCVConfig,
    generate_cpcv_folds,
    run_cpcv,
)
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


def _make_proposal(symbol: str, ts: str, weight: float = 0.05) -> ActionProposal:
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol=symbol, decision_ts=ts, side=side, target_weight=weight,
        horizon_days=5, conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.02 if weight != 0 else 0.0,
        rationale="CPCV test proposal with enough text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _make_ohlcv(start: str = "2023-01-02", n: int = 400, drift: float = 0.0005) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B")
    rng = random.Random(99)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(drift, 0.01)))
    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low": [c * 0.995 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    })


def _proposals(df: pd.DataFrame, n: int = 24):
    return [
        _make_proposal("AAA", df["Date"].iloc[80 + i * 10].strftime("%Y-%m-%dT16:00:00+00:00"),
                       weight=0.05 if i % 2 == 0 else 0.0)
        for i in range(n)
    ]


def test_run_cpcv_produces_one_path_per_fold():
    df = _make_ohlcv()
    props = _proposals(df, 24)
    folds = generate_cpcv_folds(24, CPCVConfig(n_groups=6, k_test=2, embargo_size=1))
    assert len(folds) == 15
    res = run_cpcv(proposals=props, ohlcv_by_symbol={"AAA": df}, cpcv_folds=folds)
    assert len(res.paths) == len(folds)
    assert len(res.oos_sharpes) == len(res.paths)


def test_run_cpcv_distribution_stats_are_consistent():
    df = _make_ohlcv()
    props = _proposals(df, 24)
    folds = generate_cpcv_folds(24, CPCVConfig(n_groups=6, k_test=2, embargo_size=1))
    res = run_cpcv(proposals=props, ohlcv_by_symbol={"AAA": df}, cpcv_folds=folds)
    # Mean/median lie within [min, max] of the OOS Sharpe distribution.
    assert min(res.oos_sharpes) - 1e-9 <= res.mean_oos_sharpe <= max(res.oos_sharpes) + 1e-9
    assert min(res.oos_sharpes) - 1e-9 <= res.median_oos_sharpe <= max(res.oos_sharpes) + 1e-9
    assert res.sharpe_std >= 0.0


def test_run_cpcv_is_deterministic():
    df = _make_ohlcv()
    props = _proposals(df, 24)
    folds = generate_cpcv_folds(24, CPCVConfig(n_groups=6, k_test=2, embargo_size=1))
    r1 = run_cpcv(proposals=props, ohlcv_by_symbol={"AAA": df}, cpcv_folds=folds)
    r2 = run_cpcv(proposals=props, ohlcv_by_symbol={"AAA": df}, cpcv_folds=folds)
    assert r1.oos_sharpes == r2.oos_sharpes
    assert r1.combined_equity == r2.combined_equity


def test_run_cpcv_aggregate_metrics_have_benchmark():
    df = _make_ohlcv()
    props = _proposals(df, 24)
    folds = generate_cpcv_folds(24, CPCVConfig(n_groups=5, k_test=2, embargo_size=1))
    res = run_cpcv(proposals=props, ohlcv_by_symbol={"AAA": df}, cpcv_folds=folds)
    # aggregate metrics computed over concatenated OOS returns w/ benchmark.
    assert res.aggregate_metrics.n_periods >= 1
    assert len(res.combined_benchmark) == len(res.combined_equity)
