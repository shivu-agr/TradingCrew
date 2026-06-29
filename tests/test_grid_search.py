"""Tests for the walk-forward grid-search harness (agentic training L3).

Strategy: use synthetic OHLCV + a handful of crafted ActionProposals so we
can verify (a) the cartesian product is correct, (b) parameters flow into
SizingConfig/GateConfig, (c) the best-by-rank selection actually picks
the highest scorer, and (d) failures in individual grid points don't
take down the whole sweep.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import pytest

from trading_crew.agentic.backtest import (
    BacktestMetrics,
    Fold,
    WalkForwardConfig,
)
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    OrderTimeInForce,
    SizingBasis,
    ValidityCheck,
)
from trading_crew.agentic.grid_search import (
    GridAxis,
    GridPoint,
    GridSearchResult,
    default_grid,
    run_grid_search,
    _argmax,
    _build_configs,
    _extract_rank,
)
from trading_crew.agentic.risk import GateConfig, SizingConfig


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _proposal(decision_ts: str, *, side: ActionSide = ActionSide.BUY,
              target_weight: float = 0.05) -> ActionProposal:
    return ActionProposal(
        symbol="TST",
        decision_ts=decision_ts,
        side=side,
        target_weight=target_weight,
        horizon_days=10,
        tif=OrderTimeInForce.DAY,
        conviction_score=0.6,
        conviction_tier=ConvictionTier.MEDIUM,
        sizing_basis=SizingBasis.TARGET_WEIGHT,
        expected_return_pct=0.05,
        rationale="synthetic proposal for unit test " + "x" * 100,
        validity_check=ValidityCheck(
            data_timestamps_valid=True,
            fits_risk_budget=True,
            survives_transaction_costs=True,
            liquidity_sufficient=True,
            notes="ok",
        ),
    )


def _ohlcv(n_days: int = 60, start_price: float = 100.0, daily_drift: float = 0.001) -> pd.DataFrame:
    """Deterministic-rising OHLCV for backtest replay."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    closes = [start_price * ((1.0 + daily_drift) ** i) for i in range(n_days)]
    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low": [c * 0.995 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n_days,
    })


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_default_grid_has_three_axes():
    grid = default_grid()
    assert len(grid) == 3
    assert {ax.name for ax in grid} == {"kelly_fraction", "vol_target", "max_position_weight"}


def test_build_configs_routes_params():
    sizing, gate = _build_configs({"kelly_fraction": 0.5, "max_leverage": 2.0, "max_position_weight": 0.15})
    assert isinstance(sizing, SizingConfig)
    assert isinstance(gate, GateConfig)
    assert sizing.kelly_fraction == 0.5
    assert sizing.max_position_weight == 0.15
    assert gate.max_leverage == 2.0
    # max_position_weight is mirrored to both
    assert gate.max_position_weight == 0.15


def test_extract_rank_prefers_deflated_sharpe_but_falls_back_to_sharpe():
    m = BacktestMetrics(
        total_return_pct=0.1, cagr=0.1, annualised_vol=0.1,
        sharpe=1.5, sortino=2.0, calmar=1.2, max_drawdown=0.1,
        deflated_sharpe=None, n_periods=20, periods_per_year=252,
    )
    assert _extract_rank(m, "deflated_sharpe") == 1.5  # falls back to sharpe
    assert _extract_rank(m, "sortino") == 2.0


def test_extract_rank_returns_neg_inf_on_nan_or_missing():
    m = BacktestMetrics(
        total_return_pct=float("nan"), cagr=0.0, annualised_vol=0.0,
        sharpe=float("nan"), sortino=0.0, calmar=0.0, max_drawdown=0.0,
        deflated_sharpe=None, n_periods=0, periods_per_year=252,
    )
    assert _extract_rank(m, "total_return_pct") == float("-inf")
    assert _extract_rank(None, "sharpe") == float("-inf")  # noqa: type-ignore


def test_argmax_picks_best():
    points = [
        GridPoint(sizing_config={}, gate_config={}, n_folds=1, metrics=None, rank_metric=0.1),  # type: ignore[arg-type]
        GridPoint(sizing_config={}, gate_config={}, n_folds=1, metrics=None, rank_metric=0.9),  # type: ignore[arg-type]
        GridPoint(sizing_config={}, gate_config={}, n_folds=1, metrics=None, rank_metric=0.5),  # type: ignore[arg-type]
    ]
    assert _argmax(points) == 1


def test_argmax_returns_none_when_all_neg_inf():
    points = [
        GridPoint(sizing_config={}, gate_config={}, n_folds=0, metrics=None, rank_metric=float("-inf"), error="x")  # type: ignore[arg-type]
    ]
    assert _argmax(points) is None


# ---------------------------------------------------------------------------
# Driver — end-to-end
# ---------------------------------------------------------------------------


def test_run_grid_search_with_no_proposals_returns_empty():
    result = run_grid_search(
        proposals=[],
        ohlcv_by_symbol={"TST": _ohlcv()},
    )
    assert result.n_proposals == 0
    assert result.n_points == 0
    assert result.best() is None


def test_run_grid_search_returns_cartesian_product_size():
    proposals = [
        _proposal("2025-01-03T00:00:00"),
        _proposal("2025-01-10T00:00:00"),
        _proposal("2025-01-17T00:00:00"),
        _proposal("2025-01-24T00:00:00"),
    ]
    custom_grid = [
        GridAxis("kelly_fraction", (0.10, 0.25)),
        GridAxis("vol_target", (0.07, 0.10)),
        GridAxis("max_position_weight", (0.10,)),
    ]
    result = run_grid_search(
        proposals=proposals,
        ohlcv_by_symbol={"TST": _ohlcv(n_days=90)},
        walk_forward_config=WalkForwardConfig(train_size=2, embargo_size=0, test_size=1),
        grid=custom_grid,
    )
    assert result.n_proposals == 4
    # 2 × 2 × 1 = 4 grid points
    assert result.n_points == 4
    assert result.best_index is not None
    assert 0 <= result.best_index < 4


def test_run_grid_search_writes_configs_into_points():
    proposals = [
        _proposal("2025-01-03T00:00:00"),
        _proposal("2025-01-10T00:00:00"),
        _proposal("2025-01-17T00:00:00"),
    ]
    custom_grid = [
        GridAxis("kelly_fraction", (0.10, 0.50)),
    ]
    result = run_grid_search(
        proposals=proposals,
        ohlcv_by_symbol={"TST": _ohlcv(n_days=90)},
        walk_forward_config=WalkForwardConfig(train_size=1, embargo_size=0, test_size=1),
        grid=custom_grid,
    )
    kellys = sorted(p.sizing_config["kelly_fraction"] for p in result.points)
    assert kellys == [0.10, 0.50]


def test_run_grid_search_rank_by_total_return_picks_highest():
    proposals = [
        _proposal("2025-01-03T00:00:00"),
        _proposal("2025-01-10T00:00:00"),
        _proposal("2025-01-17T00:00:00"),
    ]
    # Sweep kelly_fraction — higher kelly = bigger sized positions =
    # higher PnL on a monotonically rising market.
    custom_grid = [GridAxis("kelly_fraction", (0.05, 0.50))]
    result = run_grid_search(
        proposals=proposals,
        ohlcv_by_symbol={"TST": _ohlcv(n_days=90, daily_drift=0.005)},
        walk_forward_config=WalkForwardConfig(train_size=1, embargo_size=0, test_size=1),
        grid=custom_grid,
        rank_by="total_return_pct",
    )
    assert result.best_index is not None
    # The high-kelly point should win on a monotonically rising market
    # (more aggressive sizing captures the drift faster).
    best = result.points[result.best_index]
    losing = result.points[1 - result.best_index]
    assert best.metrics.total_return_pct >= losing.metrics.total_return_pct


def test_grid_point_to_dict_serialises_cleanly():
    proposals = [
        _proposal("2025-01-03T00:00:00"),
        _proposal("2025-01-10T00:00:00"),
        _proposal("2025-01-17T00:00:00"),
    ]
    result = run_grid_search(
        proposals=proposals,
        ohlcv_by_symbol={"TST": _ohlcv(n_days=90)},
        walk_forward_config=WalkForwardConfig(train_size=1, embargo_size=0, test_size=1),
        grid=[GridAxis("kelly_fraction", (0.25,))],
    )
    d = result.to_dict()
    # Must be JSON-serialisable
    import json
    s = json.dumps(d)
    assert "best" in s
    assert "rank_metric" in s
    assert "sizing_config" in s
