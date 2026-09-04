"""Particle Swarm Optimizer tests (generic optimiser + backtest-wired search)."""

from __future__ import annotations

import random

import pandas as pd

from trading_crew.agentic.swarm import (
    PSOConfig,
    particle_swarm_optimize,
    run_swarm_search,
)
from trading_crew.agentic.swarm.pso import DEFAULT_BOUNDS
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


# ============================================================================
# Generic optimiser on an analytic objective
# ============================================================================


def test_pso_converges_to_analytic_optimum():
    """Maximise a concave quadratic with a known peak at 0.3; the swarm
    should land close to it."""
    def objective(params):
        x = params["x"]
        return -((x - 0.3) ** 2)

    best, score, history, evals = particle_swarm_optimize(
        objective, {"x": (0.0, 1.0)}, PSOConfig(swarm_size=15, iterations=25, seed=1),
    )
    assert abs(best["x"] - 0.3) < 0.05
    assert score <= 0.0
    # global-best is non-decreasing over iterations.
    assert all(history[i] <= history[i + 1] + 1e-12 for i in range(len(history) - 1))
    assert len(evals) == 15 * (25 + 1)


def test_pso_is_deterministic_for_seed():
    def objective(params):
        return -((params["x"] - 0.3) ** 2)

    a = particle_swarm_optimize(objective, {"x": (0.0, 1.0)}, PSOConfig(swarm_size=8, iterations=6, seed=7))
    b = particle_swarm_optimize(objective, {"x": (0.0, 1.0)}, PSOConfig(swarm_size=8, iterations=6, seed=7))
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_pso_respects_bounds():
    def objective(params):
        # reward pushing x as high as possible -> clamp must hold it in-box.
        return params["x"]

    best, _score, _hist, evals = particle_swarm_optimize(
        objective, {"x": (0.0, 1.0)}, PSOConfig(swarm_size=10, iterations=10, seed=2),
    )
    assert 0.0 <= best["x"] <= 1.0
    for params, _ in evals:
        assert 0.0 <= params["x"] <= 1.0


# ============================================================================
# Backtest-wired swarm search
# ============================================================================


def _make_proposal(ts: str, weight: float) -> ActionProposal:
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol="AAA", decision_ts=ts, side=side, target_weight=weight,
        horizon_days=5, conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.02 if weight != 0 else 0.0,
        rationale="Swarm search proposal with enough text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _fixture():
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = random.Random(5)
    closes = [100.0]
    for _ in range(399):
        closes.append(closes[-1] * (1 + rng.gauss(0.0006, 0.012)))
    df = pd.DataFrame({
        "Date": dates, "Open": closes,
        "High": [c * 1.005 for c in closes], "Low": [c * 0.995 for c in closes],
        "Close": closes, "Volume": [1_000_000] * 400,
    })
    props = [
        _make_proposal(dates[80 + i * 10].strftime("%Y-%m-%dT16:00:00+00:00"),
                       0.06 if i % 2 == 0 else 0.0)
        for i in range(24)
    ]
    return props, df


def test_run_swarm_search_runs_and_reports_history():
    props, df = _fixture()
    res = run_swarm_search(
        proposals=props, ohlcv_by_symbol={"AAA": df},
        pso_config=PSOConfig(swarm_size=6, iterations=4, seed=1),
    )
    assert res.n_evaluations == 6 * (4 + 1)
    assert len(res.history) == 4 + 1
    assert set(res.best_params.keys()) == set(DEFAULT_BOUNDS.keys())
    # every evaluated config stays inside the search box.
    for e in res.evaluations:
        for name, (lo, hi) in DEFAULT_BOUNDS.items():
            assert lo - 1e-9 <= e.params[name] <= hi + 1e-9


def test_run_swarm_search_is_deterministic():
    props, df = _fixture()
    kw = dict(proposals=props, ohlcv_by_symbol={"AAA": df},
              pso_config=PSOConfig(swarm_size=6, iterations=4, seed=1))
    r1 = run_swarm_search(**kw)
    r2 = run_swarm_search(**kw)
    assert r1.best_params == r2.best_params
    assert r1.history == r2.history


def test_run_swarm_search_empty_proposals():
    _props, df = _fixture()
    res = run_swarm_search(proposals=[], ohlcv_by_symbol={"AAA": df})
    assert res.n_proposals == 0
    assert res.best_metrics is None
