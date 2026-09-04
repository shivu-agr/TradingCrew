"""Consensus-ensemble (strategy swarm) tests."""

from __future__ import annotations

import random

import pandas as pd

from trading_crew.agentic.swarm import ConsensusConfig, run_consensus_backtest
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


def _make_proposal(ts: str, weight: float) -> ActionProposal:
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol="AAA", decision_ts=ts, side=side, target_weight=weight,
        horizon_days=5, conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.02 if weight != 0 else 0.0,
        rationale="Consensus proposal with enough text.",
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


# A few sizing/gate member configs (dicts, as PSO/grid expose them).
def _members(scores):
    base_sizing = {
        "kelly_fraction": 0.25, "risk_free_rate": 0.04, "vol_target": 0.10,
        "max_cvar_pct": 0.02, "max_position_weight": 0.20, "vol_lookback_days": 63,
        "risk_mult_floor": 0.5, "risk_mult_ceiling": 1.0,
    }
    members = []
    for i, sc in enumerate(scores):
        sizing = dict(base_sizing)
        sizing["max_position_weight"] = 0.10 + 0.05 * i
        members.append((sizing, {}, sc))
    return members


def test_consensus_runs_and_reports_agreement():
    props, df = _fixture()
    members = _members([1.2, 0.8, 0.4, -0.1])
    res = run_consensus_backtest(
        proposals=props, ohlcv_by_symbol={"AAA": df}, members=members,
        config=ConsensusConfig(top_k=3),
    )
    assert res.n_members == 3          # top_k trims the 4th (worst) member
    assert 0.0 <= res.directional_agreement <= 1.0
    assert res.weight_dispersion >= 0.0
    assert res.metrics is not None


def test_consensus_directional_agreement_is_one_for_all_long():
    """All members are long-only here, so the crowd of strategies always
    agrees on the sign -> directional agreement 1.0."""
    props, df = _fixture()
    members = _members([1.0, 0.9, 0.8])
    res = run_consensus_backtest(
        proposals=props, ohlcv_by_symbol={"AAA": df}, members=members,
        config=ConsensusConfig(top_k=3),
    )
    assert res.directional_agreement == 1.0


def test_consensus_empty_members():
    props, df = _fixture()
    res = run_consensus_backtest(
        proposals=props, ohlcv_by_symbol={"AAA": df}, members=[],
    )
    assert res.n_members == 0
    assert res.metrics is None


def test_consensus_is_deterministic():
    props, df = _fixture()
    members = _members([1.2, 0.8, 0.4])
    kw = dict(proposals=props, ohlcv_by_symbol={"AAA": df}, members=members,
              config=ConsensusConfig(top_k=3))
    r1 = run_consensus_backtest(**kw)
    r2 = run_consensus_backtest(**kw)
    assert r1.equity_curve == r2.equity_curve
    assert r1.directional_agreement == r2.directional_agreement
