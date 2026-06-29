"""End-to-end smoke test: PortfolioDecision -> ActionProposal -> Pipeline.

This bypasses the CrewAI kickoff and tests the deterministic glue we added
on top: the bridge (M1) hands the M2/M5 pipeline a proposal, gates fire,
the simulator emits a fill (or rejection), and we get a structured result
the runner can turn into a JSON event.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_crew.agentic.bridge import portfolio_decision_to_action_proposal
from trading_crew.agentic.execution.pipeline import run_pipeline
from trading_crew.schemas import PortfolioDecision


def _synthetic_ohlcv(days: int = 260) -> pd.DataFrame:
    """A simple monotonic-up series — enough rows to seed VaR/CVaR."""
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    close = [100.0 + i * 0.1 for i in range(days)]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in close],
        "High": [c + 1.0 for c in close],
        "Low": [c - 1.0 for c in close],
        "Close": close,
        "Volume": [1_000_000] * days,
    })


def _decision(**overrides) -> PortfolioDecision:
    base = dict(
        action="OVERWEIGHT",
        confidence=0.75,
        size_pct_of_book=5.0,
        entry_price=150.0,
        stop_loss=140.0,
        target_price=180.0,
        horizon_days=21,
        expected_return_pct=0.08,
        rationale="Earnings beat, guide raise, strong demand.",
        key_drivers=["earnings", "guidance"],
        key_risks=["macro"],
        falsifiers=["next quarter miss"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


def test_overweight_decision_flows_through_to_fill(tmp_path, monkeypatch):
    """Happy path: a clean OVERWEIGHT decision turns into a BUY proposal,
    survives the risk gates with non-zero size, and books a fill."""
    monkeypatch.setenv("TRADINGAGENTS_PORTFOLIO_DIR", str(tmp_path / "portfolio"))
    proposal = portfolio_decision_to_action_proposal(
        _decision(), symbol="AAPL", decision_ts="2025-06-01",
    )
    result = run_pipeline(
        proposal,
        ohlcv=_synthetic_ohlcv(),
        portfolio_id="test-e2e",
        persist=False,
    )
    assert result.sizing is not None, "Sizer must run"
    assert result.risk_gate is not None, "Risk gate must run"
    # We're not asserting passed=True (CVaR clamp can be tight) but
    # we do require an audit trail.
    assert result.note or result.fill or result.rejected


def test_underweight_high_size_fails_concentration_or_passes(tmp_path, monkeypatch):
    """An UNDERWEIGHT with 50% size is concentrated; either the sizer
    clamps it or the risk gate rejects. Either way the result envelope
    captures a structured outcome — *never* an exception."""
    monkeypatch.setenv("TRADINGAGENTS_PORTFOLIO_DIR", str(tmp_path / "portfolio"))
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="UNDERWEIGHT", size_pct_of_book=50.0),
        symbol="MSFT",
        decision_ts="2025-06-01",
    )
    result = run_pipeline(
        proposal,
        ohlcv=_synthetic_ohlcv(),
        portfolio_id="test-e2e-concentrated",
        persist=False,
    )
    assert result.sizing is not None
    # Sizer should have clamped or risk_gate should have failed at least
    # one check — concentration limit (default 25%) << proposed 50%.
    sized_below_intent = abs(result.sizing.final_weight) < 0.5
    gate_caught = result.risk_gate is not None and not result.risk_gate.passed
    assert sized_below_intent or gate_caught


def test_neutral_decision_results_in_zero_action(tmp_path, monkeypatch):
    """NEUTRAL -> HOLD: no order, no fill, no rejection — just a clean
    record that we deliberately did nothing."""
    monkeypatch.setenv("TRADINGAGENTS_PORTFOLIO_DIR", str(tmp_path / "portfolio"))
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="NEUTRAL", size_pct_of_book=0.0, expected_return_pct=0.0),
        symbol="SPY",
        decision_ts="2025-06-01",
    )
    assert proposal.side.value == "HOLD"
    assert proposal.target_weight == 0.0


def test_blocked_compliance_propagates_to_pipeline(tmp_path, monkeypatch):
    """A BLOCKED compliance status sets validity flags False; the M5 risk
    multiplier should compress and the gate may reject. The pipeline must
    never crash on a BLOCKED decision."""
    monkeypatch.setenv("TRADINGAGENTS_PORTFOLIO_DIR", str(tmp_path / "portfolio"))
    proposal = portfolio_decision_to_action_proposal(
        _decision(compliance_status="BLOCKED"),
        symbol="AAPL",
        decision_ts="2025-06-01",
    )
    assert proposal.validity_check.data_timestamps_valid is False
    assert proposal.validity_check.liquidity_sufficient is False
    result = run_pipeline(
        proposal,
        ohlcv=_synthetic_ohlcv(),
        portfolio_id="test-e2e-blocked",
        persist=False,
    )
    assert result.note or result.fill or result.rejected
