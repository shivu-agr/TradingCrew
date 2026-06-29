"""Tests for the PortfolioDecision -> ActionProposal bridge."""

from __future__ import annotations

import pytest

from trading_crew.agentic.bridge import portfolio_decision_to_action_proposal
from trading_crew.agentic.execution.contracts import (
    ActionSide,
    ConvictionTier,
)
from trading_crew.schemas import PortfolioDecision


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
        rationale="Strong demand growth and improving margins.",
        key_drivers=["earnings beat", "guide raise"],
        key_risks=["macro slowdown"],
        falsifiers=["miss next quarter"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


def test_overweight_with_size_maps_to_buy():
    proposal = portfolio_decision_to_action_proposal(_decision(), symbol="AAPL")
    assert proposal.side == ActionSide.BUY
    assert proposal.target_weight == pytest.approx(0.05)
    assert proposal.symbol == "AAPL"


def test_underweight_with_size_maps_to_sell_with_negative_weight():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="UNDERWEIGHT", size_pct_of_book=3.0),
        symbol="msft",
    )
    assert proposal.side == ActionSide.SELL
    assert proposal.target_weight == pytest.approx(-0.03)
    assert proposal.symbol == "MSFT"  # uppercased


def test_neutral_maps_to_hold_with_zero_weight():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="NEUTRAL", size_pct_of_book=0.0, expected_return_pct=0.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.HOLD
    assert proposal.target_weight == 0.0


def test_overweight_with_zero_size_falls_back_to_abstain():
    """A contradictory decision (OVERWEIGHT but size=0) should not pretend to be a buy."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="OVERWEIGHT", size_pct_of_book=0.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.ABSTAIN
    assert proposal.target_weight == 0.0


def test_confidence_maps_to_correct_tier():
    high = portfolio_decision_to_action_proposal(_decision(confidence=0.9), symbol="AAPL")
    medium = portfolio_decision_to_action_proposal(_decision(confidence=0.5), symbol="AAPL")
    low = portfolio_decision_to_action_proposal(_decision(confidence=0.2), symbol="AAPL")
    assert high.conviction_tier == ConvictionTier.HIGH
    assert medium.conviction_tier == ConvictionTier.MEDIUM
    assert low.conviction_tier == ConvictionTier.LOW


def test_blocked_compliance_marks_validity_flags_false():
    proposal = portfolio_decision_to_action_proposal(
        _decision(compliance_status="BLOCKED"),
        symbol="AAPL",
    )
    assert proposal.validity_check.data_timestamps_valid is False
    assert proposal.validity_check.liquidity_sufficient is False


def test_low_expected_return_fails_cost_check():
    proposal = portfolio_decision_to_action_proposal(
        _decision(expected_return_pct=0.005),  # 0.5% < 1% threshold
        symbol="AAPL",
    )
    assert proposal.validity_check.survives_transaction_costs is False


def test_entry_price_propagated_and_levels_surfaced_in_rationale():
    proposal = portfolio_decision_to_action_proposal(_decision(), symbol="AAPL")
    assert proposal.limit_price == 150.0
    # The contract doesn't carry stop/target directly; we surface them in rationale.
    assert "140" in proposal.rationale
    assert "180" in proposal.rationale


def test_geopolitical_flags_become_tags():
    proposal = portfolio_decision_to_action_proposal(
        _decision(geopolitical_flags=["china_export", "sanctions"]),
        symbol="AAPL",
    )
    assert "china_export" in proposal.tags
    assert "sanctions" in proposal.tags


def test_short_rationale_is_padded_not_rejected():
    """A rationale shorter than the schema minimum is padded with drivers."""
    short = _decision(rationale="ok", key_drivers=["a", "b"])
    proposal = portfolio_decision_to_action_proposal(short, symbol="AAPL")
    # Schema requires len(rationale) >= 20; padding must satisfy it
    assert len(proposal.rationale) >= 20
