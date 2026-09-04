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
        action="BUY",
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


def test_buy_with_size_maps_to_buy():
    proposal = portfolio_decision_to_action_proposal(_decision(), symbol="AAPL")
    assert proposal.side == ActionSide.BUY
    assert proposal.target_weight == pytest.approx(0.05)
    assert proposal.symbol == "AAPL"


def test_strong_buy_maps_to_buy_side():
    """STRONG_BUY collapses to ``ActionSide.BUY`` (the conviction tier
    is preserved separately in ``conviction_score`` / ``target_weight``)."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="STRONG_BUY"), symbol="AAPL",
    )
    assert proposal.side == ActionSide.BUY


def test_sell_with_size_maps_to_sell_with_negative_weight():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="SELL", size_pct_of_book=3.0),
        symbol="msft",
    )
    assert proposal.side == ActionSide.SELL
    assert proposal.target_weight == pytest.approx(-0.03)
    assert proposal.symbol == "MSFT"  # uppercased


def test_strong_sell_maps_to_sell_side():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="STRONG_SELL", size_pct_of_book=3.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.SELL
    assert proposal.target_weight == pytest.approx(-0.03)


def test_hold_maps_to_hold_with_zero_weight():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="HOLD", size_pct_of_book=0.0, expected_return_pct=0.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.HOLD
    assert proposal.target_weight == 0.0


def test_legacy_neutral_action_still_maps_to_hold():
    """Legacy runs persisted the flat tier as ``NEUTRAL``; the schema's
    coercion validator normalises it to ``HOLD`` at load time, so the
    bridge sees HOLD and maps it accordingly.  This test guards the
    end-to-end round-trip (validator + bridge)."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="NEUTRAL", size_pct_of_book=0.0, expected_return_pct=0.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.HOLD


def test_buy_with_zero_size_falls_back_to_abstain():
    """A contradictory decision (BUY but size=0) should not pretend to be a buy."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="BUY", size_pct_of_book=0.0),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.ABSTAIN
    assert proposal.target_weight == 0.0


def test_confidence_maps_to_correct_tier(monkeypatch):
    """Under the legacy ``raw`` mode the conviction score is the LLM's own
    confidence, so it buckets straight through the tier boundaries."""
    monkeypatch.setenv("TRADINGCREW_CONVICTION_MODE", "raw")
    high = portfolio_decision_to_action_proposal(_decision(confidence=0.9), symbol="AAPL")
    medium = portfolio_decision_to_action_proposal(_decision(confidence=0.5), symbol="AAPL")
    low = portfolio_decision_to_action_proposal(_decision(confidence=0.2), symbol="AAPL")
    assert high.conviction_tier == ConvictionTier.HIGH
    assert medium.conviction_tier == ConvictionTier.MEDIUM
    assert low.conviction_tier == ConvictionTier.LOW


@pytest.mark.parametrize("confidence", [0.33, 0.34, 0.65, 0.659, 0.66])
def test_boundary_confidence_does_not_abstain(monkeypatch, confidence):
    """Regression: scores in the old drift zones ([0.33,0.35) and
    [0.65,0.66)) used to make the bridge emit a tier the ActionProposal
    validator rejected, so a valid BUY was silently synthesized into an
    ABSTAIN.  The shared ``ConvictionTier.from_score`` bucketing must keep
    the side intact and the tier internally consistent with the score.
    (Pinned to ``raw`` mode where conviction == confidence.)"""
    monkeypatch.setenv("TRADINGCREW_CONVICTION_MODE", "raw")
    proposal = portfolio_decision_to_action_proposal(
        _decision(confidence=confidence), symbol="AAPL",
    )
    assert proposal.side == ActionSide.BUY          # not downgraded to ABSTAIN
    assert proposal.conviction_score == pytest.approx(confidence)
    assert proposal.conviction_tier == ConvictionTier.from_score(confidence)


# --- Evidence-based conviction (default mode) ------------------------------


def test_evidence_conviction_default_is_not_raw_confidence():
    """In the default ``evidence`` mode the score is fused from action
    strength / evidence balance / size / confidence — not the raw number."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="BUY", confidence=0.75), symbol="AAPL",
    )
    assert proposal.conviction_score != pytest.approx(0.75)
    assert 0.0 <= proposal.conviction_score <= 1.0


def test_strong_action_raises_conviction_over_plain():
    """STRONG_BUY must score higher than a plain BUY with identical evidence —
    the strength tier is real information the raw path threw away."""
    strong = portfolio_decision_to_action_proposal(
        _decision(action="STRONG_BUY"), symbol="AAPL",
    ).conviction_score
    plain = portfolio_decision_to_action_proposal(
        _decision(action="BUY"), symbol="AAPL",
    ).conviction_score
    assert strong > plain


def test_evidence_balance_moves_conviction():
    """More falsifiers/risks vs drivers should lower conviction."""
    strong_support = portfolio_decision_to_action_proposal(
        _decision(key_drivers=["a", "b", "c", "d"], key_risks=[], falsifiers=[]),
        symbol="AAPL",
    ).conviction_score
    balanced = portfolio_decision_to_action_proposal(
        _decision(key_drivers=["a"], key_risks=["x", "y"], falsifiers=["z", "w"]),
        symbol="AAPL",
    ).conviction_score
    assert strong_support > balanced


def test_evidence_mode_spans_tiers():
    """The whole point of the recalibration: a weak call lands LOW/MEDIUM while
    a strong, well-supported, well-sized call lands HIGH — no collapse."""
    weak = portfolio_decision_to_action_proposal(
        _decision(
            action="SELL", size_pct_of_book=0.5, confidence=0.55,
            key_drivers=["a"], key_risks=["x", "y", "z"], falsifiers=["p", "q"],
        ),
        symbol="AAPL",
    )
    strong = portfolio_decision_to_action_proposal(
        _decision(
            action="STRONG_BUY", size_pct_of_book=5.0, confidence=0.85,
            key_drivers=["a", "b", "c", "d"], key_risks=["x"], falsifiers=[],
        ),
        symbol="AAPL",
    )
    assert weak.conviction_tier in (ConvictionTier.LOW, ConvictionTier.MEDIUM)
    assert strong.conviction_tier == ConvictionTier.HIGH
    assert strong.conviction_score > weak.conviction_score


def test_conviction_score_and_tier_stay_consistent():
    """Whatever the mode, the emitted tier must match the emitted score so the
    ActionProposal validator never rejects a valid decision."""
    proposal = portfolio_decision_to_action_proposal(_decision(action="STRONG_BUY"), symbol="AAPL")
    assert proposal.conviction_tier == ConvictionTier.from_score(proposal.conviction_score)


def test_expected_return_is_signed_to_side_positive_short_corrected():
    """A SELL carrying a (contradictory) positive expected_return_pct must be
    flipped negative so M5 Kelly / M7 allocator see a coherent edge."""
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="SELL", size_pct_of_book=3.0, expected_return_pct=0.10),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.SELL
    assert proposal.expected_return_pct == pytest.approx(-0.10)


def test_expected_return_is_signed_to_side_long_kept_positive():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="BUY", size_pct_of_book=3.0, expected_return_pct=0.12),
        symbol="AAPL",
    )
    assert proposal.expected_return_pct == pytest.approx(0.12)


def test_expected_return_zero_for_hold():
    proposal = portfolio_decision_to_action_proposal(
        _decision(action="HOLD", size_pct_of_book=0.0, expected_return_pct=0.09),
        symbol="AAPL",
    )
    assert proposal.side == ActionSide.HOLD
    assert proposal.expected_return_pct == 0.0


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
