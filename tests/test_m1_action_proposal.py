"""M1 — ActionProposal contract validation and consistency checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    OrderTimeInForce,
    ValidityCheck,
)


def _good_vc() -> ValidityCheck:
    return ValidityCheck(
        data_timestamps_valid=True,
        fits_risk_budget=True,
        survives_transaction_costs=True,
        liquidity_sufficient=True,
    )


def _kwargs(**overrides):
    base = {
        "symbol": "AAPL",
        "decision_ts": "2026-01-15T20:00:00+00:00",
        "side": ActionSide.BUY,
        "target_weight": 0.08,
        "horizon_days": 20,
        "conviction_score": 0.7,
        "conviction_tier": ConvictionTier.HIGH,
        "expected_return_pct": 0.04,
        "rationale": "Strong fundamentals + positive sentiment.",
        "validity_check": _good_vc(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


def test_valid_buy_proposal_constructs():
    ap = ActionProposal(**_kwargs())
    assert ap.side == ActionSide.BUY
    assert ap.target_weight == 0.08
    assert ap.tif == OrderTimeInForce.DAY


def test_render_markdown_includes_all_key_fields():
    ap = ActionProposal(**_kwargs(limit_price=151.50))
    md = ap.render_markdown()
    assert "**Rating**: Buy" in md
    assert "+8.00%" in md
    assert "HIGH" in md
    assert "+4.00%" in md
    assert "151.5" in md
    assert "Pre-flight checks" in md


def test_serialises_to_json_via_model_dump_json():
    ap = ActionProposal(**_kwargs())
    payload = ap.model_dump(mode="json")
    assert payload["side"] == "BUY"
    assert payload["conviction_tier"] == "HIGH"
    assert payload["validity_check"]["data_timestamps_valid"] is True


# ---------------------------------------------------------------------------
# Cross-field invariants
# ---------------------------------------------------------------------------


def test_buy_with_negative_weight_rejected():
    with pytest.raises(ValidationError, match="BUY with negative target_weight"):
        ActionProposal(**_kwargs(target_weight=-0.05))


def test_sell_with_positive_weight_rejected():
    with pytest.raises(ValidationError, match="SELL with positive target_weight"):
        ActionProposal(**_kwargs(side=ActionSide.SELL, target_weight=0.05))


@pytest.mark.parametrize("score,tier", [
    (0.2, ConvictionTier.MEDIUM),
    (0.5, ConvictionTier.HIGH),
    (0.8, ConvictionTier.LOW),
])
def test_conviction_score_and_tier_must_agree(score, tier):
    with pytest.raises(ValidationError, match="implies"):
        ActionProposal(**_kwargs(conviction_score=score, conviction_tier=tier))


@pytest.mark.parametrize("score,tier", [
    (0.1, ConvictionTier.LOW),
    (0.5, ConvictionTier.MEDIUM),
    (0.9, ConvictionTier.HIGH),
])
def test_conviction_score_and_tier_aligned_pass(score, tier):
    # Need to pick a side consistent with the tier — for LOW we drop weight to 0
    side = ActionSide.BUY if score >= 0.33 else ActionSide.HOLD
    weight = 0.05 if side == ActionSide.BUY else 0.0
    ap = ActionProposal(**_kwargs(
        side=side, target_weight=weight,
        conviction_score=score, conviction_tier=tier,
    ))
    assert ap.conviction_tier == tier


# ---------------------------------------------------------------------------
# Field range validation (pydantic)
# ---------------------------------------------------------------------------


def test_target_weight_outside_bounds_rejected():
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(target_weight=1.5))
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(target_weight=-1.5))


def test_conviction_score_outside_bounds_rejected():
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(conviction_score=1.2))
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(conviction_score=-0.1))


def test_horizon_days_must_be_positive_and_capped():
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(horizon_days=0))
    with pytest.raises(ValidationError):
        ActionProposal(**_kwargs(horizon_days=500))


# ---------------------------------------------------------------------------
# ABSTAIN behaviour
# ---------------------------------------------------------------------------


def test_abstain_with_zero_weight_is_legal_and_carries_diagnostic_notes():
    ap = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.ABSTAIN, target_weight=0.0,
        horizon_days=5,
        conviction_score=0.0, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.0,
        rationale="Abstain — analyst reports inconsistent.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=False, liquidity_sufficient=True,
            notes="Compiler could not resolve direction.",
        ),
        tags=["abstain"],
    )
    md = ap.render_markdown()
    assert "**Rating**: Abstain" in md
    assert ap.validity_check.survives_transaction_costs is False
