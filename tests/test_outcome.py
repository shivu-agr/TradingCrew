"""Prediction-vs-tape labels and desk attribution."""

from __future__ import annotations

from trading_crew.agentic.outcome import (
    OutcomeLabel,
    attribute_desks,
    hold_threshold_from_episodes,
    label_outcome,
    lean_verdict,
    market_direction,
    wrong_prediction_desks,
)


class _Ep:
    def __init__(self, side, alpha, status="RESOLVED"):
        self.outcome_status = status
        self.action_proposal = {"side": side}
        self.alpha_return = alpha


def test_buy_up_is_win():
    assert label_outcome("BUY", 0.08, 0.08) == OutcomeLabel.WIN


def test_buy_down_is_loss():
    assert label_outcome("BUY", -0.05, -0.05) == OutcomeLabel.LOSS


def test_sell_down_is_win_because_return_is_side_signed():
    # Short: price fell, signed realised is positive.
    assert label_outcome("SELL", 0.10, -0.09) == OutcomeLabel.WIN


def test_hold_unscored_without_threshold():
    assert label_outcome("HOLD", 0.0, 0.12, hold_abs_threshold=None) == OutcomeLabel.HOLD_UNSCORED


def test_hold_miss_when_move_exceeds_measured_bar():
    assert label_outcome("HOLD", 0.0, 0.12, hold_abs_threshold=0.05) == OutcomeLabel.HOLD_MISS
    assert label_outcome("HOLD", 0.0, 0.01, hold_abs_threshold=0.05) == OutcomeLabel.HOLD_CORRECT


def test_hold_threshold_needs_cohort():
    few = [_Ep("BUY", 0.04) for _ in range(3)]
    assert hold_threshold_from_episodes(few) is None
    many = [_Ep("BUY", 0.04 if i % 2 == 0 else 0.10) for i in range(10)]
    thr = hold_threshold_from_episodes(many)
    assert thr is not None and thr > 0


def test_desk_attribution_hurt_helped():
    rows = attribute_desks(
        {"news_analyst": "bullish", "macro_analyst": "bearish", "market_analyst": "neutral"},
        "bearish",
    )
    by = {r["desk"]: r["verdict"] for r in rows}
    assert by["news_analyst"] == "hurt"
    assert by["macro_analyst"] == "helped"
    assert by["market_analyst"] == "neutral"
    assert wrong_prediction_desks(rows) == ["news_analyst"]
    assert lean_verdict("bullish", "bullish") == "helped"
    assert market_direction(-0.03) == "bearish"
