"""Tests for the deterministic horizon-aware stop sanity layer."""

import math

from trading_crew.agentic.risk.stops import (
    ATR_FLOOR_MULT,
    HARD_CAP_PCT,
    HARD_FLOOR_PCT,
    annualised_vol_from_close,
    assess_stop,
    compute_atr,
    min_stop_pct,
    side_from_action,
)


def test_side_from_action():
    assert side_from_action("STRONG_SELL") == "short"
    assert side_from_action("sell") == "short"
    assert side_from_action("BUY") == "long"
    assert side_from_action("STRONG_BUY") == "long"
    assert side_from_action("HOLD") == "flat"
    assert side_from_action("") == "flat"
    # legacy tiers still map
    assert side_from_action("OVERWEIGHT") == "long"
    assert side_from_action("UNDERWEIGHT") == "short"


def test_compute_atr_and_vol_need_enough_data():
    assert compute_atr([1, 2], [1, 1], [1, 2]) is None
    assert annualised_vol_from_close([1, 2, 3]) is None
    # a clean synthetic series produces finite positive numbers
    close = [100 + math.sin(i / 5) * 3 for i in range(300)]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    atr = compute_atr(high, low, close)
    vol = annualised_vol_from_close(close)
    assert atr is not None and atr > 0
    assert vol is not None and vol > 0


def test_ntnx_case_tight_short_stop_is_widened():
    """The reported NTNX bug: a ~2% stop on a 756d short must be widened."""
    a = assess_stop(
        action="STRONG_SELL", entry=54.79, stop=56.0, target=35.0,
        horizon_days=756, current_price=54.79, atr=1.30, ann_vol=0.45,
    )
    assert a.assessable and a.adjusted
    assert a.provided_stop_pct < 0.03
    # widened well beyond the original 2.2%
    assert a.final_stop_pct > 0.15
    # short stop sits ABOVE entry
    assert a.stop > a.entry
    assert "too tight" in a.reason
    assert a.as_tag() is not None


def test_reasonable_long_stop_passes_unchanged():
    a = assess_stop(
        action="BUY", entry=100.0, stop=78.0, target=140.0,
        horizon_days=252, current_price=100.0, atr=2.0, ann_vol=0.40,
    )
    assert a.assessable and not a.adjusted
    assert a.stop == 78.0
    assert a.as_tag() is None


def test_wrong_side_stop_is_fixed():
    # A long whose "stop" is above entry is nonsensical -> corrected below entry
    a = assess_stop(
        action="BUY", entry=100.0, stop=105.0, target=130.0,
        horizon_days=120, current_price=100.0, atr=2.5, ann_vol=0.45,
    )
    assert a.adjusted
    assert a.stop < a.entry
    assert "wrong side" in a.reason


def test_missing_entry_anchors_to_current_price():
    a = assess_stop(
        action="SELL", entry=0.0, stop=0.0, target=40.0,
        horizon_days=90, current_price=50.0, atr=1.0, ann_vol=0.35,
    )
    assert a.entry == 50.0
    assert a.stop > a.entry  # short stop above entry


def test_flat_needs_no_stop():
    a = assess_stop(
        action="HOLD", entry=0, stop=0, target=0,
        horizon_days=90, current_price=54.0, atr=1.0, ann_vol=0.4,
    )
    assert a.side == "flat"
    assert not a.adjusted


def test_no_volatility_is_not_assessable_and_untouched():
    a = assess_stop(
        action="SELL", entry=54.79, stop=56.0, target=35.0,
        horizon_days=756, current_price=54.79, atr=None, ann_vol=None,
    )
    assert not a.assessable
    assert not a.adjusted
    assert a.stop == 56.0  # never fabricates a correction


def test_min_stop_pct_respects_clamps():
    # tiny vol -> hard floor
    assert min_stop_pct(atr_pct=0.001, daily_vol=0.001, horizon_days=252) == HARD_FLOOR_PCT
    # huge vol -> hard cap
    assert min_stop_pct(atr_pct=0.5, daily_vol=0.5, horizon_days=252) == HARD_CAP_PCT
    # no inputs -> None (caller must skip, not fabricate)
    assert min_stop_pct(atr_pct=None, daily_vol=None, horizon_days=252) is None


def test_atr_floor_dominates_for_short_horizon():
    # 5-day horizon: ATR floor (2xATR) should drive the min, not horizon noise
    m = min_stop_pct(atr_pct=0.03, daily_vol=0.02, horizon_days=5)
    assert m >= ATR_FLOOR_MULT * 0.03 - 1e-9


# --- target validation ---------------------------------------------------

def test_wrong_side_target_is_set_to_default():
    # long with a target BELOW entry is nonsensical -> set above entry
    a = assess_stop(
        action="BUY", entry=100.0, stop=80.0, target=90.0,
        horizon_days=180, current_price=100.0, atr=2.0, ann_vol=0.40,
    )
    assert a.target_adjusted
    assert a.target > a.entry
    assert "wrong-side" in a.target_reason or "degenerate" in a.target_reason
    # expected_return recomputed to match the corrected target
    assert a.expected_return_pct == round((a.target - a.entry) / a.entry * 100, 1)


def test_degenerate_target_is_set():
    a = assess_stop(
        action="BUY", entry=100.0, stop=80.0, target=100.2,  # ~0.2% => degenerate
        horizon_days=180, current_price=100.0, atr=2.0, ann_vol=0.40,
    )
    assert a.target_adjusted
    assert (a.target - a.entry) / a.entry > 0.01


def test_moonshot_target_is_capped():
    # +400% target on a 30-day long is implausible -> capped
    a = assess_stop(
        action="BUY", entry=100.0, stop=90.0, target=500.0,
        horizon_days=30, current_price=100.0, atr=2.0, ann_vol=0.40,
    )
    assert a.target_adjusted
    assert "implausibly large" in a.target_reason
    assert a.target < 500.0
    assert (a.target - a.entry) / a.entry <= a.max_target_pct + 1e-9


def test_short_target_capped_below_100pct():
    a = assess_stop(
        action="STRONG_SELL", entry=100.0, stop=110.0, target=1.0,  # -99% target
        horizon_days=2520, current_price=100.0, atr=3.0, ann_vol=0.60,
    )
    assert a.target_adjusted
    # a short can't realistically gain ~100%; capped below entry but above 0
    assert 0 < a.target < a.entry
    assert (a.entry - a.target) / a.entry <= 0.90 + 1e-9


def test_reasonable_target_passes_unchanged():
    a = assess_stop(
        action="BUY", entry=100.0, stop=78.0, target=140.0,
        horizon_days=252, current_price=100.0, atr=2.0, ann_vol=0.40,
    )
    assert not a.target_adjusted
    assert a.target == 140.0
    assert not a.any_adjusted  # neither stop nor target changed


def test_as_tag_reports_both_adjustments():
    a = assess_stop(
        action="SELL", entry=100.0, stop=0.0, target=0.0,  # both missing
        horizon_days=180, current_price=100.0, atr=2.0, ann_vol=0.45,
    )
    assert a.adjusted and a.target_adjusted and a.any_adjusted
    tag = a.as_tag()
    assert "stop_normalized" in tag and "target_normalized" in tag
