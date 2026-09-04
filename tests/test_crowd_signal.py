"""Crowd-positioning signal + fade-the-crowd tilt tests (pure, offline)."""

from __future__ import annotations

from trading_crew.agentic.swarm import (
    CrowdConfig,
    CrowdInputs,
    compute_crowd_signal,
    crowd_size_multiplier,
)


# ============================================================================
# compute_crowd_signal
# ============================================================================


def test_no_inputs_is_balanced():
    sig = compute_crowd_signal(CrowdInputs())
    assert sig.n_components == 0
    assert sig.score == 0.0
    assert sig.label == "balanced"


def test_score_in_unit_interval():
    sig = compute_crowd_signal(CrowdInputs(
        put_call_ratio=2.0, iv_skew=0.1, insider_net_ratio=-1.0,
        analyst_bullish_frac=0.0, institutional_net_drift=-0.5, cot_net_positioning=-1.0,
    ))
    assert -1.0 <= sig.score <= 1.0
    assert sig.n_components == 6


def test_bearish_inputs_yield_crowded_short():
    sig = compute_crowd_signal(CrowdInputs(
        put_call_ratio=2.5, iv_skew=0.08, analyst_bullish_frac=0.1,
    ))
    assert sig.score < 0
    assert sig.label == "crowded short"


def test_bullish_inputs_yield_crowded_long():
    sig = compute_crowd_signal(CrowdInputs(
        put_call_ratio=0.4, iv_skew=-0.05, insider_net_ratio=1.0,
        analyst_bullish_frac=0.95,
    ))
    assert sig.score > 0
    assert sig.label == "crowded long"


def test_missing_components_are_dropped_not_zero_filled():
    # Only one component present -> score equals that component's mapping.
    sig = compute_crowd_signal(CrowdInputs(analyst_bullish_frac=1.0))
    assert sig.n_components == 1
    assert sig.score == 1.0   # (1.0 - 0.5) * 2 clamped


# ============================================================================
# crowd_size_multiplier (fade-the-crowd)
# ============================================================================


def test_multiplier_is_one_when_disabled():
    sig = compute_crowd_signal(CrowdInputs(put_call_ratio=3.0, analyst_bullish_frac=0.0))
    assert crowd_size_multiplier(sig, "SELL", CrowdConfig(enabled=False)) == 1.0


def test_multiplier_is_one_for_balanced_crowd():
    sig = compute_crowd_signal(CrowdInputs(analyst_bullish_frac=0.55))  # near 0
    assert crowd_size_multiplier(sig, "BUY", CrowdConfig(enabled=True)) == 1.0


def test_aligned_trade_with_crowd_is_trimmed():
    # Crowded long, going long -> aligned -> multiplier < 1.
    sig = compute_crowd_signal(CrowdInputs(analyst_bullish_frac=1.0, insider_net_ratio=1.0))
    cfg = CrowdConfig(enabled=True, crowd_threshold=0.5, max_fade=0.5)
    mult = crowd_size_multiplier(sig, "BUY", cfg)
    assert mult < 1.0
    assert mult >= cfg.max_fade - 1e-9


def test_contrarian_trade_default_no_boost():
    # Crowded long, going short -> contrarian; default max_boost=1.0 -> no boost.
    sig = compute_crowd_signal(CrowdInputs(analyst_bullish_frac=1.0, insider_net_ratio=1.0))
    cfg = CrowdConfig(enabled=True, crowd_threshold=0.5, max_boost=1.0)
    assert crowd_size_multiplier(sig, "SELL", cfg) == 1.0


def test_contrarian_trade_can_boost_when_configured():
    sig = compute_crowd_signal(CrowdInputs(analyst_bullish_frac=1.0, insider_net_ratio=1.0))
    cfg = CrowdConfig(enabled=True, crowd_threshold=0.5, max_boost=1.5)
    mult = crowd_size_multiplier(sig, "SELL", cfg)
    assert mult > 1.0
    assert mult <= 1.5 + 1e-9
