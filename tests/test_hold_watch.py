"""HOLD sit-out keeps size 0 but gets ATR watch levels + as-if Kelly size."""

from __future__ import annotations

import math

from trading_crew.agentic.risk.hold_watch import (
    apply_hold_watch_plan,
    infer_watch_side,
    recommended_watch_size_pct,
    synthesize_hold_watch,
)
from trading_crew.guardrails import confidence_guardrail
from trading_crew.schemas import PortfolioDecision


def _hold(**overrides) -> PortfolioDecision:
    base = dict(
        action="HOLD",
        confidence=0.55,
        size_pct_of_book=0.0,
        entry_price=67.84,
        stop_loss=0.0,
        target_price=0.0,
        horizon_days=21,
        expected_return_pct=0.0,
        rationale="Desks split; sit out.",
        key_drivers=["mixed"],
        key_risks=["volatility"],
        falsifiers=["breakout"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


def _ohlcv(n: int = 300, last: float = 67.84):
    close = [last + math.sin(i / 8.0) * 1.5 for i in range(n)]
    high = [c + 0.8 for c in close]
    low = [c - 0.8 for c in close]
    return close, high, low


def test_infer_watch_side_majority_and_split():
    assert infer_watch_side({"a": "bearish", "b": "bearish", "c": "bullish"}) == "short"
    assert infer_watch_side({"a": "bullish", "b": "bearish"}) == "long"
    assert infer_watch_side({}) == "long"


def test_synthesize_hold_watch_fills_levels():
    a = synthesize_hold_watch(
        entry=67.84,
        horizon_days=21,
        current_price=67.84,
        atr=1.40,
        ann_vol=0.35,
        watch_side="long",
    )
    assert a.assessable
    assert a.stop > 0 and a.stop < 67.84
    assert a.target > 67.84
    assert a.expected_return_pct is not None and a.expected_return_pct > 0


def test_hold_watch_ntnx_shaped_keeps_book_size_zero():
    close, high, low = _ohlcv()
    decision = _hold()
    out = apply_hold_watch_plan(
        decision,
        close=close,
        high=high,
        low=low,
        desk_biases={"News Analyst": "bearish", "Market Analyst": "bullish"},
        ticker="NTNX",
        decision_ts="2026-08-22T00:00:00",
    )
    assert out.action == "HOLD"
    assert out.size_pct_of_book == 0.0
    assert out.stop_loss > 0
    assert out.target_price > 0
    assert out.expected_return_pct != 0
    assert out.recommended_size_pct > 0
    assert out.watch_side == "long"
    assert "HOLD watch levels" in out.rationale
    assert "not an order" in out.rationale


def test_hold_watch_skips_when_ohlcv_too_short():
    close, high, low = _ohlcv(n=10)
    decision = _hold()
    out = apply_hold_watch_plan(
        decision,
        close=close,
        high=high,
        low=low,
        desk_biases={"a": "bullish"},
        ticker="NTNX",
        decision_ts="2026-08-22T00:00:00",
    )
    assert out is decision
    assert out.stop_loss == 0.0
    assert out.recommended_size_pct == 0.0


def test_recommended_size_zero_without_vol():
    assert recommended_watch_size_pct(
        symbol="NTNX",
        decision_ts="2026-08-22T00:00:00",
        watch_side="long",
        expected_return_pct=0.08,
        horizon_days=21,
        conviction=0.55,
        realised_vol_annualised=0.0,
    ) == 0.0


def test_guardrail_still_rejects_hold_with_book_size():
    from types import SimpleNamespace

    ok, msg = confidence_guardrail(SimpleNamespace(pydantic=_hold(size_pct_of_book=1.4)))
    assert ok is False
    assert "HOLD" in msg

    watched = _hold(recommended_size_pct=1.4, watch_side="long", stop_loss=64.0, target_price=72.0)
    ok, out = confidence_guardrail(SimpleNamespace(pydantic=watched))
    assert ok is True
    assert out.size_pct_of_book == 0.0
