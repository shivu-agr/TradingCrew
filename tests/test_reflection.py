"""Tests for the outcome-resolution + reflection module (agentic training L2).

Strategy:
- Use a tmp-path JSONL store so no shared state leaks between tests.
- Inject a deterministic OHLCV stub (no yfinance over the wire).
- Inject a stub LLM that records what it was asked and returns a known
  ReflectionResponse — proves the prompt is well-formed and the writeback
  loop is correct without spending tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, List

import pandas as pd
import pytest

from trading_crew.agentic.memory import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
    Regime,
)
from trading_crew.agentic.reflection import (
    ReflectionResponse,
    ResolutionRecord,
    _max_drawdown_over_holding,
    _signed_realised_return,
    resolve_pending_episodes,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubLLM:
    """Records call args; returns a fixed ReflectionResponse for assertions."""

    def __init__(self, *, response: ReflectionResponse | None = None) -> None:
        self.calls: List[dict] = []
        self.response = response or ReflectionResponse(
            summary="The long thesis was vindicated by the earnings beat, but the stop sat below the prior support pivot and would have been hit had the post-print drift stalled. Tighten size next time.",
            confidence_score=0.65,
            key_lesson_tag="stop_too_loose",
        )

    def call(self, messages, response_format):
        self.calls.append({"messages": messages, "response_format": response_format})
        return self.response


def _ohlcv_stub_factory(symbol_returns: dict[str, float]):
    """Return a fake _fetch_ohlcv that produces a deterministic price series
    so the math is exactly verifiable.

    symbol_returns maps symbol -> total return over the window.  The stub
    builds a linearly interpolated daily close series so the start/end
    closes match the requested return, with a small dip in the middle to
    give max_drawdown a non-trivial value.
    """
    def _stub(symbol: str, end_date: datetime, lookback_days: int) -> pd.DataFrame:
        if symbol not in symbol_returns:
            return pd.DataFrame()
        ret = symbol_returns[symbol]
        # 35 trading-day window covering [end-30d, end+5d], generous enough
        # so _price_at_or_after lands on the correct day for any test date.
        start = end_date - timedelta(days=lookback_days)
        dates = pd.date_range(start=start, periods=lookback_days, freq="D")
        start_px = 100.0
        end_px = start_px * (1.0 + ret)
        n = len(dates)
        closes = [start_px + (end_px - start_px) * (i / max(n - 1, 1)) for i in range(n)]
        # Inject a 5% dip at the 60% mark so drawdown math has something to find.
        dip_idx = int(n * 0.6)
        closes[dip_idx] *= 0.95
        df = pd.DataFrame({
            "Date": dates,
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n,
        })
        return df
    return _stub


def _episode(
    symbol: str = "NTNX",
    decision_ts: str = "2026-05-01T12:00:00",
    outcome_ts: str = "2026-06-01T12:00:00",
    side: str = "BUY",
    status: OutcomeStatus = OutcomeStatus.PENDING,
) -> Episode:
    return Episode(
        episode_id=f"{symbol}-{decision_ts}",
        symbol=symbol,
        decision_ts=decision_ts,
        state_summary="Strong demand growth + improving margins; technicals trending.",
        regime=Regime.TREND,
        action_proposal={
            "side": side,
            "target_weight": 0.05,
            "conviction_score": 0.7,
            "horizon_days": 21,
        },
        outcome_ts=outcome_ts,
        outcome_status=status,
        embargo_days=21,
    )


# ---------------------------------------------------------------------------
# Pure helpers (no IO)
# ---------------------------------------------------------------------------


def test_signed_realised_return_buy_long():
    assert _signed_realised_return("BUY", 100.0, 110.0) == pytest.approx(0.10)


def test_signed_realised_return_sell_short():
    # Short going from 100 to 90 = +10% PnL on the short
    assert _signed_realised_return("SELL", 100.0, 90.0) == pytest.approx(0.1111, abs=1e-3)


def test_signed_realised_return_abstain_zero():
    # No position taken — no PnL contribution to Sharpe/Sortino
    assert _signed_realised_return("ABSTAIN", 100.0, 200.0) == 0.0


def test_signed_realised_return_handles_zero_entry():
    assert _signed_realised_return("BUY", 0.0, 110.0) == 0.0


# ---------------------------------------------------------------------------
# Resolution sweep — full pipeline with stubs
# ---------------------------------------------------------------------------


def test_resolve_skips_future_outcomes(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    # Outcome 30 days in the future relative to test "now"
    mem.add(_episode(outcome_ts="2099-01-01T00:00:00"))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.05}),
        skip_llm=True,
    )
    assert len(records) == 1
    assert records[0].status == "SKIPPED_PENDING"
    # The episode should remain PENDING — nothing was written.
    assert mem.all_episodes()[0].outcome_status == OutcomeStatus.PENDING


def test_resolve_writes_outcome_without_llm(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="BUY"))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.10, "SPY": 0.03}),
        skip_llm=True,
    )
    assert len(records) == 1
    r = records[0]
    assert r.status == "RESOLVED"
    assert r.realised_return is not None and r.realised_return > 0.05  # ~10%
    assert r.alpha_return is not None and r.alpha_return < r.realised_return  # alpha < raw
    assert r.reflection is None  # skip_llm
    persisted = mem.all_episodes()[0]
    assert persisted.outcome_status == OutcomeStatus.RESOLVED


def test_resolve_calls_llm_and_persists_reflection(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="BUY"))
    llm = _StubLLM()
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.10, "SPY": 0.03}),
        llm=llm,
    )
    assert len(llm.calls) == 1
    # Prompt must reference the actual realised number, not a placeholder.
    prompt_text = llm.calls[0]["messages"][0]["content"]
    assert "Realised return on the position:" in prompt_text
    assert "Benchmark (SPY) return" in prompt_text
    assert "NTNX" in prompt_text

    r = records[0]
    assert r.status == "RESOLVED"
    assert r.reflection is not None
    # The tag prefix must survive serialisation so retrieval can cluster
    # on it.
    assert r.reflection.startswith("[stop_too_loose")
    persisted = mem.all_episodes()[0]
    assert persisted.reflection == r.reflection


def test_resolve_sell_inverts_pnl(tmp_path):
    """A short on a falling market = positive realised return."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="SELL"))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": -0.10, "SPY": 0.0}),
        skip_llm=True,
    )
    r = records[0]
    assert r.realised_return is not None and r.realised_return > 0.0


def test_resolve_ticker_filter_only_touches_matching(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(symbol="NTNX", decision_ts="2026-05-01T00:00:00",
                     outcome_ts="2026-06-01T00:00:00"))
    mem.add(_episode(symbol="MSFT", decision_ts="2026-05-01T00:00:00",
                     outcome_ts="2026-06-01T00:00:00"))
    records = resolve_pending_episodes(
        memory=mem,
        ticker="NTNX",
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.05, "SPY": 0.02}),
        skip_llm=True,
    )
    assert len(records) == 1
    assert records[0].symbol == "NTNX"
    # MSFT episode must remain PENDING
    msft = [e for e in mem.all_episodes() if e.symbol == "MSFT"][0]
    assert msft.outcome_status == OutcomeStatus.PENDING


def test_resolve_abandons_when_no_data_and_window_passed(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(symbol="GHOST", outcome_ts="2026-01-01T00:00:00"))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),  # 7 months later
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.05, "SPY": 0.02}),  # no GHOST
        skip_llm=True,
        abandon_after_days=14,
    )
    assert records[0].status == "ABANDONED"
    persisted = mem.all_episodes()[0]
    assert persisted.outcome_status == OutcomeStatus.ABANDONED


def test_resolve_idempotent_on_already_resolved(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    ep = _episode(side="BUY")
    ep.outcome_status = OutcomeStatus.RESOLVED
    ep.realised_return = 0.07
    mem.add(ep)
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.99, "SPY": 0.0}),
        skip_llm=True,
    )
    # RESOLVED episodes are skipped entirely (no record emitted).
    assert records == []
    persisted = mem.all_episodes()[0]
    assert persisted.realised_return == pytest.approx(0.07)


def test_max_drawdown_long_position():
    """A long position with a mid-window 5% dip should report ~-5% mdd."""
    dates = pd.date_range("2026-05-01", periods=30, freq="D")
    closes = [100.0 + i for i in range(30)]
    closes[20] = closes[20] * 0.95  # 5% dip on day 20
    df = pd.DataFrame({"Date": dates, "Close": closes})
    mdd = _max_drawdown_over_holding(df, "2026-05-01", "2026-05-30", "BUY")
    assert mdd is not None
    assert -0.10 < mdd <= 0.0


def test_max_drawdown_short_position_inverts():
    """For a short, the same falling price series produces NO drawdown
    (the position is profitable on the way down)."""
    dates = pd.date_range("2026-05-01", periods=10, freq="D")
    closes = [100.0 - i for i in range(10)]  # monotonically falling
    df = pd.DataFrame({"Date": dates, "Close": closes})
    mdd = _max_drawdown_over_holding(df, "2026-05-01", "2026-05-10", "SELL")
    # Short on a falling series = no drawdown
    assert mdd == pytest.approx(0.0, abs=1e-6)


def test_resolution_record_dict_roundtrip():
    r = ResolutionRecord(
        episode_id="X", symbol="X", decision_ts="2026-01-01",
        outcome_ts="2026-02-01", side="BUY",
        realised_return=0.05, alpha_return=0.02, max_drawdown=-0.03,
        reflection="hello", status="RESOLVED",
    )
    d = r.to_dict()
    assert d["realised_return"] == 0.05
    assert d["status"] == "RESOLVED"
