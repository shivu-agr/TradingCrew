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
    _price_at_or_after,
    _signed_realised_return,
    resolve_pending_episodes,
)
from trading_crew.agentic.resolve_investigate import InvestigationPlan, ToolCallSpec


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

    def call(self, messages, response_format=None, **kwargs):
        self.calls.append({"messages": messages, "response_format": response_format})
        if response_format is InvestigationPlan:
            return InvestigationPlan(need_tools=False, calls=[])
        return self.response


class _NoResponseFormatLLM:
    """Mimics a provider that rejects ``response_format`` and returns JSON text."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls: List[dict] = []

    def call(self, messages, **kwargs):
        if "response_format" in kwargs:
            raise TypeError("call() got an unexpected keyword argument 'response_format'")
        self.calls.append({"messages": messages})
        return self.payload


def test_generate_reflection_compat_path_parses_json_text():
    """When the LLM rejects response_format we retry with a JSON hint + parse."""
    from trading_crew.agentic.reflection import _generate_reflection

    payload = (
        'Here you go:\n```json\n'
        '{"summary": "News catalyst was mispriced.", '
        '"confidence_score": 0.7, "key_lesson_tag": "news_overweight", '
        '"desks": ["news_analyst"], "error_class": "missed_catalyst", '
        '"recommended_change": "Discount faded catalysts."}\n```'
    )
    llm = _NoResponseFormatLLM(payload)
    ep = Episode(
        episode_id="X-1", symbol="X", decision_ts="2023-01-01T00:00:00+00:00",
        state_summary="tech breakout + news catalyst", regime=Regime.UNKNOWN,
        action_proposal={"side": "BUY", "target_weight": 0.1, "conviction_score": 0.6, "horizon_days": 90},
        outcome_ts="2023-04-01T00:00:00+00:00", outcome_status=OutcomeStatus.PENDING,
    )
    resp = _generate_reflection(ep, 0.05, 0.02, -0.03, 0.01, llm)
    assert "News catalyst" in resp.summary
    assert resp.confidence_score == 0.7
    assert resp.key_lesson_tag == "news_overweight"
    assert resp.desks == ["news_analyst"]
    assert resp.error_class == "missed_catalyst"
    # The JSON hint must have been appended to the retried prompt.
    assert "JSON object" in llm.calls[0]["messages"][0]["content"]


def test_extract_json_object_handles_prose_and_fences():
    from trading_crew.agentic.reflection import _extract_json_object

    assert _extract_json_object('noise {"a": 1} tail') == {"a": 1}
    assert _extract_json_object('```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    assert _extract_json_object("no json here") is None


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


def test_resolve_stores_tape_reason(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="BUY"))

    def fake_reason(ticker, outcome_date, market_return, side):
        assert ticker == "NTNX"
        return "Guidance cut [source: https://example.com/8k]"

    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.10, "SPY": 0.03}),
        skip_llm=True,
        skip_tape_reason=False,
        tape_reason_fn=fake_reason,
    )
    assert records[0].market_reason.startswith("Guidance cut")
    persisted = mem.all_episodes()[0]
    assert persisted.market_reason.startswith("Guidance cut")


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
    assert persisted.outcome_label == "WIN"
    assert persisted.market_return is not None


def test_resolve_calls_llm_and_persists_reflection(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="BUY"))
    llm = _StubLLM()
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.10, "SPY": 0.03}),
        llm=llm,
        skip_tape_reason=True,
    )
    assert len(llm.calls) >= 2
    reflect_calls = [
        c for c in llm.calls if c["response_format"] is ReflectionResponse
    ]
    assert reflect_calls
    prompt_text = reflect_calls[0]["messages"][0]["content"]
    assert "Realised return on the position:" in prompt_text
    assert "Benchmark (SPY) return" in prompt_text
    assert "NTNX" in prompt_text
    assert "Outcome label" in prompt_text

    r = records[0]
    assert r.status == "RESOLVED"
    assert r.reflection is not None
    # The tag prefix must survive serialisation so retrieval can cluster
    # on it.
    assert r.reflection.startswith("[stop_too_loose")
    persisted = mem.all_episodes()[0]
    assert persisted.reflection == r.reflection
    assert persisted.outcome_label == "WIN"


def test_resolve_persists_structured_reflection_meta(tmp_path):
    """The structured reflection (desks / error_class) is written to the
    episode so cross-desk feedback routing can use it."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(side="BUY"))
    llm = _StubLLM(response=ReflectionResponse(
        summary="Sizing was too large for the realised volatility.",
        confidence_score=0.7,
        key_lesson_tag="oversized",
        desks=["risk"],
        error_class="sizing_too_large",
        recommended_change="Halve size when ATR is elevated.",
    ))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.10, "SPY": 0.03}),
        llm=llm,
        skip_tape_reason=True,
    )
    meta = records[0].reflection_meta
    assert meta is not None
    assert meta["desks"] == ["risk"]
    assert meta["error_class"] == "sizing_too_large"
    persisted = mem.all_episodes()[0]
    assert persisted.reflection_meta["error_class"] == "sizing_too_large"
    # Round-trips through the JSONL store.
    assert persisted.reflection_meta["desks"] == ["risk"]


def test_reflection_response_drops_unknown_desks():
    """The validator keeps only canonical desk tags (never invents a desk)."""
    resp = ReflectionResponse(
        summary="x", confidence_score=0.5, key_lesson_tag="t",
        desks=["news_analyst", "not_a_desk", "RISK"], error_class="bogus",
    )
    assert resp.desks == ["news_analyst", "risk"]
    # An out-of-vocab error_class coerces to 'noise'.
    assert resp.error_class == "noise"


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


def test_price_at_or_after_uses_calendar_day_not_intraday_clock():
    """A Friday 02:54Z review must still hit that Friday's midnight bar."""
    dates = pd.date_range("2026-08-17", periods=5, freq="D")  # Mon–Fri
    df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 5,
        "Close": [101.0] * 5,
        "High": [102.0] * 5,
    })
    hit = _price_at_or_after(df, "2026-08-21T02:54:39+00:00")
    assert hit is not None
    assert hit[0] == "2026-08-21"
    assert _price_at_or_after(df, "2026-08-21T23:59:59+00:00")[0] == "2026-08-21"


def test_skip_llm_still_attributes_wrong_desk(tmp_path):
    """Even without an LLM, a LOSS is labelled and routed to the desk whose
    lean disagreed with the tape."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    ep = _episode(side="BUY")
    ep.desk_leans = {"news_analyst": "bullish", "macro_analyst": "bearish"}
    mem.add(ep)
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
        # Negative name return → BUY is a LOSS; tape is bearish.
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": -0.12, "SPY": 0.0}),
        skip_llm=True,
    )
    r = records[0]
    assert r.status == "RESOLVED"
    assert r.outcome_label == "LOSS"
    meta = r.reflection_meta or {}
    assert "news_analyst" in meta.get("desks", [])
    by_desk = {d["desk"]: d for d in meta.get("desk_lessons") or []}
    assert by_desk["news_analyst"]["verdict"] == "hurt"
    assert by_desk["macro_analyst"]["verdict"] == "helped"
    persisted = mem.all_episodes()[0]
    assert persisted.outcome_label == "LOSS"


def test_short_review_resolves_while_thesis_stays_pending(tmp_path):
    """A 180-day thesis can still teach at 30 days."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_episode(
        decision_ts="2026-05-01T00:00:00+00:00",
        outcome_ts="2026-11-01T00:00:00+00:00",
        side="BUY",
    ))
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 6, 10, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": 0.04, "SPY": 0.01}),
        skip_llm=True,
    )
    assert any(r.status == "REVIEW_RESOLVED" for r in records)
    assert not any(r.status == "RESOLVED" for r in records)
    persisted = mem.all_episodes()[0]
    assert persisted.outcome_status == OutcomeStatus.PENDING
    assert persisted.review_status == "RESOLVED"
    assert persisted.review is not None
    assert persisted.horizon_scores and "short" in persisted.horizon_scores
    lessons = (persisted.review.get("reflection_meta") or {}).get("desk_lessons") or []
    assert isinstance(lessons, list)


def test_resolve_investigation_writes_review_lessons(tmp_path, monkeypatch):
    """Resolve LLM may call tools, then write why-wrong fields for Hermes."""
    calls = []

    def _fake_tool(name, arguments):
        calls.append((name, arguments))
        return "Guidance cut in the 8-K [source: https://example.com/8k]"

    monkeypatch.setattr(
        "trading_crew.agentic.resolve_investigate._invoke_tool", _fake_tool,
    )

    class _InvestigateLLM:
        def __init__(self):
            self.prompts: List[str] = []

        def call(self, messages, response_format=None, **kwargs):
            text = messages[0]["content"]
            self.prompts.append(text)
            if response_format is InvestigationPlan:
                return InvestigationPlan(
                    need_tools=True,
                    tape_question="Why did NTNX fall after the call?",
                    calls=[ToolCallSpec(
                        tool="research_web",
                        arguments={"query": "NTNX guidance cut"},
                        why="catalyst",
                    )],
                )
            return ReflectionResponse(
                summary="Reports over-weighted a stale 10-Q; the 8-K cut guidance.",
                confidence_score=0.7,
                key_lesson_tag="filings_overweight",
                desks=["reports_analyst"],
                error_class="missed_catalyst",
                recommended_change="Discount stale filings vs the latest 8-K.",
                market_reason="Guidance cut in the 8-K.",
                unhelpful_tools=["get_report_insights"],
                overemphasized=["filings"],
                suggested_playbook="skip_reports",
            )

    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    ep = _episode(
        decision_ts="2026-05-01T00:00:00+00:00",
        outcome_ts="2026-11-01T00:00:00+00:00",
        side="BUY",
    )
    ep.desk_leans = {"reports_analyst": "bullish", "macro_analyst": "bearish"}
    mem.add(ep)
    llm = _InvestigateLLM()
    records = resolve_pending_episodes(
        memory=mem,
        now=datetime(2026, 6, 10, tzinfo=timezone.utc),
        fetch_ohlcv=_ohlcv_stub_factory({"NTNX": -0.12, "SPY": 0.0}),
        llm=llm,
        skip_tape_reason=True,
    )
    assert any(r.status == "REVIEW_RESOLVED" for r in records)
    assert calls == [("research_web", {"query": "NTNX guidance cut"})]
    assert any("Investigation briefing" in p and "Guidance cut" in p for p in llm.prompts)
    persisted = mem.all_episodes()[0]
    meta = (persisted.review or {}).get("reflection_meta") or {}
    assert meta.get("unhelpful_tools") == ["get_report_insights"]
    assert meta.get("overemphasized") == ["filings"]
    assert meta.get("suggested_playbook") == "skip_reports"
    assert "Guidance cut" in (meta.get("market_reason") or "")
    assert (meta.get("investigation") or {}).get("need_tools") is True


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
