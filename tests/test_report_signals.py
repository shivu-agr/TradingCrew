"""Tests for ``trading_crew.report_signals``.

Covers the tolerant signal-card parser, the confidence-weighted guidance-bias
score, and the deterministic financial-trend computer (screener + yfinance),
all offline (no network — the yfinance path is monkeypatched)."""

from __future__ import annotations

from trading_crew import report_signals as rs


def test_render_pm_ledger_carries_signal_and_is_nondirective():
    """The PM ledger guarantees the parsed guidance signal reaches the decision.
    It must surface NET_GUIDANCE_BIAS + the component signals and demand they be
    weighed — but must NOT prescribe a side (no hardcoded bias)."""
    txt = (
        "## REPORT TREND SIGNALS\n"
        "SIGNAL | REVENUE_GUIDANCE | RAISED | conf=0.80 | mgmt raised FY guide [source: http://x]\n"
        "SIGNAL | NET_GUIDANCE_BIAS | POSITIVE | conf=0.75 | net positive\n"
    )
    ledger = rs.render_pm_ledger(rs.parse_signal_card(txt))
    assert "NET_GUIDANCE_BIAS" in ledger and "POSITIVE" in ledger
    assert "REVENUE_GUIDANCE" in ledger
    assert "do not ignore" in ledger.lower()
    assert "does not prescribe a side" in ledger.lower()
    # It must not inject a trade instruction.
    low = ledger.lower()
    assert "you must buy" not in low and "you must sell" not in low


def test_render_pm_ledger_empty_when_nothing_decisive():
    txt = "SIGNAL | REVENUE_GUIDANCE | INSUFFICIENT_DATA | conf=0.0 | no filings on record\n"
    assert rs.render_pm_ledger(rs.parse_signal_card(txt)) == ""


def test_parse_signal_card_lifts_lines_and_source():
    txt = (
        "## REPORT TREND SIGNALS\n"
        "SIGNAL | OUTLOOK_TONE | BULLISH | conf=0.80 | strong demand [source: http://a]\n"
        "- SIGNAL | REVENUE_GUIDANCE | RAISED | conf=0.7 | guided up\n"
        "SIGNAL | CREDIT_TRAJECTORY | INSUFFICIENT_DATA | conf=0.0 |\n"
        "SIGNAL | NET_GUIDANCE_BIAS | POSITIVE | conf=0.75 | synthesis\n"
        "some prose that is not a signal line\n"
    )
    sigs = rs.parse_signal_card(txt)

    assert len(sigs.signals) == 4
    tone = sigs.get("OUTLOOK_TONE")
    assert tone is not None and tone.direction == "BULLISH"
    assert tone.confidence == 0.8
    assert tone.source == "http://a"
    assert sigs.get("REVENUE_GUIDANCE").direction == "RAISED"  # leading "- " tolerated
    assert sigs.net_bias == "POSITIVE"
    assert sigs.net_confidence == 0.75
    assert not sigs.is_empty()


def test_parse_dedupes_by_name_keeping_first():
    txt = (
        "SIGNAL | OUTLOOK_TONE | BULLISH | conf=0.9 | canonical card\n"
        "SIGNAL | OUTLOOK_TONE | DEFENSIVE | conf=0.2 | analyst paraphrase\n"
    )
    sigs = rs.parse_signal_card(txt)
    assert len(sigs.signals) == 1
    assert sigs.get("OUTLOOK_TONE").direction == "BULLISH"


def test_empty_card_is_empty():
    txt = (
        "SIGNAL | OUTLOOK_TONE | INSUFFICIENT_DATA | conf=0.0 |\n"
        "SIGNAL | NET_GUIDANCE_BIAS | INSUFFICIENT_DATA | conf=0.0 |\n"
    )
    sigs = rs.parse_signal_card(txt)
    assert sigs.is_empty()
    assert rs.guidance_bias_score(sigs) is None


def test_guidance_bias_anchors_on_net_bias_when_confident():
    txt = (
        "SIGNAL | OUTLOOK_TONE | DEFENSIVE | conf=0.9 | cautious\n"
        "SIGNAL | NET_GUIDANCE_BIAS | POSITIVE | conf=0.8 | net positive\n"
    )
    sigs = rs.parse_signal_card(txt)
    # net bias anchor (POSITIVE * 0.8) wins over the component average
    assert rs.guidance_bias_score(sigs) == 0.8


def test_guidance_bias_averages_components_when_net_weak():
    txt = (
        "SIGNAL | OUTLOOK_TONE | BULLISH | conf=1.0 | +1.0\n"
        "SIGNAL | MARGIN_GUIDANCE | CONTRACTING | conf=1.0 | -1.0\n"
        "SIGNAL | NET_GUIDANCE_BIAS | NEUTRAL | conf=0.1 | weak\n"
    )
    sigs = rs.parse_signal_card(txt)
    # net bias conf < 0.3 → fall back to component average: (+1 -1)/2 = 0
    assert rs.guidance_bias_score(sigs) == 0.0


def test_to_float_handles_indian_grouping_and_parens():
    assert rs._to_float("2,35,000") == 235000.0
    assert rs._to_float("(45)") == -45.0
    assert rs._to_float("12.5%") == 12.5
    assert rs._to_float("") is None
    assert rs._to_float("-") is None


class _FakeQuarters:
    rows = [
        ("Sales", ["100", "110", "120", "130", "120", "132", "150", "165", "160"]),
        ("Net Profit", ["10", "11", "12", "13", "12", "14", "16", "18", "20"]),
    ]


def test_compute_from_screener_quarters():
    sigs = rs.compute_financial_trend_signals("RELIANCE.NS", _FakeQuarters())
    names = {s.name: s for s in sigs}
    assert names["REVENUE_TREND"].direction in {"ACCELERATING", "DECELERATING", "STABLE"}
    assert names["EARNINGS_TREND"].confidence == 0.8
    assert names["MARGIN_TREND"].direction in {"EXPANDING", "STABLE", "CONTRACTING"}
    # all carry the computed provenance + flag
    assert all(s.computed for s in sigs)
    assert all("computed" in (s.source or "") for s in sigs)


def test_compute_returns_empty_when_no_numbers(monkeypatch):
    monkeypatch.setattr(rs, "_yfinance_quarterly", lambda t: ([], []))
    assert rs.compute_financial_trend_signals("XYZ", None) == []


def test_insufficient_history_is_not_fabricated():
    class _Thin:
        rows = [("Sales", ["100", "110"])]  # < 5 quarters → no YoY

    sigs = rs.compute_financial_trend_signals("X.NS", _Thin())
    # nothing decisive → empty (never a fabricated direction)
    assert sigs == []


def test_yfinance_path_used_when_no_screener(monkeypatch):
    monkeypatch.setattr(
        rs, "_yfinance_quarterly",
        lambda t: (
            [80, 90, 100, 110, 100, 115, 130, 145, 150],
            [8, 9, 10, 11, 10, 12, 14, 16, 18],
        ),
    )
    sigs = rs.compute_financial_trend_signals("AAPL", None)
    assert {s.name for s in sigs} == {"REVENUE_TREND", "EARNINGS_TREND", "MARGIN_TREND"}
    assert all("yfinance" in (s.source or "") for s in sigs)
