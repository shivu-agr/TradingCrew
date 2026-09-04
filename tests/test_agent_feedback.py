"""Phase 3 — desk-feedback routing: lessons reach the right agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from trading_crew.agentic import agent_feedback as af


@dataclass
class _Row:
    symbol: str
    decision_ts: str
    action: str
    realised_return: float
    alpha_return: Optional[float]
    reward: float
    reflection: Optional[str]
    desks: List[str] = field(default_factory=list)
    error_class: Optional[str] = None
    recommended_change: Optional[str] = None
    outcome_label: Optional[str] = None
    desk_lessons: List[dict] = field(default_factory=list)


def _rows():
    return [
        _Row("AAA", "2024-01-01", "BUY", 0.05, 0.02, 0.9,
             "Entered on a strong news catalyst that faded; sentiment was overweighted.",
             desks=["news_analyst"], error_class="missed_catalyst"),
        _Row("BBB", "2024-02-01", "SELL", -0.08, -0.03, 0.8,
             "Position sizing was too large for the volatility; drawdown breached the risk budget.",
             desks=["risk"], error_class="sizing_too_large"),
        _Row("CCC", "2024-03-01", "BUY", 0.01, 0.0, 0.1,
             "Valuation looked cheap but earnings guidance was weak; fundamental read was off.",
             desks=["fundamentals_analyst"], error_class="thesis_wrong"),
        _Row("DDD", "2024-04-01", "HOLD", 0.0, 0.0, 0.05,
             "Macro regime shifted with an RBI rate move we ignored.",
             desks=["macro_analyst"], error_class="macro_regime_missed"),
    ]


def test_expand_desks_fans_out_risk():
    assert af.expand_desks(["risk"]) == {
        "risk_aggressive", "risk_neutral", "risk_conservative",
    }
    # Unknown / blank tags are dropped, never invented.
    assert af.expand_desks(["news_analyst", "", "bogus_desk"]) == {"news_analyst"}


def test_news_lesson_routes_to_news_analyst_only_where_relevant():
    rows = _rows()
    news = af.build_role_feedback(rows, "news_analyst", k=3)
    assert "catalyst" in news.lower()
    assert "AAA" in news
    # A pure sizing lesson should NOT show up on the news desk.
    assert "BBB" not in news
    # The error_class tag is surfaced for the desk.
    assert "missed_catalyst" in news


def test_sizing_lesson_routes_to_all_risk_desks():
    rows = _rows()
    for desk in ("risk_aggressive", "risk_neutral", "risk_conservative"):
        block = af.build_role_feedback(rows, desk, k=3)
        assert "BBB" in block, f"{desk} should see the sizing lesson"
        assert "drawdown" in block.lower()


def test_fundamentals_and_macro_routing():
    rows = _rows()
    fund = af.build_role_feedback(rows, "fundamentals_analyst", k=3)
    assert "CCC" in fund and "valuation" in fund.lower()
    macro = af.build_role_feedback(rows, "macro_analyst", k=3)
    assert "DDD" in macro and "rbi" in macro.lower()


def test_no_relevant_lessons_returns_empty():
    # A lesson tagged to no desk (pure noise) reaches nobody.
    rows = [_Row("ZZZ", "2024-05-01", "HOLD", 0.0, 0.0, 0.0, "Nothing notable.",
                 desks=[], error_class="noise")]
    assert af.build_role_feedback(rows, "news_analyst") == ""


def test_ranking_prefers_high_absolute_reward():
    rows = [
        _Row("LOW", "2024-01-01", "BUY", 0.01, 0.0, 0.05, "news catalyst small win",
             desks=["news_analyst"]),
        _Row("HIGH", "2024-01-02", "SELL", -0.2, -0.1, 0.95, "news sentiment big miss",
             desks=["news_analyst"]),
    ]
    block = af.build_role_feedback(rows, "news_analyst", k=1)
    assert "HIGH" in block and "LOW" not in block


def test_descriptive_lesson_reaches_the_hurt_desk_only():
    row = _Row(
        "EEE", "2024-06-01", "BUY", -0.12, -0.09, 1.1,
        "Global summary the PM already saw.",
        desks=["portfolio_manager"],
        error_class="thesis_wrong",
        outcome_label="LOSS",
        desk_lessons=[{
            "desk": "news_analyst",
            "lean_at_decision": "bullish",
            "verdict": "hurt",
            "what_they_got_wrong": "Treated a faded catalyst as fresh demand.",
            "recommended_change": "Require a second source before a BULLISH news lean.",
        }],
    )
    news = af.build_role_feedback([row], "news_analyst", k=3)
    assert "EEE" in news
    assert "LOSS" in news
    assert "your lean was bullish" in news
    assert "faded catalyst" in news
    assert "Require a second source" in news
    # The PM-tagged desks list must not dump this lesson onto fundamentals.
    assert af.build_role_feedback([row], "fundamentals_analyst") == ""
    # A helped desk with no instructional text is not pestered.
    helped = _Row(
        "EEE", "2024-06-01", "BUY", -0.12, -0.09, 1.1, "",
        desks=["news_analyst"],
        desk_lessons=[{
            "desk": "macro_analyst",
            "lean_at_decision": "bearish",
            "verdict": "helped",
            "what_they_got_wrong": "",
            "recommended_change": "",
        }],
    )
    assert af.build_role_feedback([helped], "macro_analyst") == ""
