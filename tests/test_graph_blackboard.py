"""Evidence blackboard + bounded request_evidence queue."""

from __future__ import annotations

from trading_crew.agentic.graph.blackboard import (
    Blackboard,
    begin_run,
    current,
    end_run,
)


def test_queue_rejects_unknown_desk():
    bb = Blackboard(ticker="NTNX")
    msg = bb.queue_request("trader", "why?", "get_news")
    assert msg.startswith("Rejected")
    assert bb.requests == []


def test_queue_rejects_wrong_tool():
    bb = Blackboard(ticker="NTNX")
    msg = bb.queue_request("news_analyst", "any buyback?", "get_stock_data")
    assert "not owned" in msg
    assert bb.requests == []


def test_hard_cap_two_requests():
    bb = Blackboard(ticker="NTNX")
    assert "Queued" in bb.queue_request("news_analyst", "q1", "get_news")
    assert "Queued" in bb.queue_request("macro_analyst", "q2", "get_macro_data")
    third = bb.queue_request("market_analyst", "q3", "get_indicators")
    assert third.startswith("Rejected")
    assert len(bb.requests) == 2


def test_contextvar_begin_end():
    end_run()
    assert current() is None
    bb = begin_run("AAPL")
    assert current() is bb
    long_body = "TLDR:\n- foo\nDESK_BIAS: BEARISH\n\n" + ("x" * 2000)
    bb.record_desk("News Analyst", long_body, "bearish")
    assert bb.desk_leans["news_analyst"] == "bearish"
    rendered = bb.render()
    assert "news_analyst" in rendered
    assert "x" * 2000 in rendered, "full render (UI) must not clip the desk report"
    compact = bb.render_compact()
    assert "foo" in compact
    assert "x" * 2000 not in compact, "compact lookup must not dump the full report"
    payload = bb.to_payload()
    assert payload["desks"][0]["report"] == long_body
    end_run()
    assert current() is None
