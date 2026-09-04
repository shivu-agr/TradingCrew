"""v3 LangGraph routers: skip reports, extra debate, Neutral reopen."""

from __future__ import annotations

from trading_crew.graph.routers import (
    after_debate,
    after_risk,
    debate_rounds_from_action,
    evidence_budget_from_action,
    preview_role_order,
    risk_rounds_from_action,
)
from trading_crew.graph.state import RunState, leans_are_split, parse_stance


def test_after_debate_drains_when_queued_and_budget_left():
    st = RunState(ticker="NTNX", debate_rounds_target=2, debate_round=1,
                  evidence_queued=True, evidence_budget=2, evidence_used=0)
    assert after_debate(st) == "drain_evidence"


def test_after_debate_loops_until_target():
    st = RunState(ticker="NTNX", debate_rounds_target=2, debate_round=1)
    assert after_debate(st) == "debate"
    st.debate_round = 2
    assert after_debate(st) == "research_manager"


def test_after_debate_extra_on_split_once():
    st = RunState(
        ticker="NTNX", debate_rounds_target=2, debate_round=2,
        desk_split=True, extra_debate_used=False,
    )
    assert after_debate(st) == "extra_debate"
    st.extra_debate_used = True
    assert after_debate(st) == "research_manager"


def test_after_debate_extra_on_mixed_stance():
    st = RunState(
        ticker="NTNX", debate_rounds_target=1, debate_round=1,
        stance="MIXED", extra_debate_used=False,
    )
    assert after_debate(st) == "extra_debate"


def test_after_risk_reopens_trader_once():
    st = RunState(ticker="NTNX")
    st.risk_notes["Neutral Risk Analyst"] = "Size is too large.\nREOPEN_TRADER\n"
    assert after_risk(st) == "reopen_trader"
    st.trader_reopened = True
    assert after_risk(st) == "portfolio_manager"


def test_after_risk_respects_no_trader_reopen():
    st = RunState(ticker="NTNX", playbook_action="no_trader_reopen")
    st.risk_notes["Neutral Risk Analyst"] = "REOPEN_TRADER"
    assert after_risk(st) == "portfolio_manager"


def test_after_risk_routes_to_reviewers_when_enabled():
    st = RunState(ticker="NTNX", quality_review_enabled=True)
    assert after_risk(st) == "reviewers"
    st.quality_review_enabled = False
    st.compliance_enabled = True
    assert after_risk(st) == "reviewers"


def test_after_reopen_then_reviewers():
    from trading_crew.graph.routers import after_reopen

    st = RunState(ticker="NTNX", trader_reopened=True, quality_review_enabled=True)
    assert after_reopen(st) == "reviewers"
    st.quality_review_enabled = False
    assert after_reopen(st) == "portfolio_manager"


def test_use_stock_graph_honors_run_mode():
    from trading_crew.graph.routers import use_stock_graph

    assert use_stock_graph("stock", "autonomous") is True
    assert use_stock_graph("stock", "deterministic") is False
    assert use_stock_graph("commodity", "autonomous") is False
    assert use_stock_graph("commodity", "deterministic") is False
    assert use_stock_graph("stock", "") is True
    assert use_stock_graph("stock", "harness") is False


def test_preview_skips_reports_desk():
    roles = preview_role_order(skip_reports=True, debate_rounds_target=1,
                               allow_extra_debate=False, allow_trader_reopen=False)
    assert "Reports Analyst" not in roles
    assert roles.count("Bullish Researcher") == 1
    assert roles.count("Aggressive Risk Analyst") == 1
    assert "Quality Reviewer" not in roles
    assert "Compliance Officer" not in roles
    assert roles[-1] == "Portfolio Manager"


def test_preview_repeats_risk_and_includes_reviewers():
    roles = preview_role_order(
        skip_reports=False,
        debate_rounds_target=1,
        allow_extra_debate=False,
        allow_trader_reopen=False,
        risk_rounds=2,
        quality_review=True,
        compliance=True,
    )
    assert roles.count("Aggressive Risk Analyst") == 2
    assert roles.count("Neutral Risk Analyst") == 2
    assert roles.index("Quality Reviewer") < roles.index("Portfolio Manager")
    assert roles.index("Compliance Officer") < roles.index("Portfolio Manager")
    assert roles[-1] == "Portfolio Manager"


def test_action_knobs():
    assert debate_rounds_from_action("debate_rounds_1", 2) == 1
    assert debate_rounds_from_action("debate_rounds_2", 1) == 2
    assert evidence_budget_from_action("evidence_budget_0") == 0
    assert evidence_budget_from_action("evidence_budget_1") == 1
    assert risk_rounds_from_action("risk_rounds_2", 1) == 2
    assert risk_rounds_from_action("default", 1) == 1


def test_langgraph_partial_update_keeps_ticker():
    """A debate node returns only ``evidence_queued``; ticker must survive."""
    from langgraph.graph import END, START, StateGraph

    from trading_crew.graph.state import GraphState

    def debate_bear(s):
        return {"evidence_queued": True}

    def route(s):
        assert s["ticker"] == "NTNX"
        assert s["debate_round"] == 1
        return "done"

    g = StateGraph(GraphState)
    g.add_node("debate_bear", debate_bear)
    g.add_conditional_edges("debate_bear", route, {"done": END})
    g.add_edge(START, "debate_bear")
    out = g.compile().invoke({
        "ticker": "NTNX",
        "debate_round": 1,
        "evidence_queued": False,
    })
    assert out["ticker"] == "NTNX"
    assert out["evidence_queued"] is True
    assert out["debate_round"] == 1


def test_stock_graph_opens_debate_in_parallel():
    from unittest.mock import MagicMock

    from trading_crew.graph.app import compile_stock_graph

    compiled = compile_stock_graph(MagicMock())
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("analysts", "debate_open") in edges
    assert ("analysts", "debate_bull") not in edges
    assert ("debate_bull", "debate_bear") in edges
    assert ("drain_evidence", "debate_bull") in edges
    assert ("mark_extra", "debate_bull") in edges
    for hop in ("drain_evidence", "mark_extra", "debate_bull", "research_manager"):
        assert ("debate_open", hop) in edges
        assert ("debate_bear", hop) in edges


def test_run_opening_debate_uses_bundled_crew():
    from types import SimpleNamespace
    from unittest.mock import patch

    from trading_crew.graph.nodes import GraphRuntime

    edges: list[tuple[str, str, str]] = []
    rt = GraphRuntime(
        SimpleNamespace(ticker="NTNX", tasks_config={}),
        {},
        on_edge=lambda src, dst, kind: edges.append((src, dst, kind)),
    )

    def _fake_bundled(self, prior=""):
        self.emit_edge("Bullish Researcher", "Bearish Researcher", "activated")
        self.emit_edge("Bearish Researcher", "Bullish Researcher", "cycled")
        self.emit_edge("Bullish Researcher", "Research Manager", "activated")
        self.emit_edge("Bearish Researcher", "Research Manager", "activated")
        return "bull-open", "bear-rebut"

    with patch.object(GraphRuntime, "run_bundled_debate", _fake_bundled):
        bull, bear = rt.run_opening_debate(1, 2)
    assert bull == "bull-open"
    assert bear == "bear-rebut"
    assert ("Bullish Researcher", "Bearish Researcher", "activated") in edges
    assert ("Bullish Researcher", "Research Manager", "activated") in edges
    assert ("Bearish Researcher", "Research Manager", "activated") in edges


def test_run_bundled_debate_reads_task_outputs():
    from types import SimpleNamespace
    from unittest.mock import patch

    from trading_crew.graph.nodes import GraphRuntime

    class _TaskOut:
        def __init__(self, raw):
            self.raw = raw

    class _Result:
        tasks_output = [
            _TaskOut("BULL OPEN"),
            _TaskOut("BEAR REBUT"),
            _TaskOut("BULL CLOSE"),
        ]
        raw = "BEAR REBUT"

    rt = GraphRuntime(
        SimpleNamespace(ticker="BYND", tasks_config={}, run_mode="harness"),
        {
            "bull_researcher": object(),
            "bear_researcher": object(),
        },
    )

    def _fmt(self, name, **fmt):
        return f"desc {name} {fmt.get('ticker')}", f"out {name}"

    with patch.object(GraphRuntime, "_format", _fmt), patch(
        "trading_crew.graph.nodes.Task", return_value=object()
    ), patch("trading_crew.graph.nodes.Crew") as crew_cls:
        crew_cls.return_value.kickoff.return_value = _Result()
        bull, bear = rt.run_bundled_debate("prior briefs")
    assert "BULL OPEN" in bull
    assert "BULL CLOSE" in bull
    assert bear == "BEAR REBUT"
    assert "Bear rebuttal" in rt.last_raw


def test_run_bundled_risk_reads_task_outputs():
    from types import SimpleNamespace
    from unittest.mock import patch

    from trading_crew.graph.nodes import GraphRuntime

    class _TaskOut:
        def __init__(self, raw):
            self.raw = raw

    class _Result:
        tasks_output = [
            _TaskOut("AGGR OPEN"),
            _TaskOut("CONS OPEN"),
            _TaskOut("NEUT OPEN"),
            _TaskOut("AGGR REBUT"),
            _TaskOut("CONS REBUT"),
            _TaskOut("NEUT REBUT"),
        ]
        raw = "NEUT REBUT"

    rt = GraphRuntime(
        SimpleNamespace(ticker="BYND", tasks_config={}, run_mode="harness"),
        {
            "risk_aggressive": object(),
            "risk_conservative": object(),
            "risk_neutral": object(),
        },
    )

    def _fmt(self, name, **fmt):
        return f"desc {name} {fmt.get('ticker')}", f"out {name}"

    with patch.object(GraphRuntime, "_format", _fmt), patch(
        "trading_crew.graph.nodes.Task", return_value=object()
    ), patch("trading_crew.graph.nodes.Crew") as crew_cls:
        crew_cls.return_value.kickoff.return_value = _Result()
        aggr, cons, neut = rt.run_bundled_risk("prior briefs")
    assert "AGGR OPEN" in aggr and "AGGR REBUT" in aggr
    assert "CONS OPEN" in cons and "CONS REBUT" in cons
    assert "NEUT OPEN" in neut and "NEUT REBUT" in neut
    assert "Conservative open" in rt.last_raw


def test_run_opening_risk_submits_both_wings_without_handoff():
    from types import SimpleNamespace
    from unittest.mock import patch

    from trading_crew.graph.nodes import GraphRuntime

    edges: list[tuple[str, str, str]] = []
    rt = GraphRuntime(
        SimpleNamespace(ticker="NTNX", tasks_config={}),
        {},
        on_edge=lambda src, dst, kind: edges.append((src, dst, kind)),
    )
    priors: list[tuple[str, str]] = []

    def _fake_format(self, name, **kw):
        return f"{name}", "out"

    def _fake_run(self, key, desc, out, *, prior=""):
        priors.append((key, prior))
        return f"{key}-open"

    with patch.object(GraphRuntime, "_format", _fake_format), patch.object(
        GraphRuntime, "_run_agent", _fake_run
    ):
        notes = rt.run_risk_wings(1, 2, thesis="T", trade_plan="P")
    assert notes["Aggressive Risk Analyst"] == "risk_aggressive-open"
    assert notes["Conservative Risk Analyst"] == "risk_conservative-open"
    assert {k for k, _ in priors} == {"risk_aggressive", "risk_conservative"}
    for _key, prior in priors:
        assert "This-round Aggressive Risk turn" not in prior
        assert "Prior Aggressive Risk turn" not in prior
    assert (
        "Aggressive Risk Analyst",
        "Conservative Risk Analyst",
        "activated",
    ) not in edges
    assert ("Aggressive Risk Analyst", "Neutral Risk Analyst", "activated") in edges
    assert ("Conservative Risk Analyst", "Neutral Risk Analyst", "activated") in edges


def test_later_risk_round_is_sequential_rebuttal():
    from types import SimpleNamespace
    from unittest.mock import patch

    from trading_crew.graph.nodes import GraphRuntime

    edges: list[tuple[str, str, str]] = []
    rt = GraphRuntime(
        SimpleNamespace(ticker="NTNX", tasks_config={}),
        {},
        on_edge=lambda src, dst, kind: edges.append((src, dst, kind)),
    )
    priors: list[tuple[str, str]] = []

    def _fake_format(self, name, **kw):
        return f"{name}", "out"

    def _fake_run(self, key, desc, out, *, prior=""):
        priors.append((key, prior))
        return f"{key}-r2"

    with patch.object(GraphRuntime, "_format", _fake_format), patch.object(
        GraphRuntime, "_run_agent", _fake_run
    ):
        notes = rt.run_risk_wings(
            2,
            2,
            thesis="T",
            trade_plan="P",
            prior_wings={
                "Aggressive Risk Analyst": "aggr-r1",
                "Conservative Risk Analyst": "cons-r1",
            },
        )
    assert notes["Aggressive Risk Analyst"] == "risk_aggressive-r2"
    assert notes["Conservative Risk Analyst"] == "risk_conservative-r2"
    by_key = {k: p for k, p in priors}
    assert "Prior Aggressive Risk turn" in by_key["risk_aggressive"]
    assert "aggr-r1" in by_key["risk_aggressive"]
    assert "This-round Aggressive Risk turn" not in by_key["risk_aggressive"]
    assert "This-round Aggressive Risk turn" in by_key["risk_conservative"]
    assert "risk_aggressive-r2" in by_key["risk_conservative"]
    assert (
        "Aggressive Risk Analyst",
        "Conservative Risk Analyst",
        "activated",
    ) in edges


def test_leans_and_stance():
    assert leans_are_split({"a": "bullish", "b": "bearish"})
    assert not leans_are_split({"a": "bullish", "b": "neutral"})
    assert parse_stance("foo\nSTANCE: MIXED\n") == "MIXED"
    assert parse_stance("STANCE: BULL") == "BULL"


def test_format_pm_decision_keeps_horizon_set_notation():
    """NTNX died at PM: ``str.format`` treated ``{SHORT, MEDIUM}`` as a field."""
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from trading_crew.graph.nodes import GraphRuntime

    tasks = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "trading_crew" / "config" / "tasks.yaml")
        .read_text()
    )
    rt = GraphRuntime(SimpleNamespace(ticker="NTNX", tasks_config=tasks), {})
    desc, _out = rt._format("pm_decision", ticker="NTNX")
    assert "{SHORT, MEDIUM}" in desc
    assert "{ticker}" not in desc
    assert "on NTNX" in desc


def test_pack_prior_skips_empty_and_neutralizes_placeholders():
    from trading_crew.graph.nodes import neutralize_placeholders, pack_prior_outputs

    packed = pack_prior_outputs([
        ("News Analyst report", "OEM hardware shortage"),
        ("Empty desk", ""),
        ("Macro / Geopolitical Analyst report", "tariff risk {ticker} and {action}"),
    ])
    assert "OEM hardware shortage" in packed
    assert "Empty desk" not in packed
    assert "{ ticker}" in neutralize_placeholders("use {ticker} here")
    # CrewAI interpolate_only must not see {action} in stuffed reports.
    assert "{action}" not in neutralize_placeholders("{\"action\": 1} and {action}")


def test_citation_index_lists_unique_source_tags():
    from trading_crew.graph.nodes import extract_source_tags, format_citation_index

    tags = extract_source_tags(
        "cut [source: https://www.cnbctv18.com/deal.htm] and "
        "[source: screener.in #profit-loss]",
        "repeat [source: https://www.cnbctv18.com/deal.htm]",
    )
    assert tags == [
        "https://www.cnbctv18.com/deal.htm",
        "screener.in #profit-loss",
    ]
    idx = format_citation_index(tags)
    assert "Citation index" in idx
    assert "https://www.cnbctv18.com/deal.htm" in idx
    assert "do not tell the PM to drop" in idx
    assert format_citation_index([]) == ""
