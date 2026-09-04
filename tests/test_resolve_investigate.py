"""Resolve investigation tools + playbook vote sanitizer."""

from __future__ import annotations

from trading_crew.agentic.resolve_investigate import (
    InvestigationPlan,
    ToolCallSpec,
    run_investigation,
    sanitize_playbook,
)


def test_sanitize_playbook_drops_unknown():
    assert sanitize_playbook("skip_reports") == "skip_reports"
    assert sanitize_playbook("invent_a_desk") == ""
    assert sanitize_playbook("default") == ""


def test_run_investigation_executes_named_tool(monkeypatch):
    calls = []

    def _fake(name, arguments):
        calls.append((name, arguments))
        return "Guidance cut [source: https://example.com/8k]"

    monkeypatch.setattr(
        "trading_crew.agentic.resolve_investigate._invoke_tool", _fake,
    )
    plan = InvestigationPlan(
        need_tools=True,
        tape_question="Why did NTNX fall?",
        calls=[ToolCallSpec(
            tool="research_web",
            arguments={"query": "NTNX why shares fell"},
            why="catalyst",
        )],
    )
    briefing, executed = run_investigation(
        plan, ticker="NTNX", allow_tools=True,
    )
    assert calls == [("research_web", {"query": "NTNX why shares fell"})]
    assert "Guidance cut" in briefing
    assert executed[0]["tool"] == "research_web"


def test_run_investigation_skips_when_tools_disallowed():
    plan = InvestigationPlan(
        need_tools=True,
        calls=[ToolCallSpec(tool="research_web", arguments={"query": "x"})],
    )
    briefing, executed = run_investigation(
        plan, ticker="NTNX", allow_tools=False,
    )
    assert briefing == ""
    assert executed == []


def test_unknown_tool_dropped_from_plan():
    spec = ToolCallSpec(tool="not_a_tool", arguments={"query": "x"})
    assert spec.tool == ""
