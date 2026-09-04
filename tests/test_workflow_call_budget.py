"""Structural guard on the lean-workflow LLM-call budget.

The workflow was consolidated from 8 analysts to 4 merged desks, the
Quality Reviewer + Compliance Officer became optional (default off,
async advisory to the PM), and memory + the critic consistency vote
default off / single-pass.  These tests lock in the leaner task graph
so a future edit can't silently re-inflate the per-run LLM-call count.
"""

from __future__ import annotations

import pytest


def _ensure_env(monkeypatch):
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")


def _build(monkeypatch, **kwargs):
    _ensure_env(monkeypatch)
    from trading_crew.crew import TradingCrew

    defaults = dict(ticker="NTNX", debate_rounds=2, risk_rounds=1, memory=False)
    defaults.update(kwargs)
    return TradingCrew(**defaults).crew()


def _roles(crew):
    return [a.role for a in crew.agents]


def test_default_workflow_has_four_analyst_desks(monkeypatch):
    crew = _build(monkeypatch)
    roles = set(_roles(crew))
    # The merged desks — the single biggest call-reduction lever.  The
    # Reports Analyst (primary-filing guidance / target-vs-actual) was added
    # back deliberately on top of the four merged desks.
    for desk in ("Market Analyst", "News Analyst",
                 "Fundamentals Analyst", "Reports Analyst", "Macro Analyst"):
        assert desk in roles, f"missing merged desk {desk!r}"
    # The 4 old narrow desks must be gone.
    for removed in ("Social Analyst", "Geopolitical Analyst",
                    "Sector / Peer Analyst", "Quant / Options Analyst"):
        assert removed not in roles, f"{removed!r} should have been merged away"


def test_default_workflow_excludes_optional_reviewers(monkeypatch):
    crew = _build(monkeypatch)
    roles = _roles(crew)
    assert "Quality Reviewer" not in roles, "Quality Review must be off by default"
    assert "Compliance Officer" not in roles, "Compliance must be off by default"


def test_default_task_count_is_lean(monkeypatch):
    # 5 analysts + (2 debate rounds * 2) + research_synthesis + trader
    # + (1 risk round * 3) + pm = 15 tasks.
    crew = _build(monkeypatch)
    assert len(crew.tasks) == 15


def test_pm_is_terminal_and_sync(monkeypatch):
    crew = _build(monkeypatch)
    pm = crew.tasks[-1]
    assert pm.agent.role == "Portfolio Manager"
    assert not pm.async_execution, (
        "The PM must be the terminal SYNC task (CrewAI requires the crew "
        "to end with at most one async task)."
    )


def test_analyst_fan_out_is_async(monkeypatch):
    crew = _build(monkeypatch)
    first_five = crew.tasks[:5]
    assert all(t.async_execution for t in first_five), (
        "Deterministic matches v2: the 5 analyst desks fan out (async_execution)."
    )


def test_debate_is_sync_and_does_not_redump_reports(monkeypatch):
    """Bull/bear cannot be async: CrewAI forbids an async task whose
    context is the preceding async analyst batch (NTNX ValidationError).

    Debate context stays empty — they share a scratch pad instead of
    redumping the five analyst reports or prior rounds.
    """
    crew = _build(monkeypatch, debate_rounds=2)
    debate = [
        t for t in crew.tasks
        if t.agent.role in ("Bullish Researcher", "Bearish Researcher")
    ]
    assert len(debate) == 4
    assert all(not t.async_execution for t in debate)
    analyst_desks = {
        "Market Analyst", "News Analyst", "Fundamentals Analyst",
        "Reports Analyst", "Macro Analyst",
    }
    def _ctx_roles(task):
        raw = getattr(task, "context", None)
        if not isinstance(raw, (list, tuple)):
            return set()
        return {c.agent.role for c in raw}

    for t in debate:
        ctx = _ctx_roles(t)
        assert ctx.isdisjoint(analyst_desks), (
            f"{t.agent.role} must not receive full analyst reports in context"
        )
        assert "Bullish Researcher" not in ctx

    rm = next(t for t in crew.tasks if t.agent.role == "Research Manager")
    rm_ctx = [c.agent.role for c in (rm.context or [])]
    assert rm_ctx == ["Bullish Researcher", "Bearish Researcher"]

    trader = next(t for t in crew.tasks if t.agent.role == "Trader")
    trader_ctx = [c.agent.role for c in (trader.context or [])]
    assert trader_ctx == ["Research Manager"]


def test_reviewers_present_and_async_when_enabled(monkeypatch):
    crew = _build(
        monkeypatch, quality_review_enabled=True, compliance_enabled=True
    )
    roles = _roles(crew)
    assert "Quality Reviewer" in roles
    assert "Compliance Officer" in roles
    # +2 tasks over the lean default.
    assert len(crew.tasks) == 17
    # Both reviewers run async so they overlap each other before the PM.
    by_role = {t.agent.role: t for t in crew.tasks}
    assert by_role["Quality Reviewer"].async_execution
    assert by_role["Compliance Officer"].async_execution
    # PM still terminal + sync.
    assert crew.tasks[-1].agent.role == "Portfolio Manager"
    assert not crew.tasks[-1].async_execution


def test_quality_review_not_wired_into_trader(monkeypatch):
    """Quality Review is advisory to the PM, not a trader gate — so it
    can overlap the trader/risk stretch instead of blocking it."""
    crew = _build(monkeypatch, quality_review_enabled=True)
    trader = next(t for t in crew.tasks if t.agent.role == "Trader")
    ctx_roles = [c.agent.role for c in (trader.context or [])]
    assert "Quality Reviewer" not in ctx_roles
    # But the PM must see it.
    pm = crew.tasks[-1]
    pm_ctx_roles = [c.agent.role for c in (pm.context or [])]
    assert "Quality Reviewer" in pm_ctx_roles


def test_risk_opening_is_parallel_later_round_is_rebuttal(monkeypatch):
    """Round-1 wings are independent first thoughts; later conservative
    waits on this-round aggressive the way later bear waits on bull."""
    crew = _build(monkeypatch, debate_rounds=1, risk_rounds=2)
    wings = [
        t for t in crew.tasks
        if t.agent.role in (
            "Aggressive Risk Analyst", "Conservative Risk Analyst",
        )
    ]
    assert len(wings) == 4
    r1_aggr, r1_cons, r2_aggr, r2_cons = wings

    def _ctx_roles(task):
        raw = getattr(task, "context", None)
        if not isinstance(raw, (list, tuple)):
            return set()
        return {c.agent.role for c in raw}

    assert r1_aggr.async_execution and r1_cons.async_execution
    assert r1_aggr not in (r1_cons.context or [])
    assert "Aggressive Risk Analyst" not in _ctx_roles(r1_cons)
    assert not r2_aggr.async_execution
    assert not r2_cons.async_execution
    assert r2_aggr in (r2_cons.context or [])
