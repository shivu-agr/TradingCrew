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
    # The 4 merged desks — the single biggest call-reduction lever.
    for desk in ("Market Analyst", "News Analyst",
                 "Fundamentals Analyst", "Macro Analyst"):
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
    # 4 analysts + (2 debate rounds * 2) + research_synthesis + trader
    # + (1 risk round * 3) + pm = 14 tasks.
    crew = _build(monkeypatch)
    assert len(crew.tasks) == 14


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
    first_four = crew.tasks[:4]
    assert all(t.async_execution for t in first_four), (
        "The 4 analyst desks must fan out in parallel (async_execution)."
    )


def test_reviewers_present_and_async_when_enabled(monkeypatch):
    crew = _build(
        monkeypatch, quality_review_enabled=True, compliance_enabled=True
    )
    roles = _roles(crew)
    assert "Quality Reviewer" in roles
    assert "Compliance Officer" in roles
    # +2 tasks over the lean default.
    assert len(crew.tasks) == 16
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
