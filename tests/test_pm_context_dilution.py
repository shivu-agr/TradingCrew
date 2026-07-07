"""Regression guard for the pipeline-dilution fix.

The user observed that by the time information reached the final
decision it had been diluted.  Root cause: the Portfolio Manager's
context only ever saw the Research Manager's 3-driver / 3-risk
compression, never the raw analyst lenses (news, geopolitical, macro,
sector, …).  The fix adds ``analyst_tasks`` directly to the PM context
so the final decision sees the full multi-lens evidence.

Post-consolidation there are FOUR merged analyst desks (was 8):
market/positioning, news/sentiment, fundamentals/sector, macro/geopolitical.
"""

from __future__ import annotations

import pytest


def _ensure_env(monkeypatch):
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")


# The unique opening phrase of each of the 4 merged analyst task
# descriptions.
_ANALYST_MARKERS = [
    "Analyze the price action",       # market_task (price + positioning)
    "Find material news catalysts",   # news_task (news + sentiment)
    "Evaluate the fundamentals",      # fundamentals_task (fundamentals + sector)
    "Assess the macro regime",        # macro_task (macro + geopolitical)
]


def test_pm_context_includes_raw_analyst_reports(monkeypatch):
    """The PM decision task must have every analyst report in its
    ``context`` so the final call sees the raw multi-lens evidence, not
    just the manager's compressed synthesis."""
    _ensure_env(monkeypatch)
    from trading_crew.crew import TradingCrew

    # Enable compliance so we can assert it is threaded into the PM
    # context when on (it's advisory + optional / default off now).
    crew = TradingCrew(
        ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=False,
        compliance_enabled=True,
    ).crew()

    # The PM decision is the terminal task.
    pm = crew.tasks[-1]
    ctx_descriptions = [t.description for t in (pm.context or [])]

    for marker in _ANALYST_MARKERS:
        assert any(marker in d for d in ctx_descriptions), (
            f"PM context is missing the analyst report identified by "
            f"{marker!r} — the final decision would only see the "
            "manager's compressed synthesis (information dilution)."
        )

    # When enabled, the compliance review is an advisory PM-context input.
    assert any("compliance" in d.lower() for d in ctx_descriptions), (
        "PM context must include the compliance review when it is enabled."
    )
