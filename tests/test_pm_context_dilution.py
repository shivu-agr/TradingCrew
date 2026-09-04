"""Regression guard for the pipeline-dilution fix.

The user observed that by the time information reached the final
decision it had been diluted.  Root cause: the Portfolio Manager's
context only ever saw the Research Manager's 3-driver / 3-risk
compression, never the raw analyst lenses (news, geopolitical, macro,
sector, …).  The fix adds ``analyst_tasks`` directly to the PM context
so the final decision sees the full multi-lens evidence.

Post-consolidation there are FIVE analyst desks (four merged from 8,
plus Reports/Guidance): market/positioning, news/sentiment,
fundamentals/sector, reports/guidance, macro/geopolitical.
"""

from __future__ import annotations

import pytest


def _ensure_env(monkeypatch):
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")


# The unique opening phrase of each of the 5 analyst task
# descriptions.
_ANALYST_MARKERS = [
    "Analyze the price action",       # market_task (price + positioning)
    "Find material news catalysts",   # news_task (news + sentiment)
    "Evaluate the fundamentals",      # fundamentals_task (fundamentals + sector)
    "Read {ticker}'s OWN primary filings",  # reports_task (guidance / target-vs-actual)
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


def test_graph_pm_receives_raw_news_and_macro_reports():
    """v3 one-task Crews have no CrewAI context. If the graph PM node
    does not stuff the analyst reports into its description, the
    decision only sees runner-injected price-structure + guidance
    ledgers (the NTNX BUY that ignored hardware / geo news)."""
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from trading_crew.graph.nodes import GraphRuntime

    tasks = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "trading_crew" / "config" / "tasks.yaml")
        .read_text()
    )
    rt = GraphRuntime(SimpleNamespace(ticker="NTNX", tasks_config=tasks, _pm_task=None), {})
    captured: dict = {}

    def _capture(agent_key, description, expected_output, output_pydantic=None, prior=""):
        captured["desc"] = description
        captured["prior"] = prior
        return "ok"

    rt._run_agent = _capture  # type: ignore[method-assign]
    rt.run_pm(
        reports={
            "news_analyst": "Hardware shortage at a key OEM supplier [source: reuters.com/ntnx-supply]",
            "macro_analyst": "Export-control / geopolitical tariff risk [source: get_geopolitical_news]",
            "market_analyst": "Last close 67.64 [source: yfinance OHLCV]",
        },
        thesis="STANCE: BULL — guidance raise plus supply tightness",
        trade_plan="BUY NTNX at 67.64, stop 64.50",
        risk_notes={"Neutral Risk Analyst": "Size 1.5% of book"},
    )
    blob = (captured.get("desc") or "") + "\n" + (captured.get("prior") or "")
    assert "Hardware shortage" in blob
    assert "geopolitical tariff" in blob
    assert "STANCE: BULL" in blob
    assert "67.64" in blob
    assert "Size 1.5%" in blob
    assert "## Quality Review" not in (captured.get("prior") or "")
    assert "## Compliance review" not in (captured.get("prior") or "")


def test_graph_pm_packs_reviewers_only_when_present():
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from trading_crew.graph.nodes import GraphRuntime

    tasks = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "trading_crew" / "config" / "tasks.yaml")
        .read_text()
    )
    rt = GraphRuntime(SimpleNamespace(ticker="NTNX", tasks_config=tasks, _pm_task=None), {})
    captured: dict = {}

    def _capture(agent_key, description, expected_output, output_pydantic=None, prior=""):
        captured["desc"] = description
        captured["prior"] = prior
        return "ok"

    rt._run_agent = _capture  # type: ignore[method-assign]
    rt.run_pm(
        reports={"news_analyst": "OEM hardware shortage"},
        thesis="STANCE: BULL",
        trade_plan="BUY NTNX",
        risk_notes={"Neutral Risk Analyst": "Size 1.5%"},
        quality_notes="Score 7/10 — thesis grounded",
        compliance_notes="CLEAR — no SDN hit",
    )
    blob = (captured.get("desc") or "") + "\n" + (captured.get("prior") or "")
    assert "Score 7/10" in blob
    assert "CLEAR — no SDN hit" in blob
    assert "## Quality Review" in (captured.get("prior") or "")
    assert "## Compliance review" in (captured.get("prior") or "")


def test_pm_task_handle_is_terminal(monkeypatch):
    """The crew must expose the PM task so the runner can append the parsed
    guidance ledger to it before it executes."""
    _ensure_env(monkeypatch)
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=False)
    crew = tc.crew()
    assert tc._pm_task is crew.tasks[-1]


def test_decision_desks_run_deterministic_while_debate_explores(monkeypatch):
    """Aggregation/decision desks run near-zero temperature (same evidence ->
    same call), while analyst/debate desks keep the exploration temperature.
    This is the workflow-level lever on the intra-day flip rate — not a
    hardcoded side."""
    _ensure_env(monkeypatch)
    from trading_crew.crew import TradingCrew, _DECISION_AGENTS

    tc = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=False)
    for key in _DECISION_AGENTS:
        assert tc._llm_for(key).temperature == pytest.approx(0.0), key
    for key in ("bull_researcher", "bear_researcher", "market_analyst", "risk_aggressive"):
        assert tc._llm_for(key).temperature == pytest.approx(0.3), key


def test_decision_and_debate_temperature_env_override(monkeypatch):
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_DECISION_TEMPERATURE", "0.15")
    monkeypatch.setenv("TRADINGCREW_DEBATE_TEMPERATURE", "0.4")
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", memory=False)
    assert tc._llm_for("portfolio_manager").temperature == pytest.approx(0.15)
    assert tc._llm_for("bull_researcher").temperature == pytest.approx(0.4)


def test_planning_enabled_only_on_decision_desks(monkeypatch):
    """ReAct-style planning must attach to the synthesis/decision desks only;
    the tool-heavy analysts and the debate/risk desks must be left alone
    (planning on a tool loop invites the degraded-loop failure mode)."""
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_DECISION_PLANNING", "1")
    from trading_crew.crew import TradingCrew, _PLANNING_AGENTS

    assert _PLANNING_AGENTS == {"research_manager", "trader", "portfolio_manager"}

    tc = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=False)
    planned = {
        "research_manager": tc.research_manager(),
        "trader": tc.trader(),
        "portfolio_manager": tc.portfolio_manager(),
    }
    for key, ag in planned.items():
        assert ag.planning is True, key
        assert ag.planning_config is not None, key

    for key, ag in {
        "market_analyst": tc.market_analyst(),
        "bull_researcher": tc.bull_researcher(),
        "risk_aggressive": tc.risk_aggressive(),
    }.items():
        assert ag.planning is False, key
        assert ag.planning_config is None, key


def test_planning_can_be_disabled_by_env(monkeypatch):
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_DECISION_PLANNING", "0")
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", memory=False)
    assert tc.portfolio_manager().planning is False


def test_planning_effort_env_override(monkeypatch):
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_DECISION_PLANNING", "1")
    monkeypatch.setenv("TRADINGCREW_PLANNING_EFFORT", "medium")
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", memory=False)
    assert tc.portfolio_manager().planning_config.reasoning_effort == "medium"


def test_planning_observe_steps_off_by_default(monkeypatch):
    """The per-step PlannerObserver LLM call must stay OFF by default — our OSS
    model can't emit the nested StepObservation schema, so leaving it on just
    burns tokens on retried validation failures.  Opt back in only via env."""
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_DECISION_PLANNING", "1")
    monkeypatch.delenv("TRADINGCREW_PLANNING_OBSERVE_STEPS", raising=False)
    monkeypatch.delenv("TRADINGCREW_PLANNING_EFFORT", raising=False)
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", memory=False)
    pc = tc.portfolio_manager().planning_config
    assert pc.observe_steps is False
    assert pc.reasoning_effort == "low"

    monkeypatch.setenv("TRADINGCREW_PLANNING_OBSERVE_STEPS", "1")
    tc2 = TradingCrew(ticker="NTNX", memory=False)
    assert tc2.portfolio_manager().planning_config.observe_steps is True
