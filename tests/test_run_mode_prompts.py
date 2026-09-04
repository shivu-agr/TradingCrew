"""Autonomous vs deterministic prompt contracts — packaged, no invented copy."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import yaml

from trading_crew.agentic.harness import apply_harness
from trading_crew.agentic.run_mode_prompts import (
    apply_run_mode_prompts,
    prompt_run_mode,
    task_addendum,
    task_override,
)


class _Agent:
    def __init__(self, role: str, backstory: str = "yaml backstory") -> None:
        self.role = role
        self.backstory = backstory


def _agents() -> dict:
    return {
        "bull_researcher": _Agent("Bullish Researcher"),
        "research_manager": _Agent("Research Manager"),
        "portfolio_manager": _Agent("Portfolio Manager"),
        "market_analyst": _Agent("Market Analyst"),
    }


def test_prompt_run_mode_commodity_is_deterministic():
    assert prompt_run_mode("commodity", "autonomous") == "deterministic"
    assert prompt_run_mode("equity", "deterministic") == "deterministic"
    assert prompt_run_mode("equity", "autonomous") == "autonomous"
    assert prompt_run_mode("equity", "harness") == "harness"
    assert prompt_run_mode("commodity", "harness") == "deterministic"


def test_autonomous_and_deterministic_backstories_differ():
    auto = _agents()
    det = _agents()
    assert apply_run_mode_prompts(auto, book="equity", run_mode="autonomous")
    assert apply_run_mode_prompts(det, book="equity", run_mode="deterministic")
    assert "RUN MODE — AUTONOMOUS" in auto["bull_researcher"].backstory
    assert "MIXED" in auto["bull_researcher"].backstory
    assert "RUN MODE — DETERMINISTIC" in det["bull_researcher"].backstory
    assert "assigned round" in det["bull_researcher"].backstory
    assert auto["bull_researcher"].backstory != det["bull_researcher"].backstory
    assert "yaml backstory" in auto["bull_researcher"].backstory


def test_commodity_uses_deterministic_contract():
    agents = _agents()
    apply_run_mode_prompts(agents, book="commodity", run_mode="autonomous")
    assert "RUN MODE — DETERMINISTIC" in agents["research_manager"].backstory
    assert "RUN MODE — AUTONOMOUS" not in agents["research_manager"].backstory


def test_unknown_desk_is_skipped():
    agents = {"other": _Agent("Not A Desk")}
    assert apply_run_mode_prompts(agents, book="equity", run_mode="autonomous") == []
    assert agents["other"].backstory == "yaml backstory"


def test_apply_harness_keeps_operator_overlay_and_appends_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from trading_crew.agentic.prompt_overlay import set_prompt

    set_prompt("equity", "trader", goal="Operator goal.", backstory="Operator trader backstory.")
    agents = {
        "trader": _Agent("Trader"),
        "research_manager": _Agent("Research Manager"),
        "portfolio_manager": _Agent("Portfolio Manager"),
        "market_analyst": _Agent("Market Analyst"),
    }
    apply_harness(agents, journal_file=tmp_path / "journal.json", run_mode="autonomous")
    assert "Operator trader backstory." in agents["trader"].backstory
    assert "RUN MODE — AUTONOMOUS" in agents["trader"].backstory
    assert "## DESK STRATEGY" in agents["trader"].backstory


def test_autonomous_task_addendum_on_debate_and_not_on_deterministic():
    extra = task_addendum("bull_round", book="equity", run_mode="autonomous")
    assert "RUN MODE — AUTONOMOUS" in extra
    assert "stop rule" in extra
    assert "bundled crew" in extra
    extra_risk = task_addendum(
        "risk_aggressive_round", book="equity", run_mode="autonomous"
    )
    assert "independent opening" in extra_risk
    assert "other wing" in extra_risk
    extra_trader = task_addendum(
        "trader_task", book="equity", run_mode="autonomous"
    )
    extra_pm = task_addendum(
        "pm_decision", book="equity", run_mode="autonomous"
    )
    extra_rm = task_addendum(
        "research_synthesis", book="equity", run_mode="autonomous"
    )
    assert "not a sit-out" in extra_trader
    assert "true split" in extra_trader
    assert "Neutrals do not create a split" in extra_rm
    assert "3–1" in extra_pm or "3-1" in extra_pm
    assert task_addendum("bull_round", book="equity", run_mode="deterministic") == ""
    assert task_addendum("bull_round", book="equity", run_mode="harness") == ""
    assert task_addendum("no_such_task", book="equity", run_mode="autonomous") == ""
    body = task_override("market_task", book="equity", run_mode="harness")
    assert body and "You pick the tools" in body["description"]
    assert "Tool budget" not in body["description"]
    assert "AT MOST ONCE" not in body["description"]
    assert task_override("market_task", book="equity", run_mode="autonomous") is None


def test_graph_format_appends_autonomous_pm_addendum():
    from trading_crew.graph.nodes import GraphRuntime

    tasks = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "trading_crew" / "config" / "tasks.yaml")
        .read_text()
    )
    rt = GraphRuntime(
        SimpleNamespace(ticker="NTNX", tasks_config=tasks, run_mode="autonomous"),
        {},
    )
    desc, _out = rt._format("pm_decision", ticker="NTNX")
    assert "{SHORT, MEDIUM}" in desc
    assert "RUN MODE — AUTONOMOUS" in desc
    assert "on NTNX" in desc


def test_apply_harness_harness_mode_appends_harness_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    agents = _agents()
    apply_harness(agents, journal_file=tmp_path / "journal.json", run_mode="harness")
    assert "RUN MODE — HARNESS" in agents["market_analyst"].backstory
    assert "round quota" in agents["market_analyst"].backstory
    assert "RUN MODE — AUTONOMOUS" not in agents["market_analyst"].backstory


def test_apply_harness_harness_mode_skips_experiment_addenda(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from trading_crew.agentic import harness as hermes

    monkeypatch.setattr(
        hermes,
        "active_experiment",
        lambda *a, **k: {"desk_prompts": {"trader": "use two debate rounds"}},
    )
    agents = {
        "trader": _Agent("Trader"),
        "market_analyst": _Agent("Market Analyst"),
    }
    applied = apply_harness(agents, journal_file=tmp_path / "journal.json", run_mode="harness")
    assert applied["experiment"] is None
    assert "use two debate rounds" not in agents["trader"].backstory
    assert "RUN MODE — HARNESS" in agents["trader"].backstory


def test_trading_crew_preserves_harness_run_mode():
    from trading_crew.crew import TradingCrew

    tc = TradingCrew(ticker="NTNX", memory=False, run_mode="harness")
    assert tc.run_mode == "harness"


def test_graph_format_harness_replaces_task_scripts():
    from trading_crew.graph.nodes import GraphRuntime

    tasks = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "trading_crew" / "config" / "tasks.yaml")
        .read_text()
    )
    rt = GraphRuntime(
        SimpleNamespace(ticker="NTNX", tasks_config=tasks, run_mode="harness"),
        {},
    )
    market, _out = rt._format("market_task", ticker="NTNX")
    assert "NTNX" in market
    assert "You pick the tools" in market
    assert "Call get_market_context" not in market
    assert "AT MOST ONCE" not in market
    assert "Mandatory order" not in market
    assert "Tool budget" not in market
    bull, bull_out = rt._format("bull_round", ticker="NTNX")
    assert "NTNX" in bull
    assert "Debate round {" not in bull
    assert "{round_num}" not in bull
    assert "{total_rounds}" not in bull
    assert "{round_num}" not in bull_out
    assert "DESK_BIAS" in bull_out
    pm, _ = rt._format("pm_decision", ticker="NTNX")
    assert "Do not default to BUY" in pm
