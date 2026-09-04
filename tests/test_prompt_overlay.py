"""Hermes prompt overlay — set / unset / apply. No invented agents."""

from __future__ import annotations

import pytest

from trading_crew.agentic.harness import apply_harness, apply_prompt_overlay
from trading_crew.agentic.prompt_overlay import (
    PromptOverlayError,
    list_prompts,
    set_prompt,
    unset_prompt,
)


class _Agent:
    def __init__(self, role: str, backstory: str = "yaml backstory", goal: str = "yaml goal") -> None:
        self.role = role
        self.backstory = backstory
        self.goal = goal


def test_set_and_unset_trader_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    rows = list_prompts("equity")
    assert any(r["key"] == "trader" and r["overlay"] is False for r in rows)
    set_prompt("equity", "trader", goal="Trade the thesis.", backstory="You size from evidence.")
    rows = {r["key"]: r for r in list_prompts("equity")}
    assert rows["trader"]["overlay"] is True
    assert rows["trader"]["backstory"] == "You size from evidence."
    assert unset_prompt("equity", "trader") is True
    rows = {r["key"]: r for r in list_prompts("equity")}
    assert rows["trader"]["overlay"] is False


def test_unknown_agent_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    with pytest.raises(PromptOverlayError, match="unknown agent"):
        set_prompt("equity", "not_a_desk", goal="x", backstory="y")


def test_empty_prompt_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    with pytest.raises(PromptOverlayError, match="non-empty"):
        set_prompt("equity", "trader", goal="", backstory="y")


def test_apply_overlay_then_strategy_prepend(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    set_prompt(
        "equity", "trader",
        goal="Operator goal.",
        backstory="Operator trader backstory.",
    )
    agents = {
        "trader": _Agent("Trader"),
        "research_manager": _Agent("Research Manager"),
        "portfolio_manager": _Agent("Portfolio Manager"),
        "market_analyst": _Agent("Market Analyst"),
    }
    apply_harness(agents, journal_file=tmp_path / "journal.json")
    assert agents["trader"].goal == "Operator goal."
    assert "Operator trader backstory." in agents["trader"].backstory
    assert "## DESK STRATEGY" in agents["trader"].backstory
    assert "Operator trader backstory." not in agents["market_analyst"].backstory
    applied = apply_prompt_overlay({"trader": _Agent("Trader")}, book="equity")
    assert applied == ["trader"]
