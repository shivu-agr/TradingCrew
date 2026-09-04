"""Code-only graph predicates (MANTA-style: no LLM vibes)."""

from __future__ import annotations

from typing import Optional

from trading_crew.agentic.graph.playbook import PLAYBOOK_ACTIONS, preferred_action
from trading_crew.graph.state import RunState


def resolve_playbook_action(playbook_key: str) -> str:
    """Hermes playbook experiment, then per-key playbook, then legacy Mass."""
    from trading_crew.agentic.self_learn import active_experiment

    exp = active_experiment()
    if exp:
        live = (exp.get("playbook_action") or "").strip()
        if live in PLAYBOOK_ACTIONS:
            return live
    action = preferred_action(playbook_key)
    if action != "default":
        return action
    from trading_crew.agentic.mass import load_promoted  # legacy

    promoted = load_promoted().get("promoted_action")
    if promoted in PLAYBOOK_ACTIONS:
        return str(promoted)
    return "default"


def debate_rounds_from_action(action: str, ui_rounds: int) -> int:
    if action == "debate_rounds_1":
        return 1
    if action == "debate_rounds_2":
        return 2
    return max(1, int(ui_rounds or 2))


def evidence_budget_from_action(action: str, default: int = 2) -> int:
    if action == "evidence_budget_0":
        return 0
    if action == "evidence_budget_1":
        return 1
    return max(0, int(default))


def should_skip_reports(ticker: str, playbook_action: str) -> bool:
    if playbook_action == "skip_reports":
        return True
    try:
        from trading_crew.data_sources import is_indian_ticker
        from trading_crew.data_sources.sec_edgar_source import is_edgar_ticker
        from trading_crew.market_context import resolve_ticker
        from trading_crew.tools import _screener_available_for

        canonical = resolve_ticker(ticker)
        if _screener_available_for(canonical) or is_edgar_ticker(canonical):
            return False
        if is_indian_ticker(canonical):
            return False
        return True
    except Exception:
        return False


def after_debate(state: RunState) -> str:
    """Next hop after a completed debate hop (parallel opening or bear turn)."""
    if state.evidence_queued and state.evidence_used < state.evidence_budget:
        return "drain_evidence"
    want_extra = (state.desk_split or state.stance == "MIXED") and not state.extra_debate_used
    if want_extra and state.debate_round >= state.debate_rounds_target:
        return "extra_debate"
    if state.debate_round < state.debate_rounds_target:
        return "debate"
    return "research_manager"


def use_stock_graph(asset_class: str, run_mode: str = "autonomous") -> bool:
    """True when the stock LangGraph scheduler should run.

    Commodity is always sequential. Stock LangGraph runs only for
    ``autonomous``. ``deterministic`` is sequential CrewAI. ``harness``
    is the observe/dispatch loop in ``trading_crew.harness``.
    """
    if (asset_class or "stock").strip().lower() == "commodity":
        return False
    return str(run_mode or "autonomous").strip().lower() == "autonomous"


def reviewers_enabled(state: RunState) -> bool:
    return bool(state.quality_review_enabled or state.compliance_enabled)


def risk_rounds_from_action(action: str, ui_rounds: int) -> int:
    if action == "risk_rounds_2":
        return 2
    return max(1, int(ui_rounds or 1))


def after_risk(state: RunState) -> str:
    note = (state.risk_notes.get("Neutral Risk Analyst") or "")
    reopen_ok = state.playbook_action != "no_trader_reopen"
    if (
        reopen_ok
        and (not state.trader_reopened)
        and "REOPEN_TRADER" in note.upper()
    ):
        return "reopen_trader"
    if reviewers_enabled(state):
        return "reviewers"
    return "portfolio_manager"


def after_reopen(state: RunState) -> str:
    if reviewers_enabled(state):
        return "reviewers"
    return "portfolio_manager"


def preview_role_order(
    *,
    skip_reports: bool,
    debate_rounds_target: int,
    allow_extra_debate: bool = True,
    allow_trader_reopen: bool = True,
    risk_rounds: int = 1,
    quality_review: bool = False,
    compliance: bool = False,
) -> list:
    analysts = [
        "Market Analyst",
        "News Analyst",
        "Fundamentals Analyst",
        "Reports Analyst",
        "Macro Analyst",
    ]
    if skip_reports:
        analysts = [r for r in analysts if r != "Reports Analyst"]
    roles = list(analysts)
    rounds = debate_rounds_target + (1 if allow_extra_debate else 0)
    for _ in range(rounds):
        roles += ["Bullish Researcher", "Bearish Researcher"]
    roles += ["Research Manager", "Trader"]
    if allow_trader_reopen:
        roles.append("Trader")
    for _ in range(max(1, int(risk_rounds or 1))):
        roles += [
            "Aggressive Risk Analyst",
            "Conservative Risk Analyst",
            "Neutral Risk Analyst",
        ]
    if quality_review:
        roles.append("Quality Reviewer")
    if compliance:
        roles.append("Compliance Officer")
    roles.append("Portfolio Manager")
    return roles
