"""Compile and run the v3 stock StateGraph."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from trading_crew.agentic.graph.playbook import playbook_key
from trading_crew.graph.nodes import GraphRuntime
from trading_crew.graph.routers import (
    after_debate,
    after_reopen,
    after_risk,
    debate_rounds_from_action,
    evidence_budget_from_action,
    resolve_playbook_action,
    risk_rounds_from_action,
    should_skip_reports,
)
from trading_crew.graph.state import (
    GraphKickoffResult,
    GraphState,
    RunState,
    leans_are_split,
    parse_stance,
)

logger = logging.getLogger(__name__)


def _state_to_run(s: Dict[str, Any]) -> RunState:
    return RunState(
        ticker=s["ticker"],
        playbook_key=s.get("playbook_key") or "",
        playbook_action=s.get("playbook_action") or "default",
        skip_reports=bool(s.get("skip_reports")),
        debate_rounds_target=int(s.get("debate_rounds_target") or 2),
        debate_round=int(s.get("debate_round") or 0),
        extra_debate_used=bool(s.get("extra_debate_used")),
        evidence_budget=int(s.get("evidence_budget") or 2),
        evidence_used=int(s.get("evidence_used") or 0),
        evidence_queued=bool(s.get("evidence_queued")),
        desk_split=bool(s.get("desk_split")),
        stance=s.get("stance") or "",
        trader_reopened=bool(s.get("trader_reopened")),
        reports=dict(s.get("reports") or {}),
        leans=dict(s.get("leans") or {}),
        last_bull=s.get("last_bull") or "",
        last_bear=s.get("last_bear") or "",
        thesis=s.get("thesis") or "",
        trade_plan=s.get("trade_plan") or "",
        risk_notes=dict(s.get("risk_notes") or {}),
        quality_notes=s.get("quality_notes") or "",
        compliance_notes=s.get("compliance_notes") or "",
        quality_review_enabled=bool(s.get("quality_review_enabled")),
        compliance_enabled=bool(s.get("compliance_enabled")),
        ui_risk_rounds=int(s.get("ui_risk_rounds") or 1),
        decision=s.get("decision"),
        decision_raw=s.get("decision_raw") or "",
        activated_edges=list(s.get("activated_edges") or []),
    )


def compile_stock_graph(runtime: GraphRuntime):
    def analysts(s: Dict[str, Any]) -> Dict[str, Any]:
        reports = runtime.run_analysts(bool(s.get("skip_reports")))
        leans = dict(s.get("leans") or {})
        from trading_crew.agentic.graph.blackboard import current

        bb = current()
        if bb is not None:
            for key, lean in (bb.desk_leans or {}).items():
                leans[key] = lean
        return {
            "reports": reports,
            "leans": leans,
            "desk_split": leans_are_split(leans),
        }

    def debate_open(s: Dict[str, Any]) -> Dict[str, Any]:
        rnd = 1
        total = max(int(s.get("debate_rounds_target") or 2), rnd)
        bull, bear = runtime.run_opening_debate(rnd, total)
        queued = runtime.evidence_is_queued()
        return {
            "debate_round": rnd,
            "last_bull": bull,
            "last_bear": bear,
            "evidence_queued": queued,
        }

    def debate_bull(s: Dict[str, Any]) -> Dict[str, Any]:
        rnd = int(s.get("debate_round") or 0) + 1
        total = max(int(s.get("debate_rounds_target") or 2), rnd)
        text = runtime.run_debate_turn("bull", rnd, total)
        return {"debate_round": rnd, "last_bull": text}

    def debate_bear(s: Dict[str, Any]) -> Dict[str, Any]:
        rnd = int(s.get("debate_round") or 1)
        total = max(int(s.get("debate_rounds_target") or 2), rnd)
        text = runtime.run_debate_turn("bear", rnd, total)
        queued = runtime.evidence_is_queued()
        return {"evidence_queued": queued, "last_bear": text}

    def drain_evidence(s: Dict[str, Any]) -> Dict[str, Any]:
        n = runtime.drain_evidence()
        return {
            "evidence_used": int(s.get("evidence_used") or 0) + n,
            "evidence_queued": False,
        }

    def mark_extra(s: Dict[str, Any]) -> Dict[str, Any]:
        runtime.emit_edge("Bearish Researcher", "Bullish Researcher", "cycled")
        runtime.emit_edge("Research Manager", "Bullish Researcher", "cycled")
        return {"extra_debate_used": True}

    def research_manager(s: Dict[str, Any]) -> Dict[str, Any]:
        text = runtime.run_research_manager(
            last_bull=s.get("last_bull") or "",
            last_bear=s.get("last_bear") or "",
        )
        return {"thesis": text, "stance": parse_stance(text)}

    def trader(s: Dict[str, Any]) -> Dict[str, Any]:
        return {"trade_plan": runtime.run_trader(thesis=s.get("thesis") or "")}

    def risk_wings(s: Dict[str, Any]) -> Dict[str, Any]:
        rounds = max(1, int(s.get("ui_risk_rounds") or 1))
        notes = dict(s.get("risk_notes") or {})
        thesis = s.get("thesis") or ""
        plan = s.get("trade_plan") or ""
        last: Dict[str, str] = {}
        for i in range(1, rounds + 1):
            last = runtime.run_risk_wings(
                round_num=i,
                total_rounds=rounds,
                thesis=thesis,
                trade_plan=plan,
                prior_wings=notes if i > 1 else None,
            )
            notes.update(last)
        return {"risk_notes": notes}

    def risk_neutral(s: Dict[str, Any]) -> Dict[str, Any]:
        rounds = max(1, int(s.get("ui_risk_rounds") or 1))
        notes = dict(s.get("risk_notes") or {})
        thesis = s.get("thesis") or ""
        plan = s.get("trade_plan") or ""
        text = ""
        for i in range(1, rounds + 1):
            text = runtime.run_risk_neutral(
                round_num=i,
                total_rounds=rounds,
                thesis=thesis,
                trade_plan=plan,
                wing_notes=notes,
            )
            notes["Neutral Risk Analyst"] = text
        qr = bool(s.get("quality_review_enabled"))
        co = bool(s.get("compliance_enabled"))
        if qr:
            next_role = "Quality Reviewer"
        elif co:
            next_role = "Compliance Officer"
        else:
            next_role = "Portfolio Manager"
        runtime.emit_edge("Neutral Risk Analyst", next_role, "activated")
        return {"risk_notes": notes}

    def reviewers(s: Dict[str, Any]) -> Dict[str, Any]:
        notes = runtime.run_reviewers(
            quality_enabled=bool(s.get("quality_review_enabled")),
            compliance_enabled=bool(s.get("compliance_enabled")),
            reports=s.get("reports") or {},
            thesis=s.get("thesis") or "",
            trade_plan=s.get("trade_plan") or "",
            risk_notes=s.get("risk_notes") or {},
        )
        return {
            "quality_notes": notes.get("quality_notes") or "",
            "compliance_notes": notes.get("compliance_notes") or "",
        }

    def reopen_trader(s: Dict[str, Any]) -> Dict[str, Any]:
        runtime.emit_edge("Neutral Risk Analyst", "Trader", "cycled")
        return {
            "trade_plan": runtime.run_trader(thesis=s.get("thesis") or ""),
            "trader_reopened": True,
        }

    def portfolio_manager(s: Dict[str, Any]) -> Dict[str, Any]:
        raw = runtime.run_pm(
            reports=s.get("reports") or {},
            thesis=s.get("thesis") or "",
            trade_plan=s.get("trade_plan") or "",
            risk_notes=s.get("risk_notes") or {},
            quality_notes=s.get("quality_notes") or "",
            compliance_notes=s.get("compliance_notes") or "",
        )
        payload = None
        if runtime.last_pydantic is not None:
            payload = runtime.last_pydantic.model_dump()
        return {"decision_raw": raw, "decision": payload}

    def route_after_debate(s: Dict[str, Any]) -> str:
        st = _state_to_run(s)
        st.evidence_queued = runtime.evidence_is_queued()
        hop = after_debate(st)
        logger.info("v3 after_debate -> %s (round=%s split=%s queued=%s)",
                    hop, st.debate_round, st.desk_split, st.evidence_queued)
        return hop

    def route_after_rm(s: Dict[str, Any]) -> str:
        if (s.get("stance") == "MIXED") and not s.get("extra_debate_used"):
            return "extra_debate"
        return "trader"

    def route_after_risk(s: Dict[str, Any]) -> str:
        return after_risk(_state_to_run(s))

    def route_after_reopen(s: Dict[str, Any]) -> str:
        return after_reopen(_state_to_run(s))

    g = StateGraph(GraphState)
    g.add_node("analysts", analysts)
    g.add_node("debate_open", debate_open)
    g.add_node("debate_bull", debate_bull)
    g.add_node("debate_bear", debate_bear)
    g.add_node("drain_evidence", drain_evidence)
    g.add_node("mark_extra", mark_extra)
    g.add_node("research_manager", research_manager)
    g.add_node("trader", trader)
    g.add_node("risk_wings", risk_wings)
    g.add_node("risk_neutral", risk_neutral)
    g.add_node("reopen_trader", reopen_trader)
    g.add_node("reviewers", reviewers)
    g.add_node("portfolio_manager", portfolio_manager)

    g.add_edge(START, "analysts")
    g.add_edge("analysts", "debate_open")
    g.add_edge("debate_bull", "debate_bear")
    after_debate_hops = {
        "drain_evidence": "drain_evidence",
        "extra_debate": "mark_extra",
        "debate": "debate_bull",
        "research_manager": "research_manager",
    }
    g.add_conditional_edges("debate_open", route_after_debate, after_debate_hops)
    g.add_conditional_edges("debate_bear", route_after_debate, after_debate_hops)
    g.add_edge("drain_evidence", "debate_bull")
    g.add_edge("mark_extra", "debate_bull")
    g.add_conditional_edges(
        "research_manager",
        route_after_rm,
        {"extra_debate": "mark_extra", "trader": "trader"},
    )
    g.add_edge("trader", "risk_wings")
    g.add_edge("risk_wings", "risk_neutral")
    g.add_conditional_edges(
        "risk_neutral",
        route_after_risk,
        {
            "reopen_trader": "reopen_trader",
            "reviewers": "reviewers",
            "portfolio_manager": "portfolio_manager",
        },
    )
    g.add_conditional_edges(
        "reopen_trader",
        route_after_reopen,
        {"reviewers": "reviewers", "portfolio_manager": "portfolio_manager"},
    )
    g.add_edge("reviewers", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()


def initial_state(
    ticker: str,
    *,
    ui_debate_rounds: int = 2,
    ui_risk_rounds: int = 1,
    quality_review_enabled: bool = False,
    compliance_enabled: bool = False,
    market: str = "",
) -> Dict[str, Any]:
    key = playbook_key(market=market)
    action = resolve_playbook_action(key)
    return {
        "ticker": ticker,
        "playbook_key": key,
        "playbook_action": action,
        "skip_reports": should_skip_reports(ticker, action),
        "debate_rounds_target": debate_rounds_from_action(action, ui_debate_rounds),
        "debate_round": 0,
        "extra_debate_used": False,
        "evidence_budget": evidence_budget_from_action(action),
        "evidence_used": 0,
        "evidence_queued": False,
        "desk_split": False,
        "stance": "",
        "trader_reopened": False,
        "reports": {},
        "leans": {},
        "last_bull": "",
        "last_bear": "",
        "thesis": "",
        "trade_plan": "",
        "risk_notes": {},
        "quality_notes": "",
        "compliance_notes": "",
        "quality_review_enabled": bool(quality_review_enabled),
        "compliance_enabled": bool(compliance_enabled),
        "ui_risk_rounds": risk_rounds_from_action(action, ui_risk_rounds),
        "decision": None,
        "decision_raw": "",
        "activated_edges": [],
    }


def run_stock_graph(
    trading_crew: Any,
    agents: Dict[str, Any],
    *,
    step_callback: Optional[Callable] = None,
    task_callback: Optional[Callable] = None,
    on_edge: Optional[Callable[[str, str, str], None]] = None,
    ui_debate_rounds: int = 2,
    ui_risk_rounds: int = 1,
    quality_review_enabled: bool = False,
    compliance_enabled: bool = False,
    market: str = "",
) -> GraphKickoffResult:
    runtime = GraphRuntime(
        trading_crew,
        agents,
        step_callback=step_callback,
        task_callback=task_callback,
        on_edge=on_edge,
    )
    graph = compile_stock_graph(runtime)
    init = initial_state(
        trading_crew.ticker,
        ui_debate_rounds=ui_debate_rounds,
        ui_risk_rounds=ui_risk_rounds,
        quality_review_enabled=quality_review_enabled,
        compliance_enabled=compliance_enabled,
        market=market,
    )
    runtime.emit_edge("Playbook Router", "Shared State", "activated")
    if init["skip_reports"]:
        runtime.emit_edge("Playbook Router", "Reports Analyst", "skipped")
        runtime.emit_edge("Reports Analyst", "Shared State", "skipped")
    final = graph.invoke(init, config={"recursion_limit": 40})
    pydantic = runtime.last_pydantic
    raw = (final or {}).get("decision_raw") or runtime.last_raw
    if pydantic is None and (final or {}).get("decision"):
        from trading_crew.schemas import PortfolioDecision

        try:
            pydantic = PortfolioDecision.model_validate(final["decision"])
        except Exception:
            pydantic = None
    return GraphKickoffResult(pydantic=pydantic, raw=raw)
