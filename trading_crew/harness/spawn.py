"""Spawn catalog desks or a briefed generic investigator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from crewai import Agent, Crew, Process, Task

from trading_crew._common import get_llm
from trading_crew.agentic.graph.scratchpad import compact_excerpt
from trading_crew.graph.nodes import ANALYST_KEYS, GraphRuntime, pack_prior_outputs
from trading_crew.llm_presets import submit_in_active_context
from trading_crew.tools import ALL_TOOLS

from .conductor import SpawnSpec
from .planner import plan_spawn
from .safety import (
    DEBATE_DESKS,
    RISK_DESKS,
    HarnessGateError,
    check_spawn,
    is_skeptical_spawn,
)
from .state import HarnessState, is_context_overflow

DESK_TASK = {
    "market_analyst": "market_task",
    "news_analyst": "news_task",
    "fundamentals_analyst": "fundamentals_task",
    "reports_analyst": "reports_task",
    "macro_analyst": "macro_task",
    "debate": "debate",
    "bull_researcher": "bull_round",
    "bear_researcher": "bear_round",
    "research_manager": "research_synthesis",
    "trader": "trader_task",
    "risk": "risk",
    "risk_aggressive": "risk_aggressive_round",
    "risk_conservative": "risk_conservative_round",
    "risk_neutral": "risk_neutral_round",
}

DESK_ROLE = {
    "market_analyst": "Market Analyst",
    "news_analyst": "News Analyst",
    "fundamentals_analyst": "Fundamentals Analyst",
    "reports_analyst": "Reports Analyst",
    "macro_analyst": "Macro Analyst",
    "debate": "Bull/Bear Debate",
    "bull_researcher": "Bullish Researcher",
    "bear_researcher": "Bearish Researcher",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "risk": "Risk Debate",
    "risk_aggressive": "Aggressive Risk Analyst",
    "risk_conservative": "Conservative Risk Analyst",
    "risk_neutral": "Neutral Risk Analyst",
    "generic_investigator": "Investigator",
}


def _prior_for(desk: str, state: HarnessState, brief: str = "") -> str:
    sections: List[Tuple[str, str]] = []
    if brief.strip():
        sections.append(("Conductor brief for this spawn", brief.strip()))
    ctx = _market_context_excerpt(state)
    if ctx:
        sections.append(("Market brief already gathered", ctx))
    if desk in ANALYST_KEYS:
        for key in ANALYST_KEYS:
            if key == desk:
                continue
            text = compact_excerpt(state.reports.get(key, ""), limit=600)
            if text:
                sections.append((f"{key} (compact)", text))
    if desk in DEBATE_DESKS or desk == "debate":
        for key in ANALYST_KEYS:
            text = compact_excerpt(state.reports.get(key, ""), limit=600)
            if text:
                sections.append((f"{key} (compact)", text))
        if desk in DEBATE_DESKS:
            other = "bear_researcher" if desk == "bull_researcher" else "bull_researcher"
            opp = (state.reports.get(other) or "").strip()
            if opp:
                sections.append(
                    (
                        "Opposing debate — rebut this; do not ignore it",
                        compact_excerpt(opp, limit=2000) or opp,
                    )
                )
    if desk == "research_manager":
        sections.extend([
            ("Bullish Researcher", state.reports.get("bull_researcher", "")),
            ("Bearish Researcher", state.reports.get("bear_researcher", "")),
        ])
    elif desk == "trader":
        sections.append(("Research Manager thesis", state.reports.get("research_manager", "")))
    elif desk in RISK_DESKS or desk == "risk":
        sections.extend([
            ("Research Manager thesis", state.reports.get("research_manager", "")),
            ("Trader plan", state.reports.get("trader", "")),
        ])
        if desk == "risk":
            for key, title in (
                ("risk_aggressive", "Prior Aggressive Risk"),
                ("risk_conservative", "Prior Conservative Risk"),
                ("risk_neutral", "Prior Neutral Risk"),
            ):
                text = compact_excerpt(state.reports.get(key, ""), limit=800)
                if text:
                    sections.append((title, text))
        elif desk == "risk_neutral":
            sections.extend([
                ("Aggressive Risk Analyst", state.reports.get("risk_aggressive", "")),
                ("Conservative Risk Analyst", state.reports.get("risk_conservative", "")),
            ])
        else:
            others = sorted(RISK_DESKS - {desk})
            for key in others:
                text = compact_excerpt(state.reports.get(key, ""), limit=800)
                if text:
                    sections.append((f"Other risk wing ({key})", text))
    return pack_prior_outputs(sections)


def _market_context_excerpt(state: HarnessState) -> str:
    for row in state.observations:
        if row.get("kind") == "tool" and row.get("title") == "get_market_context":
            return compact_excerpt(str(row.get("body") or ""), limit=800)
    return ""


def _pack_child_context(desk: str, state: HarnessState, instruction: str) -> str:
    evidence = _prior_for(desk, state, "")
    if evidence.strip():
        return f"{instruction}\n\n{evidence}"
    return instruction


def _run_desk(
    runtime: GraphRuntime,
    desk: str,
    state: HarnessState,
    instruction: str,
) -> str:
    if desk == "debate":
        raise HarnessGateError("bundled debate is run via _run_bundled_debate")
    if desk == "risk":
        raise HarnessGateError("bundled risk is run via _run_bundled_risk")
    task = DESK_TASK.get(desk)
    if not task:
        raise HarnessGateError(f"cannot run desk: {desk}")
    return runtime.run_spawned(desk, task, _pack_child_context(desk, state, instruction))


def _run_bundled_debate(
    runtime: GraphRuntime,
    state: HarnessState,
    instruction: str,
) -> Tuple[str, str, str]:
    prior = _pack_child_context("debate", state, instruction)
    bull, bear = runtime.run_bundled_debate(prior)
    combined = runtime.last_raw or (
        f"## Bull\n{bull or ''}\n\n## Bear\n{bear or ''}"
    )
    return bull or "", bear or "", combined


def _run_bundled_risk(
    runtime: GraphRuntime,
    state: HarnessState,
    instruction: str,
) -> Tuple[str, str, str, str]:
    prior = _pack_child_context("risk", state, instruction)
    aggr, cons, neut = runtime.run_bundled_risk(prior)
    combined = runtime.last_raw or (
        f"## Aggressive\n{aggr or ''}\n\n"
        f"## Conservative\n{cons or ''}\n\n"
        f"## Neutral\n{neut or ''}"
    )
    return aggr or "", cons or "", neut or "", combined


def _run_generic(
    spec: SpawnSpec,
    *,
    ticker: str,
    cfg: Dict[str, Any],
    prior: str,
    temperature: float,
    instruction: str,
) -> str:
    names = [n for n in spec.tools if n in cfg["tool_allowlist"] or n == "query_knowledge"]
    if not names:
        raise HarnessGateError(
            "generic investigator requires a tools list from the allowlist"
        )
    tools = [ALL_TOOLS[n] for n in names if n in ALL_TOOLS]
    brief = (spec.brief or "").strip()
    if not brief:
        raise HarnessGateError("generic investigator requires a brief")
    agent = Agent(
        role="Investigator",
        goal=brief[:400],
        backstory=(
            "You investigate one assigned question. Pick which of your "
            "assigned tools to call and write your own queries from facts. "
            "There is no mandatory tool order or round quota. Do not invent "
            "figures. Cite [source: …]."
        ),
        tools=tools,
        llm=get_llm(temperature=temperature),
        allow_delegation=False,
        max_iter=6,
        respect_context_window=True,
        verbose=True,
    )
    description = f"{instruction}\n\n{prior}"
    task = Task(
        description=description,
        expected_output="A cited briefing that answers only the assigned brief.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs={"ticker": ticker})
    if result is None:
        return ""
    return getattr(result, "raw", None) or str(result)


def spawn_one(
    spec: SpawnSpec,
    *,
    runtime: GraphRuntime,
    state: HarnessState,
    cfg: Dict[str, Any],
    depth: int,
    prior: str,
) -> Tuple[str, str]:
    desk = (spec.desk or "").strip()
    if spec.kind == "generic":
        desk = "generic_investigator"
    check_spawn(
        desk,
        depth=depth,
        cfg=cfg,
        spawn_used=state.spawn_used,
    )
    role = DESK_ROLE.get(desk, "Investigator")
    instruction = plan_spawn(spec=spec, state=state, cfg=cfg, runtime=runtime)
    state.add_observation(
        "planner",
        desk,
        f"Wrote {len(instruction)}-char spawn contract for {desk}.",
    )
    child_prior = prior
    if len(child_prior) > int(cfg["context_chars"]):
        child_prior = state.render(int(cfg["context_chars"]))
    if spec.kind == "generic":
        try:
            text = _run_generic(
                spec,
                ticker=state.ticker,
                cfg=cfg,
                prior=child_prior,
                temperature=float(cfg["conductor_temperature"]),
                instruction=instruction,
            )
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            state.compact_in_place()
            text = _run_generic(
                spec,
                ticker=state.ticker,
                cfg=cfg,
                prior=state.render(int(cfg["context_chars"])),
                temperature=float(cfg["conductor_temperature"]),
                instruction=instruction,
            )
        key = f"investigator_{state.spawn_used + 1}"
    elif desk == "debate":
        try:
            bull, bear, text = _run_bundled_debate(
                runtime, state, instruction
            )
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            state.compact_in_place()
            bull, bear, text = _run_bundled_debate(
                runtime, state, instruction
            )
        key = "debate"
        state.reports["bull_researcher"] = bull
        state.reports["bear_researcher"] = bear
    elif desk == "risk":
        try:
            aggr, cons, neut, text = _run_bundled_risk(
                runtime, state, instruction
            )
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            state.compact_in_place()
            aggr, cons, neut, text = _run_bundled_risk(
                runtime, state, instruction
            )
        key = "risk"
        state.reports["risk_aggressive"] = aggr
        state.reports["risk_conservative"] = cons
        state.reports["risk_neutral"] = neut
    else:
        try:
            text = _run_desk(runtime, desk, state, instruction)
        except Exception as exc:
            if not is_context_overflow(exc):
                raise
            state.compact_in_place()
            text = _run_desk(runtime, desk, state, instruction)
        key = desk
    state.spawn_used += 1
    if desk in DEBATE_DESKS or desk == "debate":
        state.debate_used += 1
    if desk in RISK_DESKS or desk == "risk":
        state.risk_used += 1
    if is_skeptical_spawn(spec.kind, desk, spec.brief):
        state.skeptical = True
    state.spawned.append(key)
    state.reports[key] = text or ""
    state.add_edge("Conductor", role, "activated")
    return key, text or ""


def spawn_many(
    specs: List[SpawnSpec],
    *,
    runtime: GraphRuntime,
    state: HarnessState,
    cfg: Dict[str, Any],
    depth: int,
    prior: str,
) -> List[Tuple[str, str]]:
    """Run every requested spawn. Extra agents beyond max_parallel are
    queued in later batches — they are not dropped."""
    if not specs:
        return []
    out: List[Tuple[str, str]] = []
    remaining = list(specs)
    while remaining:
        n = min(len(remaining), int(cfg["max_parallel"]))
        batch = remaining[:n]
        remaining = remaining[n:]
        if n <= 1:
            out.append(
                spawn_one(
                    batch[0],
                    runtime=runtime,
                    state=state,
                    cfg=cfg,
                    depth=depth,
                    prior=prior,
                )
            )
            continue
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {
                submit_in_active_context(
                    pool,
                    spawn_one,
                    spec,
                    runtime=runtime,
                    state=state,
                    cfg=cfg,
                    depth=depth,
                    prior=prior,
                ): spec
                for spec in batch
            }
            for fut in as_completed(futs):
                out.append(fut.result())
    return out
