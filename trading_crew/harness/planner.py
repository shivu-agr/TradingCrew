"""Spawn planner — writes a full child instruction from live session state."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel

from trading_crew.agentic.graph.scratchpad import compact_excerpt
from trading_crew.tools import DEFAULT_AGENT_TOOLS

from .config import HarnessConfigError
from .conductor import SpawnSpec
from .plugins import skill_for
from .state import HarnessState

logger = logging.getLogger(__name__)


class SpawnAssignment(BaseModel):
    """Numbered, ticker-specific tasks for this spawn only."""

    user_request: str = ""


_ASSIGN_HINT = (
    "\n\nReturn ONLY a JSON object with key "
    '"user_request": a markdown numbered list (1. 2. 3.) of concrete '
    "tasks for THIS spawn and ticker. Include the gap, which required "
    "tools to call, and 'do not assume prior desk text — re-check with "
    "tools'. No other keys."
)


def _last_thought(state: HarnessState) -> str:
    for row in reversed(state.observations):
        if row.get("kind") == "thought":
            return str(row.get("body") or "").strip()
    return ""


def _prior_findings(state: HarnessState, desk: str) -> str:
    lines: List[str] = [state.coverage_board()]
    for key, text in state.reports.items():
        if key == desk:
            continue
        excerpt = compact_excerpt(str(text or ""), limit=400)
        if excerpt:
            lines.append(f"- {key}: {excerpt}")
    return "\n".join(lines) if lines else "(none yet)"


def _bound_tools(
    desk: str,
    spec: SpawnSpec,
    runtime: Any,
) -> List[str]:
    if spec.kind == "generic":
        return [str(n) for n in spec.tools if str(n).strip()]
    if desk == "debate":
        defaults = list(
            dict.fromkeys(
                list(DEFAULT_AGENT_TOOLS.get("bull_researcher", []))
                + list(DEFAULT_AGENT_TOOLS.get("bear_researcher", []))
            )
        )
    elif desk == "risk":
        defaults = list(
            dict.fromkeys(
                list(DEFAULT_AGENT_TOOLS.get("risk_aggressive", []))
                + list(DEFAULT_AGENT_TOOLS.get("risk_conservative", []))
                + list(DEFAULT_AGENT_TOOLS.get("risk_neutral", []))
            )
        )
    else:
        defaults = list(DEFAULT_AGENT_TOOLS.get(desk, []))
    crew = getattr(runtime, "tc", None)
    enabled = getattr(crew, "_tools_enabled", None)
    if isinstance(enabled, dict) and desk in enabled:
        allowed = enabled[desk]
        if not isinstance(allowed, list):
            raise HarnessConfigError(f"tools_enabled.{desk} must be a list")
        return [t for t in allowed if t in defaults]
    return list(defaults)


def _llm_assignment(
    *,
    llm: Any,
    ticker: str,
    desk: str,
    brief: str,
    thought: str,
    skill: Dict[str, Any],
    board: str,
    findings: str,
    bound: Sequence[str],
) -> str:
    from trading_crew.agentic.resolve_investigate import call_structured

    required = ", ".join(skill.get("required_tools") or []) or "(none listed)"
    prompt = (
        f"Write a spawn assignment for the stock-prediction harness.\n"
        f"Ticker: {ticker}\n"
        f"Skill / desk: {desk}\n"
        f"Conductor brief: {brief or '(none)'}\n"
        f"Conductor thought: {thought or '(none)'}\n"
        f"Required tools (prompt-only): {required}\n"
        f"Bound tools: {', '.join(bound) or '(none)'}\n"
        f"Skill success: {skill.get('success') or ''}\n"
        f"Skill hints: {skill.get('hints') or ''}\n\n"
        f"Coverage board:\n{board}\n\n"
        f"Prior findings (may be stale):\n{findings}\n\n"
        "Write 3–8 numbered tasks for THIS spawn only. Be specific to "
        "the ticker and the gap. Tell the child to re-check priors with "
        "tools. Do not invent product rules."
    )
    parsed = call_structured(llm, prompt, SpawnAssignment, _ASSIGN_HINT)
    return str(parsed.user_request or "").strip()


def assemble_instruction(
    *,
    ticker: str,
    desk: str,
    assignment: str,
    skill: Dict[str, Any],
    bound: Sequence[str],
    findings: str,
) -> str:
    required = skill.get("required_tools") or []
    req = ", ".join(required) if required else "(none listed on this skill)"
    returns = skill.get("return_to_parent") or []
    if not isinstance(returns, list) or not returns:
        raise HarnessConfigError(f"skill {desk!r} return_to_parent is empty")
    ret = "\n".join(f"- {item}" for item in returns)
    hints = str(skill.get("hints") or "").strip()
    success = str(skill.get("success") or "").strip()
    if not success:
        raise HarnessConfigError(f"skill {desk!r} success is empty")
    if not assignment.strip():
        raise HarnessConfigError("planner returned an empty user_request")
    parts = [
        f"You are working the {desk} lens on {ticker} (stock harness).",
        "",
        "## User request (do this spawn thoroughly)",
        "",
        assignment.strip(),
        "",
        "## Constraints",
        "",
        "- Cite every concrete figure with [source: …]. Do not invent numbers.",
        "- HOLD / NEUTRAL / MIXED are first-class when the evidence is split.",
        "- You cannot spawn children. Return when this assignment is done.",
        f"- Required tools for this skill (honor them): {req}",
        f"- Tools bound on this spawn: {', '.join(bound) or '(none)'}",
        "",
        "## Prior findings (re-verify; may be stale)",
        "",
        findings.strip() or "(none yet)",
        "",
        "## Hints",
        "",
        hints or "(none)",
        "",
        "## Verify",
        "",
        f"- {success}",
        "- Every number is tagged. Last line DESK_BIAS unless the skill says otherwise.",
        "",
        "## Return to parent",
        "",
        ret,
    ]
    return "\n".join(parts)


def plan_spawn(
    *,
    spec: SpawnSpec,
    state: HarnessState,
    cfg: Dict[str, Any],
    runtime: Any = None,
    llm: Any = None,
    writer: Optional[Callable[..., str]] = None,
) -> str:
    """Build the full child instruction. Empty assignment is an error."""
    desk = (spec.desk or "").strip()
    if spec.kind == "generic":
        desk = "generic_investigator"
    if not desk:
        raise HarnessConfigError("planner needs a desk or generic spawn")
    skill = skill_for(cfg, desk)
    bound = _bound_tools(desk, spec, runtime)
    findings = _prior_findings(state, desk)
    thought = _last_thought(state)
    brief = (spec.brief or "").strip()
    write = writer or cfg.get("planner_writer")
    if write is None:
        from trading_crew._common import get_llm

        conductor_llm = llm or cfg.get("planner_llm") or get_llm(
            temperature=float(cfg["conductor_temperature"])
        )

        def write(**kwargs: Any) -> str:
            return _llm_assignment(llm=conductor_llm, **kwargs)

    assignment = write(
        ticker=state.ticker,
        desk=desk,
        brief=brief,
        thought=thought,
        skill=skill,
        board=state.coverage_board(),
        findings=findings,
        bound=bound,
    )
    text = assemble_instruction(
        ticker=state.ticker,
        desk=desk,
        assignment=assignment,
        skill=skill,
        bound=bound,
        findings=findings,
    )
    logger.info("harness planner wrote %d chars for %s on %s", len(text), desk, state.ticker)
    return text
