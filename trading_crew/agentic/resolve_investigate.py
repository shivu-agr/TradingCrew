"""Resolve-time investigation: why the tape disagreed with the call.

The prediction desks do not retrieve past episodes. After a path is
priced, the Resolve LLM may call the same live tools (web, news,
filings, forwards) to name the catalyst and which desk or tool
misread it. Hermes consumes that structured text — it does not let
the LLM pick a knob freely.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from trading_crew.agentic.graph.playbook import PLAYBOOK_ACTIONS

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3

# Tools the Resolve investigator may name. Arguments are taken only from
# the plan (plus ticker filled from the episode when omitted).
RESOLVE_TOOLS: Dict[str, Tuple[str, ...]] = {
    "research_web": ("query",),
    "get_news": ("ticker", "query"),
    "get_global_news": ("ticker", "query"),
    "get_report_insights": ("ticker",),
    "get_forward_estimates": ("ticker",),
    "get_commodity_news": ("ticker",),
}


class ToolCallSpec(BaseModel):
    tool: str = ""
    arguments: Dict[str, str] = Field(default_factory=dict)
    why: str = ""

    @field_validator("tool", mode="before")
    @classmethod
    def _clean_tool(cls, v):
        name = (v or "").strip()
        return name if name in RESOLVE_TOOLS else ""


class InvestigationPlan(BaseModel):
    need_tools: bool = False
    calls: List[ToolCallSpec] = Field(default_factory=list)
    tape_question: str = ""

    @field_validator("calls", mode="before")
    @classmethod
    def _clean_calls(cls, v):
        if not v:
            return []
        if isinstance(v, dict):
            v = [v]
        return list(v)[:MAX_TOOL_CALLS]


def _invoke_tool(name: str, arguments: Dict[str, str]) -> str:
    if name == "get_commodity_news":
        from commodity_crew.tools import get_commodity_news as fn
    else:
        from trading_crew import tools as equity

        fn = getattr(equity, name, None)
        if fn is None:
            raise KeyError(name)
    allowed = RESOLVE_TOOLS[name]
    kwargs = {k: arguments[k] for k in allowed if arguments.get(k)}
    raw = getattr(fn, "func", None)
    if raw is None:
        return str(fn.run(**kwargs) if hasattr(fn, "run") else fn(**kwargs))
    return str(raw(**kwargs) or "")


def run_investigation(
    plan: InvestigationPlan,
    *,
    ticker: str,
    allow_tools: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    """Execute the plan. Unknown / empty tools are skipped, not invented."""
    if not allow_tools or not plan.need_tools:
        return "", []
    executed: List[Dict[str, str]] = []
    chunks: List[str] = []
    if plan.tape_question.strip():
        chunks.append(f"Question: {plan.tape_question.strip()}")
    for spec in plan.calls:
        if not spec.tool:
            continue
        args = dict(spec.arguments or {})
        if "ticker" in RESOLVE_TOOLS[spec.tool] and not args.get("ticker"):
            args["ticker"] = ticker
        if spec.tool == "research_web" and not args.get("query"):
            continue
        try:
            text = _invoke_tool(spec.tool, args)
        except Exception:
            logger.warning("resolve investigate tool %s failed", spec.tool, exc_info=True)
            continue
        if not (text or "").strip():
            continue
        executed.append({"tool": spec.tool, "why": spec.why, **args})
        chunks.append(f"## {spec.tool}\n{text.strip()[:4000]}")
        if len(executed) >= MAX_TOOL_CALLS:
            break
    return "\n\n".join(chunks), executed


def plan_prompt(
    *,
    symbol: str,
    decision_ts: str,
    as_of: str,
    side: str,
    outcome_label: str,
    market_dir: str,
    market_return: Optional[float],
    state_summary: str,
    desk_leans: Optional[Dict[str, Any]],
    hurt_desks: Sequence[str],
) -> str:
    leans = desk_leans or {}
    lean_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(leans.items())) or "(none)"
    mkt = f"{market_return:+.2%}" if market_return is not None else "n/a"
    tools = ", ".join(RESOLVE_TOOLS)
    return (
        "You are investigating a priced prediction. Decide whether you need "
        "live tools to explain WHY the tape moved and which desk or tool "
        "misread or over-weighted the evidence.\n"
        "This is after the holding window — not a live feature. Do not invent "
        "numbers. If state_summary already names the catalyst, set need_tools "
        "false.\n\n"
        f"Symbol: {symbol}\n"
        f"Decision: {decision_ts}  As-of (path end): {as_of}\n"
        f"Predicted side: {side}  Label: {outcome_label}\n"
        f"Tape: {mkt} ({market_dir or 'n/a'})\n"
        f"Hurt desks (lean vs tape): {', '.join(hurt_desks) or '(none)'}\n"
        f"Desk leans:\n{lean_lines}\n\n"
        f"What the desks saw (state_summary):\n{state_summary}\n\n"
        f"Allowed tools: {tools}\n"
        "research_web needs query. News/reports/forwards need ticker "
        f"(default {symbol}). At most "
        f"{MAX_TOOL_CALLS} calls.\n"
    )


_PLAN_JSON_HINT = (
    "\n\nReturn ONLY a JSON object with keys:\n"
    '  "need_tools": boolean,\n'
    '  "tape_question": string,\n'
    '  "calls": array of {tool, arguments, why} — empty if need_tools is false.\n'
)


def call_structured(llm: Any, prompt: str, model: type, hint: str) -> Any:
    """CrewAI LLM.call with optional response_format, then validate."""
    try:
        raw = llm.call(
            messages=[{"role": "user", "content": prompt}],
            response_format=model,
        )
    except TypeError:
        raw = llm.call(
            messages=[{"role": "user", "content": prompt + hint}],
        )
    if isinstance(raw, model):
        return raw
    if isinstance(raw, dict):
        return model.model_validate(raw)
    if isinstance(raw, str):
        import json

        text = raw.strip()
        try:
            return model.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            obj = _first_json_object(text)
            if obj is None:
                raise
            return model.model_validate(obj)
    if hasattr(raw, "model_dump"):
        return model.model_validate(raw.model_dump())
    raise TypeError(f"unsupported structured response: {type(raw)!r}")


def plan_investigation(llm: Any, **kwargs: Any) -> InvestigationPlan:
    return call_structured(
        llm,
        plan_prompt(**kwargs),
        InvestigationPlan,
        _PLAN_JSON_HINT,
    )


def _first_json_object(text: str) -> Optional[dict]:
    import json

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def sanitize_playbook(value: Any) -> str:
    key = (value or "").strip()
    return key if key in PLAYBOOK_ACTIONS and key != "default" else ""
