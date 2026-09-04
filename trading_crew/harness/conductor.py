"""Conductor prompt and structured turn parsing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from trading_crew.agentic.resolve_investigate import call_structured
from trading_crew.horizons import Horizon
from trading_crew.schemas import HorizonForecasts

TurnAction = Literal["call_tool", "spawn", "spawn_parallel", "finish", "compact"]


class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class SpawnSpec(BaseModel):
    kind: Literal["desk", "generic"] = "desk"
    desk: str = ""
    brief: str = ""
    tools: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("desk") is None:
            out["desk"] = ""
        if out.get("brief") is None:
            out["brief"] = ""
        if out.get("tools") is None:
            out["tools"] = []
        if out.get("kind") in (None, ""):
            out["kind"] = "generic" if out.get("brief") else "desk"
        return out


_CATALOG_KINDS = {
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "reports_analyst",
    "macro_analyst",
    "debate",
    "bull_researcher",
    "bear_researcher",
    "research_manager",
    "trader",
    "risk",
    "risk_aggressive",
    "risk_conservative",
    "risk_neutral",
    "generic_investigator",
}


class ConductorTurn(BaseModel):
    thought: str = ""
    action: TurnAction
    tool: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    agents: List[SpawnSpec] = Field(default_factory=list)
    horizon_forecasts: Optional[HorizonForecasts] = None
    primary_horizon: Optional[Horizon] = None
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_turn(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("thought") is None:
            out["thought"] = ""
        if out.get("tool") is None:
            out["tool"] = ""
        if out.get("notes") is None:
            out["notes"] = ""
        if out.get("arguments") is None:
            out["arguments"] = {}
        if out.get("agents") is None:
            out["agents"] = []
        horizon = out.get("primary_horizon")
        if horizon in ("", None):
            out["primary_horizon"] = None
        action = str(out.get("action") or "").strip()
        if not action:
            if out.get("agents"):
                action = "spawn_parallel" if len(out["agents"]) > 1 else "spawn"
            elif out.get("tool"):
                action = "call_tool"
            elif out.get("horizon_forecasts"):
                action = "finish"
        out["action"] = action
        agents = []
        for row in out.get("agents") or []:
            if isinstance(row, str):
                agents.append({"kind": "desk", "desk": row})
                continue
            if not isinstance(row, dict):
                continue
            item = dict(row)
            kind = str(item.get("kind") or "").strip()
            if kind in _CATALOG_KINDS:
                item["desk"] = item.get("desk") or kind
                item["kind"] = "desk"
            elif kind not in ("desk", "generic", ""):
                item["brief"] = item.get("brief") or kind.replace("_", " ")
                item["kind"] = "generic"
                item["desk"] = "generic_investigator"
            agents.append(item)
        out["agents"] = agents
        hf = out.get("horizon_forecasts")
        if isinstance(hf, dict):
            missing = any(hf.get(k) in (None, "", {}) for k in ("SHORT", "MEDIUM", "LONG"))
            if missing and action != "finish":
                out["horizon_forecasts"] = None
        return out


_TURN_HINT = (
    "\n\nReturn ONLY a JSON object with keys:\n"
    '  "thought": string,\n'
    '  "action": "call_tool" | "spawn" | "spawn_parallel" | "finish" | "compact",\n'
    '  "tool": string (call_tool),\n'
    '  "arguments": object (call_tool),\n'
    '  "agents": array of {kind, desk, brief, tools} (spawn / spawn_parallel),\n'
    '  "horizon_forecasts": {SHORT, MEDIUM, LONG} each with stance, '
    "horizon_days, expected_return_pct, confidence, drivers, risks, "
    "falsifiers, sources (finish),\n"
    '  "primary_horizon": "SHORT" | "MEDIUM" | "LONG" (finish),\n'
    '  "notes": string.\n'
)


def build_system_prompt(
    *,
    ticker: str,
    standing: str,
    allowlist: List[str],
    catalog: List[str],
    spawn_left: int,
    steps_left: int,
    coverage: str = "",
) -> str:
    tools = ", ".join(allowlist + ["query_knowledge"])
    desks = ", ".join(catalog + ["generic_investigator"])
    board = f"\nCoverage board:\n{coverage}\n" if coverage.strip() else ""
    return (
        f"{standing}\n\n"
        f"Ticker: {ticker}\n"
        f"Step budget remaining: {steps_left} (ceiling, not a rush signal). "
        f"Sub-agent spawns remaining: {spawn_left}. "
        "There is no debate-round or risk-round quota.\n"
        f"{board}"
        f"Tools: {tools}\n"
        f"Spawn catalog: {desks}\n"
        "For spawn kind=desk set desk to a catalog key and write a short "
        "gap brief (what is missing, what to verify). A spawn planner "
        "expands that brief into a full child instruction — do not write "
        "the whole child prompt. For kind=generic write a brief and a "
        "tool subset; the child cannot spawn.\n"
        "call_tool arguments must include ticker when the tool needs it. "
        "Search tools must include query you wrote.\n"
        "action=compact shrinks earlier observations when the transcript "
        "is large — then keep investigating. Do not finish because context "
        "is long.\n"
        "finish only when the coverage board shows no missing required "
        "lenses, SHORT / MEDIUM / LONG forecasts are each cited, and a "
        "skeptical lens has run. Then the portfolio manager writes the "
        "primary ticket from the gathered evidence."
        + (
            " When few steps remain, finish from the evidence you have. "
            "Do not open another bundled debate."
            if steps_left <= 4
            else ""
        )
    )


def build_closing_prompt(
    *,
    ticker: str,
    standing: str,
    coverage: str = "",
    steps_left: int,
) -> str:
    """Prompt after max_steps — finish from the desks that already ran."""
    board = f"\nCoverage board:\n{coverage}\n" if coverage.strip() else ""
    return (
        f"{standing}\n\n"
        f"Ticker: {ticker}\n"
        "CLOSING PHASE. The investigation step budget is spent. "
        f"Closing turns left: {steps_left}. "
        "Do not spawn. Do not call_tool. compact is allowed once if the "
        "transcript is huge. Your job is action=finish.\n"
        f"{board}"
        "Write SHORT, MEDIUM, and LONG forecasts from the observations. "
        "They may disagree. If bull and bear or the horizons conflict, "
        "that is a HOLD — cite the specific desk findings in drivers and "
        "risks. Then the portfolio manager writes the ticket. "
        "Do not write that the step budget failed as the thesis."
    )


def next_turn(
    llm: Any,
    system: str,
    transcript: str,
    *,
    closing: bool = False,
) -> ConductorTurn:
    if closing:
        choose = (
            "The investigation budget is spent. Emit action=finish with "
            "independent SHORT / MEDIUM / LONG forecasts from the observations. "
            "Conflict is a HOLD. compact once if the transcript is huge."
        )
    else:
        choose = (
            "This is a long-horizon investigation. Choose the next action. "
            "Prefer a missing required lens on the coverage board over repeating "
            "a tool you already ran. Do not finish while the coverage board "
            "lists missing lenses. If the transcript is getting long, compact, "
            "then continue."
        )
    prompt = (
        f"{system}\n\n"
        f"## Observations so far\n{transcript or '(none yet)'}\n\n"
        f"{choose}"
    )
    return call_structured(llm, prompt, ConductorTurn, _TURN_HINT)


def parse_turn_dict(raw: Dict[str, Any]) -> ConductorTurn:
    return ConductorTurn.model_validate(raw)


def dumps_turn(turn: ConductorTurn) -> str:
    return json.dumps(turn.model_dump(mode="json"), indent=2)
