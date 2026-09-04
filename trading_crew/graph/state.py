"""LangGraph run state for the v3 stock workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph channel schema. Each key is its own channel so a node
    that returns ``{evidence_queued: True}`` does not wipe ``ticker``.
    """

    ticker: str
    playbook_key: str
    playbook_action: str
    skip_reports: bool
    debate_rounds_target: int
    debate_round: int
    extra_debate_used: bool
    evidence_budget: int
    evidence_used: int
    evidence_queued: bool
    desk_split: bool
    stance: str
    trader_reopened: bool
    reports: Dict[str, str]
    leans: Dict[str, str]
    last_bull: str
    last_bear: str
    thesis: str
    trade_plan: str
    risk_notes: Dict[str, str]
    quality_notes: str
    compliance_notes: str
    quality_review_enabled: bool
    compliance_enabled: bool
    ui_risk_rounds: int
    decision: Optional[Dict[str, Any]]
    decision_raw: str
    activated_edges: List[Dict[str, str]]


@dataclass
class RunState:
    ticker: str
    playbook_key: str = ""
    playbook_action: str = "default"
    skip_reports: bool = False
    debate_rounds_target: int = 2
    debate_round: int = 0
    extra_debate_used: bool = False
    evidence_budget: int = 2
    evidence_used: int = 0
    evidence_queued: bool = False
    desk_split: bool = False
    stance: str = ""
    trader_reopened: bool = False
    reports: Dict[str, str] = field(default_factory=dict)
    leans: Dict[str, str] = field(default_factory=dict)
    last_bull: str = ""
    last_bear: str = ""
    thesis: str = ""
    trade_plan: str = ""
    risk_notes: Dict[str, str] = field(default_factory=dict)
    quality_notes: str = ""
    compliance_notes: str = ""
    quality_review_enabled: bool = False
    compliance_enabled: bool = False
    ui_risk_rounds: int = 1
    decision: Optional[Dict[str, Any]] = None
    decision_raw: str = ""
    activated_edges: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class GraphKickoffResult:
    """CrewAI-shaped result so the existing runner M1–M7 path stays intact."""

    pydantic: Any = None
    raw: str = ""

    def __str__(self) -> str:
        return self.raw or ""


def leans_are_split(leans: Dict[str, str]) -> bool:
    sides = {str(v).strip().lower() for v in leans.values() if v}
    sides.discard("neutral")
    sides.discard("")
    return "bullish" in sides and "bearish" in sides


def parse_stance(text: str) -> str:
    raw = (text or "").upper()
    for tag in ("STANCE: MIXED", "STANCE: BEAR", "STANCE: BULL"):
        if tag in raw:
            return tag.split(": ", 1)[1]
    return ""
