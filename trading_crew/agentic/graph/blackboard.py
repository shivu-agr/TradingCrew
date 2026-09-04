"""Run-scoped evidence blackboard + bounded request_evidence queue.

The graph runtime (and the researcher tools) share one blackboard per
kickoff via a contextvar so tools do not need a ticker-keyed global.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trading_crew.agentic.graph.spec import (
    MAX_EVIDENCE_REQUESTS_PER_RUN,
    RESEARCHER_EVIDENCE_TOOLS,
    role_to_agent_key,
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRequest:
    target_desk: str  # agent_key
    question: str
    tool_name: str = ""
    status: str = "queued"  # queued | filled | rejected
    excerpt: str = ""


@dataclass
class Blackboard:
    ticker: str
    desk_reports: Dict[str, str] = field(default_factory=dict)  # agent_key → text
    desk_leans: Dict[str, str] = field(default_factory=dict)    # agent_key → lean
    addenda: Dict[str, List[str]] = field(default_factory=dict)  # extra tool pulls
    requests: List[EvidenceRequest] = field(default_factory=list)

    def record_desk(self, role_or_key: str, text: str, lean: Optional[str] = None) -> None:
        key = role_to_agent_key(role_or_key) or (role_or_key or "").strip().lower()
        if not key:
            return
        self.desk_reports[key] = text or ""
        if lean:
            self.desk_leans[key] = lean.strip().lower()

    def queue_request(self, target_desk: str, question: str, tool_name: str = "") -> str:
        key = (role_to_agent_key(target_desk) or target_desk or "").strip().lower()
        allowed_tools = RESEARCHER_EVIDENCE_TOOLS.get(key) or []
        if not key or key not in RESEARCHER_EVIDENCE_TOOLS:
            return (
                f"Rejected: '{target_desk}' is not an analyst desk that can "
                "re-query. Use market_analyst, news_analyst, fundamentals_analyst, "
                "reports_analyst, or macro_analyst."
            )
        pending = [r for r in self.requests if r.status == "queued"]
        if len(self.requests) >= MAX_EVIDENCE_REQUESTS_PER_RUN:
            return (
                f"Rejected: this run already used {MAX_EVIDENCE_REQUESTS_PER_RUN} "
                "evidence requests (hard cap)."
            )
        tool = (tool_name or "").strip()
        if tool and tool not in allowed_tools:
            return (
                f"Rejected: tool '{tool}' is not owned by {key}. "
                f"Allowed: {', '.join(allowed_tools)}."
            )
        if not tool:
            tool = allowed_tools[0]
        req = EvidenceRequest(
            target_desk=key, question=(question or "").strip(), tool_name=tool
        )
        self.requests.append(req)
        return (
            f"Queued evidence request #{len(self.requests)}: {key}.{tool} "
            f"— {req.question or '(no question)'}. The owning desk will be "
            "re-queried before the next debate round; call lookup_blackboard "
            "or read_scratchpad then to read the addendum."
        )

    def render(self) -> str:
        """Full board — no silent clip. The UI consumes this; agents use compact."""
        return self._render(compact=False)

    def render_compact(self, tldr_chars: int = 800) -> str:
        """Leans + TLDR excerpts + addenda — what debate tools should see."""
        return self._render(compact=True, tldr_chars=tldr_chars)

    def _render(self, compact: bool, tldr_chars: int = 800) -> str:
        from trading_crew.agentic.graph.scratchpad import compact_excerpt

        lines = [f"## EVIDENCE BLACKBOARD ({self.ticker})"]
        if compact:
            lines[0] += " — compact"
        if not self.desk_leans and not self.desk_reports:
            lines.append("No analyst reports captured yet.")
            return "\n".join(lines)
        lines.append("Desk leans:")
        for desk, lean in sorted(self.desk_leans.items()):
            lines.append(f"- {desk}: {lean}")
        for desk, text in sorted(self.desk_reports.items()):
            lean = self.desk_leans.get(desk)
            header = f"\n### {desk}"
            if lean:
                header += f"  (DESK_BIAS: {lean.upper()})"
            lines.append(header)
            body = (text or "").strip() or "(empty report)"
            if compact:
                body = compact_excerpt(body, tldr_chars) or "(empty report)"
            lines.append(body)
            extras = self.addenda.get(desk) or []
            for i, ex in enumerate(extras, 1):
                lines.append(f"\n#### {desk} addendum {i}\n{ex}")
        filled = [r for r in self.requests if r.status == "filled" and r.excerpt]
        for r in filled:
            lines.append(
                f"\n### filled request {r.target_desk}.{r.tool_name}\n"
                f"Q: {r.question}\n{r.excerpt}"
            )
        return "\n".join(lines)

    def to_payload(self) -> Dict[str, object]:
        """Structured snapshot for the UI + run record (full text)."""
        desks = []
        for desk, text in sorted(self.desk_reports.items()):
            desks.append({
                "desk": desk,
                "lean": self.desk_leans.get(desk),
                "report": text or "",
                "addenda": list(self.addenda.get(desk) or []),
            })
        return {
            "ticker": self.ticker,
            "markdown": self.render(),
            "desks": desks,
            "requests": [
                {
                    "target_desk": r.target_desk,
                    "question": r.question,
                    "tool_name": r.tool_name,
                    "status": r.status,
                    "excerpt": r.excerpt,
                }
                for r in self.requests
            ],
        }


_CURRENT: ContextVar[Optional[Blackboard]] = ContextVar(
    "tradingcrew_blackboard", default=None
)
_TOKEN: Optional[Token] = None


def current() -> Optional[Blackboard]:
    return _CURRENT.get()


def begin_run(ticker: str) -> Blackboard:
    global _TOKEN
    bb = Blackboard(ticker=ticker)
    _TOKEN = _CURRENT.set(bb)
    return bb


def end_run() -> None:
    global _TOKEN
    tok = _TOKEN
    _TOKEN = None
    if tok is not None:
        try:
            _CURRENT.reset(tok)
        except Exception:
            _CURRENT.set(None)
    else:
        _CURRENT.set(None)


def drain_queued_requests(ticker: str) -> int:
    """Execute queued tool pulls without an LLM.  Returns how many filled."""
    bb = current()
    if bb is None:
        return 0
    filled = 0
    for req in bb.requests:
        if req.status != "queued":
            continue
        excerpt = _run_tool(req.tool_name, ticker)
        if excerpt.startswith("Rejected:") or excerpt.startswith("Tool failed"):
            req.status = "rejected"
            req.excerpt = excerpt
            continue
        req.status = "filled"
        req.excerpt = excerpt
        bb.addenda.setdefault(req.target_desk, []).append(
            f"[{req.tool_name} | q: {req.question}]\n{excerpt}"
        )
        filled += 1
    return filled


def _run_tool(tool_name: str, ticker: str) -> str:
    try:
        from trading_crew.tools import ALL_TOOLS
    except Exception as exc:  # noqa: BLE001
        return f"Tool failed: {exc}"
    tool = ALL_TOOLS.get(tool_name)
    if tool is None:
        return f"Rejected: unknown tool '{tool_name}'."
    try:
        if hasattr(tool, "run"):
            out = tool.run(ticker=ticker)
        elif callable(tool):
            out = tool(ticker)
        else:
            return f"Rejected: tool '{tool_name}' is not callable."
    except TypeError:
        try:
            out = tool.run(ticker)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("evidence tool %s failed: %s", tool_name, exc)
            return f"Tool failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence tool %s failed: %s", tool_name, exc)
        return f"Tool failed: {exc}"
    return str(out or "")
