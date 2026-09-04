"""Run transcript, reports, and compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from trading_crew.agentic.graph.scratchpad import compact_excerpt

_OVERFLOW_NEEDLES = (
    "context length",
    "maximum context",
    "context_window",
    "too many tokens",
    "prompt is too long",
    "context overflow",
    "token limit",
    "reduce the length",
    "context window",
)


def is_context_overflow(exc: BaseException) -> bool:
    """True when an LLM / provider error is a context-window overflow."""
    msg = str(exc).lower()
    return any(n in msg for n in _OVERFLOW_NEEDLES)


@dataclass
class HarnessState:
    ticker: str
    observations: List[Dict[str, Any]] = field(default_factory=list)
    reports: Dict[str, str] = field(default_factory=dict)
    tool_names: List[str] = field(default_factory=list)
    families: Set[str] = field(default_factory=set)
    spawned: List[str] = field(default_factory=list)
    edges: List[Dict[str, str]] = field(default_factory=list)
    skeptical: bool = False
    debate_used: int = 0
    risk_used: int = 0
    spawn_used: int = 0
    steps: int = 0
    compact_count: int = 0

    def add_observation(self, kind: str, title: str, body: str) -> None:
        self.observations.append({"kind": kind, "title": title, "body": body or ""})

    def add_edge(self, src: str, dst: str, kind: str = "activated") -> None:
        self.edges.append({"from": src, "to": dst, "kind": kind})

    def coverage_board(self) -> str:
        spawned = ", ".join(self.spawned) or "(none)"
        tools = ", ".join(dict.fromkeys(self.tool_names)) or "(none)"
        missing: List[str] = []
        if "market_analyst" not in self.spawned:
            missing.append("market_analyst")
        if "news_analyst" not in self.spawned:
            missing.append("news_analyst")
        if "fundamentals_analyst" not in self.spawned:
            missing.append("fundamentals_analyst")
        if "macro_analyst" not in self.spawned:
            missing.append("macro_analyst")
        if (
            "reports_analyst" not in self.spawned
            and "get_report_insights" not in self.tool_names
        ):
            missing.append("reports_analyst / get_report_insights")
        if "debate" not in self.spawned and not (self.reports.get("debate") or "").strip():
            missing.append("bundled debate (desk=debate)")
        if not (self.reports.get("bull_researcher") or "").strip():
            missing.append("bull_researcher")
        if not (self.reports.get("bear_researcher") or "").strip():
            missing.append("bear_researcher")
        if "risk" not in self.spawned and not (self.reports.get("risk") or "").strip():
            missing.append("bundled risk (desk=risk)")
        if not any(
            (self.reports.get(k) or "").strip()
            for k in ("risk_aggressive", "risk_conservative", "risk_neutral")
        ):
            missing.append("risk desk")
        miss = ", ".join(missing) or "(none — coverage complete)"
        return (
            f"Spawned: {spawned}\n"
            f"Tools: {tools}\n"
            f"Missing required lenses: {miss}\n"
            f"Steps used: {self.steps}. Debate spawns: {self.debate_used}. "
            f"Risk spawns: {self.risk_used}."
        )

    def compact_in_place(self, keep_recent: int = 4) -> int:
        """Replace older observation bodies with excerpts. Returns chars saved."""
        if keep_recent < 1:
            keep_recent = 1
        if len(self.observations) <= keep_recent:
            return 0
        saved = 0
        older = self.observations[:-keep_recent]
        for row in older:
            body = str(row.get("body") or "")
            excerpt = compact_excerpt(body, limit=240)
            saved += max(0, len(body) - len(excerpt))
            row["body"] = excerpt
            row["compacted"] = True
        self.compact_count += 1
        return saved

    def render(self, context_chars: int) -> str:
        def _join() -> str:
            board = "## Coverage\n" + self.coverage_board()
            blocks: List[str] = []
            for i, row in enumerate(self.observations, start=1):
                title = row.get("title") or row.get("kind") or f"obs-{i}"
                body = str(row.get("body") or "")
                blocks.append(f"### {i}. {title}\n{body}")
            return board + "\n\n" + "\n\n".join(blocks)

        text = _join()
        if len(text) <= context_chars:
            return text
        self.compact_in_place(keep_recent=4)
        text = _join()
        if len(text) <= context_chars:
            return text
        keep = self.observations[-4:]
        older = self.observations[:-4]
        index: List[str] = []
        for row in older:
            excerpt = compact_excerpt(str(row.get("body") or ""), limit=240)
            cite = excerpt or "(empty)"
            index.append(f"- {row.get('title')}: {cite}")
        head = (
            "## Coverage\n"
            + self.coverage_board()
            + "\n\n## Compacted earlier observations (citations kept)\n"
            + "\n".join(index)
        )
        tail = "\n\n".join(
            f"### {row.get('title')}\n{row.get('body') or ''}" for row in keep
        )
        out = f"{head}\n\n{tail}"
        if len(out) > context_chars:
            out = out[:context_chars]
        return out
