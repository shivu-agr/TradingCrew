"""Run-scoped shared debate scratch pad (bull / bear / research manager).

Debate turns do not receive the five full analyst reports or prior-round
prose in CrewAI ``context``. They read this compact pad instead, then
write new claims / tool findings back so the other side does not need a
redump of the whole discussion.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_TLDR_HEAD = re.compile(
    r"(?im)^(?:#{1,3}\s*)?(?:\*\*)?tldr(?:\*\*)?\s*:?\s*\n?(.*)",
    re.DOTALL,
)
_NEXT_SECTION = re.compile(r"(?im)^(?:#{1,3}\s|DESK_BIAS:)")


def compact_excerpt(text: str, limit: int = 800) -> str:
    """Prefer a TLDR block; otherwise the first ``limit`` characters."""
    raw = (text or "").strip()
    if not raw:
        return ""
    match = _TLDR_HEAD.search(raw)
    if match:
        rest = match.group(1)
        stop = _NEXT_SECTION.search(rest)
        chunk = (rest[: stop.start()] if stop else rest).strip()
        if chunk:
            return chunk[:limit]
    return raw[:limit]


@dataclass
class ScratchEntry:
    kind: str  # seed | debate | note
    author: str
    text: str
    round_num: int = 0


@dataclass
class ScratchPad:
    ticker: str
    entries: List[ScratchEntry] = field(default_factory=list)

    def seed_analyst(self, desk: str, lean: Optional[str], report: str) -> None:
        author = (desk or "").strip() or "analyst"
        excerpt = compact_excerpt(report)
        lean_s = (lean or "").strip()
        text = excerpt if not lean_s else f"DESK_BIAS: {lean_s.upper()}\n{excerpt}"
        self._upsert(kind="seed", author=author, round_num=0, text=text)

    def append_debate(self, author: str, round_num: int, text: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        self._upsert(
            kind="debate",
            author=(author or "").strip() or "debate",
            round_num=int(round_num or 0),
            text=body,
        )

    def write_note(self, author: str, note: str) -> None:
        body = (note or "").strip()
        if not body:
            return
        self.entries.append(
            ScratchEntry(
                kind="note",
                author=(author or "").strip() or "debate",
                text=body,
                round_num=0,
            )
        )

    def _upsert(self, kind: str, author: str, round_num: int, text: str) -> None:
        for entry in self.entries:
            if (
                entry.kind == kind
                and entry.author == author
                and entry.round_num == round_num
            ):
                entry.text = text
                return
        self.entries.append(
            ScratchEntry(kind=kind, author=author, text=text, round_num=round_num)
        )

    def render(self) -> str:
        lines = [f"## DEBATE SCRATCH PAD ({self.ticker})"]
        if not self.entries:
            lines.append(
                "Empty — analysts have not finished yet, or no debate notes "
                "have been written."
            )
            return "\n".join(lines)

        seeds = [e for e in self.entries if e.kind == "seed"]
        debate = [e for e in self.entries if e.kind == "debate"]
        notes = [e for e in self.entries if e.kind == "note"]

        if seeds:
            lines.append("\n### Analyst TLDRs")
            for e in seeds:
                lines.append(f"\n#### {e.author}\n{e.text}")
        if debate:
            lines.append("\n### Debate turns")
            for e in debate:
                label = e.author
                if e.round_num:
                    label += f" · round {e.round_num}"
                lines.append(f"\n#### {label}\n{e.text}")
        if notes:
            lines.append("\n### Live notes (tool findings / claims)")
            for e in notes:
                lines.append(f"\n#### {e.author}\n{e.text}")
        return "\n".join(lines)

    def to_payload(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "markdown": self.render(),
            "entries": [
                {
                    "kind": e.kind,
                    "author": e.author,
                    "round": e.round_num,
                    "text": e.text,
                }
                for e in self.entries
            ],
        }


_CURRENT: ContextVar[Optional[ScratchPad]] = ContextVar(
    "tradingcrew_scratchpad", default=None
)
_TOKEN: Optional[Token] = None


def current() -> Optional[ScratchPad]:
    return _CURRENT.get()


def begin_run(ticker: str) -> ScratchPad:
    global _TOKEN
    pad = ScratchPad(ticker=ticker)
    _TOKEN = _CURRENT.set(pad)
    return pad


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
