"""Route resolved-decision lessons to EVERY agent, not just the PM (Phase 3).

The RLMF few-shot prefix (L5) only ever reached the Portfolio Manager — so
the analysts, researchers, trader, and risk desks never saw *why* their
inputs led a past decision astray.  This module closes that gap.

For each resolved episode the L2 reflection emits a **structured** lesson:
which desks it concerns (``desks``), what went wrong (``error_class``), and
the concrete change to make (``recommended_change``).  We route each lesson
to exactly the desks the reflection named — a news-driven miss goes to the
News Analyst, a sizing error to the risk desks, a horizon error to the
Trader — and build a compact per-role feedback block the crew prepends to
that agent's backstory.

Routing is by the reflection's explicit ``desks`` tags (a controlled
vocabulary the LLM must choose from), NOT by fuzzy keyword matching, so a
lesson reaches precisely the desk(s) responsible for it.  Deterministic +
dependency-light: reuses the RLMF dataset rows (which carry the structured
reflection fields + ``reward``).  Gated by ``TRADINGCREW_AGENT_FEEDBACK_ENABLED``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controlled vocabulary — the single source of truth for the reflection LLM,
# the RLMF dataset, and the calibration analytics.
# ---------------------------------------------------------------------------

# Desk keys the reflection may attribute a lesson to.  Each maps 1:1 to a
# crew agent key EXCEPT ``risk`` which fans out to the three risk desks (a
# sizing/risk lesson is relevant to all of Aggressive / Neutral / Conservative).
CANONICAL_DESKS: tuple[str, ...] = (
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "reports_analyst",
    "macro_analyst",
    "bull_researcher",
    "bear_researcher",
    "research_manager",
    "trader",
    "risk",
    "portfolio_manager",
)

# ``risk`` → the three concrete risk-desk agent keys.
_DESK_EXPANSION: Dict[str, tuple[str, ...]] = {
    "risk": ("risk_aggressive", "risk_neutral", "risk_conservative"),
}

# Controlled failure-mode vocabulary (what went wrong).  Lets the calibration
# / RLMF layers segment learning by error type.  ``correct`` = the decision
# and outcome aligned (a pattern to emulate rather than a miss to fix).
ERROR_CLASSES: tuple[str, ...] = (
    "thesis_wrong",
    "timing_early",
    "timing_late",
    "sizing_too_large",
    "sizing_too_small",
    "stop_too_tight",
    "target_unrealistic",
    "missed_catalyst",
    "macro_regime_missed",
    "overconfident",
    "underconfident",
    "noise",
    "correct",
)

# The concrete crew agent keys that can receive routed feedback (i.e. the
# canonical desks with ``risk`` expanded).
ALL_FEEDBACK_ROLES: tuple[str, ...] = tuple(
    k
    for d in CANONICAL_DESKS
    for k in (_DESK_EXPANSION[d] if d in _DESK_EXPANSION else (d,))
)


def expand_desks(desks) -> Set[str]:
    """Expand a list of canonical desk tags into concrete crew agent keys.

    Unknown / blank tags are dropped (never invented).  ``risk`` fans out to
    the three risk desks so a single sizing lesson reaches all of them.
    """
    valid = set(ALL_FEEDBACK_ROLES)
    out: Set[str] = set()
    for d in desks or []:
        key = (d or "").strip().lower()
        if not key:
            continue
        if key in _DESK_EXPANSION:
            out.update(_DESK_EXPANSION[key])
        elif key in valid:
            out.add(key)
    return out


def _matches_role(row, agent_key: str) -> bool:
    """True iff this row has a teaching lesson for ``agent_key``.

    Prefers per-desk ``desk_lessons``.  A lesson counts if it names this
    desk AND is actually instructional (hurt vs the tape, or it carries
    ``what_they_got_wrong`` / ``recommended_change``).  Falls back to the
    reflection's ``desks`` tags when there are no lessons.
    """
    lessons = list(getattr(row, "desk_lessons", None) or [])
    if lessons:
        hit_teaching = False
        named_here = False
        for les in lessons:
            if isinstance(les, dict):
                desk = (les.get("desk") or "").strip().lower()
                verdict = (les.get("verdict") or "").strip().lower()
                wrong = (les.get("what_they_got_wrong") or "").strip()
                change = (les.get("recommended_change") or "").strip()
            else:
                desk = (getattr(les, "desk", "") or "").strip().lower()
                verdict = (getattr(les, "verdict", "") or "").strip().lower()
                wrong = (getattr(les, "what_they_got_wrong", "") or "").strip()
                change = (getattr(les, "recommended_change", "") or "").strip()
            if not desk or agent_key not in expand_desks([desk]):
                continue
            named_here = True
            if verdict == "hurt" or wrong or change:
                hit_teaching = True
                break
        if hit_teaching:
            return True
        if named_here:
            return False
    return agent_key in expand_desks(getattr(row, "desks", None))


def _lesson_for_role(row, agent_key: str) -> Optional[dict]:
    for les in getattr(row, "desk_lessons", None) or []:
        payload = les if isinstance(les, dict) else {
            "desk": getattr(les, "desk", ""),
            "lean_at_decision": getattr(les, "lean_at_decision", ""),
            "what_they_got_wrong": getattr(les, "what_they_got_wrong", ""),
            "recommended_change": getattr(les, "recommended_change", ""),
            "verdict": getattr(les, "verdict", ""),
        }
        desk = (payload.get("desk") or "").strip().lower()
        if agent_key in expand_desks([desk] if desk else []):
            return payload
    return None


def _format_row(row, agent_key: str = "") -> str:
    symbol = getattr(row, "symbol", "?")
    ts = (getattr(row, "decision_ts", "") or "")[:10]
    action = getattr(row, "action", "?")
    realised = getattr(row, "realised_return", None)
    alpha = getattr(row, "alpha_return", None)
    error_class = (getattr(row, "error_class", None) or "").strip()
    label = (getattr(row, "outcome_label", None) or "").strip()
    reflection = getattr(row, "reflection", "") or ""
    rec = (getattr(row, "recommended_change", None) or "").strip()
    perf = ""
    if realised is not None:
        perf = f", realised {realised:+.2%}"
    if alpha is not None:
        perf += f" (alpha {alpha:+.2%})"
    tag = f" [{error_class}]" if error_class else ""
    lab = f" → {label}" if label else ""
    lesson = _lesson_for_role(row, agent_key) if agent_key else None
    lines = [f"- [{symbol} {ts}] {action}{lab}{perf}{tag}"]
    if lesson:
        lean = (lesson.get("lean_at_decision") or "").strip()
        verdict = (lesson.get("verdict") or "").strip()
        wrong = (lesson.get("what_they_got_wrong") or "").strip()
        change = (lesson.get("recommended_change") or rec).strip()
        bits = []
        if lean:
            bits.append(f"your lean was {lean}")
        if verdict:
            bits.append(f"vs tape: {verdict}")
        if bits:
            lines.append("  " + "; ".join(bits) + ".")
        if wrong:
            lines.append(f"  What you got wrong: {wrong[:400]}")
        if change:
            lines.append(f"  Do next time: {change[:200]}")
    else:
        if reflection.strip():
            lines.append(f"  {reflection.strip()[:400]}")
        if rec:
            lines.append(f"  Do next time: {rec[:200]}")
    return "\n".join(lines)


def build_role_feedback(rows, agent_key: str, *, k: int = 3) -> str:
    """Return a compact feedback block for ``agent_key`` (or "" if none).

    ``rows`` are RLMF dataset rows (or any object exposing ``reflection`` /
    ``reward`` / ``symbol`` / ``decision_ts`` / ``action`` /
    ``realised_return`` / ``alpha_return``).  We surface the ``k`` most
    informative lessons for this desk, ranked by ``|reward|`` so both big
    wins and big misses teach.
    """
    relevant = [r for r in rows if _matches_role(r, agent_key)]
    if not relevant:
        return ""
    relevant.sort(key=lambda r: abs(getattr(r, "reward", 0.0) or 0.0), reverse=True)
    top = relevant[:k]
    lines = [
        "## DESK FEEDBACK — your past predictions vs the tape",
        "After each episode resolved, the market labelled the call WIN or LOSS "
        "and attributed the miss to the desks whose lean disagreed with the tape. "
        "The items below are YOUR step's lessons — not a generic PM note. "
        "Weigh them on a similar setup; do not fabricate data to match them.",
        "",
    ]
    lines.extend(_format_row(r, agent_key) for r in top)
    lines.append("")
    return "\n".join(lines)


def build_feedback_map(
    *,
    symbol: Optional[str] = None,
    memory=None,
    k: int = 3,
    workflow_version: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, str]:
    """Build the per-role feedback blocks from resolved episodes.

    Returns ``{agent_key: block}`` for every role that has ≥1 relevant
    lesson.  Cohort filters (Phase 0) are passed through so a caller can
    restrict feedback to a trustworthy workflow version.
    """
    from trading_crew.agentic.rlmf import build_dataset

    try:
        rows = build_dataset(
            memory=memory,
            symbol=symbol,
            workflow_version=workflow_version,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001 — feedback is advisory, never fatal
        logger.warning("agent feedback: dataset build failed (%s)", exc)
        return {}
    # Rows that can teach a specific desk: LLM reflection OR deterministic
    # desk_lessons written on skip_llm resolve.
    rows = [
        r for r in rows
        if getattr(r, "reflection", None)
        or getattr(r, "desk_lessons", None)
        or getattr(r, "desks", None)
    ]
    if not rows:
        return {}
    out: Dict[str, str] = {}
    for agent_key in ALL_FEEDBACK_ROLES:
        block = build_role_feedback(rows, agent_key, k=k)
        if block:
            out[agent_key] = block
    return out


def agent_feedback_enabled() -> bool:
    """Gate for the per-agent feedback injection (default ON).

    Safe to leave on: with no resolved episodes the feedback map is empty
    and nothing is injected, so a cold install is unaffected.  Set
    ``TRADINGCREW_AGENT_FEEDBACK_ENABLED=0`` to disable (e.g. to A/B the
    effect, or while the workflow is still being stabilised).
    """
    raw = os.environ.get("TRADINGCREW_AGENT_FEEDBACK_ENABLED", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")
