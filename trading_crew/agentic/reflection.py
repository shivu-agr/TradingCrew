"""Outcome resolution + post-hoc reflection — agentic learning Level 2.

Pipeline per pending episode:

1.  **Filter** episodes whose ``outcome_ts`` has elapsed and whose
    status is still ``PENDING`` (resolving an already-RESOLVED episode
    is a no-op).
2.  **Price the outcome** by pulling OHLCV for ``[decision_ts, outcome_ts]``
    via the same yfinance helper the chart UI uses.  Sign the realised
    return by the action's side so a successful short shows up as a
    positive number.
3.  **Score vs benchmark** by pulling SPY for the same window and
    computing ``alpha_return = realised − benchmark`` — this is what
    M6's Deflated Sharpe ultimately consumes.
4.  **Investigate + reflect.** Deterministic labels always land. When an
    LLM is available it may call live tools (web, news, filings,
    forwards) to name why the tape moved and which desk or tool misread
    it. That structured lesson is written on the 30-day review (and on
    the thesis clock) so Hermes can train. Prediction desks do not
    retrieve past episodes.
5.  **Persist** via ``EpisodicMemory.update_review`` / ``update_outcome``.

This is intentionally NOT wired to a scheduler — the user (or a cron)
calls ``resolve_pending_episodes`` explicitly so it's audit-traceable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from trading_crew.agentic.agent_feedback import CANONICAL_DESKS, ERROR_CLASSES
from trading_crew.agentic.memory import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
)
from trading_crew.agentic.execution.contracts import ActionSide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution record (returned to caller for UI display)
# ---------------------------------------------------------------------------


@dataclass
class ResolutionRecord:
    """What changed for a single episode during this resolution pass."""

    episode_id: str
    symbol: str
    decision_ts: str
    outcome_ts: str
    side: str
    realised_return: Optional[float]
    alpha_return: Optional[float]
    max_drawdown: Optional[float]
    reflection: Optional[str]
    status: str  # "RESOLVED", "ABANDONED", "SKIPPED_PENDING", "SKIPPED_NO_DATA"
    note: str = ""
    reflection_meta: Optional[Dict[str, Any]] = None
    outcome_label: Optional[str] = None
    market_return: Optional[float] = None
    market_reason: Optional[str] = None
    review: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "symbol": self.symbol,
            "decision_ts": self.decision_ts,
            "outcome_ts": self.outcome_ts,
            "side": self.side,
            "realised_return": self.realised_return,
            "alpha_return": self.alpha_return,
            "max_drawdown": self.max_drawdown,
            "reflection": self.reflection,
            "reflection_meta": self.reflection_meta,
            "status": self.status,
            "note": self.note,
            "outcome_label": self.outcome_label,
            "market_return": self.market_return,
            "market_reason": self.market_reason,
            "review": self.review,
        }


# ---------------------------------------------------------------------------
# LLM-side structured output for the reflection
# ---------------------------------------------------------------------------


class DeskLesson(BaseModel):
    """One descriptive lesson aimed at a single desk that drove the miss (or win)."""

    desk: str = Field(
        description=(
            "Crew agent key this lesson is for. Choose ONLY from: "
            f"{', '.join(CANONICAL_DESKS)}. Use 'risk' for sizing / stop / "
            "drawdown lessons."
        ),
    )
    lean_at_decision: str = Field(
        default="",
        description=(
            "That desk's directional lean at decision time if known "
            "(bullish / bearish / neutral). Empty if unknown."
        ),
    )
    what_they_got_wrong: str = Field(
        default="",
        max_length=400,
        description=(
            "Concrete, evidence-grounded description of what THIS desk "
            "predicted or emphasised and why it did not match the realised "
            "tape. Quote features from state_summary / desk leans — do NOT "
            "invent numbers. Empty only if the desk was correct."
        ),
    )
    recommended_change: str = Field(
        default="",
        max_length=200,
        description=(
            "One concrete change this desk should make next time it sees a "
            "similar setup (≤ 200 chars)."
        ),
    )

    @field_validator("desk", mode="before")
    @classmethod
    def _clean_desk(cls, v):
        key = (v or "").strip().lower()
        return key if key in set(CANONICAL_DESKS) else ""


class ReflectionResponse(BaseModel):
    """A bounded, evidence-grounded lesson the LLM emits after seeing
    the realised outcome.  Field descriptions double as prompts.
    """

    summary: str = Field(
        description=(
            "One short paragraph (≤ 600 characters) describing what worked, "
            "what didn't, and what an operator should adjust next time. "
            "Reference concrete features from the original state_summary; "
            "do NOT invent new numbers."
        ),
        max_length=800,  # leeway above 600 — LLMs sometimes overshoot
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How well the outcome could be attributed to the agent's decision "
            "(vs market drift). 1.0 = clearly the agent's call drove the result; "
            "0.0 = pure market noise."
        ),
    )
    key_lesson_tag: str = Field(
        max_length=40,
        description=(
            "Short tag (1-3 words, lowercase, snake_case) summarising the lesson "
            "category — e.g. 'momentum_overweight', 'stop_too_tight', "
            "'macro_risk_missed'. Used by retrieval to cluster similar lessons."
        ),
    )
    desks: List[str] = Field(
        default_factory=list,
        description=(
            "Which desk(s) this lesson concerns — the desks that most directly "
            "drove the outcome and should weigh it next time. Choose ONLY from: "
            f"{', '.join(CANONICAL_DESKS)}. Use 'risk' for any sizing / stop / "
            "drawdown / position-risk lesson. Return an empty list only if the "
            "outcome was pure market noise attributable to no desk."
        ),
    )
    error_class: str = Field(
        default="noise",
        description=(
            "The single failure mode that best classifies this decision. Choose "
            f"ONLY from: {', '.join(ERROR_CLASSES)}. Use 'correct' when the "
            "decision and outcome aligned (a pattern to emulate)."
        ),
    )
    recommended_change: str = Field(
        default="",
        max_length=200,
        description=(
            "One concrete, actionable change the named desk(s) should make when "
            "they next see a similar setup (≤ 200 chars). Empty if error_class is "
            "'correct' or 'noise'."
        ),
    )
    desk_lessons: List[DeskLesson] = Field(
        default_factory=list,
        description=(
            "Per-desk descriptive lessons for EVERY desk that made a wrong "
            "directional prediction vs the tape (and optionally the desks "
            "that were right but overruled). Each lesson must name a canonical "
            "desk and say what that step got wrong. Empty only for pure noise."
        ),
    )
    market_reason: str = Field(
        default="",
        max_length=800,
        description=(
            "Why the tape moved over this window, grounded in state_summary "
            "and any investigation tools. Empty if unknown."
        ),
    )
    unhelpful_tools: List[str] = Field(
        default_factory=list,
        description=(
            "Tool names whose data the desks over-weighted or that pushed "
            "the call onto the wrong side (e.g. get_report_insights, "
            "research_web, get_news, request_evidence)."
        ),
    )
    overemphasized: List[str] = Field(
        default_factory=list,
        description=(
            "Desks or lenses that were over-weighted vs the realised tape "
            "(canonical desk keys or short lens names: filings, news, "
            "macro, tape, crowd)."
        ),
    )
    suggested_playbook: str = Field(
        default="",
        description=(
            "Optional vote for a bounded graph action if the miss is "
            "structural. Must be one of the playbook actions or empty. "
            "Hermes treats this as evidence, not an install."
        ),
    )

    @field_validator("desks", mode="before")
    @classmethod
    def _clean_desks(cls, v):
        """Keep only recognised canonical desk tags (never invent a desk)."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        allowed = set(CANONICAL_DESKS)
        return [d for d in ((x or "").strip().lower() for x in v) if d in allowed]

    @field_validator("error_class", mode="before")
    @classmethod
    def _clean_error_class(cls, v):
        key = (v or "").strip().lower()
        return key if key in set(ERROR_CLASSES) else "noise"

    @field_validator("desk_lessons", mode="before")
    @classmethod
    def _clean_desk_lessons(cls, v):
        if not v:
            return []
        if isinstance(v, dict):
            v = [v]
        out = []
        allowed = set(CANONICAL_DESKS)
        for item in v:
            if isinstance(item, DeskLesson):
                if item.desk in allowed:
                    out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            desk = (item.get("desk") or "").strip().lower()
            if desk not in allowed:
                continue
            item = dict(item)
            item["desk"] = desk
            out.append(item)
        return out

    @field_validator("suggested_playbook", mode="before")
    @classmethod
    def _clean_playbook(cls, v):
        from trading_crew.agentic.resolve_investigate import sanitize_playbook

        return sanitize_playbook(v)

    @field_validator("unhelpful_tools", mode="before")
    @classmethod
    def _clean_tools(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [(x or "").strip() for x in v if (x or "").strip()]

    @field_validator("overemphasized", mode="before")
    @classmethod
    def _clean_over(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [(x or "").strip().lower() for x in v if (x or "").strip()]


# ---------------------------------------------------------------------------
# Price + outcome math
# ---------------------------------------------------------------------------


def _signed_realised_return(side: str, p_in: float, p_out: float) -> float:
    """Return signed by trade direction.

    BUY: long → return = (p_out / p_in) − 1
    SELL: short → return = (p_in / p_out) − 1  (i.e. negated)
    HOLD / ABSTAIN: no position taken — return is 0 by definition; the
        opportunity cost / would-be return is surfaced via the reflection
        text instead (so it's visible but doesn't pollute Sharpe/Sortino).
    """
    if p_in <= 0:
        return 0.0
    s = (side or "").upper()
    if s == "BUY":
        return (p_out / p_in) - 1.0
    if s == "SELL":
        return (p_in / p_out) - 1.0 if p_out > 0 else 0.0
    return 0.0


def _naive_day(ts_iso: str):
    """Floor an ISO stamp to a tz-naive calendar day.

    Episode clocks are often ``…T12:00:00+00:00`` (or any intra-day time).
    Daily OHLCV rows are midnight.  Comparing the raw timestamp to those
    rows misses the bar on the clock's own date — which is exactly when
    a 30-day review lands on the last tape day before a weekend.
    """
    import pandas as pd  # local import — keeps module import light
    t = pd.to_datetime(ts_iso, utc=True)
    return t.tz_convert(None).normalize()


def _price_at_or_after(df, ts_iso: str) -> Optional[Tuple[str, float, float, float]]:
    """Return ``(date_iso, open, close, high)`` for the first trading day on
    or after ``ts_iso``.  Tolerates weekends/holidays by walking forward.

    Returns None if no trading day is available within +14 calendar days.
    """
    if df is None or df.empty:
        return None
    try:
        target = _naive_day(ts_iso)
    except Exception:
        return None
    # df["Date"] is tz-naive after _fetch_ohlcv normalisation
    candidates = df[df["Date"] >= target]
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    return (
        row["Date"].strftime("%Y-%m-%d"),
        float(row["Open"]),
        float(row["Close"]),
        float(row["High"]),
    )


def _max_drawdown_over_holding(df, decision_ts: str, outcome_ts: str, side: str) -> Optional[float]:
    """Peak-to-trough drawdown during the holding window, signed for direction.

    For a long position, drawdown is from the highest close down to the
    lowest subsequent close.  For a short, we flip the series so peak/trough
    match the position's PnL trajectory.

    Returns a non-positive float (e.g. -0.07 means a 7% drawdown), or None
    when the window has fewer than 2 trading days.
    """
    if df is None or df.empty:
        return None
    try:
        lo = _naive_day(decision_ts)
        hi = _naive_day(outcome_ts)
    except Exception:
        return None
    window = df[(df["Date"] >= lo) & (df["Date"] <= hi)]
    if len(window) < 2:
        return None
    closes = window["Close"].astype(float).reset_index(drop=True)
    if (side or "").upper() == "SELL":
        # Invert so peak/trough align with short PnL.
        closes = 1.0 / closes
    running_peak = closes.cummax()
    drawdowns = closes / running_peak - 1.0
    return float(drawdowns.min())


@dataclass
class _PricedWindow:
    realised: float
    alpha: float
    max_drawdown: Optional[float]
    market_return: float
    benchmark_return: float
    side: str


def _price_holding(
    ep: Episode,
    df,
    benchmark_df,
    ts_out: str,
) -> Optional[_PricedWindow]:
    in_quote = _price_at_or_after(df, ep.decision_ts)
    out_quote = _price_at_or_after(df, ts_out)
    if in_quote is None or out_quote is None:
        return None
    side = (ep.action_proposal or {}).get("side", "")
    realised = _signed_realised_return(side, in_quote[2], out_quote[2])
    mdd = _max_drawdown_over_holding(df, ep.decision_ts, ts_out, side)
    if benchmark_df is not None and not benchmark_df.empty:
        bench_in = _price_at_or_after(benchmark_df, ep.decision_ts)
        bench_out = _price_at_or_after(benchmark_df, ts_out)
        if bench_in and bench_out and bench_in[2] > 0:
            benchmark_return = (bench_out[2] / bench_in[2]) - 1.0
        else:
            benchmark_return = 0.0
    else:
        benchmark_return = 0.0
    in_close = in_quote[2]
    out_close = out_quote[2]
    market_ret = ((out_close / in_close) - 1.0) if in_close > 0 else 0.0
    return _PricedWindow(
        realised=realised,
        alpha=realised - benchmark_return,
        max_drawdown=mdd,
        market_return=market_ret,
        benchmark_return=benchmark_return,
        side=side,
    )


def schedule_review_ts(decision_ts: str, calendar_days: int) -> str:
    return (_parse_ts(decision_ts) + timedelta(days=int(calendar_days))).isoformat()


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------


# Appended to the prompt only on the ``response_format``-less compatibility
# path so the model still emits a parseable structured object.
_REFLECTION_JSON_HINT = (
    "\n\nReturn ONLY a JSON object (no markdown fences, no prose) with exactly "
    "these keys:\n"
    '  "summary": string (≤ 600 chars, the one-paragraph lesson),\n'
    '  "confidence_score": number in [0, 1] (attribution to the agent vs market noise),\n'
    '  "key_lesson_tag": string (1-3 words, lowercase snake_case, e.g. "macro_risk_missed"),\n'
    f'  "desks": array of strings, each ONE of [{", ".join(CANONICAL_DESKS)}] '
    "(the desks responsible for the outcome; use \"risk\" for sizing/stop/drawdown),\n"
    f'  "error_class": string, ONE of [{", ".join(ERROR_CLASSES)}],\n'
    '  "recommended_change": string (≤ 200 chars, the concrete fix for the named desks),\n'
    '  "desk_lessons": array of objects {desk, lean_at_decision, what_they_got_wrong, recommended_change} '
    "— one object per desk that predicted the wrong way vs the tape,\n"
    '  "market_reason": string (why the tape moved),\n'
    '  "unhelpful_tools": array of tool names that pushed the wrong side,\n'
    '  "overemphasized": array of desk or lens names over-weighted,\n'
    '  "suggested_playbook": one playbook action or "".\n'
)


def _format_desk_leans(desk_leans: Optional[Dict[str, Any]]) -> str:
    if not desk_leans:
        return "(no per-desk leans were recorded for this decision)"
    lines = []
    for desk, lean in sorted(desk_leans.items()):
        lines.append(f"- {desk}: {lean}")
    return "\n".join(lines)


def _build_reflection_prompt(
    ep: Episode,
    realised_return: float,
    alpha_return: float,
    max_drawdown: Optional[float],
    benchmark_return: float,
    *,
    market_return: Optional[float] = None,
    outcome_label: Optional[str] = None,
    market_dir: str = "",
    hurt_desks: Optional[List[str]] = None,
    briefing: str = "",
) -> str:
    proposal = ep.action_proposal or {}
    side = (proposal.get("side") or "").upper()
    target_weight = proposal.get("target_weight")
    conviction = proposal.get("conviction_score")
    horizon = proposal.get("horizon_days")
    hurt = ", ".join(hurt_desks or []) or "(none — no desk lean disagreed with the tape)"
    mkt = (
        f"{market_return:+.2%}" if market_return is not None else "n/a"
    )
    body = (
        "You are a sober post-trade analyst.  An agentic system made a "
        "decision; the holding period has now elapsed and the realised "
        "outcome is known.  Produce a ONE-PARAGRAPH lesson AND a per-desk "
        "lesson for every step that predicted the wrong way vs the tape.\n\n"
        f"Symbol: {ep.symbol}\n"
        f"Decision timestamp: {ep.decision_ts}\n"
        f"Outcome timestamp:  {ep.outcome_ts}\n"
        f"Regime at decision: {ep.regime.value}\n"
        f"Action taken: {side} (target_weight={target_weight}, "
        f"conviction={conviction}, horizon_days={horizon})\n"
        f"Outcome label (deterministic, vs predicted side): {outcome_label or 'n/a'}\n"
        f"Unsigned market return over the window: {mkt} "
        f"(tape direction: {market_dir or 'n/a'})\n"
        f"\nWhat each analyst desk leaned at decision time:\n"
        f"{_format_desk_leans(getattr(ep, 'desk_leans', None))}\n"
        f"\nDesks whose lean DISAGREED with the tape (wrong prediction): {hurt}\n"
        f"Write a desk_lessons[] entry for EACH of those desks.  If the "
        "call was a LOSS and a dissenting desk was right but overruled, "
        "also add a lesson telling that desk to restate the falsifier more "
        "forcefully next time.\n"
        f"\nWhat the agents *saw* at decision time (state_summary):\n"
        f"{ep.state_summary}\n"
        f"\nWhat actually happened over the holding period:\n"
        f"- Realised return on the position: {realised_return:+.2%}\n"
        f"- Benchmark (SPY) return over same window: {benchmark_return:+.2%}\n"
        f"- Alpha vs benchmark: {alpha_return:+.2%}\n"
        f"- Max drawdown during hold: "
        f"{(f'{max_drawdown:+.2%}' if max_drawdown is not None else 'n/a')}\n"
        f"\nWrite the lesson now.  Rules:\n"
        f"- Reference concrete features from the state_summary; do not invent numbers.\n"
        f"- If the action was ABSTAIN/HOLD, comment on the OPPORTUNITY COST "
        f"(what the trade would have done) without claiming credit.\n"
        f"- Be specific about which evidence type (technical / fundamental / "
        f"macro / news / risk) most directly drove the outcome.\n"
        f"- If the outcome looks like noise rather than agent skill, say so.\n"
        f"\nThen classify the lesson so it can be routed to the right desks:\n"
        f"- desks: pick ONLY from [{', '.join(CANONICAL_DESKS)}]; name the "
        f"desk(s) that most directly drove the outcome. Use 'risk' for any "
        f"sizing / stop / drawdown / position-risk lesson. Empty only for pure noise.\n"
        f"- error_class: pick ONE from [{', '.join(ERROR_CLASSES)}] "
        f"('correct' if the call and outcome aligned).\n"
        f"- recommended_change: one concrete fix (≤ 200 chars) for those desks.\n"
        f"- desk_lessons: one object per responsible desk with what THEY got "
        f"wrong and the change THEY should make.\n"
        f"- market_reason: why the tape moved (from briefing/tools if present).\n"
        f"- unhelpful_tools: tools whose data were not helpful and pulled the "
        f"call onto the wrong side.\n"
        f"- overemphasized: desks or lenses that were over-weighted.\n"
        f"- suggested_playbook: empty, or one of skip_reports, debate_rounds_1, "
        f"debate_rounds_2, evidence_budget_0, evidence_budget_1, "
        f"extra_debate_on_split, no_trader_reopen, risk_rounds_2.\n"
    )
    if briefing.strip():
        body += (
            "\nInvestigation briefing (tool results; do not invent beyond this):\n"
            f"{briefing.strip()[:12000]}\n"
        )
    return body


def _generate_reflection(
    ep: Episode,
    realised_return: float,
    alpha_return: float,
    max_drawdown: Optional[float],
    benchmark_return: float,
    llm: Any,
    *,
    market_return: Optional[float] = None,
    outcome_label: Optional[str] = None,
    market_dir: str = "",
    hurt_desks: Optional[List[str]] = None,
    briefing: str = "",
) -> ReflectionResponse:
    """Ask the LLM for a structured reflection.  Returns the validated
    :class:`ReflectionResponse` (summary + confidence + tag + structured
    desks / error_class / recommended_change / desk_lessons)."""
    prompt = _build_reflection_prompt(
        ep, realised_return, alpha_return, max_drawdown, benchmark_return,
        market_return=market_return,
        outcome_label=outcome_label,
        market_dir=market_dir,
        hurt_desks=hurt_desks,
        briefing=briefing,
    )
    # Mirror the critic.py call style — crewai.LLM exposes a ``call``
    # method that accepts ``response_format=BaseModelClass`` for
    # constrained decoding.
    try:
        raw = llm.call(
            messages=[{"role": "user", "content": prompt}],
            response_format=ReflectionResponse,
        )
    except TypeError:
        # Some OpenAI-compatible providers (e.g. certain self-hosted vLLM
        # deployments) reject the ``response_format`` kwarg. Ask for strict
        # JSON in-prompt and parse it ourselves instead.
        raw = llm.call(
            messages=[{"role": "user", "content": prompt + _REFLECTION_JSON_HINT}],
        )
    return _coerce_response(raw)


def _deterministic_desk_lessons(
    desk_attr: List[Dict[str, str]],
    *,
    mkt_dir: str,
    market_ret: Optional[float],
    side: str,
    outcome_label: str,
) -> List[Dict[str, Any]]:
    move = f"{market_ret:+.2%}" if market_ret is not None else "n/a"
    rows: List[Dict[str, Any]] = []
    for row in desk_attr:
        hurt = row["verdict"] == "hurt"
        rows.append({
            "desk": row["desk"],
            "lean_at_decision": row["lean"],
            "verdict": row["verdict"],
            "what_they_got_wrong": (
                f"Leaned {row['lean']} while the tape was {mkt_dir} "
                f"({move}); predicted side was {side or '?'} "
                f"→ {outcome_label}."
                if hurt else ""
            ),
            "recommended_change": "",
        })
    return rows


def _tools_allowed() -> bool:
    from trading_crew.data_sources import pit

    return not pit.is_backfill()


def _llm_lesson(
    ep: Episode,
    *,
    llm: Any,
    realised: float,
    alpha: float,
    mdd: Optional[float],
    benchmark_return: float,
    market_ret: Optional[float],
    outcome_label: str,
    mkt_dir: str,
    hurt: List[str],
    as_of: str,
    desk_attr: List[Dict[str, str]],
    side: str,
) -> Tuple[Optional[str], Dict[str, Any], List[Dict[str, str]]]:
    """Plan tools, investigate, reflect. Returns text, meta extras, tool rows."""
    from trading_crew.agentic.resolve_investigate import (
        plan_investigation,
        run_investigation,
    )

    plan = plan_investigation(
        llm,
        symbol=ep.symbol,
        decision_ts=ep.decision_ts,
        as_of=as_of,
        side=side,
        outcome_label=outcome_label,
        market_dir=mkt_dir,
        market_return=market_ret,
        state_summary=ep.state_summary or "",
        desk_leans=getattr(ep, "desk_leans", None),
        hurt_desks=hurt,
    )
    briefing, calls = run_investigation(
        plan, ticker=ep.symbol, allow_tools=_tools_allowed(),
    )
    resp = _generate_reflection(
        ep, realised, alpha, mdd, benchmark_return, llm,
        market_return=market_ret,
        outcome_label=outcome_label,
        market_dir=mkt_dir,
        hurt_desks=hurt,
        briefing=briefing,
    )
    text = (
        f"[{resp.key_lesson_tag} | conf {resp.confidence_score:.2f} | "
        f"{outcome_label}] {resp.summary}"
    )
    llm_desks = list(resp.desks) or list(hurt)
    llm_lessons = [
        {
            "desk": les.desk,
            "lean_at_decision": les.lean_at_decision,
            "what_they_got_wrong": les.what_they_got_wrong,
            "recommended_change": les.recommended_change,
            "verdict": next(
                (r["verdict"] for r in desk_attr if r["desk"] == les.desk),
                "hurt",
            ),
        }
        for les in (resp.desk_lessons or [])
        if les.desk
    ]
    named = {d["desk"] for d in llm_lessons}
    for row in _deterministic_desk_lessons(
        desk_attr, mkt_dir=mkt_dir, market_ret=market_ret,
        side=side, outcome_label=outcome_label,
    ):
        if row["desk"] not in named and row.get("verdict") == "hurt":
            llm_lessons.append(row)
    extras = {
        "desks": llm_desks,
        "tag": resp.key_lesson_tag,
        "confidence": float(resp.confidence_score),
        "error_class": resp.error_class,
        "recommended_change": resp.recommended_change,
        "desk_lessons": llm_lessons,
        "market_reason": (resp.market_reason or "").strip(),
        "unhelpful_tools": list(resp.unhelpful_tools or []),
        "overemphasized": list(resp.overemphasized or []),
        "suggested_playbook": resp.suggested_playbook or "",
        "investigation": {
            "tape_question": plan.tape_question,
            "need_tools": plan.need_tools,
            "briefing_chars": len(briefing),
            "tool_calls": calls,
        },
    }
    return text, extras, calls


def _coerce_response(raw: Any) -> ReflectionResponse:
    """Normalise the LLM return into a ``ReflectionResponse``.

    Different LLM providers return slightly different shapes through CrewAI:
    the model class itself (already validated), a JSON string, a dict, or a
    Pydantic instance.  We accept all three uniformly so callers don't have
    to branch.  Plain-string returns (the ``response_format``-less path) may
    wrap the JSON in markdown fences or prose, so we extract the first
    balanced ``{...}`` object before validating.
    """
    if isinstance(raw, ReflectionResponse):
        return raw
    if isinstance(raw, dict):
        return ReflectionResponse.model_validate(raw)
    if isinstance(raw, str):
        import json

        text = raw.strip()
        try:
            return ReflectionResponse.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            obj = _extract_json_object(text)
            if obj is not None:
                return ReflectionResponse.model_validate(obj)
            raise
    # last-ditch — try .model_dump() roundtrip
    if hasattr(raw, "model_dump"):
        return ReflectionResponse.model_validate(raw.model_dump())
    raise TypeError(f"Unsupported reflection response shape: {type(raw)!r}")


def _extract_json_object(text: str) -> Optional[dict]:
    """Return the first balanced ``{...}`` JSON object found in ``text``.

    Handles fenced (```json … ```) and prose-wrapped model output.
    """
    import json

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _default_memory_path() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "memory" / "episodes.jsonl"


def resolve_pending_episodes(
    *,
    ticker: Optional[str] = None,
    now: Optional[datetime] = None,
    memory: Optional[EpisodicMemory] = None,
    llm: Optional[Any] = None,
    fetch_ohlcv: Optional[Any] = None,
    benchmark_symbol: str = "SPY",
    abandon_after_days: int = 14,
    skip_llm: bool = False,
    skip_tape_reason: Optional[bool] = None,
    tape_reason_fn: Optional[Callable[..., str]] = None,
) -> List[ResolutionRecord]:
    """Resolve every pending episode whose ``outcome_ts`` has elapsed.

    Parameters
    ----------
    ticker : optional ticker filter; if set, only episodes for this symbol are touched.
    now : reference clock (defaults to ``datetime.now(timezone.utc)``).  Injected for tests.
    memory : an ``EpisodicMemory`` instance.  Defaults to the standard
             ``~/.trading_crew/memory/episodes.jsonl`` path so the runner and
             this resolver share state.
    llm : a CrewAI-compatible LLM (must expose ``.call(messages, response_format)``).
          If None, ``skip_llm=True`` is implied and reflections are not generated.
    fetch_ohlcv : OHLCV fetcher (signature: ``(symbol, end_date, lookback_days) -> DataFrame``).
                  Defaults to the chart-pipeline helper.  Injected for tests.
    benchmark_symbol : the alpha benchmark (default SPY).
    abandon_after_days : if the outcome can't be priced and ``outcome_ts`` is
                         already this many days in the past, mark the episode
                         ABANDONED so it stops blocking the queue.
    skip_llm : if True, write outcomes but skip the LLM reflection step.

    Returns
    -------
    A list of ``ResolutionRecord`` for each episode considered (resolved,
    abandoned, or skipped).
    """
    now = now or datetime.now(timezone.utc)
    memory = memory or EpisodicMemory(_default_memory_path())
    if fetch_ohlcv is None:
        from web.backend.charts import _fetch_ohlcv as fetch_ohlcv  # local import — avoids web→agentic edge in tests

    from trading_crew.agentic.strategy import load_desk_strategy
    strategy = load_desk_strategy()
    wrote_review = False
    wrote_thesis = False

    records: List[ResolutionRecord] = []
    # Cache benchmark OHLCV once per resolver invocation; the SPY window
    # subsumes every episode window we'll touch.
    benchmark_df = None

    for ep in memory.all_episodes():
        if ep.outcome_status != OutcomeStatus.PENDING:
            continue
        if ticker and ep.symbol.upper() != ticker.upper():
            continue

        if not ep.review_ts:
            ep.review_ts = schedule_review_ts(
                ep.decision_ts, strategy.learning.calendar_days,
            )
            ep.review_status = ep.review_status or "PENDING"
            ep.review_embargo_days = strategy.learning.embargo_days
            memory.add(ep)

        review_dt = _parse_ts(ep.review_ts) if ep.review_ts else None
        review_due = (
            review_dt is not None
            and review_dt <= now
            and (ep.review_status or "PENDING").upper() != "RESOLVED"
        )
        outcome_dt = _parse_ts(ep.outcome_ts)
        thesis_due = outcome_dt <= now
        if not review_due and not thesis_due:
            records.append(ResolutionRecord(
                episode_id=ep.episode_id, symbol=ep.symbol,
                decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                side=(ep.action_proposal or {}).get("side", ""),
                realised_return=None, alpha_return=None, max_drawdown=None,
                reflection=None, status="SKIPPED_PENDING",
                note=(
                    f"learning review {ep.review_ts} and thesis "
                    f"outcome_ts {ep.outcome_ts} are still in the future"
                ),
            ))
            continue

        # Fetch a window covering [decision_ts, max(review, outcome) + 7d]
        # so we always have a forward trading day even if a clock lands
        # on a weekend.
        decision_dt = _parse_ts(ep.decision_ts)
        far_dt = outcome_dt if thesis_due else (review_dt or outcome_dt)
        if review_due and review_dt and review_dt > far_dt:
            far_dt = review_dt
        if thesis_due and outcome_dt > far_dt:
            far_dt = outcome_dt
        window_days = max(int((far_dt - decision_dt).days) + 14, 30)
        end_for_fetch = (far_dt + timedelta(days=7)).replace(tzinfo=None)
        try:
            df = fetch_ohlcv(ep.symbol, end_for_fetch, window_days)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ep.symbol, exc)
            df = None

        if df is None or len(df) == 0:
            if thesis_due and (now - outcome_dt).days >= abandon_after_days:
                memory.abandon(ep.episode_id, reason="no price data available")
                records.append(ResolutionRecord(
                    episode_id=ep.episode_id, symbol=ep.symbol,
                    decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                    side=(ep.action_proposal or {}).get("side", ""),
                    realised_return=None, alpha_return=None, max_drawdown=None,
                    reflection=None, status="ABANDONED",
                    note=f"no OHLCV after {abandon_after_days} days",
                ))
            else:
                records.append(ResolutionRecord(
                    episode_id=ep.episode_id, symbol=ep.symbol,
                    decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                    side=(ep.action_proposal or {}).get("side", ""),
                    realised_return=None, alpha_return=None, max_drawdown=None,
                    reflection=None, status="SKIPPED_NO_DATA",
                    note="OHLCV temporarily unavailable; will retry next sweep",
                ))
            continue

        # Benchmark alpha — cache the SPY pull so we don't refetch per episode.
        if benchmark_df is None:
            try:
                benchmark_df = fetch_ohlcv(
                    benchmark_symbol,
                    end_for_fetch,
                    window_days,
                )
            except Exception as exc:
                logger.warning("benchmark fetch failed: %s", exc)
                benchmark_df = None

        from trading_crew.agentic.outcome import (
            OutcomeLabel,
            attribute_desks,
            hold_threshold_from_episodes,
            label_outcome,
            market_direction,
            wrong_prediction_desks,
        )
        hold_thr = hold_threshold_from_episodes(memory.all_episodes())

        if review_due:
            priced_r = _price_holding(ep, df, benchmark_df, ep.review_ts)
            if priced_r is None:
                if not thesis_due:
                    records.append(ResolutionRecord(
                        episode_id=ep.episode_id, symbol=ep.symbol,
                        decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                        side=(ep.action_proposal or {}).get("side", ""),
                        realised_return=None, alpha_return=None, max_drawdown=None,
                        reflection=None, status="SKIPPED_NO_DATA",
                        note="decision or 30-day review date outside OHLCV range",
                    ))
                    continue
            else:
                rev_label = label_outcome(
                    priced_r.side, priced_r.realised, priced_r.market_return,
                    hold_abs_threshold=hold_thr,
                )
                rev_dir = market_direction(priced_r.market_return)
                rev_attr = attribute_desks(getattr(ep, "desk_leans", None) or {}, rev_dir)
                rev_hurt = wrong_prediction_desks(rev_attr)
                rev_meta = {
                    "clock": "short_review",
                    "outcome_label": rev_label.value,
                    "market_direction": rev_dir,
                    "desks": list(rev_hurt),
                    "learning_calendar_days": strategy.learning.calendar_days,
                    "desk_lessons": _deterministic_desk_lessons(
                        rev_attr, mkt_dir=rev_dir,
                        market_ret=priced_r.market_return,
                        side=priced_r.side, outcome_label=rev_label.value,
                    ),
                    "recommended_change": "",
                    "unhelpful_tools": [],
                    "overemphasized": [],
                    "suggested_playbook": "",
                }
                rev_text = (
                    f"[short_review | {rev_label.value}] "
                    f"30-day path {priced_r.realised:+.2%} "
                    f"(α {priced_r.alpha:+.2%}) vs predicted {priced_r.side}."
                )
                if not skip_llm and llm is not None:
                    try:
                        lesson, extras, _calls = _llm_lesson(
                            ep, llm=llm,
                            realised=priced_r.realised,
                            alpha=priced_r.alpha,
                            mdd=priced_r.max_drawdown,
                            benchmark_return=priced_r.benchmark_return,
                            market_ret=priced_r.market_return,
                            outcome_label=rev_label.value,
                            mkt_dir=rev_dir,
                            hurt=rev_hurt,
                            as_of=str(ep.review_ts or "")[:10],
                            desk_attr=rev_attr,
                            side=priced_r.side,
                        )
                        if lesson:
                            rev_text = lesson
                        rev_meta.update(extras)
                        rev_meta["clock"] = "short_review"
                    except Exception:
                        logger.exception(
                            "Review investigation failed for %s", ep.episode_id,
                        )
                memory.update_review(
                    ep.episode_id,
                    review_ts=ep.review_ts,
                    realised_return=priced_r.realised,
                    alpha_return=priced_r.alpha,
                    max_drawdown=(priced_r.max_drawdown if priced_r.max_drawdown is not None else 0.0),
                    market_return=priced_r.market_return,
                    outcome_label=rev_label.value,
                    reflection=rev_text,
                    reflection_meta=rev_meta,
                    embargo_days=strategy.learning.embargo_days,
                )
                wrote_review = True
                ep = memory.get(ep.episode_id) or ep
                if not thesis_due:
                    records.append(ResolutionRecord(
                        episode_id=ep.episode_id, symbol=ep.symbol,
                        decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                        side=priced_r.side,
                        realised_return=priced_r.realised,
                        alpha_return=priced_r.alpha,
                        max_drawdown=priced_r.max_drawdown,
                        reflection=ep.review.get("reflection") if ep.review else None,
                        reflection_meta=rev_meta,
                        status="REVIEW_RESOLVED",
                        note=(
                            f"30-day learning review scored; thesis "
                            f"outcome_ts {ep.outcome_ts} still pending"
                        ),
                        outcome_label=rev_label.value,
                        market_return=priced_r.market_return,
                        review=ep.review,
                    ))
                    continue

        if not thesis_due:
            continue

        priced = _price_holding(ep, df, benchmark_df, ep.outcome_ts)
        if priced is None:
            records.append(ResolutionRecord(
                episode_id=ep.episode_id, symbol=ep.symbol,
                decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                side=(ep.action_proposal or {}).get("side", ""),
                realised_return=None, alpha_return=None, max_drawdown=None,
                reflection=None, status="SKIPPED_NO_DATA",
                note="decision or outcome date outside OHLCV range",
            ))
            continue

        side = priced.side
        realised = priced.realised
        mdd = priced.max_drawdown
        alpha = priced.alpha
        benchmark_return = priced.benchmark_return
        market_ret = priced.market_return
        hold_thr = hold_threshold_from_episodes(memory.all_episodes())
        outcome_label = label_outcome(
            side, realised, market_ret, hold_abs_threshold=hold_thr,
        )
        mkt_dir = market_direction(market_ret)
        desk_attr = attribute_desks(getattr(ep, "desk_leans", None) or {}, mkt_dir)
        hurt = wrong_prediction_desks(desk_attr)

        # Reflection — optional (skipped when no LLM, e.g. CI tests).
        # Deterministic attribution always lands in reflection_meta so a
        # skip_llm resolve still teaches the desks that leaned the wrong way.
        reflection_text: Optional[str] = None
        reflection_meta: Dict[str, Any] = {
            "outcome_label": outcome_label.value,
            "market_direction": mkt_dir,
            "desks": list(hurt),
            "error_class": (
                "correct" if outcome_label in (
                    OutcomeLabel.WIN, OutcomeLabel.HOLD_CORRECT,
                ) else ("noise" if outcome_label == OutcomeLabel.FLAT else "thesis_wrong")
            ),
            "recommended_change": "",
            "desk_lessons": _deterministic_desk_lessons(
                desk_attr, mkt_dir=mkt_dir, market_ret=market_ret,
                side=side, outcome_label=outcome_label.value,
            ),
            "unhelpful_tools": [],
            "overemphasized": [],
            "suggested_playbook": "",
        }
        if not skip_llm and llm is not None:
            try:
                lesson, extras, _calls = _llm_lesson(
                    ep, llm=llm,
                    realised=realised, alpha=alpha, mdd=mdd,
                    benchmark_return=benchmark_return,
                    market_ret=market_ret,
                    outcome_label=outcome_label.value,
                    mkt_dir=mkt_dir,
                    hurt=hurt,
                    as_of=str(ep.outcome_ts or "")[:10],
                    desk_attr=desk_attr,
                    side=side,
                )
                reflection_text = lesson
                reflection_meta.update(extras)
            except Exception:
                logger.exception("Reflection generation failed for %s", ep.episode_id)
                reflection_text = None

        memory.update_outcome(
            ep.episode_id,
            realised_return=realised,
            alpha_return=alpha,
            max_drawdown=(mdd if mdd is not None else 0.0),
            reflection=reflection_text,
            reflection_meta=reflection_meta,
            market_return=market_ret,
            outcome_label=outcome_label.value,
        )
        persisted = memory.get(ep.episode_id)
        if persisted is not None:
            from trading_crew.horizons import horizon_days_to_horizon
            hd = int((ep.action_proposal or {}).get("horizon_days") or 0)
            tier = horizon_days_to_horizon(hd).value.lower()
            scores = dict(persisted.horizon_scores or {})
            scores[tier] = {
                "realised_return": realised,
                "alpha_return": alpha,
                "max_drawdown": mdd,
                "outcome_label": outcome_label.value,
            }
            persisted.horizon_scores = scores
            memory.add(persisted)
        try:
            from pathlib import Path as _Path
            from trading_crew.agentic.graph.playbook import playbook_key, record_outcome
            from trading_crew.data_sources import is_indian_ticker
            mkt = "IN" if is_indian_ticker(ep.symbol) else "US"
            dbp = getattr(memory, "_db_path", None)
            pb_path = (_Path(dbp).parent / "topology_playbook.json") if dbp else None
            record_outcome(playbook_key(market=mkt), outcome_label, path=pb_path)
        except Exception:
            logger.warning("topology playbook update failed for %s", ep.episode_id, exc_info=True)

        do_tape = (not skip_llm) if skip_tape_reason is None else (not skip_tape_reason)
        market_reason = str(reflection_meta.get("market_reason") or "")
        if do_tape and not market_reason:
            outcome_date = str(ep.outcome_ts or "")[:10]
            try:
                if tape_reason_fn is not None:
                    market_reason = tape_reason_fn(
                        ep.symbol, outcome_date, market_ret, side,
                    ) or ""
                else:
                    from trading_crew.agentic.mass import fetch_tape_reason
                    market_reason = fetch_tape_reason(
                        ep.symbol, outcome_date, market_ret, side,
                    ) or ""
            except Exception:
                logger.warning("mass tape reason failed for %s", ep.episode_id, exc_info=True)
                market_reason = ""
        if market_reason:
            reflection_meta["market_reason"] = market_reason
            memory.update_outcome(
                ep.episode_id,
                realised_return=realised,
                alpha_return=alpha,
                max_drawdown=(mdd if mdd is not None else 0.0),
                reflection=reflection_text,
                reflection_meta=reflection_meta,
                market_return=market_ret,
                outcome_label=outcome_label.value,
                market_reason=market_reason,
            )
            ep.market_reason = market_reason
            ep.outcome_label = outcome_label.value
            ep.alpha_return = alpha
            ep.reflection_meta = reflection_meta
            try:
                from trading_crew.agentic.mass import on_episode_resolved
                on_episode_resolved(ep, skip_search=True)
            except Exception:
                logger.warning("legacy mass 1PO write failed for %s", ep.episode_id, exc_info=True)

        records.append(ResolutionRecord(
            episode_id=ep.episode_id, symbol=ep.symbol,
            decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
            side=side, realised_return=realised, alpha_return=alpha,
            max_drawdown=mdd, reflection=reflection_text,
            reflection_meta=reflection_meta,
            status="RESOLVED",
            outcome_label=outcome_label.value,
            market_return=market_ret,
            market_reason=market_reason or None,
            review=(memory.get(ep.episode_id) or ep).review,
        ))
        wrote_thesis = True

    if wrote_review or wrote_thesis:
        try:
            from trading_crew.agentic.harness import step_harness
            from trading_crew.agentic.self_learn import journal_path
            jpath = journal_path()
            dbp = getattr(memory, "_db_path", None)
            if dbp is not None:
                cache = Path(
                    os.environ.get("TRADINGCREW_CACHE_DIR")
                    or os.path.expanduser("~/.trading_crew")
                )
                parent = Path(dbp).parent
                if parent != cache:
                    jpath = parent / "self_learn_journal.json"
            step_harness(
                memory.all_episodes(),
                journal_file=jpath,
                skip_llm=skip_llm,
            )
        except Exception:
            logger.warning("harness step failed after resolve", exc_info=True)

    return records


def _parse_ts(ts: str) -> datetime:
    cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
