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
4.  **Reflect** by asking the LLM a single tightly-scoped question:
    given what the agents *knew* (``state_summary``), what they *did*
    (``action_proposal``), and what *happened* (``realised_return``,
    ``max_drawdown``), produce a one-paragraph lesson — what would a
    careful operator change next time?  The output is constrained
    (≤ 600 chars, no numbers fabricated) so it can be safely surfaced
    to the analysts via ``retrieve_past_episodes``.
5.  **Persist** the resolution back into the episode store via
    ``EpisodicMemory.update_outcome``.

Why this is "agentic training" without fine-tuning the LLM
----------------------------------------------------------
The reflections we generate here are read by future runs through the
``retrieve_past_episodes`` tool (M3 + Phase C).  Because the M3
embargo blocks retrieval until ``outcome_ts + embargo_days`` has
passed, no future leak is possible.  Over time this builds a
self-curated case-book the agents consult — the closest equivalent
of "training" the system without touching the LLM weights.

This is intentionally NOT wired to a scheduler — the user (or a cron)
calls ``resolve_pending_episodes`` explicitly so it's audit-traceable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

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
            "status": self.status,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# LLM-side structured output for the reflection
# ---------------------------------------------------------------------------


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


def _price_at_or_after(df, ts_iso: str) -> Optional[Tuple[str, float, float, float]]:
    """Return ``(date_iso, open, close, high)`` for the first trading day on
    or after ``ts_iso``.  Tolerates weekends/holidays by walking forward.

    Returns None if no trading day is available within +14 calendar days.
    """
    if df is None or df.empty:
        return None
    import pandas as pd  # local import — keeps module import light
    try:
        target = pd.to_datetime(ts_iso).tz_localize(None) if hasattr(pd.to_datetime(ts_iso), "tz_localize") else pd.to_datetime(ts_iso)
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
    import pandas as pd
    try:
        lo = pd.to_datetime(decision_ts).tz_localize(None) if hasattr(pd.to_datetime(decision_ts), "tz_localize") else pd.to_datetime(decision_ts)
        hi = pd.to_datetime(outcome_ts).tz_localize(None) if hasattr(pd.to_datetime(outcome_ts), "tz_localize") else pd.to_datetime(outcome_ts)
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


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------


def _build_reflection_prompt(
    ep: Episode,
    realised_return: float,
    alpha_return: float,
    max_drawdown: Optional[float],
    benchmark_return: float,
) -> str:
    proposal = ep.action_proposal or {}
    side = (proposal.get("side") or "").upper()
    target_weight = proposal.get("target_weight")
    conviction = proposal.get("conviction_score")
    horizon = proposal.get("horizon_days")
    return (
        "You are a sober post-trade analyst.  An agentic system made a "
        "decision; the holding period has now elapsed and the realised "
        "outcome is known.  Produce a ONE-PARAGRAPH lesson the system "
        "should remember.\n\n"
        f"Symbol: {ep.symbol}\n"
        f"Decision timestamp: {ep.decision_ts}\n"
        f"Outcome timestamp:  {ep.outcome_ts}\n"
        f"Regime at decision: {ep.regime.value}\n"
        f"Action taken: {side} (target_weight={target_weight}, "
        f"conviction={conviction}, horizon_days={horizon})\n"
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
    )


def _generate_reflection(
    ep: Episode,
    realised_return: float,
    alpha_return: float,
    max_drawdown: Optional[float],
    benchmark_return: float,
    llm: Any,
) -> Tuple[str, float, str]:
    """Ask the LLM for a structured reflection.  Returns (summary, confidence, tag)."""
    prompt = _build_reflection_prompt(
        ep, realised_return, alpha_return, max_drawdown, benchmark_return
    )
    # Mirror the critic.py call style — crewai.LLM exposes a ``call``
    # method that accepts ``response_format=BaseModelClass`` for
    # constrained decoding.
    raw = llm.call(
        messages=[{"role": "user", "content": prompt}],
        response_format=ReflectionResponse,
    )
    response = _coerce_response(raw)
    return response.summary, float(response.confidence_score), response.key_lesson_tag


def _coerce_response(raw: Any) -> ReflectionResponse:
    """Normalise the LLM return into a ``ReflectionResponse``.

    Different LLM providers return slightly different shapes through CrewAI:
    the model class itself (already validated), a JSON string, a dict, or a
    Pydantic instance.  We accept all three uniformly so callers don't have
    to branch.
    """
    if isinstance(raw, ReflectionResponse):
        return raw
    if isinstance(raw, dict):
        return ReflectionResponse.model_validate(raw)
    if isinstance(raw, str):
        import json
        return ReflectionResponse.model_validate(json.loads(raw))
    # last-ditch — try .model_dump() roundtrip
    if hasattr(raw, "model_dump"):
        return ReflectionResponse.model_validate(raw.model_dump())
    raise TypeError(f"Unsupported reflection response shape: {type(raw)!r}")


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

    records: List[ResolutionRecord] = []
    # Cache benchmark OHLCV once per resolver invocation; the SPY window
    # subsumes every episode window we'll touch.
    benchmark_df = None

    for ep in memory.all_episodes():
        if ep.outcome_status != OutcomeStatus.PENDING:
            continue
        if ticker and ep.symbol.upper() != ticker.upper():
            continue

        outcome_dt = _parse_ts(ep.outcome_ts)
        if outcome_dt > now:
            records.append(ResolutionRecord(
                episode_id=ep.episode_id, symbol=ep.symbol,
                decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                side=(ep.action_proposal or {}).get("side", ""),
                realised_return=None, alpha_return=None, max_drawdown=None,
                reflection=None, status="SKIPPED_PENDING",
                note=f"outcome_ts {ep.outcome_ts} is still in the future",
            ))
            continue

        # Fetch a window covering [decision_ts, outcome_ts + 7d] so we
        # always have a forward trading day even if outcome_ts lands on
        # a weekend.
        decision_dt = _parse_ts(ep.decision_ts)
        window_days = max(int((outcome_dt - decision_dt).days) + 14, 30)
        end_for_fetch = (outcome_dt + timedelta(days=7)).replace(tzinfo=None)
        try:
            df = fetch_ohlcv(ep.symbol, end_for_fetch, window_days)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ep.symbol, exc)
            df = None

        if df is None or len(df) == 0:
            if (now - outcome_dt).days >= abandon_after_days:
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

        in_quote = _price_at_or_after(df, ep.decision_ts)
        out_quote = _price_at_or_after(df, ep.outcome_ts)
        if in_quote is None or out_quote is None:
            records.append(ResolutionRecord(
                episode_id=ep.episode_id, symbol=ep.symbol,
                decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
                side=(ep.action_proposal or {}).get("side", ""),
                realised_return=None, alpha_return=None, max_drawdown=None,
                reflection=None, status="SKIPPED_NO_DATA",
                note="decision or outcome date outside OHLCV range",
            ))
            continue

        side = (ep.action_proposal or {}).get("side", "")
        realised = _signed_realised_return(side, in_quote[2], out_quote[2])
        mdd = _max_drawdown_over_holding(df, ep.decision_ts, ep.outcome_ts, side)

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

        if benchmark_df is not None and not benchmark_df.empty:
            bench_in = _price_at_or_after(benchmark_df, ep.decision_ts)
            bench_out = _price_at_or_after(benchmark_df, ep.outcome_ts)
            if bench_in and bench_out and bench_in[2] > 0:
                benchmark_return = (bench_out[2] / bench_in[2]) - 1.0
            else:
                benchmark_return = 0.0
        else:
            benchmark_return = 0.0
        alpha = realised - benchmark_return

        # Reflection — optional (skipped when no LLM, e.g. CI tests).
        reflection_text: Optional[str] = None
        if not skip_llm and llm is not None:
            try:
                summary, confidence, tag = _generate_reflection(
                    ep, realised, alpha, mdd, benchmark_return, llm
                )
                # Tagging the reflection makes downstream retrieval cluster
                # by lesson category — e.g. all "stop_too_tight" episodes.
                reflection_text = f"[{tag} | conf {confidence:.2f}] {summary}"
            except Exception:
                logger.exception("Reflection generation failed for %s", ep.episode_id)
                reflection_text = None

        memory.update_outcome(
            ep.episode_id,
            realised_return=realised,
            alpha_return=alpha,
            max_drawdown=(mdd if mdd is not None else 0.0),
            reflection=reflection_text,
        )

        records.append(ResolutionRecord(
            episode_id=ep.episode_id, symbol=ep.symbol,
            decision_ts=ep.decision_ts, outcome_ts=ep.outcome_ts,
            side=side, realised_return=realised, alpha_return=alpha,
            max_drawdown=mdd, reflection=reflection_text,
            status="RESOLVED",
        ))

    return records


def _parse_ts(ts: str) -> datetime:
    cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
