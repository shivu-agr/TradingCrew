"""Canonical prediction-vs-market labels for resolved episodes.

This is the Mass/HiVA ``y`` for a trading book: after ``outcome_ts`` we
compare the predicted side to what the tape did, then attribute the miss
to the desks whose lean disagreed with the market (or that backed the
losing call).

No LLM.  Sign convention matches ``reflection._signed_realised_return``:
directional realised return is already side-signed, so ``> 0`` is a WIN
for both BUY and SELL.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


class OutcomeLabel(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    FLAT = "FLAT"
    HOLD_CORRECT = "HOLD_CORRECT"
    HOLD_MISS = "HOLD_MISS"
    HOLD_UNSCORED = "HOLD_UNSCORED"


DIRECTIONAL_SIDES = frozenset({"BUY", "SELL"})
HOLD_SIDES = frozenset({"HOLD", "ABSTAIN"})

# Desk lean vs realised market direction.
VERDICT_HURT = "hurt"       # desk predicted the wrong way vs the tape
VERDICT_HELPED = "helped"   # desk predicted the tape
VERDICT_NEUTRAL = "neutral" # no directional lean to score

_FLAT_EPS = 1e-9
_MIN_HOLD_COHORT = 8


def normalize_side(side: Any) -> str:
    return str(side or "").strip().upper()


def market_direction(market_return: Optional[float]) -> str:
    """``bullish`` / ``bearish`` / ``neutral`` from the unsigned price move."""
    if market_return is None:
        return "neutral"
    r = float(market_return)
    if r > _FLAT_EPS:
        return "bullish"
    if r < -_FLAT_EPS:
        return "bearish"
    return "neutral"


def hold_threshold_from_episodes(episodes: Sequence[Any]) -> Optional[float]:
    """Median ``|alpha|`` of resolved directional episodes.

    Used as the HOLD-miss bar so we do not invent a percent cutoff.  Returns
    ``None`` until the cohort has ``_MIN_HOLD_COHORT`` directional resolves —
    HOLD stays ``HOLD_UNSCORED`` until then.
    """
    alphas: List[float] = []
    for ep in episodes:
        status = getattr(ep, "outcome_status", None)
        status_s = getattr(status, "value", status)
        if str(status_s).upper() != "RESOLVED":
            continue
        proposal = getattr(ep, "action_proposal", None) or {}
        if normalize_side(proposal.get("side")) not in DIRECTIONAL_SIDES:
            continue
        alpha = getattr(ep, "alpha_return", None)
        if alpha is None:
            continue
        alphas.append(abs(float(alpha)))
    if len(alphas) < _MIN_HOLD_COHORT:
        return None
    alphas.sort()
    return float(alphas[len(alphas) // 2])


def label_outcome(
    side: Any,
    realised_return: Optional[float],
    market_return: Optional[float],
    *,
    hold_abs_threshold: Optional[float] = None,
) -> OutcomeLabel:
    """WIN/LOSS vs the predicted side, or HOLD_CORRECT/HOLD_MISS.

    ``hold_abs_threshold`` is the measured |move| bar from
    :func:`hold_threshold_from_episodes`.  ``None`` → HOLD_UNSCORED.
    """
    s = normalize_side(side)
    if s in HOLD_SIDES:
        if hold_abs_threshold is None:
            return OutcomeLabel.HOLD_UNSCORED
        move = abs(float(market_return or 0.0))
        return (
            OutcomeLabel.HOLD_MISS
            if move >= float(hold_abs_threshold)
            else OutcomeLabel.HOLD_CORRECT
        )
    r = float(realised_return or 0.0)
    if r > _FLAT_EPS:
        return OutcomeLabel.WIN
    if r < -_FLAT_EPS:
        return OutcomeLabel.LOSS
    return OutcomeLabel.FLAT


def lean_verdict(lean: Optional[str], market_dir: str) -> str:
    """Did this desk's directional lean match the tape?"""
    key = (lean or "").strip().lower()
    md = (market_dir or "").strip().lower()
    if key not in ("bullish", "bearish") or md not in ("bullish", "bearish"):
        return VERDICT_NEUTRAL
    return VERDICT_HELPED if key == md else VERDICT_HURT


def attribute_desks(
    desk_leans: Optional[Dict[str, str]],
    market_dir: str,
) -> List[Dict[str, str]]:
    """One row per desk: ``{desk, lean, verdict}``.

    ``desk_leans`` is ``{agent_key: bullish|bearish|neutral}``.  Empty /
    unknown leans are kept as ``neutral`` so later prose can still name
    the desk if the LLM attributed a lesson to it.
    """
    rows: List[Dict[str, str]] = []
    for desk, lean in (desk_leans or {}).items():
        key = (desk or "").strip().lower()
        if not key:
            continue
        L = (lean or "").strip().lower() or "unknown"
        rows.append({
            "desk": key,
            "lean": L,
            "verdict": lean_verdict(L, market_dir),
        })
    # Hurt desks first — those are the steps that made the wrong prediction.
    order = {VERDICT_HURT: 0, VERDICT_HELPED: 1, VERDICT_NEUTRAL: 2}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["desk"]))
    return rows


def wrong_prediction_desks(attributions: Iterable[Dict[str, str]]) -> List[str]:
    """Agent keys whose lean disagreed with the tape."""
    return [r["desk"] for r in attributions if r.get("verdict") == VERDICT_HURT]
