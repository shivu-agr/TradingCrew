"""Deterministic utilization of structured signals the PM already saw.

The desk-bias board, the reports-desk guidance card, and the computed
financial-trend signals are injected into the PM prompt as completeness
guarantees — but an LLM can still emit BUY on a split or majority-opposed
board.  This module is the *enforcement* counterpart: it never invents a
side from quant signals alone (the LLM remains the mandate), it only
shrinks a directional PM call when those signals disagree with it.

Rules (symmetric long/short; no country/sector prior):

* Majority of directional desks opposite the PM side → HOLD.
* Split board (bullish count == bearish count, both > 0) → HOLD.
* STRONG_* with any opposing desk → cap at BUY / SELL.
* A decisive computed REVENUE+EARNINGS trend (both accelerating or both
  decelerating) counts as one extra virtual desk — hard numbers, not
  management tone.

Applied only when at least ``min_desks`` analyst leans were captured, so a
parse miss cannot sit the book out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

_LONG = {"BUY", "STRONG_BUY", "OVERWEIGHT"}
_SHORT = {"SELL", "STRONG_SELL", "UNDERWEIGHT"}
_FLAT = {"HOLD", "NEUTRAL", "ABSTAIN"}
_BULL_TRENDS = {"ACCELERATING", "GROWING", "EXPANDING"}
_BEAR_TRENDS = {"DECELERATING", "DECLINING", "CONTRACTING"}
_MIN_DESKS = 3


def _bias_counts(board: Optional[Sequence[Dict[str, Any]]]) -> Tuple[int, int, int]:
    bull = bear = neu = 0
    for row in board or []:
        b = str(row.get("bias") or "").strip().lower()
        if b == "bullish":
            bull += 1
        elif b == "bearish":
            bear += 1
        elif b == "neutral":
            neu += 1
    return bull, bear, neu


def _signal_rows(report_signals: Any) -> List[Any]:
    if report_signals is None:
        return []
    if isinstance(report_signals, dict):
        return list(report_signals.get("signals") or [])
    return list(getattr(report_signals, "signals", None) or [])


def _dir_of(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("direction") or "").upper()
    return str(getattr(row, "direction", "") or "").upper()


def _name_of(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("name") or "").upper()
    return str(getattr(row, "name", "") or "").upper()


def computed_trend_lean(report_signals: Any) -> Optional[str]:
    """``bullish`` / ``bearish`` when BOTH revenue and earnings trends agree.

    Mixed or missing computed trends return ``None`` (never fabricate a lean
    from one series, and never treat STABLE as a vote).
    """
    rev = ern = None
    for row in _signal_rows(report_signals):
        name = _name_of(row)
        direction = _dir_of(row)
        if name == "REVENUE_TREND":
            rev = direction
        elif name == "EARNINGS_TREND":
            ern = direction
    if not rev or not ern:
        return None
    if rev in _BULL_TRENDS and ern in _BULL_TRENDS:
        return "bullish"
    if rev in _BEAR_TRENDS and ern in _BEAR_TRENDS:
        return "bearish"
    return None


def apply_signal_alignment(
    decision: Any,
    *,
    board: Optional[Sequence[Dict[str, Any]]] = None,
    report_signals: Any = None,
    min_desks: int = _MIN_DESKS,
) -> Tuple[Any, List[str]]:
    """Return ``(decision, notes)``.  ``notes`` is empty iff nothing changed.

    ``decision`` is a ``PortfolioDecision`` (or any object with ``action``,
    ``size_pct_of_book``, ``rationale``, and ``model_copy``).  Unchanged
    decisions are returned as the same object.
    """
    action = str(getattr(decision, "action", "") or "HOLD").upper()
    if action in _FLAT or action not in (_LONG | _SHORT):
        return decision, []

    bull, bear, neu = _bias_counts(board)
    captured = bull + bear + neu
    lean = computed_trend_lean(report_signals)
    if lean == "bullish":
        bull += 1
    elif lean == "bearish":
        bear += 1

    is_long = action in _LONG
    n_for = bull if is_long else bear
    n_against = bear if is_long else bull

    notes: List[str] = []
    new_action = action

    if action.startswith("STRONG") and n_against >= 1:
        new_action = "BUY" if is_long else "SELL"
        notes.append(
            f"{action} capped to {new_action} — {n_against} opposing desk(s); "
            "STRONG requires a one-sided board."
        )

    if captured >= min_desks:
        if n_against > n_for:
            new_action = "HOLD"
            notes.append(
                f"majority of directional desks oppose {action} "
                f"(for {n_for} / against {n_against}) — sitting out."
            )
        elif n_for == n_against and n_for > 0:
            new_action = "HOLD"
            notes.append(
                f"desk board is split ({bull} bullish / {bear} bearish) — "
                f"sitting out rather than forcing {action}."
            )

    if new_action == action:
        return decision, []

    updates: Dict[str, Any] = {"action": new_action}
    if new_action == "HOLD":
        updates["size_pct_of_book"] = 0.0
    rat = str(getattr(decision, "rationale", "") or "").rstrip()
    updates["rationale"] = f"{rat}\n\n[signal alignment] " + " ".join(notes)
    copier = getattr(decision, "model_copy", None)
    if callable(copier):
        aligned = copier(update=updates)
    else:
        aligned = decision
        for k, v in updates.items():
            setattr(aligned, k, v)
    return aligned, notes
