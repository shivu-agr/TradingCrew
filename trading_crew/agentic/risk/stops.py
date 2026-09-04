"""Deterministic, horizon-aware stop-loss sanity layer.

Motivation
----------
The Portfolio Manager's ``entry_price`` / ``stop_loss`` / ``target_price``
are free-form numbers the LLM writes — nothing in the pipeline computed or
validated them.  In practice the model would attach a *tight* technical
stop (e.g. "1.5x ATR252 → $56.00", a ~2% band) to a **multi-month or
multi-year** horizon trade, so the position would be stopped out by one
ordinary day of noise long before the thesis had a chance to play out.
There is also no ATR computed anywhere in the equity crew, so any "ATR"
figure in the narrative was invented.

This module supplies the missing deterministic piece: given the realised
volatility / ATR of the underlying and the stated horizon, it computes a
**minimum sensible stop distance** and widens an implausibly tight (or
wrong-side / missing) stop to it, tagging the adjustment for the audit
trail.  It never *tightens* a conservative wide stop, and — per the
"no silent fallback" rule — it declines to assess (rather than fabricate a
volatility) when neither ATR nor annualised vol is available.

The constants below are model parameters, not fallbacks:

* ``ATR_FLOOR_MULT``      — a stop is never tighter than this many ATRs
  (≈ this many sessions of typical noise), independent of horizon.
* ``HORIZON_NOISE_MULT``  — multiple of the horizon-scaled 1-sigma move a
  stop must clear so ordinary drift over the holding period doesn't
  trigger it.
* ``HORIZON_SCALE_CAP_DAYS`` — sqrt-of-time scaling is capped here (~one
  quarter).  Beyond a quarter a "stop" is really a thesis-invalidation
  exit, not a price level, so we stop widening it toward a full drawdown.
* ``HARD_FLOOR_PCT`` / ``HARD_CAP_PCT`` — absolute clamps so the rule
  degrades sanely for very low-/high-vol names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

ATR_FLOOR_MULT: float = 2.0
HORIZON_NOISE_MULT: float = 1.0
HORIZON_SCALE_CAP_DAYS: int = 63
HARD_FLOOR_PCT: float = 0.03
HARD_CAP_PCT: float = 0.35
_TRADING_DAYS_PER_YEAR: float = 252.0

# --- target validation --------------------------------------------------
# Unlike the stop, the target's horizon scaling is NOT capped at a quarter:
# a multi-year hold legitimately targets a large total return, so we let the
# reachable band grow with the full holding period and only clamp the
# egregious moonshots (e.g. "+300% in 30 days") and degenerate/wrong-side
# targets.
TARGET_MAX_SIGMA_MULT: float = 3.0    # max plausible target = 3x horizon-sigma
TARGET_DEFAULT_SIGMA_MULT: float = 1.5  # used to synthesise a missing target
TARGET_MIN_PCT: float = 0.01          # below this the target is degenerate
TARGET_HARD_CAP_PCT: float = 5.0      # absolute ceiling (+500%)
TARGET_SHORT_MAX_PCT: float = 0.90    # a short can't gain more than ~100%

# --- staged (scale-out) targets ----------------------------------------
# Firefly-style "two or three staged profit targets": rather than one
# all-or-nothing target, split the reachable move into tiers so the desk
# can bank partial profit and de-risk as price advances.  Tiers are placed
# at fractions of the validated FINAL target distance (monotone, always
# inside the reachable band), each with a scale-out weight (fraction of the
# position to exit at that tier).  Weights sum to 1.0.
STAGED_TARGET_FRACTIONS: tuple[float, ...] = (0.5, 0.75, 1.0)
STAGED_TARGET_WEIGHTS: tuple[float, ...] = (0.4, 0.35, 0.25)

_LONG_ACTIONS = {"BUY", "STRONG_BUY", "OVERWEIGHT"}
_SHORT_ACTIONS = {"SELL", "STRONG_SELL", "UNDERWEIGHT"}

# A marketable limit sits a few percent from last close. The previous
# 0.5×–2.0× ratio let a $100 schema-example survive on a ~$68 name
# (1.48×). Anything outside ±20% of spot, or non-positive, is grounded.
ENTRY_PLAUSIBLE_BAND: float = 0.20


def entry_needs_grounding(
    entry: float,
    spot: float,
    band: float = ENTRY_PLAUSIBLE_BAND,
) -> bool:
    """True when ``entry`` is missing or too far from ``spot`` to keep.

    Returns False when spot itself is unusable — we never invent a close.
    """
    if not math.isfinite(spot) or spot <= 0:
        return False
    if not math.isfinite(entry) or entry <= 0:
        return True
    return abs(entry / spot - 1.0) > band


def compute_staged_targets(
    *,
    entry: float,
    side: str,
    final_target_pct: Optional[float],
    final_stop_pct: Optional[float],
) -> list[dict]:
    """Split the validated final target into scale-out tiers.

    Returns a list of ``{tier, price, pct, reward_risk, weight}`` dicts (one
    per :data:`STAGED_TARGET_FRACTIONS` tier) for a directional trade, or an
    empty list when the inputs are degenerate.  ``pct`` is the profit-direction
    decimal move to the tier; ``price`` is signed by ``side``; ``reward_risk``
    is the tier's move over the stop distance; ``weight`` is the fraction of
    the position to exit at that tier.  Pure + deterministic — no fabrication
    when there is no valid target.
    """
    if entry <= 0 or side not in ("long", "short"):
        return []
    if final_target_pct is None or final_target_pct <= 0:
        return []
    out: list[dict] = []
    for i, (frac, weight) in enumerate(
        zip(STAGED_TARGET_FRACTIONS, STAGED_TARGET_WEIGHTS), start=1
    ):
        pct = final_target_pct * frac
        price = entry * (1 + pct) if side == "long" else entry * (1 - pct)
        rr = (
            pct / final_stop_pct
            if (final_stop_pct and final_stop_pct > 0)
            else None
        )
        out.append({
            "tier": i,
            "price": round(price, 2),
            "pct": round(pct, 4),
            "reward_risk": round(rr, 2) if rr is not None else None,
            "weight": weight,
        })
    return out


def side_from_action(action: str) -> str:
    """Map a 5-tier (or legacy) action string to ``long`` / ``short`` / ``flat``."""
    key = (action or "").strip().upper()
    if key in _LONG_ACTIONS:
        return "long"
    if key in _SHORT_ACTIONS:
        return "short"
    return "flat"


def compute_atr(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """Wilder's Average True Range over the trailing ``period`` sessions.

    Returns ``None`` when there isn't enough clean data to compute a
    meaningful ATR (we do not guess).
    """
    n = min(len(high), len(low), len(close))
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h, l, pc = float(high[i]), float(low[i]), float(close[i - 1])
        if not all(math.isfinite(x) for x in (h, l, pc)):
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    # Wilder smoothing seeded with the simple mean of the first `period` TRs.
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr if math.isfinite(atr) and atr > 0 else None


def annualised_vol_from_close(close: Sequence[float], window: int = 252) -> Optional[float]:
    """Annualised close-to-close volatility as a fraction (0.45 == 45%).

    Uses the trailing ``window`` daily returns.  Returns ``None`` when the
    series is too short to be trustworthy.
    """
    vals = [float(c) for c in close if math.isfinite(float(c)) and float(c) > 0]
    if len(vals) < 21:
        return None
    tail = vals[-(window + 1):] if len(vals) > window + 1 else vals
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail))]
    if len(rets) < 20:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    daily = math.sqrt(var)
    ann = daily * math.sqrt(_TRADING_DAYS_PER_YEAR)
    return ann if math.isfinite(ann) and ann > 0 else None


def min_stop_pct(
    *,
    atr_pct: Optional[float],
    daily_vol: Optional[float],
    horizon_days: int,
) -> Optional[float]:
    """Minimum sensible stop distance (as a fraction of entry).

    Combines an ATR floor (never tighter than ``ATR_FLOOR_MULT`` ATRs) with
    a horizon-scaled 1-sigma move.  Returns ``None`` when neither a per-day
    ATR nor a daily vol is available — the caller must then skip the check
    rather than invent a number.
    """
    units = [u for u in (atr_pct, daily_vol) if u is not None and u > 0]
    if not units:
        return None
    sigma_unit = max(units)
    atr_term = ATR_FLOOR_MULT * (atr_pct if (atr_pct and atr_pct > 0) else sigma_unit)
    scale_days = min(max(int(horizon_days), 1), HORIZON_SCALE_CAP_DAYS)
    sigma_h = sigma_unit * math.sqrt(scale_days)
    raw = max(atr_term, HORIZON_NOISE_MULT * sigma_h)
    return max(HARD_FLOOR_PCT, min(HARD_CAP_PCT, raw))


@dataclass(frozen=True)
class StopAssessment:
    """Result of assessing (and possibly correcting) a decision's levels."""

    entry: float
    stop: float
    target: float
    side: str
    adjusted: bool                 # stop was changed
    reason: str
    min_stop_pct: Optional[float] = None
    provided_stop_pct: Optional[float] = None
    final_stop_pct: Optional[float] = None
    reward_risk: Optional[float] = None
    assessable: bool = True
    # target validation
    target_adjusted: bool = False
    target_reason: str = ""
    provided_target_pct: Optional[float] = None
    final_target_pct: Optional[float] = None
    max_target_pct: Optional[float] = None
    expected_return_pct: Optional[float] = None  # decimal fraction (target-entry)/entry
    # Scale-out tiers derived from the validated final target (see
    # ``compute_staged_targets``).  Empty for HOLD/flat or when no valid
    # target exists.  Each item: {tier, price, pct, reward_risk, weight}.
    staged_targets: list = field(default_factory=list)

    @property
    def any_adjusted(self) -> bool:
        return self.adjusted or self.target_adjusted

    def as_tag(self) -> Optional[str]:
        """Compact audit tag, or ``None`` when nothing changed."""
        parts: list[str] = []
        if self.adjusted:
            prev = f"{self.provided_stop_pct * 100:.1f}%" if self.provided_stop_pct is not None else "n/a"
            parts.append(
                f"stop_normalized:{prev}->{(self.final_stop_pct or 0) * 100:.1f}%"
                f"(min {(self.min_stop_pct or 0) * 100:.1f}% for {self.side} horizon)"
            )
        if self.target_adjusted:
            prev = f"{self.provided_target_pct * 100:.1f}%" if self.provided_target_pct is not None else "n/a"
            parts.append(
                f"target_normalized:{prev}->{(self.final_target_pct or 0) * 100:.1f}%"
            )
        return "; ".join(parts) if parts else None


def assess_stop(
    *,
    action: str,
    entry: float,
    stop: float,
    target: float,
    horizon_days: int,
    current_price: float,
    atr: Optional[float] = None,
    ann_vol: Optional[float] = None,
) -> StopAssessment:
    """Assess a decision's entry/stop and widen an implausible stop.

    Deterministic and pure.  A HOLD/flat decision needs no stop and is
    returned unchanged.  When volatility cannot be measured the result is
    marked ``assessable=False`` and levels are returned untouched (no
    fabricated volatility).
    """
    side = side_from_action(action)
    entry = float(entry)
    stop = float(stop)
    target = float(target)
    current_price = float(current_price)

    # Anchor a missing/degenerate entry to the real last close — but never
    # override a legitimate limit level the PM chose.
    if entry <= 0 and current_price > 0:
        entry = current_price

    if side == "flat" or entry <= 0:
        return StopAssessment(
            entry=entry, stop=stop, target=target, side=side,
            adjusted=False, reason="no directional stop required" if side == "flat" else "no valid entry",
            assessable=side != "flat",
        )

    atr_pct = (atr / entry) if (atr and atr > 0) else None
    daily_vol = (ann_vol / math.sqrt(_TRADING_DAYS_PER_YEAR)) if (ann_vol and ann_vol > 0) else None
    floor_pct = min_stop_pct(atr_pct=atr_pct, daily_vol=daily_vol, horizon_days=horizon_days)

    if floor_pct is None:
        return StopAssessment(
            entry=entry, stop=stop, target=target, side=side,
            adjusted=False, reason="volatility unavailable — levels not validated",
            assessable=False,
        )

    # ---- stop -----------------------------------------------------------
    # Directional distance of the provided stop; None if it is on the wrong
    # side of entry (a long stop above entry, or a short stop below it).
    provided_pct: Optional[float]
    if side == "long":
        provided_pct = (entry - stop) / entry if (stop > 0 and stop < entry) else None
    else:  # short
        provided_pct = (stop - entry) / entry if (stop > 0 and stop > entry) else None

    needs_fix = provided_pct is None or provided_pct < floor_pct
    new_stop = stop
    reason = "stop within volatility-consistent band"
    if needs_fix:
        new_stop = entry * (1 - floor_pct) if side == "long" else entry * (1 + floor_pct)
        if provided_pct is None:
            reason = (
                f"stop was missing / on the wrong side of entry for a {side}; "
                f"set to {floor_pct * 100:.1f}% (volatility+horizon floor)"
            )
        else:
            reason = (
                f"stop was {provided_pct * 100:.1f}% from entry — too tight for a "
                f"{horizon_days}d {side} at this volatility; widened to {floor_pct * 100:.1f}%"
            )
    final_stop_pct = floor_pct if needs_fix else provided_pct

    # ---- target ---------------------------------------------------------
    # Full-horizon reachable band (target scaling is NOT capped at a quarter).
    sigma_unit = max(u for u in (atr_pct, daily_vol) if u is not None and u > 0)
    sigma_h_full = sigma_unit * math.sqrt(max(int(horizon_days), 1))
    max_target_pct = min(TARGET_MAX_SIGMA_MULT * sigma_h_full, TARGET_HARD_CAP_PCT)
    if side == "short":
        max_target_pct = min(max_target_pct, TARGET_SHORT_MAX_PCT)
    default_target_pct = min(max(TARGET_DEFAULT_SIGMA_MULT * sigma_h_full, HARD_FLOOR_PCT), max_target_pct)

    # Profit-direction distance of the provided target (None if wrong side).
    if side == "long":
        provided_tgt_pct = (target - entry) / entry if (target > 0 and target > entry) else None
    else:  # short
        provided_tgt_pct = (entry - target) / entry if (target > 0 and target < entry) else None

    new_target = target
    target_adjusted = False
    target_reason = "target within reachable band"
    if provided_tgt_pct is None or provided_tgt_pct < TARGET_MIN_PCT:
        new_target = entry * (1 + default_target_pct) if side == "long" else entry * (1 - default_target_pct)
        target_adjusted = True
        target_reason = (
            f"target was missing / wrong-side / degenerate for a {side}; "
            f"set to {default_target_pct * 100:.1f}% (horizon-scaled default)"
        )
        final_target_pct = default_target_pct
    elif provided_tgt_pct > max_target_pct:
        new_target = entry * (1 + max_target_pct) if side == "long" else entry * (1 - max_target_pct)
        target_adjusted = True
        target_reason = (
            f"target was {provided_tgt_pct * 100:.1f}% — implausibly large for a "
            f"{horizon_days}d {side}; capped to {max_target_pct * 100:.1f}%"
        )
        final_target_pct = max_target_pct
    else:
        final_target_pct = provided_tgt_pct

    # ---- derived ---------------------------------------------------------
    reward_risk = (final_target_pct / final_stop_pct) if (final_stop_pct and final_stop_pct > 0) else None
    # expected_return_pct convention = DECIMAL FRACTION price move to target
    # (0.05 = +5%), signed by price direction, matching the canonical
    # PortfolioDecision / ActionProposal convention consumed by M5 sizing.
    expected_return_pct = round((new_target - entry) / entry, 4) if entry > 0 else None

    staged = compute_staged_targets(
        entry=entry,
        side=side,
        final_target_pct=final_target_pct,
        final_stop_pct=final_stop_pct,
    )

    return StopAssessment(
        entry=round(entry, 2),
        stop=round(new_stop, 2),
        target=round(new_target, 2),
        side=side,
        adjusted=needs_fix,
        reason=reason,
        min_stop_pct=floor_pct,
        provided_stop_pct=provided_pct,
        final_stop_pct=final_stop_pct,
        reward_risk=reward_risk,
        assessable=True,
        target_adjusted=target_adjusted,
        target_reason=target_reason,
        provided_target_pct=provided_tgt_pct,
        final_target_pct=final_target_pct,
        max_target_pct=max_target_pct,
        expected_return_pct=expected_return_pct,
        staged_targets=staged,
    )
