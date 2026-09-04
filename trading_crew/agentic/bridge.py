"""Bridge between CrewAI's ``PortfolioDecision`` and the M1 ``ActionProposal``.

The Portfolio Manager task in the CrewAI workflow already emits a typed
``PortfolioDecision`` (see ``trading_crew/schemas.py``).  That schema carries
the *narrative* fields a human reviewer wants to see (rationale, drivers,
risks, falsifiers, compliance_status).  Downstream M1-M7 layers need the
deterministic ``ActionProposal`` shape — ``side`` / ``target_weight`` /
``conviction_*`` / ``validity_check`` — so the cost model + risk gates +
simulator + memory all see the same structured object.

This bridge is **deterministic**: it does not call the LLM.  We map fields
1-to-1 where possible and synthesize the missing ones from existing data
(``ValidityCheck`` is filled by inspecting the proposal's own coherence so
M5's risk gate has something to check).  When the source decision is
inconsistent (e.g. ``BUY`` with ``size_pct_of_book == 0``), we fall
back to ``ABSTAIN`` rather than emitting a malformed proposal.

Action tier mapping
-------------------
The CrewAI Portfolio Manager emits the retail-friendly 5-tier
``PortfolioDecision.action`` (``STRONG_BUY`` / ``BUY`` / ``HOLD`` /
``SELL`` / ``STRONG_SELL``).  The downstream M1 ``ActionProposal`` only
carries the 4-value ``ActionSide`` (``BUY`` / ``SELL`` / ``HOLD`` /
``ABSTAIN``).  We collapse:

* ``STRONG_BUY``  → ``ActionSide.BUY``   (conviction is captured by
  ``conviction_score`` + ``target_weight`` — no separate side needed)
* ``BUY``         → ``ActionSide.BUY``
* ``HOLD``        → ``ActionSide.HOLD``
* ``SELL``        → ``ActionSide.SELL``
* ``STRONG_SELL`` → ``ActionSide.SELL``

Legacy ``OVERWEIGHT`` / ``NEUTRAL`` / ``UNDERWEIGHT`` inputs are
normalised through the schema's coercion validator before we ever see
them here, but ``_action_side_from_decision`` also handles them
defensively (belt-and-braces).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    OrderTimeInForce,
    SizingBasis,
    ValidityCheck,
)
from trading_crew.horizons import Horizon, MAX_ACTIONABLE_HORIZON_DAYS


# Alias to the single source of truth so old callers keep working.
# See ``trading_crew.horizons.MAX_ACTIONABLE_HORIZON_DAYS`` for the
# rationale (5-year cap on outcome accounting even though the LLM's
# stated horizon can be a 5-10 year narrative view).
_MAX_ACTIONABLE_HORIZON_DAYS = MAX_ACTIONABLE_HORIZON_DAYS

from trading_crew.schemas import PortfolioDecision

logger = logging.getLogger(__name__)


_LONG_TIERS = {"BUY", "STRONG_BUY", "OVERWEIGHT"}
_SHORT_TIERS = {"SELL", "STRONG_SELL", "UNDERWEIGHT"}
_FLAT_TIERS = {"HOLD", "NEUTRAL"}


def _action_side_from_decision(action: str, size_pct: float) -> ActionSide:
    """Map (action, size_pct) to an ``ActionSide``.

    Long tiers (``BUY``, ``STRONG_BUY``) → ``BUY``; short tiers
    (``SELL``, ``STRONG_SELL``) → ``SELL``; ``HOLD`` → ``HOLD``.  A
    directional tier with ``size_pct == 0`` is degenerate (the
    guardrail should have caught it upstream) and rolled back to
    ``ABSTAIN`` so M5's risk gate doesn't have to defend against
    contradictions.  Legacy ``OVERWEIGHT`` / ``UNDERWEIGHT`` /
    ``NEUTRAL`` strings are accepted defensively for records emitted by
    older builds.
    """
    action = (action or "HOLD").upper()
    if action in _LONG_TIERS:
        return ActionSide.BUY if size_pct > 0 else ActionSide.ABSTAIN
    if action in _SHORT_TIERS:
        return ActionSide.SELL if size_pct > 0 else ActionSide.ABSTAIN
    if action in _FLAT_TIERS:
        return ActionSide.HOLD
    return ActionSide.ABSTAIN


def _conviction_tier_from_score(conv: float) -> ConvictionTier:
    """Delegate to the contract's canonical bucketing so the bridge and the
    ``ActionProposal`` validator can never disagree on the thresholds."""
    return ConvictionTier.from_score(conv)


# --- Evidence-based conviction (M1) ---------------------------------------
#
# The Portfolio Manager's raw ``confidence`` is a single self-reported number
# that empirically collapses into a narrow, overconfident band (~0.60-0.80):
# on 824 production episodes 99% of scores fell in [0.60, 0.80] and only 0.5%
# were ever LOW.  A compressed conviction defeats its whole purpose — M5 sizing
# scales risk by it, so nearly every position gets the same size, and two runs
# of the same name on the same day can flip BUY<->SELL at *identical*
# conviction (pure noise, not a change of confidence).
#
# So instead of trusting one number, we fuse the *evidence the PM already
# emitted alongside it* into a conviction that actually spans [0, 1]:
#
#   * action strength   — STRONG_BUY/STRONG_SELL is an explicit high-conviction
#                         call the 4-value ActionSide throws away; we recover it.
#   * evidence balance  — key_drivers vs (key_risks + falsifiers): a thesis with
#                         many supports and few falsifiers is higher conviction
#                         than a finely-balanced one.
#   * capital committed — size_pct_of_book: how much book the PM is willing to
#                         put behind the call (well-defined 0-100, unlike the
#                         expected_return_pct whose units are still being fixed).
#   * raw confidence    — kept, but de-weighted and floor-stretched so the
#                         overconfident band no longer dominates.
#
# This is a deterministic, self-contained recalibration (no LLM, no resolved
# outcomes needed).  Once enough outcomes resolve, the RLMF/calibration layer
# can learn a data-driven correction on top; until then this restores a
# meaningful spread from information already on the decision.
def _conviction_mode() -> str:
    """``evidence`` (default) or ``raw`` — read live so it can be toggled at
    runtime (and monkeypatched in tests) without re-importing."""
    return (os.getenv("TRADINGCREW_CONVICTION_MODE", "evidence") or "evidence").strip().lower()


_CONV_W_ACTION = 0.45
_CONV_W_EVIDENCE = 0.25
_CONV_W_SIZE = 0.15
_CONV_W_CONF = 0.15
# size_pct_of_book (a 0-100 percentage) that counts as "full commitment".
_CONV_SIZE_FULL_PCT = 5.0
# Raw confidence below this floor contributes nothing; above it is stretched to
# [0, 1] so the observed 0.60-0.80 band spreads instead of acting as a constant
# offset.
_CONV_CONF_FLOOR = 0.50

_ACTION_STRENGTH = {
    "STRONG_BUY": 1.0,
    "STRONG_SELL": 1.0,
    "BUY": 0.5,
    "SELL": 0.5,
    "OVERWEIGHT": 0.5,
    "UNDERWEIGHT": 0.5,
    "HOLD": 0.25,
    "NEUTRAL": 0.25,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _evidence_conviction(decision: PortfolioDecision, side: ActionSide) -> float:
    """Fuse the PM's own evidence into a conviction score in [0, 1].

    See the module note above for the rationale.  The result is deterministic
    and depends only on fields already present on ``decision``.
    """
    action = (decision.action or "HOLD").upper()
    s_action = _ACTION_STRENGTH.get(action, 0.0)

    n_drivers = len(decision.key_drivers or [])
    n_against = len(decision.key_risks or []) + len(decision.falsifiers or [])
    denom = n_drivers + n_against
    # No evidence either way -> neutral 0.5 (not a failure fallback; it is the
    # correct prior when the PM listed neither supports nor falsifiers).
    s_evidence = (n_drivers / denom) if denom else 0.5

    size_pct = max(0.0, float(decision.size_pct_of_book or 0.0))
    s_size = _clamp01(size_pct / _CONV_SIZE_FULL_PCT)

    raw_conf = _clamp01(float(decision.confidence or 0.0))
    s_conf = _clamp01((raw_conf - _CONV_CONF_FLOOR) / (1.0 - _CONV_CONF_FLOOR))

    conv = (
        _CONV_W_ACTION * s_action
        + _CONV_W_EVIDENCE * s_evidence
        + _CONV_W_SIZE * s_size
        + _CONV_W_CONF * s_conf
    )
    # A degenerate directional call rolled back to ABSTAIN carries no real
    # conviction; cap it low so downstream sizing can't act on it.
    if side == ActionSide.ABSTAIN:
        conv = min(conv, 0.15)
    return _clamp01(conv)


def _signed_expected_return(er: float, side: ActionSide) -> float:
    """Force ``expected_return_pct`` (a canonical DECIMAL fraction) to agree
    in sign with the trade side.

    Convention: ``expected_return_pct`` is the expected move of the
    *underlying* — positive for a long thesis, negative for a short.  The PM
    occasionally emits a contradictory sign (e.g. a positive number on a
    SELL); we resolve it deterministically from the side magnitude rather
    than trusting the stray sign, so M5 Kelly sizing and the M7 allocator see
    a coherent edge.  HOLD / ABSTAIN carry no directional edge.
    """
    mag = abs(er)
    if side == ActionSide.BUY:
        return mag
    if side == ActionSide.SELL:
        return -mag
    return 0.0


def _target_weight_from_decision(side: ActionSide, size_pct: float) -> float:
    """``size_pct_of_book`` is a 0-100 percentage; convert to a fraction in
    the proposal's signed weight convention.

    SELL flips the sign; HOLD / ABSTAIN clamp to 0 regardless of the
    upstream size_pct (which the PM sometimes leaves stale).
    """
    size_fraction = max(0.0, min(1.0, float(size_pct) / 100.0))
    if side == ActionSide.SELL:
        return -size_fraction
    if side in (ActionSide.HOLD, ActionSide.ABSTAIN):
        return 0.0
    return size_fraction


def _validity_check_from_decision(
    decision: PortfolioDecision,
    side: ActionSide,
    size_pct: float,
) -> ValidityCheck:
    """Synthesize a ``ValidityCheck`` from the CrewAI decision.

    We can't ask the agent to fill these flags retroactively, but we can
    derive them from fields it already produced:

    - ``data_timestamps_valid``: True iff ``compliance_status != BLOCKED``
      (Compliance Officer blocks on stale data among other things).
    - ``fits_risk_budget``: True iff size_pct is within [0, 100] and the
      decision is internally consistent (action <-> side agree).
    - ``survives_transaction_costs``: True iff the expected return is
      strictly above 1% absolute (we use 1% as a coarse cost floor; the
      real check happens in M2's cost model regardless).
    - ``liquidity_sufficient``: True iff ``compliance_status != BLOCKED``.

    Any False flag will reduce the M5 risk-multiplier; multiple Falses
    can collapse a proposal to zero size at the sizer.
    """
    compliance_ok = (decision.compliance_status or "CLEAR").upper() != "BLOCKED"
    expected_abs = abs(decision.expected_return_pct or 0.0)
    consistent_with_side = side in (ActionSide.HOLD, ActionSide.ABSTAIN) or size_pct > 0
    return ValidityCheck(
        data_timestamps_valid=compliance_ok,
        fits_risk_budget=consistent_with_side and 0 <= size_pct <= 100,
        survives_transaction_costs=expected_abs >= 0.01,
        liquidity_sufficient=compliance_ok,
        notes=(
            f"Synthesized from PortfolioDecision: compliance={decision.compliance_status}, "
            f"size%={size_pct:.2f}, expected_return%={decision.expected_return_pct:.4f}"
        ),
    )


def portfolio_decision_to_action_proposal(
    decision: PortfolioDecision,
    *,
    symbol: str,
    decision_ts: Optional[str] = None,
) -> ActionProposal:
    """Translate a CrewAI ``PortfolioDecision`` into an M1 ``ActionProposal``.

    ``decision_ts`` defaults to ``datetime.utcnow().isoformat()``; callers
    should pass the actual trade-date timestamp when running historical
    analyses so episodic memory tags the right outcome window.

    The function is pure — no LLM call, no I/O.  Any error during
    construction is logged and a structured ``ABSTAIN`` proposal is
    returned so the downstream pipeline can record the run without
    crashing the kickoff.
    """
    decision_ts = decision_ts or datetime.now(timezone.utc).isoformat()
    try:
        side = _action_side_from_decision(decision.action, decision.size_pct_of_book)
        target_weight = _target_weight_from_decision(side, decision.size_pct_of_book)
        # Derive the conviction score, then bucket the tier from the *same*
        # value so the two can never disagree.  ``evidence`` (default) fuses the
        # PM's own drivers/risks/action-strength/size into a score that actually
        # spans [0, 1]; ``raw`` preserves the legacy "conviction == confidence"
        # behaviour for A/B comparison and easy rollback.
        if _conviction_mode() == "raw":
            conviction = _clamp01(float(decision.confidence or 0.0))
        else:
            conviction = _evidence_conviction(decision, side)
        tier = _conviction_tier_from_score(conviction)
        validity = _validity_check_from_decision(decision, side, decision.size_pct_of_book)

        rationale = (decision.rationale or "").strip()
        # The ActionProposal schema doesn't carry stop / target directly,
        # so we surface them in the rationale where M5's audit panel can
        # display them alongside the trade-thesis text.  We pad the
        # rationale anyway when it's shorter than the schema minimum so
        # validation doesn't reject a valid PM output that was terse.
        targets = (
            f"stop_loss={decision.stop_loss:.2f}, "
            f"target_price={decision.target_price:.2f}"
        )
        drivers = ", ".join(decision.key_drivers or []) or "n/a"
        rationale = (
            f"{rationale}\nKey drivers: {drivers}.\nLevels: {targets}."
        ).strip()

        # Clamp horizon_days for the ActionProposal outcome window.  For
        # SHORT/MEDIUM this is a no-op; for LONG (up to 2520d in the
        # decision) we cap at 5y = 1260d so the outcome-resolution
        # loop still finishes within the RLMF training window.  We
        # emit a tag so downstream consumers (memory, UI) know the
        # decision's *narrative* horizon differs from the *accounting*
        # horizon.
        raw_horizon = max(1, int(decision.horizon_days or 21))
        accounting_horizon = min(raw_horizon, _MAX_ACTIONABLE_HORIZON_DAYS)
        tags = list(decision.geopolitical_flags or [])
        if accounting_horizon < raw_horizon:
            tags.append(
                f"horizon_capped:{raw_horizon}d→{accounting_horizon}d"
            )

        return ActionProposal(
            symbol=symbol.upper(),
            decision_ts=decision_ts,
            side=side,
            target_weight=target_weight,
            horizon_days=accounting_horizon,
            limit_price=decision.entry_price if decision.entry_price > 0 else None,
            tif=OrderTimeInForce.DAY,
            conviction_score=conviction,
            conviction_tier=tier,
            sizing_basis=SizingBasis.TARGET_WEIGHT,
            expected_return_pct=_signed_expected_return(
                float(decision.expected_return_pct or 0.0), side
            ),
            rationale=rationale,
            validity_check=validity,
            tags=tags,
        )
    except Exception as exc:
        logger.warning("PortfolioDecision -> ActionProposal failed (%s); abstaining", exc)
        return ActionProposal(
            symbol=symbol.upper(),
            decision_ts=decision_ts,
            side=ActionSide.ABSTAIN,
            target_weight=0.0,
            horizon_days=21,
            conviction_score=0.0,
            conviction_tier=ConvictionTier.LOW,
            expected_return_pct=0.0,
            rationale=f"Bridge fallback: could not translate PortfolioDecision: {exc}",
            validity_check=ValidityCheck(
                data_timestamps_valid=False,
                fits_risk_budget=False,
                survives_transaction_costs=False,
                liquidity_sufficient=False,
                notes="Synthesized ABSTAIN on translation failure.",
            ),
        )
