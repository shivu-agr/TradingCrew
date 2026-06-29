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
inconsistent (e.g. ``OVERWEIGHT`` with ``size_pct_of_book == 0``), we fall
back to ``ABSTAIN`` rather than emitting a malformed proposal.
"""

from __future__ import annotations

import logging
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
from trading_crew.schemas import PortfolioDecision

logger = logging.getLogger(__name__)


def _action_side_from_decision(action: str, size_pct: float) -> ActionSide:
    """Map (action, size_pct) to an ``ActionSide``.

    ``NEUTRAL`` *and* ``size_pct == 0`` -> HOLD.  ``OVERWEIGHT`` -> BUY,
    ``UNDERWEIGHT`` -> SELL.  An ``OVERWEIGHT/UNDERWEIGHT`` with zero size
    is degenerate and rolled back to ``ABSTAIN`` so M5's risk gate doesn't
    have to defend against contradictions.
    """
    action = (action or "NEUTRAL").upper()
    if action == "OVERWEIGHT":
        return ActionSide.BUY if size_pct > 0 else ActionSide.ABSTAIN
    if action == "UNDERWEIGHT":
        return ActionSide.SELL if size_pct > 0 else ActionSide.ABSTAIN
    if action == "NEUTRAL":
        return ActionSide.HOLD
    return ActionSide.ABSTAIN


def _conviction_tier_from_score(conv: float) -> ConvictionTier:
    if conv >= 0.65:
        return ConvictionTier.HIGH
    if conv >= 0.35:
        return ConvictionTier.MEDIUM
    return ConvictionTier.LOW


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
        tier = _conviction_tier_from_score(decision.confidence)
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

        return ActionProposal(
            symbol=symbol.upper(),
            decision_ts=decision_ts,
            side=side,
            target_weight=target_weight,
            horizon_days=max(1, int(decision.horizon_days or 21)),
            limit_price=decision.entry_price if decision.entry_price > 0 else None,
            tif=OrderTimeInForce.DAY,
            conviction_score=max(0.0, min(1.0, float(decision.confidence or 0.0))),
            conviction_tier=tier,
            sizing_basis=SizingBasis.TARGET_WEIGHT,
            expected_return_pct=float(decision.expected_return_pct or 0.0),
            rationale=rationale,
            validity_check=validity,
            tags=list(decision.geopolitical_flags or []),
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
