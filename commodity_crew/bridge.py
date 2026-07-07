"""Convert ``FuturesDecision`` into a ``PortfolioDecision`` so the
existing M1 bridge (``trading_crew.agentic.bridge``) can lift it into
an ``ActionProposal``.

We intentionally route through ``PortfolioDecision`` rather than
shortcutting straight to ``ActionProposal`` because:

1. The reflective critic (M4) is implemented against
   ``PortfolioDecision`` — going through that schema lets the critic
   work on commodity decisions unchanged.
2. Run persistence (Phase E) snapshots ``PortfolioDecision`` natively;
   if we shipped ``FuturesDecision`` end-to-end we'd need a parallel
   schema in the run record.
3. The futures-specific fields (``contract_month``, ``contract_size``,
   ``curve_view``, ``roll_yield_pct_annualised``) are preserved as text
   in the ``rationale`` and as a list entry in ``key_drivers`` so the
   audit trail loses nothing.
"""

from __future__ import annotations

from typing import Dict

from trading_crew.schemas import PortfolioDecision

from .schemas import FuturesDecision


# FuturesDecision uses the futures-idiomatic ``LONG`` / ``NEUTRAL`` /
# ``SHORT`` action alphabet.  PortfolioDecision (the shared schema) now
# uses the retail 5-tier ``STRONG_BUY`` / ``BUY`` / ``HOLD`` / ``SELL`` /
# ``STRONG_SELL``.  We collapse LONG→BUY and SHORT→SELL (not to the
# STRONG_* tiers, since the FuturesDecision doesn't carry a separate
# conviction bit — it only exposes ``confidence``, which the equity
# schema already carries).  Downstream code that wants the higher tier
# for a very confident futures bet can promote the label post-hoc based
# on ``confidence`` + ``size_pct_of_book``, but by default we keep it
# conservative to avoid over-signalling to the UI.
_ACTION_MAP: Dict[str, str] = {
    "LONG": "BUY",
    "NEUTRAL": "HOLD",
    "SHORT": "SELL",
}


def futures_decision_to_portfolio_decision(fd: FuturesDecision) -> PortfolioDecision:
    """Lossless-on-the-surface adapter: every futures field is preserved
    either in the typed PortfolioDecision schema or embedded into the
    rationale + key_drivers/key_risks so nothing is silently dropped.
    """
    action = _ACTION_MAP.get(fd.action, "HOLD")
    futures_context = (
        f"Contract: {fd.contract_month} ({fd.contract_size:.0f}/contract). "
        f"Curve: {fd.curve_view} (annualised roll {fd.roll_yield_pct_annualised:+.2f}%)."
    )
    rationale = f"{fd.rationale}\n\n{futures_context}".strip()

    # Surface the futures-specific bits in key_drivers so the M1 bridge's
    # ActionProposal carries them as ``tags`` — that's how the UI's
    # order-ticket panel renders them.
    extra_drivers = [
        f"contract_month:{fd.contract_month}",
        f"contract_size:{fd.contract_size:.0f}",
        f"curve_view:{fd.curve_view}",
    ]
    if abs(fd.roll_yield_pct_annualised) >= 0.1:
        extra_drivers.append(f"roll_yield:{fd.roll_yield_pct_annualised:+.2f}%/y")

    return PortfolioDecision(
        action=action,
        confidence=fd.confidence,
        size_pct_of_book=fd.size_pct_of_book,
        entry_price=fd.entry_price,
        stop_loss=fd.stop_loss,
        target_price=fd.target_price,
        horizon_days=fd.horizon_days,
        expected_return_pct=fd.expected_return_pct,
        rationale=rationale,
        key_drivers=list(fd.key_drivers) + extra_drivers,
        key_risks=list(fd.key_risks),
        falsifiers=list(fd.falsifiers),
        geopolitical_flags=list(fd.geopolitical_flags or []),
        compliance_status=fd.compliance_status,
        # Forward provenance untouched so the Reflective Critic sees the
        # same identifiers the inline ``[source: …]`` tags reference.
        sources=list(getattr(fd, "sources", None) or []),
    )
