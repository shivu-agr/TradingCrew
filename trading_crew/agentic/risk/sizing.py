"""Position sizing: fractional Kelly + vol target + CVaR clamp (paper §7.2).

The sizer takes the agent's typed intent (``ActionProposal``) and produces
a *deterministic* size multiplier that respects three independent caps:

1. **Fractional Kelly** — ``f* = (expected_return - r_f) / variance``
   scaled by a fraction (default 0.25) because full Kelly is empirically
   too aggressive on noisy edge estimates.  See Thorp (1969), MacLean
   et al. (2010).
2. **Vol-target** — cap |target_weight| so the position contributes at most
   ``vol_target`` to portfolio annualised vol.  Standard institutional
   practice; keeps sizing stable across vol regimes.
3. **CVaR clamp** — cap |target_weight| so the position's expected tail
   loss (CVaR at 95%) is below ``max_cvar`` of NAV.

The binding constraint is the *minimum* of all three.  Confidence and
risk-debate sentiment are folded in as a multiplier in ``[0.5, 1.0]`` so
a low-conviction proposal gets sized half what a high-conviction one
would, without ever exceeding the deterministic caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
)


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingConfig:
    """Sizing parameters — all explicit, no silent defaults that hide policy.

    - ``kelly_fraction``: fraction of full Kelly to use.  0.25 = quarter-Kelly.
    - ``risk_free_rate``: annualised, used in the Kelly numerator.
    - ``vol_target``: annualised portfolio-vol cap *per position*.  0.10 = 10%.
    - ``max_cvar_pct``: max position CVaR as a fraction of NAV.  0.02 = 2%.
    - ``max_position_weight``: hard cap on |target_weight| regardless of
      the formulas above.  0.20 = 20% of NAV.
    - ``vol_lookback_days``: trading days used to estimate annualised vol.
    - ``risk_mult_floor``: minimum debate-derived multiplier (0.5 default).
    - ``risk_mult_ceiling``: maximum debate multiplier (1.0 default — debate
      can only *reduce* size, never amplify it).
    """

    kelly_fraction: float = 0.25
    risk_free_rate: float = 0.04
    vol_target: float = 0.10
    max_cvar_pct: float = 0.02
    max_position_weight: float = 0.20
    vol_lookback_days: int = 63
    risk_mult_floor: float = 0.5
    risk_mult_ceiling: float = 1.0


@dataclass
class SizingResult:
    """Sizing decomposition surfaced to the UI's risk panel.

    Each ``*_cap`` field shows the size cap imposed by that constraint, so
    users can see *which* constraint bound the final weight.  ``final_weight``
    is the signed weight (positive long, negative short).
    """

    final_weight: float
    risk_mult: float
    kelly_cap: float
    vol_cap: float
    cvar_cap: float
    hard_cap: float
    binding_constraint: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Sizer
# ---------------------------------------------------------------------------


def compute_size(
    proposal: ActionProposal,
    *,
    realised_vol_annualised: float,
    cvar_one_day: float,
    risk_mult: float = 1.0,
    config: SizingConfig = SizingConfig(),
) -> SizingResult:
    """Return a deterministic position size for ``proposal``.

    ``realised_vol_annualised`` is the per-position annualised realised
    volatility (e.g. ``std(daily_returns) * sqrt(252)``).  ``cvar_one_day``
    is the 1-day CVaR as a positive fraction of price.

    All three formula caps are computed; the binding one is the minimum
    in absolute value, then signed by the proposal's direction.

    Returns the cap *and* the binding constraint so the UI can show
    which rule capped the trade (paper §7.2: "make caps transparent").
    """
    if proposal.side in (ActionSide.HOLD, ActionSide.ABSTAIN):
        return SizingResult(
            final_weight=0.0,
            risk_mult=0.0,
            kelly_cap=0.0,
            vol_cap=0.0,
            cvar_cap=0.0,
            hard_cap=config.max_position_weight,
            binding_constraint="HOLD_OR_ABSTAIN",
            notes="No size — proposal is HOLD or ABSTAIN.",
        )

    rm = max(config.risk_mult_floor, min(config.risk_mult_ceiling, risk_mult))

    # --- Kelly ----------------------------------------------------------
    # Convert horizon expected return to annualised
    annual_factor = 252 / max(1, proposal.horizon_days)
    expected_annual = proposal.expected_return_pct * annual_factor
    excess = expected_annual - config.risk_free_rate
    var_annual = max(realised_vol_annualised ** 2, 1e-6)  # avoid /0
    full_kelly = excess / var_annual
    kelly_weight = config.kelly_fraction * full_kelly

    # --- Vol-target -----------------------------------------------------
    if realised_vol_annualised > 0:
        vol_weight = config.vol_target / realised_vol_annualised
    else:
        # Flat-vol series — vol-target is non-binding, fall back to hard cap
        vol_weight = config.max_position_weight

    # --- CVaR clamp -----------------------------------------------------
    # We want: |weight| * cvar_one_day <= max_cvar_pct
    if cvar_one_day > 0:
        cvar_weight = config.max_cvar_pct / cvar_one_day
    else:
        cvar_weight = config.max_position_weight

    # --- Hard cap -------------------------------------------------------
    hard_weight = config.max_position_weight

    # --- Combine --------------------------------------------------------
    # Use absolute caps; pick the smallest
    candidates = {
        "kelly":  abs(kelly_weight),
        "vol":    abs(vol_weight),
        "cvar":   abs(cvar_weight),
        "hard":   hard_weight,
        "intent": abs(proposal.target_weight),
    }
    binding = min(candidates, key=candidates.get)
    bound_abs = candidates[binding]

    # Apply the debate multiplier and the proposal's sign
    signed = math.copysign(bound_abs, proposal.target_weight if proposal.target_weight else 1.0) * rm
    # Keep within [-hard_weight, +hard_weight] in case rm pushed us out
    signed = max(-hard_weight, min(hard_weight, signed))

    return SizingResult(
        final_weight=signed,
        risk_mult=rm,
        kelly_cap=kelly_weight,
        vol_cap=vol_weight,
        cvar_cap=cvar_weight,
        hard_cap=hard_weight,
        binding_constraint=binding,
        notes=(
            f"Kelly {kelly_weight:+.3f} | Vol {vol_weight:+.3f} | CVaR {cvar_weight:+.3f} | "
            f"Hard {hard_weight:.3f} | Intent {proposal.target_weight:+.3f}. "
            f"Bound by {binding}; debate multiplier {rm:.2f}."
        ),
    )


# ---------------------------------------------------------------------------
# Risk-debate -> multiplier
# ---------------------------------------------------------------------------


def debate_to_risk_mult(
    risk_debate_state: dict,
    *,
    proposal: ActionProposal,
    floor: float = 0.5,
    ceiling: float = 1.0,
) -> tuple[float, str]:
    """Convert qualitative risk-debate output into a [floor, ceiling] multiplier.

    Rules (each contributes a multiplicative penalty):

    - Aggressive contributions ≠ majority and Conservative wins -> ×0.8
    - Conviction LOW            -> ×0.7
    - Conviction MEDIUM         -> ×0.9
    - Validity flags any False  -> ×0.8 per flag, multiplied together,
                                   capped at floor

    Returns (multiplier, explanation).  No LLM call required — the
    debate state is consumed deterministically so the multiplier is
    reproducible across runs.
    """
    mult = 1.0
    parts = []

    tier = proposal.conviction_tier
    if tier == ConvictionTier.LOW:
        mult *= 0.7
        parts.append("LOW conviction ×0.7")
    elif tier == ConvictionTier.MEDIUM:
        mult *= 0.9
        parts.append("MEDIUM conviction ×0.9")

    vc = proposal.validity_check
    flags = [
        ("timestamps", vc.data_timestamps_valid),
        ("risk_budget", vc.fits_risk_budget),
        ("survives_costs", vc.survives_transaction_costs),
        ("liquidity", vc.liquidity_sufficient),
    ]
    for name, ok in flags:
        if not ok:
            mult *= 0.8
            parts.append(f"{name} flag failed ×0.8")

    # Risk-debate winner heuristic: presence of "Conservative" in the latest_speaker
    speaker = (risk_debate_state or {}).get("latest_speaker", "").lower()
    if "conservative" in speaker and proposal.side != ActionSide.ABSTAIN:
        mult *= 0.85
        parts.append("Conservative analyst last to speak ×0.85")

    mult = max(floor, min(ceiling, mult))
    explanation = " · ".join(parts) if parts else "No adjustments — full size."
    return mult, explanation
