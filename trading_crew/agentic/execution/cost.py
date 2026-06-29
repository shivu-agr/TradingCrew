"""Transaction-cost model used by the execution simulator.

Implements paper §6.2 ("Execution and Cost Modeling"): a credible trading
agent cannot claim profitability without an explicit, sensitivity-tested
cost layer.  The model has three components:

1. **Fees** — exchange/broker commissions, expressed as a fixed bps of notional
   plus an optional flat per-trade charge.  These are deterministic and known
   at order time.
2. **Half-spread** — the difference between mid and the side of the book the
   order has to cross.  Expressed in bps; the executor pays this on every
   marketable order.
3. **Slippage / market impact** — non-linear cost from consuming liquidity.
   We use the Almgren-Chriss square-root model: ``impact_bps = k * sqrt(participation)``,
   where ``participation = order_size / ADV``.  This is the standard
   "permanent + temporary impact" decomposition simplified to a single
   coefficient ``k`` per asset class.

The three components live in a single ``CostModel`` so callers can swap them
together (different asset classes get different ``k``, ``half_spread_bps``,
and ``fee_bps``) without touching the simulator code.

Sensitivity testing (paper §13.1.6 #2): the ``sweep`` helper runs the same
``ActionProposal`` under a grid of cost assumptions so the UI can plot how
fragile a strategy is.  This is the single most important deliverable for
moving us from R0 -> R2 reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Round-trip transaction cost model for a single asset class.

    All bps figures are *per-leg* (so a round trip pays them twice).  The
    coefficient ``impact_k`` is in bps per sqrt(participation) where
    participation is order_notional / (price * ADV).

    Reasonable defaults for US large-cap equity (used as the "standard"
    scenario in the M2 sensitivity sweep):

    - ``fee_bps = 1.0`` — modern institutional commissions
    - ``flat_fee = 0.0`` — most prime brokers don't charge flat anymore
    - ``half_spread_bps = 2.5`` — average bid/ask half-spread on liquid names
    - ``impact_k = 10.0`` — Almgren-Chriss coefficient implying ~1% impact at
      1% participation

    Crank these up for small-cap (impact_k ~ 30), down for index futures
    (half_spread_bps ~ 0.5).  None of these are hardcoded fallbacks — they
    are explicit configuration the caller chooses.
    """

    fee_bps: float
    flat_fee: float
    half_spread_bps: float
    impact_k: float
    name: str = "default"

    # -- per-leg cost decomposition ---------------------------------------

    def fees(self, notional: float) -> float:
        """Deterministic fees on a single leg: percentage + flat."""
        return abs(notional) * self.fee_bps / 1e4 + self.flat_fee

    def spread_cost(self, notional: float) -> float:
        """Half-spread cost on a single leg (always positive — you pay to cross)."""
        return abs(notional) * self.half_spread_bps / 1e4

    def impact_bps(self, participation: float) -> float:
        """Almgren-Chriss square-root impact in bps.

        ``participation`` is the fraction of ADV the order represents
        (notional / (price * ADV)).  Returns 0 if participation is 0 or
        negative (defensive — but caller should never pass <0).
        """
        if participation <= 0:
            return 0.0
        return self.impact_k * math.sqrt(participation)

    def impact_cost(self, notional: float, participation: float) -> float:
        """Slippage cost in dollars given notional and participation."""
        return abs(notional) * self.impact_bps(participation) / 1e4

    # -- composite --------------------------------------------------------

    def total_cost(
        self,
        notional: float,
        participation: float = 0.0,
    ) -> Dict[str, float]:
        """Total per-leg cost, broken into components for the UI.

        Returns a dict so the execution panel can display each line item.
        ``total`` is the sum the caller should debit from cash.
        """
        fees = self.fees(notional)
        spread = self.spread_cost(notional)
        impact = self.impact_cost(notional, participation)
        return {
            "fees": fees,
            "spread": spread,
            "impact": impact,
            "total": fees + spread + impact,
            "total_bps": (fees + spread + impact) / abs(notional) * 1e4 if notional else 0.0,
        }


# ---------------------------------------------------------------------------
# Stock presets — explicit configuration, not silent fallbacks
# ---------------------------------------------------------------------------


COST_MODEL_LIBRARY: Dict[str, CostModel] = {
    # Low-friction: optimistic / index-future-like assumptions.  Use this to
    # see "best case" strategy returns when stress-testing.
    "low": CostModel(
        fee_bps=0.5, flat_fee=0.0, half_spread_bps=0.5, impact_k=4.0, name="low",
    ),
    # Realistic US large-cap equity book (Goldman, Morgan, mid-size HF).
    "standard": CostModel(
        fee_bps=1.0, flat_fee=0.0, half_spread_bps=2.5, impact_k=10.0, name="standard",
    ),
    # High-friction: small-cap, illiquid, or retail-style execution.
    "high": CostModel(
        fee_bps=2.0, flat_fee=0.0, half_spread_bps=8.0, impact_k=25.0, name="high",
    ),
    # ---- Futures presets ------------------------------------------------
    # Futures cost geometry is different from equities: exchange fees are
    # tiny (CME charges ~$1-2/contract on round trip ≈ 0.1-0.3 bps on
    # notional), half-spreads on flagship contracts (CL, GC, ZC) are ~1
    # tick which on a 100k notional is ~1 bp, and the leverage embedded
    # in futures means impact at small participation is comparable to
    # equities — but the relevant ADV is much larger.
    #
    # The "futures_*" presets express *per-leg* costs in basis points of
    # notional (not margin) so they slot into the same sizer / simulator
    # the equity flow uses.  Roll-yield is a separate carry cost handled
    # by the curve analyst and surfaced in FuturesDecision — it is NOT
    # baked into these transaction-cost models because it depends on
    # holding period rather than execution.
    "futures_low": CostModel(
        fee_bps=0.2, flat_fee=2.0, half_spread_bps=0.5, impact_k=6.0,
        name="futures_low",  # index / energy front-month, deep liquidity
    ),
    "futures_standard": CostModel(
        fee_bps=0.3, flat_fee=2.0, half_spread_bps=1.5, impact_k=12.0,
        name="futures_standard",  # typical CL/GC/ZS during US session
    ),
    "futures_high": CostModel(
        fee_bps=0.5, flat_fee=2.5, half_spread_bps=4.0, impact_k=22.0,
        name="futures_high",  # softs, livestock, back-month, off-hours
    ),
}


# Carry-cost helpers ---------------------------------------------------------
#
# For a long futures position, roll-yield represents the per-period drag
# (contango) or lift (backwardation) of holding the position through a
# contract rollover.  It's NOT a transaction cost in the per-leg sense
# above — it's a *carry* cost that scales with holding period.  Exposing
# it here keeps the M2 sweep + risk panel honest about long-dated futures
# trades that "look cheap" on round-trip costs but bleed via roll.


def roll_yield_carry_cost(
    notional: float,
    annualised_roll_yield_pct: float,
    holding_days: int,
) -> float:
    """Dollar carry cost over the holding period.

    Negative roll yield (contango) returns a positive cost (the trader pays it).
    Positive roll yield (backwardation) returns a negative cost (the trader
    receives it — modelled as a carry credit).

    Returns the signed dollar amount the position will accrue from rolling
    over ``holding_days`` calendar days at the given annualised rate.
    """
    if notional == 0.0 or holding_days <= 0:
        return 0.0
    daily_rate = (annualised_roll_yield_pct / 100.0) / 365.0
    carry = -abs(notional) * daily_rate * float(holding_days)
    return carry


def get_cost_model(name: str) -> CostModel:
    """Return a preset cost model by name (``low`` / ``standard`` / ``high``).

    Raises ``KeyError`` if the name isn't registered — the caller must pick
    explicitly.  No silent default.
    """
    if name not in COST_MODEL_LIBRARY:
        raise KeyError(
            f"Unknown cost model '{name}'. Available: {list(COST_MODEL_LIBRARY)}"
        )
    return COST_MODEL_LIBRARY[name]


# ---------------------------------------------------------------------------
# Sensitivity sweep — the deliverable that gets us to R2 reproducibility
# ---------------------------------------------------------------------------


@dataclass
class CostScenario:
    """A single cost scenario inside a sweep.

    ``label`` is shown in the UI; ``model`` is the cost model applied;
    ``net_pnl`` and ``edge_survives`` are populated by the caller after
    running the strategy under this scenario.
    """

    label: str
    model: CostModel
    notional: float
    participation: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)


def cost_sweep(
    notional: float,
    participation: float,
    models: Iterable[str] = ("low", "standard", "high"),
) -> List[CostScenario]:
    """Compute cost-decomposition under each named preset.

    The UI calls this with the proposal's notional and participation and
    displays a table of "what would this trade cost under low/std/high?"
    so the user can eyeball edge survival before any LLM call has fired.
    """
    out: List[CostScenario] = []
    for name in models:
        model = get_cost_model(name)
        out.append(
            CostScenario(
                label=name,
                model=model,
                notional=notional,
                participation=participation,
                cost_breakdown=model.total_cost(notional, participation),
            )
        )
    return out
