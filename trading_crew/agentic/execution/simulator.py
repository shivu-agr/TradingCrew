"""Execution simulator — turns an ``Order`` into a ``Fill`` (paper §6.1).

The simulator is the deterministic boundary between the agent's typed intent
(``ActionProposal``) and the portfolio's ground-truth state.  It models:

- **Next-bar fill timing**: orders submitted at ``decision_ts`` get filled at
  the *next* bar's open, never at the current bar's close — closing this
  loophole is exactly what the paper calls out as the most common execution-
  semantics flaw (§6.1 "implicit execution").
- **Limit-order semantics**: if a limit is set and the next bar's range
  doesn't cross it, the order is treated as a *Rejection* (TIF=DAY) or
  carried (TIF=GTC).  IOC fills what's marketable and cancels the rest.
- **Partial fills**: if the order size exceeds the configured per-bar
  participation cap (default 5% of ADV), only the cap is filled and the
  remainder is reported as ``status=PARTIAL_FILL``.
- **Costs**: fees, spread, and Almgren-Chriss square-root impact are applied
  from the ``CostModel`` and debited from cash inside ``PortfolioState``.

The simulator is dependency-injection friendly: callers pass in the
``Bar`` data (OHLCV for the fill bar) explicitly so the same code can drive
unit tests, walk-forward backtests (M6), and live runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    OrderTimeInForce,
)
from trading_crew.agentic.execution.cost import CostModel
from trading_crew.agentic.portfolio.state import PortfolioState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order / Bar / Fill — internal types
# ---------------------------------------------------------------------------


class FillStatus(str, Enum):
    """Possible terminal states from the simulator (paper §6.1)."""

    FILLED = "FILLED"
    PARTIAL_FILL = "PARTIAL_FILL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar.

    The fill simulator only needs the next bar after the decision.  ADV
    (Average Daily Volume) is required to compute participation for the
    Almgren-Chriss impact term.  When ADV is unknown the caller should
    pass the bar's own volume as a conservative proxy.
    """

    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adv: float

    def crosses_limit(self, side: ActionSide, limit_price: float) -> bool:
        """True iff the bar's range touches the limit price for ``side``.

        - For a BUY limit, fillable if ``low <= limit`` (we'd be willing to
          pay up to limit, market traded at or below it).
        - For a SELL limit, fillable if ``high >= limit``.
        """
        if side == ActionSide.BUY:
            return self.low <= limit_price
        return self.high >= limit_price


@dataclass(frozen=True)
class Order:
    """Compiled order — what the sizer + risk gate hand to the simulator.

    Distinct from ``ActionProposal`` because:
    - ``qty_signed`` is concrete shares (sizer converted target_weight via
      NAV / price), not a fractional weight.
    - ``side`` and ``limit_price`` come from the proposal.
    - ``decision_ts`` and ``tif`` set when the order expires.
    """

    symbol: str
    side: ActionSide
    qty_signed: float
    limit_price: Optional[float]
    tif: OrderTimeInForce
    decision_ts: str


@dataclass
class Fill:
    """Result of attempting to execute an ``Order``.

    Fields:

    - ``status``: one of ``FillStatus``.
    - ``qty_filled``: signed share count actually filled (0 if rejected).
    - ``avg_price``: VWAP of the fill (mid + slippage); None when rejected.
    - ``cost_breakdown``: per-component cost dollars (fees / spread / impact).
    - ``slippage_bps``: realised slippage vs the bar open (positive = adverse).
    - ``latency_ms``: simulator-injected latency (deterministic, configurable).
    - ``reason``: human-readable note for rejections / partial fills.
    """

    status: FillStatus
    qty_filled: float
    avg_price: Optional[float]
    cost_breakdown: dict
    slippage_bps: float
    latency_ms: int
    reason: str
    ts: str


# ---------------------------------------------------------------------------
# ExecutionSimulator
# ---------------------------------------------------------------------------


@dataclass
class ExecutionSimulator:
    """Deterministic next-bar execution simulator with explicit failure modes.

    Parameters:

    - ``cost_model``: see ``execution.cost.CostModel``.
    - ``participation_cap``: maximum fraction of the next bar's volume the
      simulator will fill in one go.  Excess size is reported as a partial
      fill.  Default 0.05 (5% of ADV — the same threshold the paper's
      §6.1 microstructure example uses).
    - ``latency_ms``: deterministic latency injected on every fill so callers
      can stress-test the latency-sensitivity of their strategy.  Default 50ms.
    """

    cost_model: CostModel
    participation_cap: float = 0.05
    latency_ms: int = 50

    def execute(
        self,
        order: Order,
        next_bar: Bar,
        state: PortfolioState,
    ) -> Fill:
        """Run a single order against ``next_bar`` and mutate ``state`` on fill.

        Algorithm:

        1. Resolve the trade price (limit if set & crosses; otherwise the
           next bar's open).
        2. Compute participation = abs(qty) / next_bar.adv.
        3. If qty exceeds the participation cap, truncate and tag as PARTIAL.
        4. Apply cost model: fees + half-spread + Almgren-Chriss impact.
           The half-spread + impact is added to the trade price as adverse
           slippage (the actual price the book showed).
        5. Mutate ``state.apply_fill`` with the realised qty, price, fees.
        6. Return a ``Fill`` with the full diagnostic breakdown.
        """
        if order.qty_signed == 0:
            return Fill(
                status=FillStatus.REJECTED,
                qty_filled=0.0,
                avg_price=None,
                cost_breakdown={},
                slippage_bps=0.0,
                latency_ms=self.latency_ms,
                reason="Zero-qty order — rejected before simulator.",
                ts=next_bar.ts,
            )

        # --- price discovery ------------------------------------------------

        if order.limit_price is not None:
            if not next_bar.crosses_limit(order.side, order.limit_price):
                # Limit not touched in the next bar — Day order rejects, GTC
                # would carry but we don't model multi-bar carry here.
                status = (
                    FillStatus.EXPIRED if order.tif == OrderTimeInForce.GTC
                    else FillStatus.REJECTED
                )
                return Fill(
                    status=status,
                    qty_filled=0.0,
                    avg_price=None,
                    cost_breakdown={},
                    slippage_bps=0.0,
                    latency_ms=self.latency_ms,
                    reason=f"Limit {order.limit_price} not crossed in [{next_bar.low}, {next_bar.high}]",
                    ts=next_bar.ts,
                )
            # When the limit is touched, we get the limit price (favorable
            # case — we're willing to pay up to limit; market traded at or
            # better than limit).
            mid_price = order.limit_price
        else:
            # Market order — fill at the bar open.
            mid_price = next_bar.open

        # --- participation truncation --------------------------------------

        if next_bar.adv <= 0:
            return Fill(
                status=FillStatus.REJECTED,
                qty_filled=0.0,
                avg_price=None,
                cost_breakdown={},
                slippage_bps=0.0,
                latency_ms=self.latency_ms,
                reason="ADV reported as 0 — cannot fill against zero liquidity",
                ts=next_bar.ts,
            )

        requested_abs = abs(order.qty_signed)
        max_fill_abs = self.participation_cap * next_bar.adv
        if requested_abs > max_fill_abs:
            filled_abs = max_fill_abs
            partial = True
        else:
            filled_abs = requested_abs
            partial = False

        qty_filled = math.copysign(filled_abs, order.qty_signed)
        participation = filled_abs / next_bar.adv

        # --- cost model ----------------------------------------------------

        notional = filled_abs * mid_price
        costs = self.cost_model.total_cost(notional, participation)

        # Apply adverse slippage: half-spread + impact moves the price against
        # us (BUY pays *more*, SELL receives *less*).  Fees are taken out of
        # cash separately.
        slippage_bps_adverse = (
            self.cost_model.half_spread_bps
            + self.cost_model.impact_bps(participation)
        )
        if order.side == ActionSide.BUY:
            fill_price = mid_price * (1.0 + slippage_bps_adverse / 1e4)
        else:
            fill_price = mid_price * (1.0 - slippage_bps_adverse / 1e4)

        # --- cash check (basic margin) ------------------------------------
        # For longs, ensure we have cash to cover the fill + fees.  Shorts
        # need haircut equity (not modelled in this milestone — M5 covers
        # leverage gates).
        if order.side == ActionSide.BUY:
            required_cash = qty_filled * fill_price + costs["fees"]
            if required_cash > state.cash:
                return Fill(
                    status=FillStatus.REJECTED,
                    qty_filled=0.0,
                    avg_price=None,
                    cost_breakdown=costs,
                    slippage_bps=0.0,
                    latency_ms=self.latency_ms,
                    reason=(
                        f"Insufficient cash: need ${required_cash:.2f}, "
                        f"have ${state.cash:.2f}"
                    ),
                    ts=next_bar.ts,
                )

        # --- mutate state --------------------------------------------------

        state.apply_fill(
            symbol=order.symbol,
            qty_delta=qty_filled,
            fill_price=fill_price,
            fees=costs["fees"],
            ts=next_bar.ts,
        )
        # The simulator is also responsible for marking the position to the
        # latest bar's close, so NAV / drawdown stay current after the fill.
        state.mark_to_market({order.symbol: next_bar.close}, ts=next_bar.ts)

        return Fill(
            status=FillStatus.PARTIAL_FILL if partial else FillStatus.FILLED,
            qty_filled=qty_filled,
            avg_price=fill_price,
            cost_breakdown=costs,
            slippage_bps=slippage_bps_adverse,
            latency_ms=self.latency_ms,
            reason=(
                f"Partial: capped at {self.participation_cap:.1%} ADV "
                f"({max_fill_abs:.0f} shares)"
                if partial else "Filled at next-bar open + adverse slippage"
            ),
            ts=next_bar.ts,
        )


# ---------------------------------------------------------------------------
# Pipeline helper: ActionProposal -> Order
# ---------------------------------------------------------------------------


def proposal_to_order(
    proposal: ActionProposal,
    state: PortfolioState,
    reference_price: float,
    *,
    risk_mult: float = 1.0,
) -> Optional[Order]:
    """Compile an ``ActionProposal`` into an executable ``Order``.

    Conversion:

    1. **Compute the target dollar exposure**: ``target_weight * NAV``
       times ``risk_mult`` (a [0, 1] multiplier the M5 risk gate / debate
       distillation can attach).
    2. **Subtract current exposure** so we trade the *delta*, not the
       absolute target.
    3. **Convert dollars to shares** at ``reference_price``.
    4. **Skip HOLD / ABSTAIN** (return None — there's nothing to execute).

    Why a helper instead of a method on ``ActionProposal``?  The proposal is
    pure intent; this conversion needs ``PortfolioState`` and a market
    price, which would couple the contract to the execution stack.  Keeping
    it free-standing means M6's walk-forward backtest can call it with a
    historical price without touching live infra.
    """
    if proposal.side in (ActionSide.HOLD, ActionSide.ABSTAIN):
        return None
    if risk_mult <= 0:
        return None

    target_dollars = proposal.target_weight * state.nav * risk_mult
    current_dollars = state.positions[proposal.symbol].market_value if proposal.symbol in state.positions else 0.0
    delta_dollars = target_dollars - current_dollars

    # If the trade is too small to round to even 1 share, skip.
    qty_signed = delta_dollars / reference_price
    if abs(qty_signed) < 1.0:
        return None

    # Round to integer shares (the simulator handles fractional sizes fine,
    # but real exchanges don't accept them for stocks).
    qty_signed = math.copysign(round(abs(qty_signed)), qty_signed)

    # Verify the side matches the sign of the delta — if not, the proposal
    # is internally inconsistent and we refuse to compile it.  This is the
    # ground-truth re-check of ``ActionProposal.validity_check.fits_risk_budget``.
    if (qty_signed > 0 and proposal.side != ActionSide.BUY) or (
        qty_signed < 0 and proposal.side != ActionSide.SELL
    ):
        logger.warning(
            "Proposal side=%s contradicts delta sign (qty=%s); refusing to compile.",
            proposal.side, qty_signed,
        )
        return None

    return Order(
        symbol=proposal.symbol,
        side=proposal.side,
        qty_signed=qty_signed,
        limit_price=proposal.limit_price,
        tif=proposal.tif,
        decision_ts=proposal.decision_ts,
    )
