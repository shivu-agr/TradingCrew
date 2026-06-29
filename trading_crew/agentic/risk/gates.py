"""Pre-trade hard risk gates (paper §7.3).

Each gate is a deterministic predicate over (proposal, portfolio_state,
risk_metrics).  The gate returns ``passed=False`` with a diagnostic
reason; ``run_risk_gates`` aggregates results so the caller sees *all*
failures, not just the first.

Gates implemented:

1. **Concentration**     — |target_weight| <= max_position_weight
2. **Leverage**          — gross_exposure / NAV <= max_leverage post-fill
3. **Drawdown stop**     — max_drawdown < drawdown_kill_threshold
                           (the kill-switch — paper §7.3 emergency stop)
4. **Single-position CVaR** — proposed_notional * cvar_one_day <= max_cvar_dollars
5. **Stale data**        — last bar's date must be within ``max_stale_days``
                           of decision_ts (no trading on month-old data)
6. **Cash sufficiency**  — long buy must have cash >= notional + estimated fees

The kill-switch (gate #3) is a *blocking* condition: when triggered, the
entire risk gate returns ``passed=False`` regardless of whether other
gates pass.  This is the explicit "regime stop-loss" the paper §7.3
recommends — once portfolio drawdown crosses the threshold, no new
positions can be opened until manual reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from trading_crew.agentic.execution.contracts import ActionProposal, ActionSide
from trading_crew.agentic.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Hard limits applied to every proposal before it can be executed."""

    max_position_weight: float = 0.20
    max_leverage: float = 1.5
    drawdown_kill_threshold: float = 0.20  # 20% peak-to-trough triggers kill-switch
    max_position_cvar_pct: float = 0.02   # 2% of NAV as single-position 1d CVaR
    max_stale_days: int = 5
    min_cash_buffer_pct: float = 0.02     # keep 2% NAV in cash always


@dataclass
class GateResult:
    """Aggregate verdict of all gates.

    ``failures`` is a list of (gate_name, reason) tuples so the UI can
    show every breach, not just the first.  Empty failures + passed=True
    means the trade may proceed.
    """

    passed: bool
    failures: List[tuple[str, str]] = field(default_factory=list)
    kill_switch_triggered: bool = False


class RiskGate:
    """Single-shot gate evaluator, configurable per portfolio.

    Methods are short and side-effect-free so they're easy to test
    individually.  ``run`` aggregates them into a ``GateResult``.
    """

    def __init__(self, config: GateConfig = GateConfig()) -> None:
        self.config = config

    # -- individual gates ----------------------------------------------

    def check_concentration(self, proposal: ActionProposal) -> Optional[str]:
        if abs(proposal.target_weight) > self.config.max_position_weight:
            return (
                f"|target_weight|={abs(proposal.target_weight):.3f} exceeds "
                f"max_position_weight={self.config.max_position_weight:.3f}"
            )
        return None

    def check_drawdown(self, state: PortfolioState) -> Optional[str]:
        if state.max_drawdown >= self.config.drawdown_kill_threshold:
            return (
                f"Kill-switch — max_drawdown={state.max_drawdown:.3f} >= "
                f"threshold={self.config.drawdown_kill_threshold:.3f}"
            )
        return None

    def check_leverage(self, state: PortfolioState, proposal: ActionProposal) -> Optional[str]:
        """Projected gross leverage post-fill.  We approximate the new gross
        exposure by adding the proposed delta to current gross — accurate
        for the directionally-consistent case (same-sign add); an
        approximation when crossing sides, but conservative."""
        if state.nav == 0:
            return None
        existing_pos = state.positions.get(proposal.symbol)
        current_weight = state.weight(proposal.symbol) if existing_pos else 0.0
        delta = proposal.target_weight - current_weight
        projected_gross = state.gross_exposure + abs(delta * state.nav)
        projected_leverage = projected_gross / state.nav
        if projected_leverage > self.config.max_leverage:
            return (
                f"Projected gross leverage={projected_leverage:.3f} exceeds "
                f"max_leverage={self.config.max_leverage:.3f}"
            )
        return None

    def check_position_cvar(self, proposal: ActionProposal, state: PortfolioState, cvar_one_day: float) -> Optional[str]:
        """Single-position 1-day CVaR vs NAV cap."""
        if state.nav == 0:
            return None
        position_cvar = abs(proposal.target_weight) * cvar_one_day * state.nav
        cap = self.config.max_position_cvar_pct * state.nav
        if position_cvar > cap:
            return (
                f"Position CVaR=${position_cvar:.2f} exceeds cap=${cap:.2f} "
                f"({self.config.max_position_cvar_pct:.1%} of NAV)"
            )
        return None

    def check_stale_data(self, decision_ts: str, last_bar_ts: str) -> Optional[str]:
        """Reject if the latest market data is more than max_stale_days old."""
        from datetime import datetime, timezone

        def _parse(ts: str) -> datetime:
            cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        decision = _parse(decision_ts)
        last_bar = _parse(last_bar_ts)
        delta_days = (decision - last_bar).total_seconds() / 86400.0
        if delta_days > self.config.max_stale_days:
            return (
                f"Latest market data is {delta_days:.1f} days old "
                f"(> {self.config.max_stale_days})"
            )
        return None

    def check_cash_buffer(self, proposal: ActionProposal, state: PortfolioState, reference_price: float, est_fees: float) -> Optional[str]:
        """For longs, ensure post-trade cash > min_cash_buffer_pct * NAV."""
        if proposal.side != ActionSide.BUY:
            return None
        existing_pos = state.positions.get(proposal.symbol)
        current_weight = state.weight(proposal.symbol) if existing_pos else 0.0
        delta_weight = proposal.target_weight - current_weight
        if delta_weight <= 0:
            return None  # not actually a net buy
        delta_dollars = delta_weight * state.nav
        cash_after = state.cash - delta_dollars - est_fees
        min_cash = self.config.min_cash_buffer_pct * state.nav
        if cash_after < min_cash:
            return (
                f"Insufficient cash post-fill: ${cash_after:.2f} < "
                f"${min_cash:.2f} ({self.config.min_cash_buffer_pct:.1%} of NAV)"
            )
        return None

    # -- aggregate ---------------------------------------------------

    def run(
        self,
        proposal: ActionProposal,
        state: PortfolioState,
        *,
        cvar_one_day: float,
        reference_price: float,
        est_fees: float = 0.0,
        last_bar_ts: Optional[str] = None,
    ) -> GateResult:
        """Run every gate and aggregate the verdict.

        Order matters only insofar as the kill-switch sets the
        ``kill_switch_triggered`` flag — it doesn't short-circuit the
        other checks because we want users to see *all* failures.
        """
        failures: List[tuple[str, str]] = []
        kill = False

        # HOLD/ABSTAIN bypasses everything (nothing to size or execute)
        if proposal.side in (ActionSide.HOLD, ActionSide.ABSTAIN):
            return GateResult(passed=True, failures=[], kill_switch_triggered=False)

        msg = self.check_concentration(proposal)
        if msg:
            failures.append(("concentration", msg))

        msg = self.check_drawdown(state)
        if msg:
            failures.append(("drawdown_kill_switch", msg))
            kill = True

        msg = self.check_leverage(state, proposal)
        if msg:
            failures.append(("leverage", msg))

        msg = self.check_position_cvar(proposal, state, cvar_one_day)
        if msg:
            failures.append(("position_cvar", msg))

        msg = self.check_cash_buffer(proposal, state, reference_price, est_fees)
        if msg:
            failures.append(("cash_buffer", msg))

        if last_bar_ts:
            msg = self.check_stale_data(proposal.decision_ts, last_bar_ts)
            if msg:
                failures.append(("stale_data", msg))

        return GateResult(
            passed=(not failures),
            failures=failures,
            kill_switch_triggered=kill,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def run_risk_gates(
    proposal: ActionProposal,
    state: PortfolioState,
    *,
    cvar_one_day: float,
    reference_price: float,
    est_fees: float = 0.0,
    last_bar_ts: Optional[str] = None,
    config: GateConfig = GateConfig(),
) -> GateResult:
    """Functional alias of ``RiskGate(config).run(...)``."""
    return RiskGate(config).run(
        proposal, state,
        cvar_one_day=cvar_one_day,
        reference_price=reference_price,
        est_fees=est_fees,
        last_bar_ts=last_bar_ts,
    )
