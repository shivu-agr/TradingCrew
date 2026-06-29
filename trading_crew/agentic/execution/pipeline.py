"""End-to-end pipeline: ActionProposal -> Fill -> updated PortfolioState.

This module is the deterministic continuation of a TradingAgents run.  After
the agent graph emits a typed ``ActionProposal`` (M1 / Action Compiler), the
pipeline:

1. Loads the next available OHLCV bar after ``decision_ts`` from the shared
   cache (point-in-time safe — we already enforced that in M0's load_ohlcv
   refactor).
2. Compiles the proposal into an ``Order`` via ``proposal_to_order``.
3. Hands the order to ``ExecutionSimulator`` against the chosen cost model.
4. Mutates the persistent ``PortfolioState`` and returns a diagnostic
   ``PipelineResult`` for the UI / audit log.

The pipeline is deliberately not part of the LangGraph workflow — it's
*after* the agent loop completes.  Putting it in the graph would tempt
future contributors to make sizing/cost decisions LLM-driven, which the
paper §9.2 explicitly warns against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from trading_crew.agentic.execution.contracts import ActionProposal
from trading_crew.agentic.execution.cost import CostModel, cost_sweep, get_cost_model, CostScenario
from trading_crew.agentic.execution.simulator import (
    Bar,
    ExecutionSimulator,
    Fill,
    FillStatus,
    Order,
    proposal_to_order,
)
from trading_crew.agentic.portfolio.state import (
    PortfolioState,
    load_portfolio_state,
    save_portfolio_state,
)
from trading_crew.agentic.risk import (
    GateConfig,
    GateResult,
    SizingConfig,
    SizingResult,
    compute_historical_var,
    compute_size,
    run_risk_gates,
    VarConfig,
)
from trading_crew.agentic.risk.sizing import debate_to_risk_mult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Everything the UI / audit log needs after a single run.

    - ``order``: compiled ``Order`` (None when the proposal was HOLD/ABSTAIN
      or rounded to < 1 share).
    - ``fill``: simulator result (None when no order was compiled).
    - ``state_after``: snapshot of the portfolio after the fill.
    - ``cost_scenarios``: low/standard/high sensitivity sweep on the trade
      that was *proposed* (not necessarily executed) — useful for the UI to
      flag fragile-edge trades even when the executor passes.
    - ``rejected``: True if the simulator returned a non-FILLED status.
    - ``sizing``: M5 sizing decomposition (Kelly / vol / CVaR caps).
    - ``risk_gate``: M5 gate verdict (concentration / leverage / drawdown).
    """

    proposal: ActionProposal
    order: Optional[Order]
    fill: Optional[Fill]
    state_after: dict
    cost_scenarios: list = field(default_factory=list)
    rejected: bool = False
    note: str = ""
    sizing: Optional[SizingResult] = None
    risk_gate: Optional[GateResult] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_bar_after(df: pd.DataFrame, decision_ts: str) -> Optional[Tuple[pd.Timestamp, pd.Series]]:
    """Return (timestamp, row) of the first bar strictly after ``decision_ts``.

    The OHLCV DataFrame has a Date column (str or datetime) and the
    standard ``Open/High/Low/Close/Volume`` columns.  We strip the time
    component of ``decision_ts`` (we don't have intraday data) and pick
    the next available trading day.

    Returns ``None`` when the cache doesn't have any bar after the
    decision date — typical when the decision is on or near "today" and
    no future data exists yet.
    """
    if df is None or df.empty:
        return None
    # Determine the date column name
    date_col = None
    for c in ("Date", "date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None and df.index.dtype == "object":
        # CSV cache uses Date as a column, but some flows use it as index.
        df = df.reset_index().rename(columns={"index": "Date"})
        date_col = "Date"
    if date_col is None:
        return None

    decision_date = pd.to_datetime(decision_ts).normalize()
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    mask = dates > decision_date
    if not mask.any():
        return None
    idx = mask.idxmax()
    row = df.loc[idx]
    return pd.to_datetime(row[date_col]), row


def _row_to_bar(date: pd.Timestamp, row: pd.Series, adv: float) -> Bar:
    """Convert a DataFrame row into a ``Bar`` dataclass."""
    return Bar(
        ts=date.isoformat(),
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
        adv=float(adv),
    )


def _compute_adv(df: pd.DataFrame, lookback: int = 20) -> float:
    """Average daily volume over the last ``lookback`` bars.

    Used by the simulator to compute participation for the impact term.
    Falls back to the current bar's volume if the DataFrame is shorter
    than the lookback window — this is intentional and explicit, not a
    silent fallback (a 5-bar history with 5-day ADV is the best we can do).
    """
    if df is None or df.empty:
        raise ValueError("Cannot compute ADV from empty DataFrame")
    if "Volume" not in df.columns:
        raise ValueError("DataFrame missing Volume column")
    return float(df["Volume"].tail(lookback).mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    proposal: ActionProposal,
    *,
    ohlcv: pd.DataFrame,
    portfolio_id: str = "default",
    starting_cash: float = 100_000.0,
    cost_model_name: str = "standard",
    participation_cap: float = 0.05,
    risk_mult: float = 1.0,
    risk_debate_state: Optional[dict] = None,
    sizing_config: Optional[SizingConfig] = None,
    gate_config: Optional[GateConfig] = None,
    persist: bool = True,
) -> PipelineResult:
    """Execute one proposal end-to-end and (optionally) persist the new book.

    Returns a ``PipelineResult`` regardless of success — the caller can
    distinguish FILLED / PARTIAL_FILL / REJECTED via ``result.fill.status``.

    When the proposal is HOLD/ABSTAIN, no order is compiled and
    ``rejected=False`` (HOLD is a successful "do nothing" outcome — see
    paper §4.1 on Layer A state).
    """
    state = load_portfolio_state(portfolio_id, starting_cash=starting_cash)

    # Pick the bar we'd fill on.
    bar_info = _next_bar_after(ohlcv, proposal.decision_ts)
    if bar_info is None:
        return PipelineResult(
            proposal=proposal,
            order=None,
            fill=None,
            state_after=state.to_snapshot(),
            rejected=False,
            note=(
                "No bar available after decision_ts — typical for runs on "
                "today/future dates. Simulator cannot fill."
            ),
        )
    bar_date, bar_row = bar_info
    adv = _compute_adv(ohlcv)
    bar = _row_to_bar(bar_date, bar_row, adv)

    # Reference price for the sizer: use the *previous* close so we don't
    # implicitly use bar.open (which would be future info from the agent's
    # POV).  The simulator separately re-fetches the open for the actual
    # fill price.
    reference_price = float(ohlcv["Close"].iloc[-1])
    if pd.to_datetime(ohlcv.iloc[-1].get("Date", proposal.decision_ts)) > pd.to_datetime(proposal.decision_ts):
        # In case the cache extends beyond decision_ts, find the close on
        # the decision date itself.
        date_col = "Date" if "Date" in ohlcv.columns else "date"
        decision_date = pd.to_datetime(proposal.decision_ts).normalize()
        dates = pd.to_datetime(ohlcv[date_col]).dt.normalize()
        decision_rows = ohlcv[dates == decision_date]
        if not decision_rows.empty:
            reference_price = float(decision_rows.iloc[0]["Close"])

    # Cost-sensitivity sweep on the *intended* notional.  This shows the user
    # how the trade survives different cost regimes even before we fire.
    intended_notional = abs(proposal.target_weight) * state.nav
    participation = (intended_notional / reference_price) / max(adv, 1.0)
    scenarios = cost_sweep(intended_notional, participation)

    # ---- M5: deterministic sizing + risk gates ------------------------
    sizing_config = sizing_config or SizingConfig()
    gate_config = gate_config or GateConfig()

    # Compute realised vol + 1-day historical VaR/CVaR from the same OHLCV.
    realised_vol_annual = 0.0
    cvar_one_day = 0.0
    try:
        import math
        closes = ohlcv["Close"].astype(float).tolist()
        log_returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if len(log_returns) >= 30:
            var_cfg = VarConfig(window=min(len(log_returns), sizing_config.vol_lookback_days), confidence=0.95)
            historical = compute_historical_var(log_returns, var_cfg)
            cvar_one_day = historical.cvar
            mean = sum(log_returns[-var_cfg.window:]) / var_cfg.window
            var_term = sum((r - mean) ** 2 for r in log_returns[-var_cfg.window:]) / max(1, var_cfg.window - 1)
            realised_vol_annual = math.sqrt(var_term) * math.sqrt(252)
    except Exception:
        logger.exception("M5: failed to compute realised vol / VaR")

    # Re-derive risk_mult from the debate if caller didn't override.
    if risk_debate_state is not None:
        rm, _ = debate_to_risk_mult(
            risk_debate_state,
            proposal=proposal,
            floor=sizing_config.risk_mult_floor,
            ceiling=sizing_config.risk_mult_ceiling,
        )
        if rm < risk_mult:
            risk_mult = rm  # debate can only reduce size, never increase

    sizing = compute_size(
        proposal,
        realised_vol_annualised=realised_vol_annual or 0.20,
        cvar_one_day=cvar_one_day or 0.02,
        risk_mult=risk_mult,
        config=sizing_config,
    )

    # If sizing collapsed to 0, return early.
    if abs(sizing.final_weight) < 1e-6:
        return PipelineResult(
            proposal=proposal,
            order=None,
            fill=None,
            state_after=state.to_snapshot(),
            cost_scenarios=scenarios,
            rejected=False,
            note=f"Sizing collapsed weight to 0 (binding={sizing.binding_constraint}).",
            sizing=sizing,
        )

    # Build a sized version of the proposal for the gate + simulator.
    sized_proposal = proposal.model_copy(update={"target_weight": sizing.final_weight})

    # Hard risk gates — refuse fills that breach concentration / leverage / DD.
    last_bar_ts = bar.ts
    est_fees = abs(sizing.final_weight * state.nav) * 0.0001  # ~1 bps fees ballpark
    gate_result = run_risk_gates(
        sized_proposal, state,
        cvar_one_day=cvar_one_day or 0.02,
        reference_price=reference_price,
        est_fees=est_fees,
        last_bar_ts=last_bar_ts,
        config=gate_config,
    )
    if not gate_result.passed:
        return PipelineResult(
            proposal=proposal,
            order=None,
            fill=None,
            state_after=state.to_snapshot(),
            cost_scenarios=scenarios,
            rejected=True,
            note="Risk-gate rejection: " + "; ".join(f"{n}: {r}" for n, r in gate_result.failures),
            sizing=sizing,
            risk_gate=gate_result,
        )

    order = proposal_to_order(sized_proposal, state, reference_price, risk_mult=1.0)
    if order is None:
        return PipelineResult(
            proposal=proposal,
            order=None,
            fill=None,
            state_after=state.to_snapshot(),
            cost_scenarios=scenarios,
            rejected=False,
            note=(
                "No order compiled — sized weight too close to current position "
                "(delta < 1 share)."
            ),
            sizing=sizing,
            risk_gate=gate_result,
        )

    cost_model = get_cost_model(cost_model_name)
    sim = ExecutionSimulator(
        cost_model=cost_model,
        participation_cap=participation_cap,
    )
    fill = sim.execute(order, bar, state)

    if persist:
        try:
            save_portfolio_state(state)
        except OSError as exc:
            logger.warning("Could not persist post-fill portfolio state: %s", exc)

    return PipelineResult(
        proposal=proposal,
        order=order,
        fill=fill,
        state_after=state.to_snapshot(),
        cost_scenarios=scenarios,
        rejected=(fill.status != FillStatus.FILLED and fill.status != FillStatus.PARTIAL_FILL),
        note=fill.reason,
        sizing=sizing,
        risk_gate=gate_result,
    )
