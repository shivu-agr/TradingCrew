"""Backtest PnL attribution — decision vs sizing vs cost (Phase 3).

A backtest that only reports a Sharpe answers *whether* the strategy made
money, not *why* it lost.  When the desk sees a bad backtest it needs to
know which layer to fix:

- **Decision** — was the directional call itself wrong?  (attributable to
  the analysts / researchers / PM — the LLM decision policy)
- **Sizing** — was the call right but the position mis-sized by the M5
  risk gate?  (attributable to L3 sizing config / risk desks)
- **Cost** — did fees + spread eat an otherwise-fine edge?  (attributable
  to execution / horizon choice)

We decompose each decision's realised PnL against a *unit reference
exposure* so the three legs sum back to the net:

    gross_pnl  = sized_weight        * asset_return          (what actually happened)
    decision   = direction * ref_w   * asset_return          (a unit-sized directional bet)
    sizing     = gross_pnl - decision                         (deviation from the unit bet)
    cost       = -|sized_weight| * round_trip_cost_fraction   (from the real CostModel)
    net_pnl    = gross_pnl + cost

Deterministic + dependency-light (numpy/pandas only via the shared
helpers).  No LLM, no I/O.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from trading_crew.agentic.execution.contracts import ActionProposal
from trading_crew.agentic.execution.cost import get_cost_model

from .engine import _normalise_ohlcv, _to_naive_utc


_LONG = 1.0
_SHORT = -1.0


def _direction(side: str) -> float:
    s = (side or "").upper()
    if s == "BUY":
        return _LONG
    if s == "SELL":
        return _SHORT
    return 0.0


@dataclass
class DecisionAttribution:
    symbol: str
    decision_ts: str
    side: str
    direction: float
    asset_return: float
    ref_weight: float
    sized_weight: float
    decision_pnl: float
    sizing_pnl: float
    cost_pnl: float
    gross_pnl: float
    net_pnl: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "decision_ts": self.decision_ts,
            "side": self.side,
            "direction": self.direction,
            "asset_return": self.asset_return,
            "ref_weight": self.ref_weight,
            "sized_weight": self.sized_weight,
            "decision_pnl": self.decision_pnl,
            "sizing_pnl": self.sizing_pnl,
            "cost_pnl": self.cost_pnl,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
        }


@dataclass
class AttributionResult:
    decisions: List[DecisionAttribution] = field(default_factory=list)
    total_decision_pnl: float = 0.0
    total_sizing_pnl: float = 0.0
    total_cost_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    n_decisions: int = 0
    n_directional: int = 0
    hit_rate: Optional[float] = None
    ref_weight: float = 0.0
    diagnosis: str = ""

    def to_dict(self) -> dict:
        return {
            "total_decision_pnl": self.total_decision_pnl,
            "total_sizing_pnl": self.total_sizing_pnl,
            "total_cost_pnl": self.total_cost_pnl,
            "total_gross_pnl": self.total_gross_pnl,
            "total_net_pnl": self.total_net_pnl,
            "n_decisions": self.n_decisions,
            "n_directional": self.n_directional,
            "hit_rate": self.hit_rate,
            "ref_weight": self.ref_weight,
            "diagnosis": self.diagnosis,
            "decisions": [d.to_dict() for d in self.decisions],
        }


def sized_weights_from_trades(trades: Sequence) -> Dict[tuple, float]:
    """Build a ``{(symbol, normalized_ts): sized_weight}`` map from trades.

    Only FILLED trades contribute an executed exposure; rejected ones map
    to zero (so the decision is scored as a flat outcome, which is exactly
    what the risk gate produced).
    """
    out: Dict[tuple, float] = {}
    for t in trades:
        try:
            key = (t.symbol.upper(), _to_naive_utc(t.ts))
        except Exception:  # noqa: BLE001
            continue
        status = str(getattr(t, "status", "")).upper()
        out[key] = float(getattr(t, "sized_weight", 0.0)) if "FILL" in status or status == "FILLED" else 0.0
    return out


def _asset_return(df: pd.DataFrame, decision_ts: str, horizon_days: int, date_col: str) -> Optional[float]:
    """Simple return from the decision-day close to the horizon-day close."""
    dts = df[date_col]
    d0 = _to_naive_utc(decision_ts)
    # First bar on/after the decision date.
    start_idx = dts.searchsorted(d0, side="left")
    if start_idx >= len(df):
        return None
    entry_close = float(df.iloc[start_idx]["Close"])
    target_date = d0 + pd.Timedelta(days=max(1, horizon_days))
    end_idx = dts.searchsorted(target_date, side="right") - 1
    end_idx = min(max(end_idx, start_idx), len(df) - 1)
    if end_idx <= start_idx:
        return None
    exit_close = float(df.iloc[end_idx]["Close"])
    if entry_close <= 0:
        return None
    return exit_close / entry_close - 1.0


def attribute_decisions(
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    *,
    sized_weights: Optional[Dict[tuple, float]] = None,
    cost_model_name: str = "standard",
    ref_weight: Optional[float] = None,
) -> AttributionResult:
    """Decompose realised PnL into decision / sizing / cost legs.

    ``sized_weights`` maps ``(symbol, normalized_ts)`` to the *executed*
    signed weight (e.g. from :func:`sized_weights_from_trades`).  When a
    decision has no entry there, the proposal's ``target_weight`` is used
    (i.e. no risk-gate adjustment is assumed).
    """
    cost_model = get_cost_model(cost_model_name)
    # Per-leg cost fraction of notional (fees + half-spread); ×2 for a round
    # trip.  Impact is excluded here — it needs participation/ADV which is
    # not part of a decision-level attribution.
    per_leg_cost = (cost_model.fee_bps + cost_model.half_spread_bps) / 1e4
    round_trip_cost = 2.0 * per_leg_cost

    normalised: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv_by_symbol.items():
        try:
            normalised[sym.upper()] = _normalise_ohlcv(df)
        except Exception:  # noqa: BLE001
            continue

    # Reference exposure = median |executed weight| over directional bets, so
    # the sizing leg isolates deviation from a "typical" position.
    exposures: List[float] = []
    for p in proposals:
        d = _direction(p.side.value if hasattr(p.side, "value") else str(p.side))
        if d == 0.0:
            continue
        key = (p.symbol.upper(), _to_naive_utc(p.decision_ts))
        e = sized_weights.get(key) if sized_weights is not None else None
        if e is None:
            e = float(p.target_weight)
        exposures.append(abs(e))
    if ref_weight is None:
        ref_weight = statistics.median(exposures) if exposures else 0.0
        if ref_weight <= 0:
            ref_weight = (sum(exposures) / len(exposures)) if exposures else 0.1

    out = AttributionResult(ref_weight=ref_weight)
    wins = 0
    for p in proposals:
        side = p.side.value if hasattr(p.side, "value") else str(p.side)
        d = _direction(side)
        df = normalised.get(p.symbol.upper())
        date_col = None
        if df is not None:
            date_col = "Date" if "Date" in df.columns else "date"
        r = _asset_return(df, p.decision_ts, int(p.horizon_days or 21), date_col) if df is not None else None
        if r is None:
            continue

        key = (p.symbol.upper(), _to_naive_utc(p.decision_ts))
        sized = sized_weights.get(key) if sized_weights is not None else None
        if sized is None:
            sized = float(p.target_weight)

        decision_pnl = d * ref_weight * r
        gross_pnl = sized * r
        sizing_pnl = gross_pnl - decision_pnl
        cost_pnl = -abs(sized) * round_trip_cost
        net_pnl = gross_pnl + cost_pnl

        out.decisions.append(DecisionAttribution(
            symbol=p.symbol.upper(), decision_ts=p.decision_ts, side=side,
            direction=d, asset_return=r, ref_weight=ref_weight, sized_weight=sized,
            decision_pnl=decision_pnl, sizing_pnl=sizing_pnl, cost_pnl=cost_pnl,
            gross_pnl=gross_pnl, net_pnl=net_pnl,
        ))
        out.total_decision_pnl += decision_pnl
        out.total_sizing_pnl += sizing_pnl
        out.total_cost_pnl += cost_pnl
        out.total_gross_pnl += gross_pnl
        out.total_net_pnl += net_pnl
        if d != 0.0:
            out.n_directional += 1
            if d * r > 0:
                wins += 1

    out.n_decisions = len(out.decisions)
    out.hit_rate = (wins / out.n_directional) if out.n_directional else None
    out.diagnosis = _diagnose(out)
    return out


def _diagnose(res: AttributionResult) -> str:
    """One-line plain-English attribution of where PnL came from / went."""
    if res.n_decisions == 0:
        return "No resolvable decisions to attribute."
    legs = {
        "directional calls": res.total_decision_pnl,
        "position sizing": res.total_sizing_pnl,
        "costs": res.total_cost_pnl,
    }
    # The leg with the largest absolute contribution dominates the story.
    driver, val = max(legs.items(), key=lambda kv: abs(kv[1]))
    sign = "added to" if val > 0 else "subtracted from"
    hit = f" Hit rate {res.hit_rate:.0%}." if res.hit_rate is not None else ""
    if res.total_decision_pnl < 0 and abs(res.total_decision_pnl) >= abs(res.total_sizing_pnl):
        head = "Losses are dominated by WRONG DIRECTIONAL CALLS — fix the decision policy (analysts/PM)."
    elif res.total_sizing_pnl < 0 and abs(res.total_sizing_pnl) > abs(res.total_decision_pnl):
        head = "Direction was net-OK but SIZING hurt — revisit the M5 risk gate / L3 sizing config."
    elif res.total_cost_pnl < 0 and abs(res.total_cost_pnl) >= abs(res.total_gross_pnl):
        head = "Edge exists gross but COSTS eat it — trade less / longer horizon / cheaper venue."
    else:
        head = "Net-positive edge."
    return f"{head} Largest leg: {driver} {sign} net PnL ({val:+.4f}).{hit}"
