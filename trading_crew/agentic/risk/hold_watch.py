"""HOLD sit-out still gets ATR watch levels + as-if-entered Kelly size.

``size_pct_of_book`` stays 0 (guardrail + paper book). Stop / target /
expected return are filled as watch levels for the inferred side, not as
live orders. Recommended size is the Kelly/vol cap that would have been
used if that side were entered.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)
from trading_crew.agentic.risk.sizing import SizingConfig, compute_size
from trading_crew.agentic.risk.stops import (
    annualised_vol_from_close,
    assess_stop,
    compute_atr,
)
from trading_crew.horizons import MAX_ACTIONABLE_HORIZON_DAYS
from trading_crew.schemas import StagedTarget


def infer_watch_side(desk_biases: Optional[Mapping[str, str]] = None) -> str:
    """Majority desk lean; a split (or empty) board watches the long side."""
    bull = 0
    bear = 0
    for raw in (desk_biases or {}).values():
        side = str(raw or "").strip().lower()
        if side == "bullish":
            bull += 1
        elif side == "bearish":
            bear += 1
    if bear > bull:
        return "short"
    return "long"


def synthesize_hold_watch(
    *,
    entry: float,
    horizon_days: int,
    current_price: float,
    atr: Optional[float],
    ann_vol: Optional[float],
    watch_side: str,
):
    """Run ``assess_stop`` as BUY/SELL so HOLD can still get ATR levels."""
    action = "BUY" if watch_side == "long" else "SELL"
    return assess_stop(
        action=action,
        entry=entry,
        stop=0.0,
        target=0.0,
        horizon_days=horizon_days,
        current_price=current_price,
        atr=atr,
        ann_vol=ann_vol,
    )


def recommended_watch_size_pct(
    *,
    symbol: str,
    decision_ts: str,
    watch_side: str,
    expected_return_pct: float,
    horizon_days: int,
    conviction: float,
    realised_vol_annualised: float,
    cvar_one_day: float = 0.0,
) -> float:
    """Kelly/vol size as if the watch side were entered (percent of book)."""
    if realised_vol_annualised <= 0:
        return 0.0
    cfg = SizingConfig()
    side = ActionSide.BUY if watch_side == "long" else ActionSide.SELL
    sign = 1.0 if watch_side == "long" else -1.0
    er = float(expected_return_pct or 0.0)
    if watch_side == "short" and er > 0:
        er = -er
    elif watch_side == "long" and er < 0:
        er = abs(er)
    horizon = max(1, min(int(horizon_days or 21), MAX_ACTIONABLE_HORIZON_DAYS))
    score = max(0.0, min(1.0, float(conviction or 0.0)))
    proposal = ActionProposal(
        symbol=symbol,
        decision_ts=decision_ts,
        side=side,
        target_weight=sign * cfg.max_position_weight,
        horizon_days=horizon,
        conviction_score=score,
        conviction_tier=ConvictionTier.from_score(score),
        expected_return_pct=er,
        rationale="HOLD watch size (not an order)",
        validity_check=ValidityCheck(
            data_timestamps_valid=True,
            fits_risk_budget=True,
            survives_transaction_costs=True,
            liquidity_sufficient=True,
            notes="Advisory watch size; HOLD is not a live order.",
        ),
    )
    result = compute_size(
        proposal,
        realised_vol_annualised=realised_vol_annualised,
        cvar_one_day=max(0.0, float(cvar_one_day or 0.0)),
        config=cfg,
    )
    return round(abs(result.final_weight) * 100.0, 2)


def apply_hold_watch_plan(
    decision: Any,
    *,
    close: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    desk_biases: Optional[Mapping[str, str]] = None,
    ticker: str,
    decision_ts: str,
) -> Any:
    """Fill HOLD watch stop/target/ER + recommended size. Size stays 0."""
    if str(getattr(decision, "action", "") or "").upper() != "HOLD":
        return decision
    if len(close) < 30:
        return decision
    atr = compute_atr(list(high), list(low), list(close))
    ann_vol = annualised_vol_from_close(list(close))
    current_price = float(close[-1]) if close else 0.0
    watch_side = infer_watch_side(desk_biases)
    assessment = synthesize_hold_watch(
        entry=float(getattr(decision, "entry_price", 0.0) or 0.0),
        horizon_days=int(getattr(decision, "horizon_days", 21) or 21),
        current_price=current_price,
        atr=atr,
        ann_vol=ann_vol,
        watch_side=watch_side,
    )
    if not assessment.assessable:
        return decision

    rec = 0.0
    if ann_vol and ann_vol > 0 and assessment.expected_return_pct is not None:
        rec = recommended_watch_size_pct(
            symbol=ticker,
            decision_ts=decision_ts,
            watch_side=watch_side,
            expected_return_pct=assessment.expected_return_pct,
            horizon_days=int(getattr(decision, "horizon_days", 21) or 21),
            conviction=float(getattr(decision, "confidence", 0.0) or 0.0),
            realised_vol_annualised=ann_vol,
        )

    note = (
        f"[risk] HOLD watch levels (not an order): {watch_side} "
        f"stop {assessment.stop:.2f} / target {assessment.target:.2f}"
        + (f" / recommended size {rec:.2f}% of book" if rec > 0 else "")
        + ". size_pct_of_book stays 0."
    )
    update: Dict[str, Any] = {
        "stop_loss": assessment.stop,
        "target_price": assessment.target,
        "watch_side": watch_side,
        "recommended_size_pct": rec,
        "size_pct_of_book": 0.0,
        "rationale": f"{(getattr(decision, 'rationale', '') or '').strip()}\n{note}".strip(),
    }
    if assessment.expected_return_pct is not None:
        update["expected_return_pct"] = assessment.expected_return_pct
    if assessment.staged_targets:
        update["staged_targets"] = [StagedTarget(**t) for t in assessment.staged_targets]
    return decision.model_copy(update=update)
