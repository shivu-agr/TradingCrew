"""Phase 3 — backtest PnL attribution (decision vs sizing vs cost)."""

from __future__ import annotations

import pandas as pd

from trading_crew.agentic.backtest import attribute_decisions, sized_weights_from_trades
from trading_crew.agentic.backtest.engine import TradeRecord
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


def _proposal(symbol, ts, weight, horizon=10):
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol=symbol, decision_ts=ts, side=side, target_weight=weight,
        horizon_days=horizon, conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.02 if weight else 0.0,
        rationale="Attribution test proposal with sufficient text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _uptrend_ohlcv(start="2024-01-02", n=60, step=0.01):
    dates = pd.date_range(start, periods=n, freq="B")
    closes = [100.0 * (1 + step) ** i for i in range(n)]
    return pd.DataFrame({
        "Date": dates, "Open": closes, "High": [c * 1.001 for c in closes],
        "Low": [c * 0.999 for c in closes], "Close": closes, "Volume": [1_000_000] * n,
    })


def test_correct_long_call_credits_decision_leg():
    df = _uptrend_ohlcv()
    ts = df["Date"].iloc[5].strftime("%Y-%m-%dT16:00:00+00:00")
    props = [_proposal("AAA", ts, weight=0.1, horizon=10)]
    res = attribute_decisions(props, {"AAA": df}, cost_model_name="standard")
    assert res.n_decisions == 1
    d = res.decisions[0]
    # Uptrend + long call => positive asset return and positive decision PnL.
    assert d.asset_return > 0
    assert d.decision_pnl > 0
    assert d.cost_pnl < 0  # costs are always a drag
    assert res.hit_rate == 1.0


def test_wrong_short_call_shows_negative_decision_leg():
    df = _uptrend_ohlcv()
    ts = df["Date"].iloc[5].strftime("%Y-%m-%dT16:00:00+00:00")
    props = [_proposal("AAA", ts, weight=-0.1, horizon=10)]
    res = attribute_decisions(props, {"AAA": df}, cost_model_name="standard")
    d = res.decisions[0]
    # Shorting into a rally: direction is negative, asset return positive => loss.
    assert d.decision_pnl < 0
    assert "DIRECTIONAL" in res.diagnosis.upper() or res.total_decision_pnl < 0


def test_oversizing_isolates_sizing_leg():
    df = _uptrend_ohlcv()
    ts1 = df["Date"].iloc[5].strftime("%Y-%m-%dT16:00:00+00:00")
    ts2 = df["Date"].iloc[6].strftime("%Y-%m-%dT16:00:00+00:00")
    # Two identical-direction calls; ref_weight = median(|exposure|).
    props = [_proposal("AAA", ts1, weight=0.1), _proposal("AAA", ts2, weight=0.5)]
    res = attribute_decisions(props, {"AAA": df})
    # The 0.5 position deviates above the median (0.3) so its sizing leg is
    # positive in an uptrend; the 0.1 position sizes below => negative sizing.
    big = next(d for d in res.decisions if d.sized_weight == 0.5)
    small = next(d for d in res.decisions if d.sized_weight == 0.1)
    assert big.sizing_pnl > 0
    assert small.sizing_pnl < 0
    # gross = decision + sizing (identity holds per decision).
    for d in res.decisions:
        assert abs(d.gross_pnl - (d.decision_pnl + d.sizing_pnl)) < 1e-9


def test_sized_weights_from_trades_uses_executed_exposure():
    df = _uptrend_ohlcv()
    ts = df["Date"].iloc[5].strftime("%Y-%m-%dT16:00:00+00:00")
    props = [_proposal("AAA", ts, weight=0.4)]
    # The risk gate cut the position to 0.05 at fill.
    trades = [TradeRecord(
        fold_id=0, ts=ts, symbol="AAA", side="BUY", intent_weight=0.4,
        sized_weight=0.05, fill_qty=10, fill_price=100.0, fees=1.0, status="FILLED",
    )]
    sw = sized_weights_from_trades(trades)
    res = attribute_decisions(props, {"AAA": df}, sized_weights=sw)
    d = res.decisions[0]
    assert d.sized_weight == 0.05  # executed, not the 0.4 intent
    # Downsized well below intent => gross PnL smaller than an un-gated bet.
    assert d.gross_pnl == 0.05 * d.asset_return
