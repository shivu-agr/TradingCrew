"""M2 — ExecutionSimulator: fill semantics, failure modes, cost application."""

from __future__ import annotations

import pytest

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    OrderTimeInForce,
    ValidityCheck,
)
from trading_crew.agentic.execution.cost import CostModel
from trading_crew.agentic.execution.simulator import (
    Bar,
    ExecutionSimulator,
    Fill,
    FillStatus,
    Order,
    proposal_to_order,
)
from trading_crew.agentic.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _zero_cost_model() -> CostModel:
    """A cost model with no fees / spread / impact — for testing pure fill mechanics."""
    return CostModel(fee_bps=0.0, flat_fee=0.0, half_spread_bps=0.0, impact_k=0.0)


def _realistic_cost_model() -> CostModel:
    return CostModel(fee_bps=1.0, flat_fee=0.0, half_spread_bps=2.5, impact_k=10.0)


def _fresh_state(starting=1_000_000.0) -> PortfolioState:
    return PortfolioState(
        portfolio_id="t", base_currency="USD",
        starting_cash=starting, cash=starting, peak_nav=starting,
    )


def _bar(open_=100.0, high=105.0, low=98.0, close=102.0, volume=100_000.0, adv=100_000.0, ts="2026-01-15T15:00:00+00:00") -> Bar:
    return Bar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume, adv=adv)


def _market_buy(qty=100, symbol="AAPL") -> Order:
    return Order(
        symbol=symbol, side=ActionSide.BUY, qty_signed=float(qty),
        limit_price=None, tif=OrderTimeInForce.DAY,
        decision_ts="2026-01-14T20:00:00+00:00",
    )


def _market_sell(qty=100, symbol="AAPL") -> Order:
    return Order(
        symbol=symbol, side=ActionSide.SELL, qty_signed=-float(qty),
        limit_price=None, tif=OrderTimeInForce.DAY,
        decision_ts="2026-01-14T20:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_market_buy_fills_at_next_bar_open_with_no_cost_model():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    fill = sim.execute(_market_buy(100), _bar(open_=100.0), state)

    assert fill.status == FillStatus.FILLED
    assert fill.qty_filled == 100
    # No spread / impact / fees -> fills exactly at bar open
    assert fill.avg_price == pytest.approx(100.0)
    assert fill.cost_breakdown["total"] == 0.0
    # State updated
    assert state.positions["AAPL"].qty == 100
    assert state.positions["AAPL"].avg_cost == pytest.approx(100.0)


def test_market_buy_applies_adverse_slippage_under_realistic_costs():
    sim = ExecutionSimulator(_realistic_cost_model())
    state = _fresh_state()
    fill = sim.execute(_market_buy(100), _bar(open_=100.0, adv=100_000.0), state)

    assert fill.status == FillStatus.FILLED
    # participation = 100/100,000 = 0.001 → impact = 10*sqrt(0.001)≈0.316bps
    # half_spread = 2.5 bps, so adverse slippage ≈ 2.816 bps
    # buy fills ABOVE bar open
    assert fill.avg_price > 100.0
    expected_slippage = 2.5 + 10.0 * (0.001 ** 0.5)
    assert fill.slippage_bps == pytest.approx(expected_slippage, abs=0.01)
    assert fill.cost_breakdown["fees"] > 0


def test_market_sell_applies_adverse_slippage_in_opposite_direction():
    sim = ExecutionSimulator(_realistic_cost_model())
    state = _fresh_state()
    # Need a position first
    state.apply_fill("AAPL", 100, 100.0, fees=0.0, ts="t0")

    fill = sim.execute(_market_sell(100), _bar(open_=100.0, adv=100_000.0), state)
    assert fill.status == FillStatus.FILLED
    # sell fills BELOW bar open
    assert fill.avg_price < 100.0


# ---------------------------------------------------------------------------
# Partial fill (participation cap)
# ---------------------------------------------------------------------------


def test_oversize_order_truncates_to_participation_cap():
    sim = ExecutionSimulator(_zero_cost_model(), participation_cap=0.05)
    state = _fresh_state()
    # ADV=10_000, cap 5% = 500 shares.  Ask for 2000.
    fill = sim.execute(
        Order("AAPL", ActionSide.BUY, 2000.0, None, OrderTimeInForce.DAY, "t0"),
        _bar(adv=10_000.0),
        state,
    )
    assert fill.status == FillStatus.PARTIAL_FILL
    assert fill.qty_filled == 500
    assert "Partial" in fill.reason


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


def test_buy_limit_rejected_when_bar_doesnt_cross():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    # Limit at 95, bar low is 98 -> never touched
    order = Order("AAPL", ActionSide.BUY, 100.0, limit_price=95.0,
                  tif=OrderTimeInForce.DAY, decision_ts="t0")
    fill = sim.execute(order, _bar(open_=100.0, low=98.0), state)
    assert fill.status == FillStatus.REJECTED
    assert fill.qty_filled == 0
    assert "Limit" in fill.reason


def test_buy_limit_filled_when_bar_dips_below_limit():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    # Limit at 99, bar low is 97 -> limit is touched
    order = Order("AAPL", ActionSide.BUY, 100.0, limit_price=99.0,
                  tif=OrderTimeInForce.DAY, decision_ts="t0")
    fill = sim.execute(order, _bar(open_=100.0, low=97.0), state)
    assert fill.status == FillStatus.FILLED
    # Fill is *at* the limit (we got the favourable price) ignoring slippage
    assert fill.avg_price == pytest.approx(99.0)


def test_sell_limit_filled_when_bar_pops_above_limit():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    state.apply_fill("AAPL", 100, 100.0, 0.0, "t0")
    order = Order("AAPL", ActionSide.SELL, -100.0, limit_price=110.0,
                  tif=OrderTimeInForce.DAY, decision_ts="t0")
    fill = sim.execute(order, _bar(open_=100.0, high=112.0), state)
    assert fill.status == FillStatus.FILLED
    assert fill.avg_price == pytest.approx(110.0)


def test_gtc_unfilled_limit_expires_rather_than_rejecting():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    order = Order("AAPL", ActionSide.BUY, 100.0, limit_price=95.0,
                  tif=OrderTimeInForce.GTC, decision_ts="t0")
    fill = sim.execute(order, _bar(open_=100.0, low=98.0), state)
    assert fill.status == FillStatus.EXPIRED


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_zero_qty_order_rejected():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    order = Order("AAPL", ActionSide.BUY, 0.0, None, OrderTimeInForce.DAY, "t0")
    fill = sim.execute(order, _bar(), state)
    assert fill.status == FillStatus.REJECTED


def test_zero_adv_bar_rejected():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state()
    fill = sim.execute(_market_buy(100), _bar(adv=0.0), state)
    assert fill.status == FillStatus.REJECTED
    assert "ADV" in fill.reason


def test_insufficient_cash_rejected_for_buy():
    sim = ExecutionSimulator(_zero_cost_model())
    state = _fresh_state(starting=500.0)  # not enough for 100*100=10k
    fill = sim.execute(_market_buy(100), _bar(open_=100.0), state)
    assert fill.status == FillStatus.REJECTED
    assert "cash" in fill.reason.lower()
    # State must NOT have been mutated
    assert state.cash == 500.0
    assert "AAPL" not in state.positions


# ---------------------------------------------------------------------------
# proposal_to_order — conversion to Order
# ---------------------------------------------------------------------------


def _good_proposal(side=ActionSide.BUY, target_weight=0.10) -> ActionProposal:
    score = 0.7
    return ActionProposal(
        symbol="AAPL", decision_ts="2026-01-14T20:00:00+00:00",
        side=side, target_weight=target_weight,
        horizon_days=21, conviction_score=score, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.04,
        rationale="Strong evidence.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def test_proposal_to_order_compiles_buy_for_long_target():
    state = _fresh_state(starting=100_000.0)
    order = proposal_to_order(_good_proposal(target_weight=0.10), state, reference_price=100.0)
    assert order is not None
    # 10% of $100k = $10k / $100 ref = 100 shares long
    assert order.qty_signed == 100
    assert order.side == ActionSide.BUY


def test_proposal_to_order_returns_none_for_hold():
    state = _fresh_state()
    p = _good_proposal(side=ActionSide.HOLD, target_weight=0.0)
    # Switch tier+score to fit HOLD invariants
    p = p.model_copy(update={"side": ActionSide.HOLD, "target_weight": 0.0})
    assert proposal_to_order(p, state, reference_price=100.0) is None


def test_proposal_to_order_returns_none_for_abstain():
    state = _fresh_state()
    # ABSTAIN: target_weight=0, conviction LOW
    p = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-14T20:00:00+00:00",
        side=ActionSide.ABSTAIN, target_weight=0.0, horizon_days=5,
        conviction_score=0.0, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.0,
        rationale="Abstain.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )
    assert proposal_to_order(p, state, reference_price=100.0) is None


def test_proposal_to_order_trades_only_delta_when_position_already_exists():
    state = _fresh_state(starting=100_000.0)
    # Already long 50 shares at 100 = $5,000 = 5% weight
    state.apply_fill("AAPL", 50, 100.0, fees=0, ts="t0")
    state.mark_to_market({"AAPL": 100.0}, ts="t0")
    # Propose 10% target — delta is 5% = $5k / $100 = 50 more shares
    order = proposal_to_order(_good_proposal(target_weight=0.10), state, reference_price=100.0)
    assert order is not None
    assert order.qty_signed == pytest.approx(50, abs=1)


def test_proposal_to_order_returns_none_when_delta_rounds_to_zero():
    state = _fresh_state(starting=100_000.0)
    # 0.1% target weight at $100 ref = $100 = 1 share — but if reference is
    # 200 then it's 0.5 shares which rounds to 0
    p = _good_proposal(target_weight=0.001)
    # Match LOW conviction tier
    p = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-14T20:00:00+00:00",
        side=ActionSide.BUY, target_weight=0.001, horizon_days=5,
        conviction_score=0.0, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.0, rationale="tiny",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=False, liquidity_sufficient=True,
        ),
    )
    order = proposal_to_order(p, state, reference_price=200.0)
    assert order is None
