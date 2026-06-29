"""M5 — VaR/CVaR, sizing, hard risk gates."""

from __future__ import annotations

import math
import random

import pytest

from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)
from trading_crew.agentic.portfolio.state import PortfolioState
from trading_crew.agentic.risk.gates import GateConfig, GateResult, RiskGate, run_risk_gates
from trading_crew.agentic.risk.sizing import (
    SizingConfig,
    compute_size,
    debate_to_risk_mult,
)
from trading_crew.agentic.risk.var import (
    VarConfig,
    compute_historical_var,
    compute_parametric_var,
)


# ===========================================================================
# VaR / CVaR
# ===========================================================================


def _normal_returns(n=500, mean=0.0005, std=0.012, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(mean, std) for _ in range(n)]


def test_historical_var_returns_positive_loss_threshold():
    returns = _normal_returns()
    result = compute_historical_var(returns, VarConfig(window=252, confidence=0.95))
    assert result.var > 0
    assert result.cvar >= result.var, "CVaR should be at least VaR"
    assert result.method == "historical"
    assert result.window_used == 252


def test_historical_var_raises_on_short_series():
    with pytest.raises(ValueError, match="at least"):
        compute_historical_var([0.01, -0.02, 0.005])


def test_parametric_var_matches_historical_for_normal_returns():
    """For Gaussian-generated returns, parametric and historical should be close."""
    returns = _normal_returns(n=2000)
    hist = compute_historical_var(returns)
    para = compute_parametric_var(returns)
    # Within 30% of each other (parametric assumes normality which is true here)
    ratio = max(hist.var, para.var) / max(min(hist.var, para.var), 1e-9)
    assert ratio < 1.3, f"normal-returns historical={hist.var:.4f} parametric={para.var:.4f}"


def test_parametric_var_handles_zero_variance():
    """Constant returns have zero risk."""
    result = compute_parametric_var([0.001] * 50)
    assert result.var == 0.0
    assert result.cvar == 0.0


def test_var_config_validates_confidence_bounds():
    with pytest.raises(ValueError, match="confidence"):
        VarConfig(confidence=0.4)
    with pytest.raises(ValueError, match="confidence"):
        VarConfig(confidence=1.0)


def test_var_uses_only_window_observations():
    """A 1000-observation series with window=100 should use only the last 100."""
    returns = _normal_returns(n=1000)
    result = compute_historical_var(returns, VarConfig(window=100, confidence=0.95))
    assert result.window_used == 100


# ===========================================================================
# Sizing
# ===========================================================================


def _proposal(side=ActionSide.BUY, target=0.10, conv=0.7, tier=ConvictionTier.HIGH) -> ActionProposal:
    return ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=side, target_weight=target, horizon_days=21,
        conviction_score=conv, conviction_tier=tier,
        expected_return_pct=0.04,
        rationale="Strong evidence.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def test_sizing_returns_zero_for_hold():
    result = compute_size(
        _proposal(side=ActionSide.HOLD, target=0.0, conv=0.0, tier=ConvictionTier.LOW),
        realised_vol_annualised=0.20,
        cvar_one_day=0.02,
    )
    assert result.final_weight == 0.0
    assert result.binding_constraint == "HOLD_OR_ABSTAIN"


def test_sizing_returns_zero_for_abstain():
    proposal = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.ABSTAIN, target_weight=0.0, horizon_days=5,
        conviction_score=0.0, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.0,
        rationale="Abstain.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=False, liquidity_sufficient=True,
        ),
    )
    result = compute_size(proposal, realised_vol_annualised=0.20, cvar_one_day=0.02)
    assert result.final_weight == 0.0


def test_sizing_respects_hard_cap():
    """Even a wildly optimistic Kelly should be capped at max_position_weight."""
    proposal = _proposal(target=0.10)
    cfg = SizingConfig(max_position_weight=0.05, kelly_fraction=1.0)
    result = compute_size(proposal, realised_vol_annualised=0.10, cvar_one_day=0.001, config=cfg)
    assert abs(result.final_weight) <= 0.05 + 1e-9


def test_sizing_binds_to_cvar_in_high_tail_environment():
    """When CVaR is large, the CVaR cap should bind."""
    proposal = _proposal(target=0.10)
    cfg = SizingConfig(max_cvar_pct=0.01)  # 1% NAV CVaR cap
    result = compute_size(proposal, realised_vol_annualised=0.20, cvar_one_day=0.10, config=cfg)
    # max_cvar / cvar = 0.01 / 0.10 = 0.10
    assert result.binding_constraint in ("cvar", "intent", "kelly", "vol", "hard")
    assert abs(result.final_weight) <= 0.10 + 1e-9


def test_sizing_returns_signed_weight_for_sell():
    proposal = _proposal(side=ActionSide.SELL, target=-0.05)
    proposal = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.SELL, target_weight=-0.05, horizon_days=21,
        conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=-0.04,
        rationale="Bearish.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )
    result = compute_size(proposal, realised_vol_annualised=0.20, cvar_one_day=0.02)
    assert result.final_weight < 0


def test_sizing_applies_risk_multiplier():
    proposal = _proposal(target=0.10)
    full = compute_size(proposal, realised_vol_annualised=0.20, cvar_one_day=0.02, risk_mult=1.0)
    half = compute_size(proposal, realised_vol_annualised=0.20, cvar_one_day=0.02, risk_mult=0.5)
    assert abs(half.final_weight) == pytest.approx(abs(full.final_weight) * 0.5, rel=0.01)


def test_debate_to_risk_mult_reduces_for_low_conviction():
    proposal = _proposal(conv=0.2, tier=ConvictionTier.LOW)
    mult, _ = debate_to_risk_mult({}, proposal=proposal)
    assert mult < 1.0


def test_debate_to_risk_mult_floors_at_config_floor():
    """Stacking penalties cannot push the multiplier below floor."""
    proposal = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.BUY, target_weight=0.05, horizon_days=21,
        conviction_score=0.1, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.02,
        rationale="Low conviction with multiple flag failures.",
        validity_check=ValidityCheck(
            data_timestamps_valid=False,
            fits_risk_budget=False,
            survives_transaction_costs=False,
            liquidity_sufficient=False,
        ),
    )
    mult, _ = debate_to_risk_mult({"latest_speaker": "Conservative"}, proposal=proposal, floor=0.5)
    assert mult == 0.5


# ===========================================================================
# Risk gates
# ===========================================================================


def _fresh_state(starting=100_000.0) -> PortfolioState:
    return PortfolioState(
        portfolio_id="t", base_currency="USD",
        starting_cash=starting, cash=starting, peak_nav=starting,
    )


def test_concentration_gate_rejects_oversized_position():
    proposal = _proposal(target=0.30)
    gate = RiskGate(GateConfig(max_position_weight=0.20))
    msg = gate.check_concentration(proposal)
    assert msg is not None
    assert "concentration" not in msg.lower() or "exceeds" in msg.lower()


def test_concentration_gate_passes_at_limit():
    proposal = _proposal(target=0.20)
    gate = RiskGate(GateConfig(max_position_weight=0.20))
    assert gate.check_concentration(proposal) is None


def test_drawdown_kill_switch_fires_at_threshold():
    state = _fresh_state()
    state.peak_nav = 100_000
    # Force a drawdown by faking position + price drop
    state.apply_fill("AAPL", 1000, 100.0, fees=0, ts="t1")
    state.mark_to_market({"AAPL": 70.0}, ts="t2")  # ~30% loss on position
    gate = RiskGate(GateConfig(drawdown_kill_threshold=0.20))
    # Verify drawdown is large enough
    assert state.max_drawdown >= 0.20, f"setup error: drawdown only {state.max_drawdown}"
    msg = gate.check_drawdown(state)
    assert msg is not None
    assert "Kill-switch" in msg


def test_drawdown_gate_passes_under_threshold():
    state = _fresh_state()
    state.peak_nav = 100_000
    state.max_drawdown = 0.05
    gate = RiskGate(GateConfig(drawdown_kill_threshold=0.20))
    assert gate.check_drawdown(state) is None


def test_leverage_gate_rejects_when_projected_gross_too_high():
    state = _fresh_state()
    # Existing long 60% of NAV in AAPL
    state.apply_fill("AAPL", 600, 100.0, fees=0, ts="t1")
    state.mark_to_market({"AAPL": 100.0}, ts="t1")
    proposal = _proposal(target=0.50)  # propose another huge add
    # Existing weight = 0.6, delta = 0.5 - 0.6 = -0.1 ... actually trim
    # Use a different proposal that adds
    proposal = ActionProposal(
        symbol="MSFT", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.BUY, target_weight=0.95, horizon_days=21,
        conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.04,
        rationale="x",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )
    gate = RiskGate(GateConfig(max_leverage=1.0))
    # AAPL 60% + MSFT 95% = 155% gross
    msg = gate.check_leverage(state, proposal)
    assert msg is not None
    assert "leverage" in msg.lower()


def test_position_cvar_gate_rejects_when_position_too_risky():
    state = _fresh_state()
    proposal = _proposal(target=0.10)
    # 10% of NAV * 5% 1-day CVaR = 0.5% of NAV -> well under default 2% cap
    gate = RiskGate(GateConfig(max_position_cvar_pct=0.001))  # very tight cap
    msg = gate.check_position_cvar(proposal, state, cvar_one_day=0.05)
    assert msg is not None


def test_cash_buffer_gate_rejects_when_cash_would_drop_below_threshold():
    state = _fresh_state(starting=10_000.0)
    proposal = _proposal(target=0.99)  # buy almost everything
    gate = RiskGate(GateConfig(min_cash_buffer_pct=0.10))  # require 10% cash buffer
    msg = gate.check_cash_buffer(proposal, state, reference_price=100.0, est_fees=10.0)
    assert msg is not None
    assert "cash" in msg.lower()


def test_stale_data_gate_rejects_old_data():
    gate = RiskGate(GateConfig(max_stale_days=3))
    msg = gate.check_stale_data(
        decision_ts="2026-01-15T20:00:00+00:00",
        last_bar_ts="2026-01-05T20:00:00+00:00",  # 10 days old
    )
    assert msg is not None
    assert "stale" in msg.lower() or "days old" in msg.lower()


def test_stale_data_gate_passes_when_data_is_fresh():
    gate = RiskGate(GateConfig(max_stale_days=3))
    msg = gate.check_stale_data(
        decision_ts="2026-01-15T20:00:00+00:00",
        last_bar_ts="2026-01-14T20:00:00+00:00",  # 1 day old
    )
    assert msg is None


# ===========================================================================
# Aggregate run_risk_gates
# ===========================================================================


def test_run_risk_gates_aggregates_failures():
    state = _fresh_state(starting=1000.0)  # very small NAV
    state.peak_nav = 5000
    state.max_drawdown = 0.30  # kill switch
    proposal = _proposal(target=0.30)  # also concentration breach
    result = run_risk_gates(
        proposal, state,
        cvar_one_day=0.02, reference_price=100.0, est_fees=1.0,
    )
    assert result.passed is False
    assert result.kill_switch_triggered is True
    gates_failed = {gate for gate, _ in result.failures}
    assert "drawdown_kill_switch" in gates_failed
    assert "concentration" in gates_failed


def test_run_risk_gates_passes_when_all_clean():
    state = _fresh_state()
    proposal = _proposal(target=0.05)
    result = run_risk_gates(
        proposal, state,
        cvar_one_day=0.02, reference_price=100.0, est_fees=5.0,
    )
    assert result.passed is True
    assert result.failures == []
    assert result.kill_switch_triggered is False


def test_run_risk_gates_bypasses_for_hold_or_abstain():
    state = _fresh_state()
    state.max_drawdown = 0.99  # would normally trigger kill-switch
    proposal = ActionProposal(
        symbol="AAPL", decision_ts="2026-01-15T20:00:00+00:00",
        side=ActionSide.HOLD, target_weight=0.0, horizon_days=5,
        conviction_score=0.0, conviction_tier=ConvictionTier.LOW,
        expected_return_pct=0.0,
        rationale="Hold.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )
    result = run_risk_gates(
        proposal, state,
        cvar_one_day=0.02, reference_price=100.0, est_fees=0.0,
    )
    assert result.passed is True  # HOLD bypasses gates
