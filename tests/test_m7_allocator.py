"""M7 — Multi-ticker allocator (HRP / mean-variance / equal-risk)."""

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
from trading_crew.agentic.portfolio.allocator import (
    AllocationMethod,
    AllocatorConfig,
    _correlation_distance,
    _correlation_from_cov,
    _hrp_allocate,
    _mean_variance_allocate,
    _risk_contributions,
    _shrunk_covariance,
    _single_linkage_order,
    allocate,
)


def _proposal(symbol: str, side: ActionSide = ActionSide.BUY, conv: float = 0.7, exp_ret: float = 0.04) -> ActionProposal:
    weight = 0.05 if side == ActionSide.BUY else (-0.05 if side == ActionSide.SELL else 0.0)
    if conv >= 0.65:
        tier = ConvictionTier.HIGH
    elif conv >= 0.35:
        tier = ConvictionTier.MEDIUM
    else:
        tier = ConvictionTier.LOW
    return ActionProposal(
        symbol=symbol, decision_ts="2026-01-15T20:00:00+00:00",
        side=side, target_weight=weight, horizon_days=21,
        conviction_score=conv,
        conviction_tier=tier,
        expected_return_pct=exp_ret if side != ActionSide.HOLD else 0.0,
        rationale="Test proposal with enough text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _random_returns(n: int = 60, mean: float = 0.0005, std: float = 0.01, seed: int = 0):
    rng = random.Random(seed)
    return [rng.gauss(mean, std) for _ in range(n)]


# ============================================================================
# Covariance + helpers
# ============================================================================


def test_shrunk_covariance_no_shrinkage_equals_sample():
    """λ=0 → pure sample covariance; diagonal entries match variances."""
    rets = {
        "A": _random_returns(seed=1),
        "B": _random_returns(seed=2),
    }
    cov = _shrunk_covariance(rets, lambda_=0.0)
    # Diagonal entries should match the variance of each series
    var_a = sum((r - sum(rets["A"]) / 60) ** 2 for r in rets["A"]) / 59
    assert cov[0][0] == pytest.approx(var_a, abs=1e-10)


def test_shrunk_covariance_full_shrinkage_is_diagonal():
    """λ=1 → off-diagonals are 0, diagonals equal the mean variance."""
    rets = {
        "A": _random_returns(seed=3),
        "B": _random_returns(seed=4),
        "C": _random_returns(seed=5),
    }
    cov = _shrunk_covariance(rets, lambda_=1.0)
    assert cov[0][1] == 0.0
    assert cov[0][2] == 0.0
    assert cov[1][2] == 0.0
    # Diagonal should equal mean of sample variances
    assert cov[0][0] == cov[1][1] == cov[2][2]


def test_correlation_from_cov_diagonal_is_one():
    cov = [[0.04, 0.01], [0.01, 0.09]]
    corr = _correlation_from_cov(cov)
    assert corr[0][0] == pytest.approx(1.0)
    assert corr[1][1] == pytest.approx(1.0)
    # Off-diagonal should be 0.01 / (0.2 * 0.3) = 0.1667
    assert corr[0][1] == pytest.approx(0.01 / (0.2 * 0.3), abs=1e-6)


def test_correlation_distance_bounds():
    corr = [[1.0, 0.5, -0.5], [0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]]
    dist = _correlation_distance(corr)
    # ρ=1 → d=0 ; ρ=-1 → d=1 ; ρ=0 → d=√0.5
    assert dist[0][0] == 0.0
    assert dist[0][2] == pytest.approx(math.sqrt(0.75), abs=1e-6)
    assert dist[1][2] == pytest.approx(math.sqrt(0.5), abs=1e-6)


def test_single_linkage_groups_correlated_pair():
    """Two highly-correlated tickers and one uncorrelated should be adjacent in the order."""
    corr = [
        [1.0, 0.95, 0.05],
        [0.95, 1.0, 0.05],
        [0.05, 0.05, 1.0],
    ]
    dist = _correlation_distance(corr)
    order = _single_linkage_order(dist)
    # Index 0 and 1 should be next to each other
    pos_0 = order.index(0)
    pos_1 = order.index(1)
    assert abs(pos_0 - pos_1) == 1


# ============================================================================
# HRP
# ============================================================================


def test_hrp_weights_sum_to_one():
    cov = [
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.002],
        [0.005, 0.002, 0.16],
    ]
    weights = _hrp_allocate(["A", "B", "C"], cov)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    for w in weights.values():
        assert w > 0


def test_hrp_gives_lower_weight_to_higher_vol():
    """Two-asset case: higher-vol asset should get lower weight."""
    cov = [
        [0.01, 0.0],
        [0.0, 0.16],  # second asset 4x the vol
    ]
    weights = _hrp_allocate(["LOW", "HIGH"], cov)
    assert weights["LOW"] > weights["HIGH"]


def test_hrp_single_symbol_returns_one():
    cov = [[0.04]]
    weights = _hrp_allocate(["A"], cov)
    assert weights == {"A": 1.0}


# ============================================================================
# Mean-variance tilt
# ============================================================================


def test_mean_variance_overweights_higher_expected_return():
    """Equal vol, but A has higher expected return → A weighted higher."""
    cov = [[0.04, 0.0], [0.0, 0.04]]
    expected = {"A": 0.08, "B": 0.02}
    weights = _mean_variance_allocate(["A", "B"], expected, cov, risk_aversion=2.0)
    assert weights["A"] > weights["B"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_mean_variance_clips_negative_tilts():
    """A symbol with very large negative tilt is clipped to 0."""
    cov = [[0.04, 0.0], [0.0, 0.04]]
    expected = {"A": -1.0, "B": 0.10}
    weights = _mean_variance_allocate(["A", "B"], expected, cov, risk_aversion=1.0)
    assert weights["A"] == 0.0
    assert weights["B"] == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# Risk contributions
# ============================================================================


def test_risk_contributions_sum_to_one_when_invested():
    cov = [[0.04, 0.01], [0.01, 0.09]]
    rc = _risk_contributions(["A", "B"], [0.4, 0.6], cov)
    assert sum(rc) == pytest.approx(1.0, abs=1e-9)


def test_risk_contributions_zero_for_zero_weights():
    cov = [[0.04, 0.0], [0.0, 0.09]]
    rc = _risk_contributions(["A", "B"], [0.0, 0.0], cov)
    assert rc == [0.0, 0.0]


# ============================================================================
# Allocate (full pipeline)
# ============================================================================


def test_allocate_respects_gross_budget():
    proposals = [_proposal(s, exp_ret=0.05) for s in ("AAPL", "MSFT", "GOOG")]
    returns = {s: _random_returns(seed=i + 10) for i, s in enumerate(("AAPL", "MSFT", "GOOG"))}
    config = AllocatorConfig(gross_budget=0.50, max_position_weight=0.20)
    result = allocate(proposals, returns, config)
    gross = sum(abs(w) for w in result.weights.values())
    assert gross <= 0.50 + 1e-9


def test_allocate_respects_per_ticker_cap():
    proposals = [_proposal(s, exp_ret=0.10) for s in ("AAPL", "MSFT")]
    returns = {s: _random_returns(seed=i + 20) for i, s in enumerate(("AAPL", "MSFT"))}
    config = AllocatorConfig(max_position_weight=0.15, gross_budget=1.0)
    result = allocate(proposals, returns, config)
    for w in result.weights.values():
        assert abs(w) <= 0.15 + 1e-9


def test_allocate_signs_sells_negative():
    proposals = [
        _proposal("AAPL", side=ActionSide.BUY),
        _proposal("MSFT", side=ActionSide.SELL),
    ]
    returns = {s: _random_returns(seed=i + 30) for i, s in enumerate(("AAPL", "MSFT"))}
    result = allocate(proposals, returns, AllocatorConfig())
    assert result.weights["AAPL"] > 0
    assert result.weights["MSFT"] < 0


def test_allocate_excludes_hold_proposals():
    proposals = [
        _proposal("AAPL", side=ActionSide.BUY),
        _proposal("MSFT", side=ActionSide.HOLD, exp_ret=0.0),
        _proposal("GOOG", side=ActionSide.ABSTAIN, exp_ret=0.0),
    ]
    returns = {s: _random_returns(seed=i + 40) for i, s in enumerate(("AAPL", "MSFT", "GOOG"))}
    result = allocate(proposals, returns, AllocatorConfig())
    assert result.weights["MSFT"] == 0.0
    assert result.weights["GOOG"] == 0.0
    assert result.weights["AAPL"] > 0


def test_allocate_drops_symbols_with_short_history():
    """A ticker with < 30 returns is excluded from the allocator."""
    proposals = [_proposal("AAPL"), _proposal("MSFT")]
    returns = {
        "AAPL": _random_returns(n=60, seed=50),
        "MSFT": _random_returns(n=10, seed=51),  # too short
    }
    result = allocate(proposals, returns, AllocatorConfig())
    assert result.weights["MSFT"] == 0.0
    assert result.weights["AAPL"] > 0


def test_allocate_dust_positions_rounded_to_zero():
    """Weights below min_position_weight should round to 0."""
    proposals = [_proposal(s, conv=0.6, exp_ret=0.04) for s in ("A", "B", "C", "D", "E")]
    returns = {s: _random_returns(seed=i + 60) for i, s in enumerate(("A", "B", "C", "D", "E"))}
    config = AllocatorConfig(
        gross_budget=0.05,  # tiny budget -> some weights below min
        min_position_weight=0.02,
        max_position_weight=0.20,
    )
    result = allocate(proposals, returns, config)
    for w in result.weights.values():
        assert w == 0.0 or abs(w) >= 0.02 - 1e-9


def test_allocate_falls_back_to_equal_risk_for_single_symbol_hrp():
    proposals = [_proposal("AAPL")]
    returns = {"AAPL": _random_returns(seed=70)}
    result = allocate(proposals, returns, AllocatorConfig(method=AllocationMethod.HRP))
    assert result.method_used == AllocationMethod.EQUAL_RISK
    assert result.weights["AAPL"] > 0


def test_allocate_method_is_recorded():
    proposals = [_proposal(s) for s in ("AAPL", "MSFT", "GOOG")]
    returns = {s: _random_returns(seed=i + 80) for i, s in enumerate(("AAPL", "MSFT", "GOOG"))}
    for method in [AllocationMethod.HRP, AllocationMethod.MEAN_VARIANCE, AllocationMethod.EQUAL_RISK]:
        result = allocate(proposals, returns, AllocatorConfig(method=method))
        assert result.method_used == method


def test_allocate_handles_all_abstain():
    proposals = [_proposal(s, side=ActionSide.ABSTAIN, exp_ret=0.0) for s in ("A", "B")]
    returns = {s: _random_returns(seed=i + 90) for i, s in enumerate(("A", "B"))}
    result = allocate(proposals, returns, AllocatorConfig())
    assert all(w == 0.0 for w in result.weights.values())


def test_allocate_diversifies_across_correlated_pair():
    """Highly-correlated pair shouldn't both get max weight under HRP."""
    proposals = [_proposal(s) for s in ("AAPL", "MSFT", "TLT")]
    # AAPL and MSFT highly correlated, TLT independent
    rng = random.Random(123)
    common = [rng.gauss(0.0005, 0.01) for _ in range(60)]
    aapl_ret = [common[i] + rng.gauss(0, 0.002) for i in range(60)]
    msft_ret = [common[i] + rng.gauss(0, 0.002) for i in range(60)]
    tlt_ret = [rng.gauss(0.0002, 0.005) for _ in range(60)]
    returns = {"AAPL": aapl_ret, "MSFT": msft_ret, "TLT": tlt_ret}
    result = allocate(proposals, returns, AllocatorConfig(method=AllocationMethod.HRP, gross_budget=1.0))
    # AAPL + MSFT (correlated) should not crowd out TLT
    assert result.weights["TLT"] > 0
    # Combined weight on the correlated pair should be reasonable, not dominating
    correlated_weight = abs(result.weights["AAPL"]) + abs(result.weights["MSFT"])
    assert correlated_weight < 1.0  # at least some weight on TLT


def test_allocate_low_conviction_reduces_size():
    """Low conviction reduces the per-ticker weight via the conv scaling."""
    high = _proposal("AAPL", conv=0.9)
    low = _proposal("AAPL", conv=0.1)
    returns = {"AAPL": _random_returns(seed=100), "MSFT": _random_returns(seed=101)}
    high_proposals = [high, _proposal("MSFT", conv=0.5)]
    low_proposals = [low, _proposal("MSFT", conv=0.5)]
    # Use a permissive cap so the conv scaling can show through.
    cfg = AllocatorConfig(max_position_weight=0.60, gross_budget=1.0)
    high_result = allocate(high_proposals, returns, cfg)
    low_result = allocate(low_proposals, returns, cfg)
    assert abs(high_result.weights["AAPL"]) > abs(low_result.weights["AAPL"])


# ============================================================================
# Phase 2B — Ledoit-Wolf shrinkage + book partition + allocator method API
# ============================================================================


def test_ledoit_wolf_default_returns_lambda_in_zero_one():
    """``lambda_=None`` (default) selects the LW optimal λ in [0, 1]."""
    rets = {
        "A": _random_returns(seed=20),
        "B": _random_returns(seed=21),
        "C": _random_returns(seed=22),
    }
    cov = _shrunk_covariance(rets)  # lambda_ omitted → LW
    sample = _shrunk_covariance(rets, lambda_=0.0)
    diag = _shrunk_covariance(rets, lambda_=1.0)
    # The LW result must sit between the sample and the diagonal targets
    # on at least one off-diagonal entry — i.e. some shrinkage happened.
    assert sample[0][1] != 0.0
    assert diag[0][1] == 0.0
    # Off-diagonal entry of LW estimator is somewhere between the two.
    lo, hi = sorted([sample[0][1], diag[0][1]])
    assert lo - 1e-9 <= cov[0][1] <= hi + 1e-9


def test_ledoit_wolf_zero_inputs_does_not_crash():
    """Edge case: all-zero series → degenerate covariance, λ clamped safely."""
    rets = {"A": [0.0] * 30, "B": [0.0] * 30}
    cov = _shrunk_covariance(rets)
    # Diagonal might be 0 (zero variance) and off-diagonal 0 — no NaN/inf.
    for row in cov:
        for c in row:
            assert math.isfinite(c)


def test_allocator_config_default_shrinkage_is_lw():
    """AllocatorConfig defaults to LW (shrinkage=None)."""
    cfg = AllocatorConfig()
    assert cfg.shrinkage is None


def test_allocate_runs_with_lw_default():
    proposals = [_proposal(s) for s in ("AAPL", "MSFT", "TLT")]
    returns = {
        "AAPL": _random_returns(seed=200),
        "MSFT": _random_returns(seed=201),
        "TLT": _random_returns(seed=202),
    }
    result = allocate(proposals, returns, AllocatorConfig())
    # All actionable weights non-negative direction, sums within gross budget.
    gross = sum(abs(w) for w in result.weights.values())
    assert gross <= AllocatorConfig().gross_budget + 1e-9


def test_portfolio_state_partitions_by_book(tmp_path, monkeypatch):
    """paper vs prod books live in different files."""
    monkeypatch.setenv("TRADINGAGENTS_CACHE_DIR", str(tmp_path))
    from trading_crew.agentic.portfolio import load_portfolio_state, save_portfolio_state

    paper = load_portfolio_state("paper")
    prod = load_portfolio_state("prod")
    assert paper.portfolio_id == "paper"
    assert prod.portfolio_id == "prod"

    # Mutating paper must not touch prod.
    paper.cash = 1234.5
    save_portfolio_state(paper)
    refreshed_prod = load_portfolio_state("prod")
    assert refreshed_prod.cash != 1234.5
