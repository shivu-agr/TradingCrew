"""Probability-edge, benchmark, and bootstrap-CI metric tests.

All deterministic + offline — they only exercise the pure-Python metric
functions in ``agentic.backtest.metrics``.
"""

from __future__ import annotations

import random

import pytest

from trading_crew.agentic.backtest.metrics import (
    bootstrap_sharpe_ci,
    compute_benchmark_stats,
    compute_metrics,
    probability_edge,
)


# ============================================================================
# probability_edge
# ============================================================================


def test_probability_edge_empty_is_zero():
    assert probability_edge([]) == (0.0, 0.0, 0.0, 0.0)


def test_probability_edge_all_wins():
    win_rate, expectancy, payoff, pf = probability_edge([0.01, 0.02, 0.03])
    assert win_rate == 1.0
    assert expectancy == pytest.approx(0.02)
    assert payoff == float("inf")   # no losses
    assert pf == float("inf")


def test_probability_edge_low_hit_rate_positive_expectancy():
    """The core Firefly claim: <50% win rate can still be net-positive when
    the winners are much bigger than the losers."""
    # 2 wins of +0.10, 3 losses of -0.02 -> win_rate 0.4 but expectancy +.
    returns = [0.10, 0.10, -0.02, -0.02, -0.02]
    win_rate, expectancy, payoff, pf = probability_edge(returns)
    assert win_rate == pytest.approx(0.4)
    assert expectancy > 0
    assert payoff == pytest.approx(0.10 / 0.02)
    assert pf == pytest.approx((0.10 * 2) / (0.02 * 3))


def test_probability_edge_all_losses():
    win_rate, expectancy, payoff, pf = probability_edge([-0.01, -0.02])
    assert win_rate == 0.0
    assert expectancy < 0
    assert payoff == 0.0
    assert pf == 0.0


# ============================================================================
# compute_benchmark_stats (alpha / beta)
# ============================================================================


def test_benchmark_beta_is_one_for_identical_series():
    r = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    bench_ret, alpha, beta = compute_benchmark_stats(r, r, periods_per_year=252)
    assert beta == pytest.approx(1.0, abs=1e-9)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_benchmark_alpha_positive_when_strategy_adds_constant():
    bench = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
    strat = [x + 0.005 for x in bench]  # +50bps every period on top of beta 1
    bench_ret, alpha, beta = compute_benchmark_stats(strat, bench, periods_per_year=252)
    assert beta == pytest.approx(1.0, abs=1e-9)
    assert alpha > 0


def test_benchmark_returns_none_for_short_series():
    assert compute_benchmark_stats([0.01], [0.01]) == (None, None, None)


# ============================================================================
# bootstrap_sharpe_ci
# ============================================================================


def test_bootstrap_ci_none_for_short_series():
    assert bootstrap_sharpe_ci([0.01] * 10) == (None, None)


def test_bootstrap_ci_is_deterministic_and_ordered():
    rng = random.Random(3)
    returns = [rng.gauss(0.001, 0.01) for _ in range(200)]
    lo1, hi1 = bootstrap_sharpe_ci(returns, seed=42, n_boot=300)
    lo2, hi2 = bootstrap_sharpe_ci(returns, seed=42, n_boot=300)
    assert (lo1, hi1) == (lo2, hi2)     # deterministic
    assert lo1 <= hi1                    # ordered band


def test_bootstrap_ci_brackets_point_estimate_loosely():
    rng = random.Random(9)
    returns = [rng.gauss(0.0015, 0.01) for _ in range(400)]
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    m = compute_metrics(equity, bootstrap=True, bootstrap_seed=1)
    assert m.sharpe_ci_low is not None and m.sharpe_ci_high is not None
    assert m.sharpe_ci_low <= m.sharpe <= m.sharpe_ci_high


# ============================================================================
# compute_metrics wiring (edge + benchmark fields populated)
# ============================================================================


def test_compute_metrics_populates_probability_edge_fields():
    rng = random.Random(11)
    daily = [rng.gauss(0.0008, 0.012) for _ in range(252)]
    equity = [100.0]
    for r in daily:
        equity.append(equity[-1] * (1 + r))
    m = compute_metrics(equity)
    assert 0.0 <= m.win_rate <= 1.0
    assert m.payoff_ratio >= 0.0
    assert m.profit_factor >= 0.0
    # No benchmark supplied -> alpha/beta stay None.
    assert m.alpha is None and m.beta is None


def test_compute_metrics_with_benchmark_populates_alpha_beta():
    rng = random.Random(13)
    bench_eq = [100.0]
    strat_eq = [100.0]
    for _ in range(252):
        br = rng.gauss(0.0005, 0.01)
        bench_eq.append(bench_eq[-1] * (1 + br))
        strat_eq.append(strat_eq[-1] * (1 + br + 0.0003))  # small steady alpha
    m = compute_metrics(strat_eq, benchmark_equity=bench_eq)
    assert m.alpha is not None and m.beta is not None
    assert m.benchmark_return_pct is not None
