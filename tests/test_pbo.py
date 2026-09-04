"""Probability of Backtest Overfitting (CSCV) tests.

Deterministic + offline: PBO is a pure function of a config-vs-time
return matrix, so we build matrices with known overfit / genuine
structure and assert the qualitative behaviour.
"""

from __future__ import annotations

import random

from trading_crew.agentic.backtest.metrics import (
    probability_of_backtest_overfitting,
)


def test_pbo_none_for_tiny_matrix():
    assert probability_of_backtest_overfitting([[0.1, 0.2]], n_splits=4) is None
    assert probability_of_backtest_overfitting([], n_splits=4) is None


def test_pbo_none_for_single_config():
    matrix = [[0.01] for _ in range(40)]  # only one column
    assert probability_of_backtest_overfitting(matrix, n_splits=8) is None


def test_pbo_in_unit_interval():
    rng = random.Random(1)
    matrix = [[rng.gauss(0, 0.01) for _ in range(6)] for _ in range(64)]
    pbo = probability_of_backtest_overfitting(matrix, n_splits=8)
    assert pbo is not None
    assert 0.0 <= pbo <= 1.0


def _dominant_matrix(n_obs: int = 96, n_cfg: int = 4, seed: int = 3):
    """Config 0 has a genuinely high Sharpe (positive mean, small noise);
    the rest are pure noise.  CSCV should keep picking config 0 OOS."""
    rng = random.Random(seed)
    matrix = []
    for _ in range(n_obs):
        row = [rng.gauss(0.02, 0.005)] + [rng.gauss(0.0, 0.02) for _ in range(n_cfg - 1)]
        matrix.append(row)
    return matrix


def test_pbo_low_when_one_config_genuinely_dominates():
    pbo = probability_of_backtest_overfitting(_dominant_matrix(), n_splits=8)
    assert pbo is not None
    assert pbo < 0.25


def test_pbo_high_when_configs_are_pure_noise():
    """Pure-noise configs: the IS-best is essentially random OOS, so PBO
    trends well above the genuinely-dominant case."""
    rng = random.Random(7)
    n_obs, n_cfg = 96, 8
    matrix = [[rng.gauss(0, 0.02) for _ in range(n_cfg)] for _ in range(n_obs)]
    pbo = probability_of_backtest_overfitting(matrix, n_splits=8)
    pbo_dom = probability_of_backtest_overfitting(_dominant_matrix(), n_splits=8)
    assert pbo is not None and pbo_dom is not None
    assert pbo > pbo_dom


def test_pbo_handles_odd_n_splits():
    rng = random.Random(2)
    matrix = [[rng.gauss(0, 0.01) for _ in range(4)] for _ in range(50)]
    # odd n_splits should be coerced down to even internally, not crash.
    pbo = probability_of_backtest_overfitting(matrix, n_splits=7)
    assert pbo is None or 0.0 <= pbo <= 1.0
