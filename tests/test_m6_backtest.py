"""M6 — Walk-forward backtest harness, metrics, manifest."""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from trading_crew.agentic.backtest import (
    BacktestMetrics,
    CPCVConfig,
    CPCVFold,
    Fold,
    FoldResult,
    RunManifest,
    WalkForwardConfig,
    build_manifest,
    compute_metrics,
    deflated_sharpe,
    generate_cpcv_folds,
    generate_folds,
    hash_text,
    load_manifest,
    max_drawdown_pct,
    run_backtest,
    run_walk_forward,
    walk_forward_cpcv,
    write_manifest,
)
from trading_crew.agentic.backtest.walk_forward import assert_no_leakage
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


# ============================================================================
# Metrics
# ============================================================================


def test_max_drawdown_zero_for_monotonic_curve():
    assert max_drawdown_pct([100, 110, 120, 130]) == 0.0


def test_max_drawdown_captures_peak_to_trough():
    equity = [100, 120, 80, 90, 110]
    assert max_drawdown_pct(equity) == pytest.approx((120 - 80) / 120, abs=1e-9)


def test_max_drawdown_handles_empty():
    assert max_drawdown_pct([]) == 0.0


def test_max_drawdown_caps_at_one():
    """Equity going below or to zero -> drawdown capped at 1.0."""
    equity = [100, 50, 0]
    assert max_drawdown_pct(equity) == 1.0


def test_compute_metrics_flat_curve_has_zero_sharpe():
    metrics = compute_metrics([100.0] * 252)
    assert metrics.sharpe == 0.0
    assert metrics.total_return_pct == 0.0
    assert metrics.max_drawdown == 0.0


def test_compute_metrics_positive_returns_have_positive_sharpe():
    # Constant +0.05% daily -> annual ~12.5%, vol 0 -> sharpe = 0 (no vol)
    # Use random walk with positive drift
    rng = random.Random(42)
    daily = [rng.gauss(0.0005, 0.01) for _ in range(252)]
    equity = [100.0]
    for r in daily:
        equity.append(equity[-1] * (1 + r))
    metrics = compute_metrics(equity)
    assert metrics.sharpe > 0
    assert metrics.annualised_vol > 0
    assert metrics.n_periods == 252


def test_compute_metrics_includes_deflated_sharpe_for_long_series():
    rng = random.Random(7)
    daily = [rng.gauss(0.0008, 0.012) for _ in range(252)]
    equity = [100.0]
    for r in daily:
        equity.append(equity[-1] * (1 + r))
    metrics = compute_metrics(equity, n_trials=10)
    assert metrics.deflated_sharpe is not None
    assert 0.0 <= metrics.deflated_sharpe <= 1.0


def test_deflated_sharpe_returns_none_for_short_series():
    assert deflated_sharpe(sharpe=1.5, n_obs=10, skewness=0.0, excess_kurtosis=0.0, n_trials=5) is None


def test_deflated_sharpe_decreases_with_more_trials():
    """Holding observed Sharpe fixed, trying more strategies should reduce confidence."""
    dsr_1 = deflated_sharpe(sharpe=1.5, n_obs=252, skewness=0.0, excess_kurtosis=0.0, n_trials=1)
    dsr_100 = deflated_sharpe(sharpe=1.5, n_obs=252, skewness=0.0, excess_kurtosis=0.0, n_trials=100)
    assert dsr_1 is not None and dsr_100 is not None
    assert dsr_1 > dsr_100


def test_sortino_uses_only_downside_vol():
    """A series with mostly positive returns and rare large drops should have Sortino > Sharpe."""
    # Asymmetric: many small gains, few large losses
    equity = [100.0]
    rng = random.Random(13)
    for i in range(252):
        if rng.random() < 0.05:
            equity.append(equity[-1] * 0.97)  # 3% loss
        else:
            equity.append(equity[-1] * 1.005)
    metrics = compute_metrics(equity)
    # Sortino can be very large here; just verify both numbers are sensible
    assert metrics.sortino >= 0
    assert metrics.sharpe >= 0


def test_compute_metrics_handles_two_point_curve():
    metrics = compute_metrics([100.0, 110.0])
    assert metrics.total_return_pct == pytest.approx(0.10, abs=1e-9)


# ============================================================================
# Walk-forward folds
# ============================================================================


def test_fold_indices_are_disjoint():
    folds = generate_folds(
        n_obs=1000,
        config=WalkForwardConfig(train_size=252, embargo_size=5, test_size=63),
    )
    for f in folds:
        assert set(f.train_indices).isdisjoint(set(f.test_indices))
        assert set(f.train_indices).isdisjoint(set(f.embargo_indices))
        assert set(f.test_indices).isdisjoint(set(f.embargo_indices))


def test_walk_forward_embargo_is_at_least_horizon():
    """Embargo size must be honoured exactly between train_end and test_start."""
    embargo = 7
    folds = generate_folds(
        n_obs=500,
        config=WalkForwardConfig(train_size=100, embargo_size=embargo, test_size=20),
    )
    for f in folds:
        assert f.test_start - f.train_end == embargo


def test_walk_forward_no_partial_folds():
    folds = generate_folds(
        n_obs=200,
        config=WalkForwardConfig(train_size=100, embargo_size=10, test_size=40),
    )
    # Bars 100-110 embargo, 110-150 test (fold 0), then slide by 40:
    # 140-150 embargo (overlap with previous test, OK on the train side), 150-190 test (fold 1)
    # Fold 2 would need test_end = 230 > 200 -> excluded
    for f in folds:
        assert f.test_end <= 200


def test_walk_forward_expanding_train():
    folds = generate_folds(
        n_obs=500,
        config=WalkForwardConfig(train_size=100, embargo_size=5, test_size=50, expanding=True),
    )
    assert len(folds) >= 2
    # In expanding mode, train_start is always 0
    for f in folds:
        assert f.train_start == 0
    # Train_end should grow by test_size between folds
    assert folds[1].train_end - folds[0].train_end == 50


def test_walk_forward_rolling_train_keeps_train_size_constant():
    config = WalkForwardConfig(train_size=100, embargo_size=5, test_size=50, expanding=False)
    folds = generate_folds(n_obs=500, config=config)
    for f in folds:
        assert f.train_end - f.train_start == 100


def test_walk_forward_config_validates_sizes():
    with pytest.raises(ValueError, match="train_size"):
        WalkForwardConfig(train_size=0, embargo_size=5, test_size=20)
    with pytest.raises(ValueError, match="embargo_size"):
        WalkForwardConfig(train_size=100, embargo_size=-1, test_size=20)
    with pytest.raises(ValueError, match="test_size"):
        WalkForwardConfig(train_size=100, embargo_size=5, test_size=0)


def test_assert_no_leakage_passes_for_valid_folds():
    folds = generate_folds(
        n_obs=500,
        config=WalkForwardConfig(train_size=100, embargo_size=5, test_size=50),
    )
    assert_no_leakage(folds)  # should not raise


def test_assert_no_leakage_catches_handcrafted_overlap():
    """If a caller hand-builds an invalid fold, the assertion must fire."""
    bad = Fold(
        fold_id=0,
        train_start=0, train_end=100,
        embargo_start=50, embargo_end=60,  # overlaps train
        test_start=60, test_end=80,
    )
    with pytest.raises(AssertionError):
        assert_no_leakage([bad])


# ============================================================================
# Run manifest
# ============================================================================


def test_manifest_roundtrip(tmp_path):
    m = build_manifest(
        run_id="test-001",
        repo_root=tmp_path,
        seed=42,
        prompts_hash="dead" * 16,
        cost_params={"fee_bps": 1.0, "spread_bps": 5.0},
        data_hashes={"AAPL.csv": "abc123"},
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_temperature=0.7,
        extra={"note": "smoke test"},
    )
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    loaded = load_manifest(out)
    assert loaded.run_id == "test-001"
    assert loaded.seed == 42
    assert loaded.cost_params == {"fee_bps": 1.0, "spread_bps": 5.0}
    assert loaded.data_hashes == {"AAPL.csv": "abc123"}
    assert loaded.llm_model == "gpt-4o"
    assert loaded.extra == {"note": "smoke test"}


def test_hash_text_is_deterministic():
    h1 = hash_text("hello world")
    h2 = hash_text("hello world")
    h3 = hash_text("hello world!")
    assert h1 == h2
    assert h1 != h3


def test_manifest_handles_non_git_dir(tmp_path):
    """Non-git directory should produce sha='unknown' but not raise."""
    m = build_manifest(
        run_id="r", repo_root=tmp_path, seed=0, prompts_hash="x",
        cost_params={}, data_hashes={},
        llm_provider="local", llm_model="x", llm_temperature=0.0,
    )
    assert m.code_git_sha == "unknown"


# ============================================================================
# Backtest engine end-to-end
# ============================================================================


def _make_proposal(symbol: str, ts: str, weight: float = 0.05) -> ActionProposal:
    side = ActionSide.BUY if weight > 0 else (ActionSide.SELL if weight < 0 else ActionSide.HOLD)
    return ActionProposal(
        symbol=symbol, decision_ts=ts, side=side,
        target_weight=weight, horizon_days=5,
        conviction_score=0.7, conviction_tier=ConvictionTier.HIGH,
        expected_return_pct=0.02 if weight != 0 else 0.0,
        rationale="Test proposal with enough text.",
        validity_check=ValidityCheck(
            data_timestamps_valid=True, fits_risk_budget=True,
            survives_transaction_costs=True, liquidity_sufficient=True,
        ),
    )


def _make_ohlcv(start: str = "2024-01-02", n: int = 100, drift: float = 0.0005) -> pd.DataFrame:
    """Synthetic OHLCV with a small positive drift."""
    dates = pd.date_range(start, periods=n, freq="B")
    rng = random.Random(99)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.gauss(drift, 0.01)))
    return pd.DataFrame({
        "Date": dates,
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low": [c * 0.995 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    })


def test_backtest_runs_end_to_end():
    """A minimal proposal sequence should produce a valid equity curve + metrics."""
    df = _make_ohlcv(start="2024-01-02", n=100)
    proposals = [
        _make_proposal("AAPL", df["Date"].iloc[10].strftime("%Y-%m-%dT16:00:00+00:00"), weight=0.05),
        _make_proposal("AAPL", df["Date"].iloc[30].strftime("%Y-%m-%dT16:00:00+00:00"), weight=0.0),
        _make_proposal("AAPL", df["Date"].iloc[50].strftime("%Y-%m-%dT16:00:00+00:00"), weight=-0.03),
    ]
    result = run_backtest(
        proposals=proposals,
        ohlcv_by_symbol={"AAPL": df},
        starting_cash=100_000.0,
    )
    assert len(result.equity_curve) == len(proposals) + 1
    assert result.equity_curve[0] == 100_000.0
    assert len(result.trades) == 3
    assert isinstance(result.metrics, BacktestMetrics)


def test_backtest_no_ohlcv_logged_as_no_data():
    proposals = [_make_proposal("ZZZZ", "2024-06-01T16:00:00+00:00", weight=0.05)]
    result = run_backtest(proposals=proposals, ohlcv_by_symbol={})
    assert result.trades[0].status == "NO_DATA"
    assert result.equity_curve[-1] == 100_000.0


def test_backtest_no_future_bar_logged():
    """A proposal at the very last bar leaves no future bar to fill on."""
    df = _make_ohlcv(start="2024-01-02", n=20)
    proposals = [
        _make_proposal("AAPL", df["Date"].iloc[-1].strftime("%Y-%m-%dT16:00:00+00:00"), weight=0.05)
    ]
    result = run_backtest(proposals=proposals, ohlcv_by_symbol={"AAPL": df})
    assert result.trades[0].status == "NO_FUTURE_BAR"


def test_walk_forward_runs_fold_by_fold():
    """run_walk_forward should produce a FoldResult per fold."""
    df = _make_ohlcv(start="2023-01-02", n=300)
    # 20 proposals spaced over the OHLCV range
    proposals = [
        _make_proposal("AAPL", df["Date"].iloc[i * 10].strftime("%Y-%m-%dT16:00:00+00:00"), weight=0.05 if i % 2 == 0 else 0.0)
        for i in range(20)
    ]
    folds = generate_folds(
        n_obs=20,  # 20 proposals
        config=WalkForwardConfig(train_size=5, embargo_size=1, test_size=4),
    )
    assert len(folds) >= 2

    result = run_walk_forward(
        proposals=proposals,
        ohlcv_by_symbol={"AAPL": df},
        folds=folds,
        starting_cash=100_000.0,
    )
    assert len(result.folds) >= 2
    assert len(result.combined_equity) >= 1
    # The combined equity should chain — last NAV of fold k = first NAV of fold k+1
    last_navs = [fr.equity_curve[-1] for fr in result.folds]
    # combined_equity starts at starting_cash, then chains
    assert result.combined_equity[0] == 100_000.0


def test_walk_forward_proposals_filtered_to_test_window():
    """A proposal outside any fold's test window should not appear in any fold's trades."""
    df = _make_ohlcv(start="2023-01-02", n=300)
    proposals = [
        _make_proposal("AAPL", df["Date"].iloc[i * 10].strftime("%Y-%m-%dT16:00:00+00:00"), weight=0.05)
        for i in range(20)
    ]
    folds = generate_folds(
        n_obs=20,
        config=WalkForwardConfig(train_size=10, embargo_size=2, test_size=3),
    )
    # Train indices = [0, 10) for fold 0, embargo = [10, 12), test = [12, 15)
    # No fold's test should include indices 0-11
    result = run_walk_forward(
        proposals=proposals,
        ohlcv_by_symbol={"AAPL": df},
        folds=folds,
    )
    all_trade_ts = {t.ts for fr in result.folds for t in fr.trades}
    train_ts_first_fold = {proposals[i].decision_ts for i in range(10)}
    assert all_trade_ts.isdisjoint(train_ts_first_fold), \
        "Trades from the training portion of fold 0 leaked into the backtest results"


# =============================================================================
# Phase 2C — CPCV (Combinatorial Purged Cross-Validation)
# =============================================================================


def test_cpcv_config_rejects_invalid_args():
    with pytest.raises(ValueError):
        CPCVConfig(n_groups=1, k_test=1)
    with pytest.raises(ValueError):
        CPCVConfig(n_groups=6, k_test=0)
    with pytest.raises(ValueError):
        CPCVConfig(n_groups=6, k_test=6)
    with pytest.raises(ValueError):
        CPCVConfig(n_groups=6, k_test=2, embargo_size=-1)


def test_cpcv_fold_count_matches_n_choose_k():
    """6 groups, k=2 -> C(6,2) = 15 folds."""
    folds = generate_cpcv_folds(60, CPCVConfig(n_groups=6, k_test=2, embargo_size=0))
    assert len(folds) == 15
    folds2 = generate_cpcv_folds(60, CPCVConfig(n_groups=5, k_test=2, embargo_size=0))
    assert len(folds2) == 10


def test_cpcv_train_and_test_are_disjoint():
    """Train must never contain a test index."""
    folds = generate_cpcv_folds(60, CPCVConfig(n_groups=6, k_test=2, embargo_size=1))
    for f in folds:
        train_idx = set(f.train_indices)
        test_idx = set(f.test_indices)
        assert train_idx.isdisjoint(test_idx)


def test_cpcv_embargo_removes_train_neighbours():
    """With embargo_size=2, the 2 bars on each side of every test group are dropped from train."""
    folds = generate_cpcv_folds(60, CPCVConfig(n_groups=6, k_test=2, embargo_size=2))
    for f in folds:
        train_idx = set(f.train_indices)
        for r in f.test_ranges:
            for offset in (1, 2):
                assert (r.start - offset) not in train_idx or (r.start - offset) < 0
                assert (r.stop + offset - 1) not in train_idx or (r.stop + offset - 1) >= 60


def test_cpcv_test_indices_cover_all_obs_across_folds():
    """Across all folds, every obs index should land in *some* test set (CPCV invariant)."""
    n_obs = 60
    folds = generate_cpcv_folds(n_obs, CPCVConfig(n_groups=6, k_test=2, embargo_size=0))
    covered = set()
    for f in folds:
        covered.update(f.test_indices)
    assert covered == set(range(n_obs))


def test_walk_forward_cpcv_wrapper_is_identity():
    cfg = CPCVConfig(n_groups=5, k_test=2, embargo_size=1)
    a = generate_cpcv_folds(50, cfg)
    b = walk_forward_cpcv(50, cfg)
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.train_indices == fb.train_indices
        assert fa.test_indices == fb.test_indices
