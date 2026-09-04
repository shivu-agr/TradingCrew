"""Tests for the conviction-calibration measurement harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from trading_crew.agentic import calibration as cal


def test_brier_score_perfect_and_worst():
    # Perfect predictions -> 0.0
    assert cal.brier_score([(1.0, 1), (0.0, 0)]) == pytest.approx(0.0)
    # Worst predictions -> 1.0
    assert cal.brier_score([(0.0, 1), (1.0, 0)]) == pytest.approx(1.0)
    # Empty -> None (no fabricated score)
    assert cal.brier_score([]) is None


def test_reliability_bins_group_and_omit_empty():
    pairs = [(0.05, 0), (0.08, 0), (0.95, 1), (0.92, 1)]
    bins = cal.reliability_bins(pairs, n_bins=10)
    # Only the two populated buckets appear.
    assert len(bins) == 2
    low, high = bins[0], bins[-1]
    assert low.n == 2 and low.hit_rate == 0.0
    assert high.n == 2 and high.hit_rate == 1.0


def test_calibration_detects_overconfidence():
    # Desk claims 0.9 conviction but only wins half the time -> positive gap.
    pairs = [(0.9, 1), (0.9, 0), (0.9, 1), (0.9, 0)]
    report = cal.calibration_from_pairs(pairs)
    assert report.n == 4
    assert report.hit_rate == pytest.approx(0.5)
    assert report.mean_conviction == pytest.approx(0.9)
    assert report.overconfidence_gap == pytest.approx(0.4)  # 0.9 - 0.5
    assert report.brier == pytest.approx(0.41, abs=1e-6)


def test_empty_report():
    r = cal.calibration_from_pairs([])
    assert r.n == 0 and r.brier is None
    d = r.to_dict()
    assert d["n"] == 0 and d["bins"] == []


@dataclass
class _Row:
    action: str
    conviction_score: float
    realised_return: Optional[float]
    alpha_return: Optional[float] = None


def test_pairs_from_rows_only_directional_and_basis():
    rows = [
        _Row("BUY", 0.7, 0.05, 0.02),
        _Row("SELL", 0.6, -0.01, -0.03),   # a losing short (return<0 -> loss)
        _Row("HOLD", 0.9, 0.10, 0.10),      # excluded: not directional
        _Row("BUY", 0.4, None, None),       # excluded: no outcome
    ]
    pairs = cal._pairs_from_rows(rows, use_alpha=False)
    assert pairs == [(0.7, 1), (0.6, 0)]
    # Alpha basis flips the second row's win/loss only if alpha differs in sign
    pairs_alpha = cal._pairs_from_rows(rows, use_alpha=True)
    assert pairs_alpha == [(0.7, 1), (0.6, 0)]
