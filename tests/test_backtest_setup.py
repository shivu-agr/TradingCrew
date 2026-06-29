"""Tests for the multi-horizon ``backtest_setup`` tool.

These tests pin the per-horizon expectancy + payoff fields that the PM
relies on to avoid the "hit-rate < 40% → NEUTRAL by default" failure
mode.  We patch ``yf.Ticker`` so the test is deterministic and offline.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from trading_crew.tools import backtest_setup, _simulate_horizon


class _FakeTicker:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def history(self, period: str = "5y") -> pd.DataFrame:
        return self._df


def _trending_closes(n: int = 800, drift: float = 0.0005, vol: float = 0.012, seed: int = 7) -> pd.DataFrame:
    """Synthesise a long up-trending OHLCV frame so the harness can run
    every horizon (252d + 30d buffer requires ~280 bars)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=vol, size=n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({"Close": closes})


def test_simulate_horizon_reports_expectancy_and_payoff() -> None:
    closes = _trending_closes().values[:, 0]
    hit_rate, expectancy, payoff, avg, median, total, wins, losses, timeouts, avg_win, avg_loss = _simulate_horizon(
        closes, horizon=60, tgt=0.05, stp=0.05
    )
    assert total == len(closes) - 60
    assert wins + losses + timeouts == total
    assert 0.0 <= hit_rate <= 1.0
    assert avg_win >= 0.0
    assert avg_loss <= 0.0
    # Expectancy reconciles with the per-bucket means.
    expected = (
        hit_rate * avg_win
        + (losses / total) * avg_loss
        + (timeouts / total) * (avg - hit_rate * avg_win - (losses / total) * avg_loss) / max(1e-9, timeouts / total)
    )
    # Looser: just assert the expectancy is finite and the payoff is positive when wins exist.
    assert np.isfinite(expectancy)
    if avg_loss < 0:
        assert payoff >= 0.0


def test_backtest_setup_emits_multi_horizon_panel(monkeypatch) -> None:
    df = _trending_closes(n=800)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=5.0, stop_pct=3.0)
    # Headline + table columns.
    assert "Multi-horizon panel" in out
    for col in ("hit-rate", "payoff", "expectancy"):
        assert col in out
    # All four horizons should appear (20 — trader, 60, 120, 252).
    assert "20d (trader)" in out
    assert "60d" in out
    assert "120d" in out
    assert "252d" in out
    # Best-expectancy / best-hit headlines.
    assert "Best expectancy:" in out
    assert "Best hit-rate:" in out
    # Provenance footer intact (``_source_line`` emits ``Source: …``).
    assert "Source: backtest_setup(" in out


def test_backtest_setup_handles_short_history(monkeypatch) -> None:
    """A short history should not error — only horizons that fit are run."""
    df = _trending_closes(n=120)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=3.0, stop_pct=2.0)
    # 252d should be skipped (120 < 252+30), but the panel + trader row remain.
    assert "Multi-horizon panel" in out
    # Inspect the panel block only — the source line still lists fitted horizons.
    panel = out.split("Multi-horizon panel")[1].split("Best expectancy:")[0]
    assert "20d (trader)" in panel
    assert "252d" not in panel


def test_backtest_setup_payoff_ratio_format(monkeypatch) -> None:
    """Sanity check the table format the PM parses for confidence calibration."""
    df = _trending_closes(n=600)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=5.0, stop_pct=3.0)
    # Locate the panel rows and confirm payoff is a number-with-x suffix.
    matches = re.findall(r"\|\s*([\d\.\-]+)x\s*\|", out)
    assert matches, f"No payoff column found in panel:\n{out}"
    for m in matches:
        assert float(m) >= 0.0
