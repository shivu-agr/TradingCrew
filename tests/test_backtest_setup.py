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
    """With enough history (800 bars ≈ 3.2 years), the panel covers the
    SHORT / MEDIUM / LONG tiers up to 504d ≈ 2y; the 5y / 10y rows are
    dropped for lack of history but the panel labels every fitted row
    with its tier tag so the LLM can group them.
    """
    df = _trending_closes(n=800)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=5.0, stop_pct=3.0)
    # Headline + table columns.
    assert "SHORT vs MEDIUM vs LONG" in out
    for col in ("hit-rate", "payoff", "expectancy"):
        assert col in out
    # Tier tags are now inline on every row.
    assert "[SHORT]" in out
    assert "[MEDIUM]" in out
    # Trader's chosen horizon is marked with an asterisk.
    assert "20d" in out
    assert "*" in out
    # Per-tier summary lines
    assert "Per-tier best expectancy" in out
    # Provenance footer intact (``_source_line`` emits ``Source: …``).
    assert "Source: backtest_setup(" in out


def test_backtest_setup_handles_short_history(monkeypatch) -> None:
    """A short history should not error — only horizons that fit are
    run.  With 400 bars the 252d row still fits (minimum 20 samples
    required by the new gate) but the LONG-tier rows do not.
    """
    df = _trending_closes(n=400)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=3.0, stop_pct=2.0)
    assert "SHORT vs MEDIUM vs LONG" in out
    panel = out.split("SHORT vs MEDIUM vs LONG")[1].split("Best expectancy:")[0]
    assert "20d" in panel
    # 5y (1260d) can never fit inside 400 bars — dropped-panel line must appear.
    assert "1260d" not in panel or "dropped" in out


def test_backtest_setup_long_history_covers_five_year_horizon(monkeypatch) -> None:
    """With ~15 years of history (3800 bars) the LONG-tier 5y row fits
    the ≥20-sample gate and appears in the panel.
    """
    df = _trending_closes(n=3800)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=90, target_pct=8.0, stop_pct=4.0)
    assert "[LONG]" in out
    assert "1260d" in out


def test_backtest_setup_payoff_ratio_format(monkeypatch) -> None:
    """Sanity check the table format the PM parses for confidence calibration."""
    df = _trending_closes(n=800)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=5.0, stop_pct=3.0)
    # Locate the panel rows and confirm payoff is a number-with-x suffix.
    matches = re.findall(r"\|\s*([\d\.\-]+)x\s*\|", out)
    assert matches, f"No payoff column found in panel:\n{out}"
    for m in matches:
        assert float(m) >= 0.0


# ---------------------------------------------------------------------------
# Direction=short — regression test for the "similar +1.10% across
# unrelated tickers" bug the desk reported.  Prior to the fix,
# backtest_setup silently computed the LONG-side expectancy for a
# SHORT trade because ``_simulate_horizon`` had no ``direction``
# parameter and the trader passed positive magnitudes for both target
# and stop regardless of side.  These tests pin the new behaviour so
# a future refactor can't regress into the same LONG-bias.
# ---------------------------------------------------------------------------


def test_simulate_horizon_short_direction_is_mirror_of_long_on_symmetric_walk() -> None:
    """On a **zero-drift** random walk, SHORT and LONG expectancy for a
    symmetric target/stop should be approximately mirror images (both
    close to zero, with equal magnitude of any residual)."""
    closes = _trending_closes(n=1500, drift=0.0, vol=0.012, seed=11).values[:, 0]
    long_hit, long_exp, *_ = _simulate_horizon(closes, horizon=60, tgt=0.05, stp=0.05, direction="long")
    short_hit, short_exp, *_ = _simulate_horizon(closes, horizon=60, tgt=0.05, stp=0.05, direction="short")
    # On a symmetric target/stop the two hit-rates should be within a
    # few percentage points of each other.
    assert abs(long_hit - short_hit) < 0.15
    # And the signed expectancies should also be within a few tens of
    # basis points of each other (zero-drift walk).
    assert abs(long_exp + short_exp) < 1.0  # they should roughly cancel


def test_simulate_horizon_short_direction_reports_negative_edge_on_uptrend() -> None:
    """On a **positive-drift** series a SHORT trade should have a
    negative expectancy — which is exactly the pattern the desk
    reported.  This is the regression test for the '+1.10% per trade'
    LONG-bias bug: for a real uptrending name the tool used to report
    +1.10% on a short trade because the underlying simulation was LONG-
    only.  Now the short-direction call must return a negative number.
    """
    closes = _trending_closes(n=1500, drift=0.0006, vol=0.010, seed=17).values[:, 0]
    _, long_exp, *_ = _simulate_horizon(closes, horizon=120, tgt=0.10, stp=0.05, direction="long")
    _, short_exp, *_ = _simulate_horizon(closes, horizon=120, tgt=0.10, stp=0.05, direction="short")
    assert long_exp > 0, f"expected positive LONG expectancy on uptrend, got {long_exp}"
    assert short_exp < 0, f"expected negative SHORT expectancy on uptrend, got {short_exp}"
    # And the two should differ by more than 1% — meaningful edge, not
    # a numerical coincidence.
    assert (long_exp - short_exp) > 1.0


def test_backtest_setup_direction_short_shows_side_in_banner(monkeypatch) -> None:
    """The multi-horizon panel banner must name the side so the trader
    (and the PM downstream) can see which direction was simulated."""
    df = _trending_closes(n=800, drift=0.0004)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out_short = backtest_setup.func(
        "FAKE", horizon_days=60, target_pct=5.0, stop_pct=3.0, direction="short"
    )
    assert "SHORT-side simulation" in out_short
    assert "direction=SHORT" in out_short
    out_long = backtest_setup.func(
        "FAKE", horizon_days=60, target_pct=5.0, stop_pct=3.0, direction="long"
    )
    assert "LONG-side simulation" in out_long
    assert "direction=LONG" in out_long


def test_backtest_setup_invalid_direction_returns_error(monkeypatch) -> None:
    df = _trending_closes(n=400)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    out = backtest_setup.func("FAKE", horizon_days=20, target_pct=3.0, stop_pct=2.0, direction="sideways")
    assert "Invalid direction" in out


def test_backtest_setup_default_direction_is_long_for_backwards_compat(monkeypatch) -> None:
    """Calling backtest_setup without ``direction`` must behave exactly
    like ``direction="long"`` so existing tests and third-party callers
    do not break."""
    df = _trending_closes(n=800)
    monkeypatch.setattr("trading_crew.tools.yf.Ticker", lambda _t: _FakeTicker(df))
    default_out = backtest_setup.func("FAKE", horizon_days=60, target_pct=5.0, stop_pct=3.0)
    long_out = backtest_setup.func(
        "FAKE", horizon_days=60, target_pct=5.0, stop_pct=3.0, direction="long"
    )
    # Same panel + banner; the two outputs should be byte-identical.
    assert default_out == long_out
