"""Lightweight regime detector — feeds the cascaded controller (M4) and
episodic memory's regime tag (M3).

The detector takes the last ~63 trading days of OHLCV and returns one of
``Regime.{TREND, RANGE, HIGH_VOL, CRISIS}``.  Heuristics:

- **CRISIS** if realised 20d vol > 2× the 252d mean *and* 20d max drawdown
  exceeds 10%.  Captures shock regimes (March 2020, Oct 2008).
- **HIGH_VOL** if realised 20d vol > 1.5× the 252d mean (sub-crisis).
- **TREND** if the 20d / 63d return is significant relative to noise
  (Sharpe-style filter: |return| > 1.0 σ).
- **RANGE** otherwise (low vol, no clear direction).

The detector is deliberately simple and explicit — paper §11.2 warns that
fancy regime models often hallucinate regime transitions.  The thresholds
above are the standard "rule of thumb" levels used in practitioner notes;
they're config-exposed so users can tune them, but no silent defaults are
applied if the data is too short.
"""

from __future__ import annotations

import math
from typing import Sequence

from .episodic import Regime


def detect_regime(
    closes: Sequence[float],
    *,
    crisis_vol_mult: float = 2.0,
    high_vol_mult: float = 1.5,
    trend_sharpe_threshold: float = 1.0,
    short_window: int = 20,
    long_window: int = 63,
    full_window: int = 252,
) -> Regime:
    """Classify the current market regime from a series of closes.

    ``closes`` must be in chronological order (oldest first, newest last).
    Returns ``Regime.UNKNOWN`` when the series is too short to compute the
    required statistics — *we do not silently fabricate a regime* when
    inputs are missing (paper §13.1 explicitly warns against this).
    """
    n = len(closes)
    if n < short_window + 1:
        return Regime.UNKNOWN

    # Daily returns
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, n)
        if closes[i - 1] > 0
    ]
    if len(returns) < short_window:
        return Regime.UNKNOWN

    def _vol(window: int) -> float:
        if len(returns) < window:
            return 0.0
        slice_ = returns[-window:]
        mean = sum(slice_) / len(slice_)
        variance = sum((r - mean) ** 2 for r in slice_) / max(1, len(slice_) - 1)
        return math.sqrt(variance)

    short_vol = _vol(short_window)
    long_vol = _vol(min(full_window, len(returns)))
    if long_vol == 0:
        return Regime.UNKNOWN

    vol_ratio = short_vol / long_vol

    # Drawdown over the short window
    short_closes = closes[-short_window:]
    peak = max(short_closes)
    trough = min(short_closes)
    drawdown = (peak - trough) / peak if peak > 0 else 0.0

    if vol_ratio >= crisis_vol_mult and drawdown >= 0.10:
        return Regime.CRISIS

    # Compute the trend filter result up-front so it can also discriminate
    # HIGH_VOL_TREND from HIGH_VOL_RANGE.
    trend_score = 0.0
    if len(returns) >= long_window:
        slice_ = returns[-long_window:]
        mean = sum(slice_) / len(slice_)
        std = _vol(long_window)
        if std > 0:
            trend_score = abs(mean) / std

    is_trending = trend_score > (trend_sharpe_threshold / math.sqrt(min(long_window, len(returns))))

    if vol_ratio >= high_vol_mult:
        # Phase 2E — split HIGH_VOL into trend / range sub-regimes so
        # the cascaded controller can route them differently.
        return Regime.HIGH_VOL_TREND if is_trending else Regime.HIGH_VOL_RANGE

    if is_trending:
        return Regime.TREND
    return Regime.RANGE
