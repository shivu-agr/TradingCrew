"""State featurizer for the L4 RL environment.

The featurizer turns the (OHLCV-history-up-to-t, position-state) pair
into a **fixed-size, scale-invariant** float vector the policy network
consumes.  Two non-negotiable properties:

1. **Look-back uses only bars up to and including ``t``** — never the
   bar that's about to be executed.  The simulator fills at ``t+1``
   open; if any feature peeked at ``t+1`` we'd be leaking.
2. **All features are normalised** to live in roughly [-3, 3].  Raw
   prices would give a different policy for every ticker; we feed
   log-returns + normalised oscillators instead.

The feature set is deliberately small (no transformer-grade context) —
the goal of L4 is a calibrated *prior* the LLM can lean on, not a
black-box that replaces it.  Smaller state == fewer parameters ==
faster training on small episodic histories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# Ordering matters — the policy network is fully connected, so a stable
# index for every feature is what makes a saved checkpoint portable
# across runs.  Keep new features at the *end* of this list.
FEATURE_NAMES: tuple[str, ...] = (
    # Past 10 daily log-returns (most recent last).
    *[f"r_lag_{i}" for i in range(10, 0, -1)],
    # Realised volatility (rolling 20-day stdev of log-returns), z-scored.
    "vol_20",
    # Momentum: 5-day and 20-day cumulative log-return.
    "mom_5",
    "mom_20",
    # RSI(14) mapped from [0, 100] to [-1, 1].
    "rsi_14",
    # MACD signal — (ema_12 - ema_26)/close, z-scored.
    "macd_norm",
    # Bollinger %b — (close - lower)/(upper - lower), centred at 0.
    "bb_pct",
    # Position-state features.
    "pos_weight",        # current weight ∈ [-1, 1]
    "pos_unrealised",    # unrealised PnL % since entry
    "pos_age_norm",      # bars-in-trade / 60, clipped to [0, 1]
    # Day-of-week as a sin/cos pair (lets the policy learn weekend gaps).
    "dow_sin",
    "dow_cos",
)

FEATURE_DIM: int = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractor:
    """Stateless OHLCV → feature-vector translator.

    Stateless so that **training and inference share the exact same
    code path** — the policy never sees an instance variable that
    drifted between rollouts.  Position-state features are passed in
    explicitly by the env on every step.
    """

    lookback: int = 60
    """Bars of history to require before any feature can be computed.
    20 is enough for the rolling stats, 60 leaves head-room for RSI/MACD
    EMAs to warm up properly so the policy doesn't see junk in epoch 0.
    """

    universe_size: int = 0
    """Phase 2D — multi-ticker policy.  When ``> 0`` the extractor
    appends a one-hot tail of length ``universe_size`` to every output
    vector, identifying *which* ticker in the policy's universe the
    state belongs to.  Default ``0`` keeps the single-ticker behaviour."""

    universe_index: int = 0
    """Index into the one-hot tail when ``universe_size > 0``.  Set by
    the env when it constructs the extractor.  Out-of-range values
    silently produce an all-zero tail (the policy must learn to ignore
    that slot)."""

    def __post_init__(self) -> None:
        if self.lookback < 30:
            raise ValueError(
                f"lookback={self.lookback} is too small; "
                "RSI(14)+MACD(26) need at least 30 bars to be meaningful."
            )
        if self.universe_size < 0:
            raise ValueError("universe_size must be >= 0")

    @property
    def total_dim(self) -> int:
        """Effective feature dim for the current configuration."""
        return FEATURE_DIM + max(0, self.universe_size)

    # -- public API --------------------------------------------------------

    def warmup_bars(self) -> int:
        """Minimum bars needed before the extractor can produce a vector."""
        return self.lookback

    def extract(
        self,
        ohlcv_history: pd.DataFrame,
        *,
        position_weight: float = 0.0,
        unrealised_pct: float = 0.0,
        bars_in_trade: int = 0,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """Build the feature vector for the bar at the *end* of ``ohlcv_history``.

        Args:
            ohlcv_history: DataFrame with columns ``open, high, low, close,
                volume`` indexed by date (oldest first).  Must contain at
                least ``warmup_bars()`` rows.
            position_weight: signed portfolio weight of the symbol right
                now, in [-1, 1].  Passed in by the env so the policy can
                avoid e.g. flipping LONG→SHORT every bar.
            unrealised_pct: unrealised PnL as a fraction of cost basis
                (e.g. ``+0.05`` for +5%).  Helps the policy learn to take
                profits / cut losers.
            bars_in_trade: number of bars since the position was opened.
                Normalised against 60 inside; longer than 60 saturates at 1.
            as_of_date: optional override for the timestamp used in the
                day-of-week features.  Defaults to the last index entry.

        Returns:
            ``np.ndarray`` of shape ``(FEATURE_DIM,)`` with dtype float32.
            All values are finite — NaNs are replaced with 0.0 so the
            policy never explodes on a malformed bar.
        """
        if len(ohlcv_history) < self.lookback:
            raise ValueError(
                f"Need at least {self.lookback} bars, got {len(ohlcv_history)}"
            )

        close = ohlcv_history["close"].astype(float).to_numpy()
        log_ret = _safe_log_returns(close)

        # --- past-10 log-returns (most recent last) ----------------------
        last10 = log_ret[-10:] if len(log_ret) >= 10 else _pad_left(log_ret, 10)

        # --- realised vol (z-scored against the 60-day reference) --------
        recent_vol = _stdev(log_ret[-20:]) if len(log_ret) >= 20 else 0.0
        ref_vol = _stdev(log_ret[-self.lookback:]) if len(log_ret) >= self.lookback else recent_vol or 1.0
        vol_z = (recent_vol - ref_vol) / (ref_vol + 1e-8)

        # --- momentum ----------------------------------------------------
        mom_5 = float(np.sum(log_ret[-5:])) if len(log_ret) >= 5 else 0.0
        mom_20 = float(np.sum(log_ret[-20:])) if len(log_ret) >= 20 else 0.0

        # --- RSI(14) mapped to [-1, 1] ----------------------------------
        rsi_raw = _rsi(close, n=14)
        rsi_pm = (rsi_raw - 50.0) / 50.0

        # --- MACD --------------------------------------------------------
        ema12 = _ema(close, span=12)
        ema26 = _ema(close, span=26)
        macd_raw = (ema12 - ema26) / (close[-1] + 1e-8)
        # Z-score using the rolling stdev of (ema12-ema26) over `lookback`.
        macd_series = _ema(close, span=12, return_series=True) - _ema(close, span=26, return_series=True)
        macd_norm = float(macd_raw / (_stdev(macd_series[-self.lookback:]) / (close[-1] + 1e-8) + 1e-8))
        macd_norm = float(np.clip(macd_norm, -5.0, 5.0))

        # --- Bollinger %b -----------------------------------------------
        bb_pct = _bollinger_pct_b(close, n=20, k=2.0)
        bb_pct = float(np.clip((bb_pct - 0.5) * 2.0, -2.0, 2.0))

        # --- Position-state features ------------------------------------
        pos_w = float(np.clip(position_weight, -1.0, 1.0))
        pos_unrealised = float(np.clip(unrealised_pct, -1.0, 1.0))
        pos_age = float(np.clip(bars_in_trade / 60.0, 0.0, 1.0))

        # --- Day-of-week sin/cos (cyclical) ------------------------------
        if as_of_date is None:
            ts = ohlcv_history.index[-1] if isinstance(ohlcv_history.index, pd.DatetimeIndex) else None
        else:
            ts = pd.Timestamp(as_of_date)
        if ts is not None:
            dow = ts.dayofweek
            dow_sin = math.sin(2 * math.pi * dow / 7.0)
            dow_cos = math.cos(2 * math.pi * dow / 7.0)
        else:
            dow_sin = 0.0
            dow_cos = 0.0

        vec = np.concatenate([
            last10.astype(np.float32),
            np.array([
                vol_z, mom_5, mom_20, rsi_pm, macd_norm, bb_pct,
                pos_w, pos_unrealised, pos_age, dow_sin, dow_cos,
            ], dtype=np.float32),
        ])
        # Final safety net — replace any NaN/Inf with 0 so a malformed
        # bar never poisons the gradient.
        vec = np.nan_to_num(vec, nan=0.0, posinf=3.0, neginf=-3.0)
        # Clip to ±5 so a black-swan day doesn't single-handedly dominate
        # the policy update.
        vec = np.clip(vec, -5.0, 5.0).astype(np.float32)
        assert vec.shape == (FEATURE_DIM,), (vec.shape, FEATURE_DIM)

        # Phase 2D — append the optional one-hot universe tail.  The
        # tail length is fixed by ``universe_size`` so all envs in a
        # multi-ticker training run produce vectors of the same size
        # (the policy network only knows about ``total_dim``).
        if self.universe_size > 0:
            tail = np.zeros(self.universe_size, dtype=np.float32)
            if 0 <= self.universe_index < self.universe_size:
                tail[self.universe_index] = 1.0
            vec = np.concatenate([vec, tail])
        return vec


# ---------------------------------------------------------------------------
# Helpers — small, intentionally local copies (avoids pulling in stockstats
# just for an RSI inside an env step that runs 1000s of times per training
# run).  These are correctness-tested in tests/test_rl_state.py.
# ---------------------------------------------------------------------------


def _safe_log_returns(close: np.ndarray) -> np.ndarray:
    """log(close_t / close_{t-1}), with the first element dropped."""
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return np.zeros(0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close[1:] / close[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    return r


def _pad_left(arr: np.ndarray, n: int) -> np.ndarray:
    """Left-pad ``arr`` with zeros so the output length is exactly ``n``."""
    if len(arr) >= n:
        return arr[-n:]
    out = np.zeros(n, dtype=arr.dtype if arr.size else float)
    out[-len(arr):] = arr
    return out


def _stdev(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _ema(close: np.ndarray, span: int, *, return_series: bool = False):
    """Exponential moving average, Pandas-compatible parametrisation."""
    if span <= 1 or len(close) == 0:
        return close.copy() if return_series else float(close[-1] if len(close) else 0.0)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(close, dtype=float)
    out[0] = close[0]
    for i in range(1, len(close)):
        out[i] = alpha * close[i] + (1 - alpha) * out[i - 1]
    if return_series:
        return out
    return float(out[-1])


def _rsi(close: np.ndarray, n: int = 14) -> float:
    """Wilder's RSI (the standard one yfinance reports)."""
    if len(close) < n + 1:
        return 50.0
    deltas = np.diff(close.astype(float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:n].mean()
    avg_loss = losses[:n].mean()
    for i in range(n, len(deltas)):
        avg_gain = (avg_gain * (n - 1) + gains[i]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i]) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _bollinger_pct_b(close: np.ndarray, n: int = 20, k: float = 2.0) -> float:
    """Bollinger %b — where the last close sits inside the band, in [0, 1] usually."""
    if len(close) < n:
        return 0.5
    window = close[-n:]
    mu = float(np.mean(window))
    sigma = float(np.std(window, ddof=1))
    if sigma == 0:
        return 0.5
    upper = mu + k * sigma
    lower = mu - k * sigma
    return (close[-1] - lower) / (upper - lower)
