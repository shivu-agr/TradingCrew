"""Value-at-Risk + Conditional VaR estimators (paper §7.1).

Two estimators, both honest about their assumptions:

- **Historical VaR/CVaR**: empirical quantile of the past N daily returns.
  No distributional assumption.  Conservative but slow-to-react.
- **Parametric VaR/CVaR**: Gaussian closed-form using sample mean + std.
  Faster to react to vol regime changes, but assumes normality which
  understates tails — the paper §7.1 warns against using only the
  parametric value as a hard gate.

Best practice (per the paper): compute both, raise an alert when they
diverge by > 1.5×, and use the *max* as the binding constraint.  This
module exposes both with their assumptions so the caller can implement
that policy in ``gates.py``.

Conventions
-----------
- Returns are *log* returns of closes, computed by the caller.
- VaR is reported as a **positive number** representing the loss
  threshold: "1-day 5% VaR = 0.022" means we expect a loss of 2.2%
  or worse on 5% of days.
- CVaR (expected shortfall) is the average loss *given* we breach VaR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarConfig:
    """VaR estimator configuration.

    - ``window``: lookback in observations (default 252 ≈ 1y trading days).
    - ``confidence``: one-sided confidence level (0.95 -> 5% tail).
    """

    window: int = 252
    confidence: float = 0.95

    def __post_init__(self):
        if not (0.5 < self.confidence < 1.0):
            raise ValueError(f"confidence must be in (0.5, 1.0); got {self.confidence}")


@dataclass(frozen=True)
class VarResult:
    """VaR + CVaR estimate at a single point in time.

    ``method`` is "historical" or "parametric"; ``window_used`` is the
    actual number of observations used (capped at len(returns)).
    """

    var: float
    cvar: float
    method: str
    window_used: int
    confidence: float


# ---------------------------------------------------------------------------
# Historical VaR/CVaR — empirical quantile
# ---------------------------------------------------------------------------


def compute_historical_var(
    returns: Sequence[float],
    config: VarConfig = VarConfig(),
) -> VarResult:
    """Historical VaR/CVaR — non-parametric, slow-to-react.

    Returns the absolute value of the empirical tail quantile so callers
    can compare directly against a position's notional.  Raises if the
    series is shorter than ``min(window, 30)`` — too short to be reliable.
    """
    if len(returns) < min(config.window, 30):
        raise ValueError(
            f"Need at least {min(config.window, 30)} observations for historical VaR; "
            f"got {len(returns)}"
        )

    slice_ = list(returns[-config.window:])
    sorted_returns = sorted(slice_)
    tail_size = max(1, int(math.floor(len(sorted_returns) * (1.0 - config.confidence))))
    tail = sorted_returns[:tail_size]
    var = abs(tail[-1])  # quantile boundary
    cvar = abs(sum(tail) / len(tail))  # average tail loss

    return VarResult(
        var=var,
        cvar=cvar,
        method="historical",
        window_used=len(slice_),
        confidence=config.confidence,
    )


# ---------------------------------------------------------------------------
# Parametric VaR/CVaR — Gaussian
# ---------------------------------------------------------------------------


def compute_parametric_var(
    returns: Sequence[float],
    config: VarConfig = VarConfig(),
) -> VarResult:
    """Parametric Gaussian VaR/CVaR.

    Formula:
        VaR_α = -μ + z_α · σ      (z_α = inverse-CDF at 1-α)
        CVaR_α = -μ + σ · φ(z_α) / (1 - α)   (φ = standard normal density)

    Uses the *negative* mean so a positive expected return *reduces* VaR
    (the convention in risk reporting).  The Gaussian assumption is
    explicit — callers should not use this as the sole input to a hard
    gate (paper §7.1, "tail-risk asymmetry").
    """
    if len(returns) < min(config.window, 30):
        raise ValueError(
            f"Need at least {min(config.window, 30)} observations for parametric VaR; "
            f"got {len(returns)}"
        )

    slice_ = list(returns[-config.window:])
    n = len(slice_)
    mean = sum(slice_) / n
    variance = sum((r - mean) ** 2 for r in slice_) / max(1, n - 1)
    std = math.sqrt(variance)

    if std == 0:
        # Degenerate input — return zero VaR (constant returns have no risk).
        # Not a silent fallback: the caller asked for an estimate on a flat
        # series and gets the correct answer.
        return VarResult(var=0.0, cvar=0.0, method="parametric", window_used=n, confidence=config.confidence)

    z = _inverse_normal_cdf(config.confidence)  # one-sided
    var = max(0.0, -mean + z * std)

    # CVaR uses standard normal pdf at z
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    cvar = max(0.0, -mean + std * phi / (1 - config.confidence))

    return VarResult(
        var=var,
        cvar=cvar,
        method="parametric",
        window_used=n,
        confidence=config.confidence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse CDF of N(0, 1) using Beasley-Springer/Moro.

    Accurate to ~1e-9 over (0.001, 0.999); we only call with p in
    (0.5, 1.0) for VaR so we stay well within range.
    Source: https://en.wikipedia.org/wiki/Probit#Computation
    (the rational approximation from Wichura 1988, AS 241).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1); got {p}")

    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
