"""Performance + risk-adjusted return metrics (paper §8.2).

The paper §8.2 critiques the literature for reporting raw return numbers
that look attractive but aren't risk-adjusted, and warns specifically
about over-fit Sharpe ratios.  This module implements:

- **Sharpe ratio**      — excess return / volatility (annualised)
- **Sortino ratio**     — Sharpe with downside-only volatility
- **Calmar ratio**      — annualised return / max drawdown
- **Max drawdown %**    — worst peak-to-trough loss
- **Deflated Sharpe**   — Bailey & López de Prado (2014) correction for
                          the number of strategy variations tried and
                          the higher moments of the return distribution.
                          A deflated-Sharpe > 0 with high probability
                          means the observed Sharpe is unlikely to be
                          the result of multiple-comparison overfit.

All inputs are *simple* periodic returns (not log returns).  Annualisation
assumes ``periods_per_year`` periods of independent returns; we use 252
for daily, 12 for monthly, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class BacktestMetrics:
    """Bundle of all metrics for a single equity-curve series."""

    total_return_pct: float
    cagr: float
    annualised_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    deflated_sharpe: Optional[float]
    n_periods: int
    periods_per_year: int


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float], ddof: int = 1) -> float:
    if len(xs) <= ddof:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - ddof)
    return math.sqrt(var)


def _skew(xs: Sequence[float]) -> float:
    """Sample skewness (Fisher-Pearson)."""
    if len(xs) < 3:
        return 0.0
    mu = _mean(xs)
    s = _std(xs)
    if s == 0:
        return 0.0
    n = len(xs)
    m3 = sum((x - mu) ** 3 for x in xs) / n
    return m3 / (s ** 3)


def _kurtosis(xs: Sequence[float]) -> float:
    """Sample (excess) kurtosis."""
    if len(xs) < 4:
        return 0.0
    mu = _mean(xs)
    s = _std(xs)
    if s == 0:
        return 0.0
    n = len(xs)
    m4 = sum((x - mu) ** 4 for x in xs) / n
    return m4 / (s ** 4) - 3.0


def max_drawdown_pct(equity: Sequence[float]) -> float:
    """Worst peak-to-trough loss as a positive fraction.

    A flat or monotonically increasing curve returns 0.  Equity values
    of zero or negative are tolerated (some strategies blow up); the
    drawdown is then bounded at 1.0.
    """
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > worst:
                worst = dd
    return min(worst, 1.0)


def deflated_sharpe(
    sharpe: float,
    n_obs: int,
    skewness: float,
    excess_kurtosis: float,
    n_trials: int,
) -> Optional[float]:
    """Bailey & López de Prado (2014) deflated Sharpe ratio.

    Returns the probability that the observed Sharpe is *not* due to
    multiple-comparison overfit, given:

    - ``sharpe``: observed annualised Sharpe
    - ``n_obs``: number of independent return observations
    - ``skewness``, ``excess_kurtosis``: of the return distribution
    - ``n_trials``: how many strategy variants were tried before
                    picking this one (the multiple-comparison count)

    Implementation references Bailey, D.H. & López de Prado, M.L.
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality", JPM 2014.

    Returns ``None`` when ``n_obs`` is too small (< 30) — the asymptotic
    distribution underlying the formula is unreliable for tiny samples.
    """
    if n_obs < 30 or n_trials < 1:
        return None

    # Expected maximum Sharpe under null of zero true-Sharpe across
    # ``n_trials`` independent random strategies.  See BlpdP Eq. 5.
    # Approximation: E[max] ≈ sqrt(2 * ln(n_trials)) - (γ + ln(ln(n_trials)) / 2) / sqrt(2 * ln(n_trials))
    # where γ = 0.5772... is Euler-Mascheroni.
    if n_trials == 1:
        sharpe_expected_max = 0.0
    else:
        ln_n = math.log(n_trials)
        sqrt_2lnN = math.sqrt(2.0 * ln_n)
        gamma = 0.5772156649015329
        sharpe_expected_max = sqrt_2lnN - (gamma + math.log(ln_n) / 2.0) / sqrt_2lnN

    # Standard error of the observed Sharpe under the higher-moment
    # correction (Eq. 3 of the paper):
    denom = 1.0 - skewness * sharpe + (excess_kurtosis / 4.0) * (sharpe ** 2)
    if denom <= 0:
        # Pathological higher moments — refuse to invent a number
        return None
    se = math.sqrt(denom / (n_obs - 1))

    if se == 0:
        return None

    # The deflated SR is the probability that observed SR exceeds the
    # expected max SR under the null.  Equivalent to Φ(z) where z is
    # the standardised deviation.
    z = (sharpe - sharpe_expected_max) / se
    return _normal_cdf(z)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the math.erf primitive."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_metrics(
    equity: Sequence[float],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    n_trials: int = 1,
) -> BacktestMetrics:
    """Full metric bundle for an equity curve.

    ``equity[0]`` is the starting NAV; subsequent values are mark-to-market
    NAVs (any positive scale, since we work with returns).  Periodic
    returns are derived from successive equity values.

    ``risk_free_rate`` is the *annualised* risk-free rate; we de-annualise
    it to per-period before subtracting from each return for Sharpe.

    ``n_trials`` is the number of strategy variations that were
    evaluated before picking this one.  Used only for the deflated-SR
    computation; if you're reporting a single strategy honestly, leave
    it at 1.
    """
    if len(equity) < 2:
        return BacktestMetrics(
            total_return_pct=0.0, cagr=0.0, annualised_vol=0.0,
            sharpe=0.0, sortino=0.0, calmar=0.0,
            max_drawdown=0.0, deflated_sharpe=None,
            n_periods=len(equity), periods_per_year=periods_per_year,
        )

    returns = [
        (equity[i] / equity[i - 1]) - 1.0
        for i in range(1, len(equity))
        if equity[i - 1] > 0
    ]
    n = len(returns)
    if n == 0:
        return BacktestMetrics(
            total_return_pct=0.0, cagr=0.0, annualised_vol=0.0,
            sharpe=0.0, sortino=0.0, calmar=0.0,
            max_drawdown=max_drawdown_pct(equity),
            deflated_sharpe=None, n_periods=len(equity),
            periods_per_year=periods_per_year,
        )

    total_ret = (equity[-1] / equity[0]) - 1.0 if equity[0] > 0 else 0.0
    years = n / periods_per_year
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + total_ret) > 0 else 0.0

    mu = _mean(returns)
    sd = _std(returns)
    rf_per = risk_free_rate / periods_per_year
    excess_mean = mu - rf_per

    annualised_vol = sd * math.sqrt(periods_per_year)
    sharpe = (excess_mean * periods_per_year) / annualised_vol if annualised_vol > 0 else 0.0

    downside = [r for r in returns if r < rf_per]
    if downside:
        ds = math.sqrt(sum((r - rf_per) ** 2 for r in downside) / len(downside))
        downside_annual = ds * math.sqrt(periods_per_year)
        sortino = (excess_mean * periods_per_year) / downside_annual if downside_annual > 0 else 0.0
    else:
        sortino = float("inf") if excess_mean > 0 else 0.0

    mdd = max_drawdown_pct(equity)
    calmar = cagr / mdd if mdd > 0 else (float("inf") if cagr > 0 else 0.0)

    dsr = deflated_sharpe(
        sharpe=sharpe,
        n_obs=n,
        skewness=_skew(returns),
        excess_kurtosis=_kurtosis(returns),
        n_trials=n_trials,
    )

    return BacktestMetrics(
        total_return_pct=total_ret,
        cagr=cagr,
        annualised_vol=annualised_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        deflated_sharpe=dsr,
        n_periods=n,
        periods_per_year=periods_per_year,
    )
