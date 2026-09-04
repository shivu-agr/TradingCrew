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

Probability-edge block (Firefly "probability edge" strategy — a system
can be net-profitable with sub-50% accuracy as long as the average win
dwarfs the average loss):

- **Win rate**          — fraction of positive-return periods.
- **Expectancy**        — average return per period (the true edge).
- **Payoff ratio**      — average win / |average loss|.
- **Profit factor**     — gross gains / gross losses.

Benchmark block (is the edge *alpha* or just market beta?):

- **Benchmark return**  — buy-and-hold return of the underlying.
- **Alpha / Beta**      — OLS of strategy returns on benchmark returns.

Robustness block:

- **Bootstrap Sharpe CI** — stationary-bootstrap confidence band so the
                          Sharpe is reported as a range, not a point.
- **PBO**               — Probability of Backtest Overfitting (Bailey,
                          Borwein, López de Prado & Zhu 2017) via the
                          Combinatorially-Symmetric Cross-Validation
                          (CSCV) algorithm.

All inputs are *simple* periodic returns (not log returns).  Annualisation
assumes ``periods_per_year`` periods of independent returns; we use 252
for daily, 12 for monthly, etc.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple


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
    # -- probability-edge block (all default 0 for backwards-compat) -------
    win_rate: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    # -- benchmark block (None when no benchmark supplied) -----------------
    benchmark_return_pct: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    # -- robustness block (None until computed) ----------------------------
    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None


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


# ---------------------------------------------------------------------------
# Probability-edge metrics (Firefly "probability edge" strategy)
# ---------------------------------------------------------------------------


def probability_edge(returns: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return ``(win_rate, expectancy, payoff_ratio, profit_factor)``.

    Computed over *periodic* returns (each period is treated as one
    "bet").  This is the equity-curve-level view of the probability-edge
    idea: a strategy is viable when ``expectancy > 0`` even if
    ``win_rate < 0.5``, provided the ``payoff_ratio`` is high enough.

    - ``win_rate``      = fraction of strictly-positive-return periods.
    - ``expectancy``    = mean periodic return (the realised per-bet edge).
    - ``payoff_ratio``  = mean win / |mean loss|; ``inf`` when there are
                          no losing periods, ``0`` when there are no wins.
    - ``profit_factor`` = sum of gains / sum of |losses|; same edge cases.
    """
    if not returns:
        return 0.0, 0.0, 0.0, 0.0
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    n = len(returns)
    win_rate = len(wins) / n
    expectancy = _mean(returns)
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = _mean(losses) if losses else 0.0  # negative
    if avg_loss < 0:
        payoff_ratio = avg_win / abs(avg_loss)
    else:
        payoff_ratio = float("inf") if avg_win > 0 else 0.0
    gross_gain = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_gain / gross_loss
    else:
        profit_factor = float("inf") if gross_gain > 0 else 0.0
    return win_rate, expectancy, payoff_ratio, profit_factor


# ---------------------------------------------------------------------------
# Benchmark comparison (alpha / beta vs buy-and-hold)
# ---------------------------------------------------------------------------


def compute_benchmark_stats(
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    periods_per_year: int = 252,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return ``(benchmark_return_pct, alpha, beta)``.

    ``alpha`` and ``beta`` come from an OLS regression of the strategy's
    periodic returns on the benchmark's periodic returns:

        strategy_t = alpha_per + beta * benchmark_t + eps_t

    ``alpha`` is annualised (``alpha_per * periods_per_year``) so it reads
    as an annual excess return over what the market beta would have
    delivered.  ``benchmark_return_pct`` is the benchmark's cumulative
    (buy-and-hold) return over the window.

    Returns ``(None, None, None)`` when the two series can't be aligned to
    a common length of at least 2 observations.
    """
    n = min(len(returns), len(benchmark_returns))
    if n < 2:
        return None, None, None
    r = list(returns[-n:])
    b = list(benchmark_returns[-n:])

    bench_cum = 1.0
    for x in b:
        bench_cum *= (1.0 + x)
    benchmark_return_pct = bench_cum - 1.0

    mean_r = _mean(r)
    mean_b = _mean(b)
    cov = sum((r[i] - mean_r) * (b[i] - mean_b) for i in range(n)) / (n - 1)
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n)) / (n - 1)
    if var_b <= 0:
        return benchmark_return_pct, None, None
    beta = cov / var_b
    alpha_per = mean_r - beta * mean_b
    alpha = alpha_per * periods_per_year
    return benchmark_return_pct, alpha, beta


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on the Sharpe ratio
# ---------------------------------------------------------------------------


def _sharpe_of(returns: Sequence[float], periods_per_year: int, rf_per: float) -> float:
    sd = _std(returns)
    if sd <= 0:
        return 0.0
    excess_mean = _mean(returns) - rf_per
    ann_vol = sd * math.sqrt(periods_per_year)
    return (excess_mean * periods_per_year) / ann_vol


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    n_boot: int = 1000,
    mean_block: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float]]:
    """Stationary-bootstrap CI on the annualised Sharpe ratio.

    Politis & Romano (1994) stationary bootstrap: resample *blocks* of
    geometrically-distributed length (mean ``mean_block``) so the
    resampled series preserves short-range autocorrelation instead of
    assuming i.i.d. returns (which would understate the CI width for
    trending / mean-reverting strategies).

    Returns ``(low, high)`` percentiles at the ``alpha`` two-sided level
    (default 5% -> 2.5th / 97.5th percentiles).  Returns ``(None, None)``
    when there are too few observations (< 30) for a meaningful band.
    """
    n = len(returns)
    if n < 30:
        return None, None
    rf_per = risk_free_rate / periods_per_year
    rng = random.Random(seed)
    p_restart = 1.0 / max(1, mean_block)
    sharpes: List[float] = []
    for _ in range(n_boot):
        sample: List[float] = []
        idx = rng.randrange(n)
        while len(sample) < n:
            sample.append(returns[idx])
            if rng.random() < p_restart:
                idx = rng.randrange(n)
            else:
                idx = (idx + 1) % n
        sharpes.append(_sharpe_of(sample, periods_per_year, rf_per))
    sharpes.sort()
    lo_i = max(0, int((alpha / 2.0) * n_boot))
    hi_i = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    return sharpes[lo_i], sharpes[hi_i]


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV — Bailey et al. 2017)
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]],
    *,
    n_splits: int = 16,
) -> Optional[float]:
    """Probability of Backtest Overfitting via CSCV.

    ``returns_matrix`` is ``T`` rows (time observations) × ``N`` columns
    (candidate configurations), where entry ``[t][n]`` is config ``n``'s
    periodic return at time ``t``.  This is exactly the matrix a grid /
    swarm search produces: one equity-return series per configuration,
    aligned in time.

    Algorithm (Bailey, Borwein, López de Prado & Zhu, "The Probability of
    Backtest Overfitting", J. Computational Finance 2017):

    1. Split the ``T`` rows into ``S`` (=``n_splits``, even) disjoint
       contiguous submatrices.
    2. For every way of choosing ``S/2`` submatrices as in-sample (IS),
       the complement is out-of-sample (OOS).
    3. On IS pick the config with the best Sharpe (``n*``).
    4. Find ``n*``'s *rank* among all configs on OOS; the relative rank
       ``w`` maps to a logit ``lambda = ln(w / (1 - w))``.
    5. PBO = fraction of combinations whose ``lambda < 0`` — i.e. how
       often the IS-optimal config lands in the bottom half OOS.

    A PBO near 0 means the selection procedure generalises; near 0.5+
    means the "best" config is indistinguishable from luck.  Returns
    ``None`` when the matrix is too small to split.
    """
    if not returns_matrix:
        return None
    T = len(returns_matrix)
    N = len(returns_matrix[0])
    if N < 2 or T < n_splits or n_splits < 2:
        return None
    if n_splits % 2 != 0:
        n_splits -= 1
    if n_splits < 2:
        return None

    # Partition rows into n_splits contiguous, near-equal submatrices.
    base = T // n_splits
    rem = T % n_splits
    blocks: List[List[int]] = []
    cursor = 0
    for s in range(n_splits):
        size = base + (1 if s < rem else 0)
        if size == 0:
            return None
        blocks.append(list(range(cursor, cursor + size)))
        cursor += size

    def _col_sharpe(rows: List[int], col: int) -> float:
        series = [returns_matrix[t][col] for t in rows]
        return _sharpe_of(series, periods_per_year=1, rf_per=0.0)

    half = n_splits // 2
    below = 0
    total = 0
    for combo in combinations(range(n_splits), half):
        is_rows: List[int] = []
        for s in combo:
            is_rows.extend(blocks[s])
        oos_rows: List[int] = []
        for s in range(n_splits):
            if s not in combo:
                oos_rows.extend(blocks[s])

        is_perf = [_col_sharpe(is_rows, n) for n in range(N)]
        n_star = max(range(N), key=lambda n: is_perf[n])
        oos_perf = [_col_sharpe(oos_rows, n) for n in range(N)]
        # Rank of n_star among OOS performances (1 = worst, N = best).
        rank = 1 + sum(1 for n in range(N) if oos_perf[n] < oos_perf[n_star])
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        lam = math.log(w / (1.0 - w))
        if lam < 0:
            below += 1
        total += 1

    if total == 0:
        return None
    return below / total


def compute_metrics(
    equity: Sequence[float],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    n_trials: int = 1,
    benchmark_equity: Optional[Sequence[float]] = None,
    bootstrap: bool = False,
    bootstrap_seed: int = 0,
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

    ``benchmark_equity`` (optional) is a buy-and-hold equity curve aligned
    to ``equity`` (same length); when supplied, alpha/beta vs the
    benchmark are populated.  ``bootstrap`` (optional) turns on the
    stationary-bootstrap Sharpe CI (adds a little compute per call).
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

    win_rate, expectancy, payoff_ratio, profit_factor = probability_edge(returns)

    benchmark_return_pct: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    if benchmark_equity is not None and len(benchmark_equity) >= 2:
        bench_returns = [
            (benchmark_equity[i] / benchmark_equity[i - 1]) - 1.0
            for i in range(1, len(benchmark_equity))
            if benchmark_equity[i - 1] > 0
        ]
        benchmark_return_pct, alpha, beta = compute_benchmark_stats(
            returns, bench_returns, periods_per_year=periods_per_year,
        )

    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None
    if bootstrap:
        sharpe_ci_low, sharpe_ci_high = bootstrap_sharpe_ci(
            returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            seed=bootstrap_seed,
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
        win_rate=win_rate,
        expectancy=expectancy,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        benchmark_return_pct=benchmark_return_pct,
        alpha=alpha,
        beta=beta,
        sharpe_ci_low=sharpe_ci_low,
        sharpe_ci_high=sharpe_ci_high,
    )
