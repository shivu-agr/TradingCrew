# Decision impact — RELIANCE.NS (target +8.0% / stop −3.0% / trader horizon 20d)

_Generated 20260626T093645Z._  Source: yfinance 5y OHLCV; M1 bridge + M5 sizer; M6 walk-forward.

## Multi-horizon backtest panel

| Horizon | Hit-rate | Payoff | Expectancy | Avg %  | Median % | N    |
|--------:|---------:|-------:|-----------:|-------:|---------:|-----:|
| **20d (trader)** | 16.8% | 2.27x | +0.25% | +0.25% | -3.02% | 1217 |
| 60d | 32.1% | 2.28x | +0.28% | +0.28% | -3.28% | 1177 |
| 120d | 34.6% | 2.28x | +0.54% | +0.54% | -3.25% | 1117 |
| 252d | 34.0% | 2.32x | +0.50% | +0.50% | -3.26% | 985 |

- **Best expectancy**: +0.54% on 120d (hit-rate 34.6%, payoff 2.28x).
- **Best hit-rate**:   34.6% on 120d (expectancy +0.54%).

## Decision impact (same panel, old vs new PM rules)

| Rules | Action | Confidence | Size % | Rationale |
|---|---|---:|---:|---|
| Old (pre-Phase-2F) | **NEUTRAL** | 0.55 | 0.00 | hit-rate 16.8% < 40% caps confidence at 0.60; quality penalty halves size; both binding => NEUTRAL by default. |
| New (current)      | **OVERWEIGHT** | 0.81 | 1.12 | best-expectancy horizon = 120d (exp +0.54%, hit-rate 34.6%, payoff 2.28x). Positive expectancy with payoff ≥ 1.5 — hit-rate cap not binding. |

**How much did the levers change?**

- Action moved: **NEUTRAL → OVERWEIGHT**.
- Confidence Δ: **+0.258** (0.55 → 0.81).
- Size Δ: **+1.12%** of book (0.00% → 1.12%).

## M6 walk-forward backtest

_Only 0 proposals for RELIANCE.NS; walk-forward needs ≥ 5._

### How the walk-forward changes the decision

Without a sufficient proposal history for this ticker the walk-forward is silent — the only signal at this point comes from the multi-horizon `backtest_setup` panel above. Run an analysis on this ticker, accumulate ≥ 5 episodes, then re-run this script to see the empirical fold-level deltas.
