# Decision impact — NTNX (target +6.0% / stop −3.0% / trader horizon 20d)

_Generated 20260626T094006Z._  Source: yfinance 5y OHLCV; M1 bridge + M5 sizer; M6 walk-forward.

## Multi-horizon backtest panel

| Horizon | Hit-rate | Payoff | Expectancy | Avg %  | Median % | N    |
|--------:|---------:|-------:|-----------:|-------:|---------:|-----:|
| **20d (trader)** | 37.4% | 1.72x | +0.26% | +0.26% | -3.33% | 1234 |
| 60d | 37.2% | 1.72x | +0.06% | +0.06% | -3.42% | 1194 |
| 120d | 38.0% | 1.72x | +0.18% | +0.18% | -3.39% | 1134 |
| 252d | 39.7% | 1.74x | +0.43% | +0.43% | -3.33% | 1002 |

- **Best expectancy**: +0.43% on 252d (hit-rate 39.7%, payoff 1.74x).
- **Best hit-rate**:   39.7% on 252d (expectancy +0.43%).

## Decision impact (same panel, old vs new PM rules)

| Rules | Action | Confidence | Size % | Rationale |
|---|---|---:|---:|---|
| Old (pre-Phase-2F) | **OVERWEIGHT** | 0.60 | 1.50 | hit-rate 37.4% — confidence capped at 0.60 (old hit-rate rule) |
| New (current)      | **OVERWEIGHT** | 0.79 | 1.50 | best-expectancy horizon = 252d (exp +0.43%, hit-rate 39.7%, payoff 1.74x). Positive expectancy with payoff ≥ 1.5 — hit-rate cap not binding. |

**How much did the levers change?**

- Action unchanged (OVERWEIGHT).
- Confidence Δ: **+0.187** (0.60 → 0.79).
- Size Δ: **+0.00%** of book (1.50% → 1.50%).

## M6 walk-forward backtest

- Proposals replayed: **8**, folds: **4**
- Total return: **-0.00%**
- Sharpe (annualised): **-7.94**
- Deflated Sharpe: _n/a_ (need ≥ 2 folds to estimate the inflation factor).
- Max drawdown: **+0.00%**

### How the walk-forward changes the decision

- Sharpe -7.94 and Deflated Sharpe n/a (single fold): the historical proposals **lost money out-of-sample**. The PM should require a much stronger fresh-evidence case before going long; the right baseline is NEUTRAL.
