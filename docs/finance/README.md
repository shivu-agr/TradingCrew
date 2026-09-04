# Finance primer

> Part of the TradingCrew docs — back to the [full documentation](../README.md) or the [project README](../../README.md).

## Finance basics (for non-finance readers)

If you've come at this from the engineering side and want the finance vocabulary the system uses, this section walks through every concept in order and links each one back to a file you can read.

### Stocks vs commodities vs futures

A **stock** is a claim on a company's future cash flows (Apple → Apple's earnings). A **commodity** is a physical good (oil, copper, wheat). You don't usually trade physical commodities — you trade **futures contracts**, which are standardised agreements to buy/sell a commodity at a fixed price on a fixed future date. `CL=F` (the yfinance symbol) is the next-expiry WTI crude oil futures contract; `MAZDOCK.NS` is a stock.

This repo runs two dashboards on the same backbone: `/stock` for equities and `/commodity` for futures.

### Buying long vs selling short

- **Long**: you own the asset and profit when it goes up. `+10%` weight = "10% of my portfolio is long this stock".
- **Short**: you borrow the asset, sell it now, hope to buy it back cheaper later. `-10%` weight = "10% of my portfolio is short this stock". You profit when it goes down.
- **Flat / Neutral**: zero weight, no position.

The PM's `PortfolioDecision.action` is on a **retail 5-tier scale**: `STRONG_BUY` / `BUY` (long, conviction-graded), `HOLD` (flat), or `SELL` / `STRONG_SELL` (short, conviction-graded).  Older records persisted with the legacy 3-tier alphabet (`OVERWEIGHT` / `NEUTRAL` / `UNDERWEIGHT`) are silently coerced to `BUY` / `HOLD` / `SELL` at load time by the schema's `@field_validator`, so run history stays browsable without a one-shot data migration.

### What price action tells you (Market Analyst lens)

- **OHLCV** = Open, High, Low, Close, Volume — the 5 numbers for each trading day. The system pulls them via yfinance.
- **SMA / EMA** = Simple/Exponential Moving Average — a smoothed line through the closes. SMA20 is the 20-day average. When price > SMA20 > SMA50 the trend is up; when price < SMA20 < SMA50 the trend is down.
- **RSI(14)** = Relative Strength Index over 14 days, bounded [0, 100]. >70 ≈ overbought (recent gains have been concentrated, expect mean-reversion). <30 ≈ oversold.
- **MACD** = momentum oscillator (difference of two EMAs). When MACD crosses above its signal line, that's typically read as bullish.
- **Bollinger Bands** = the moving average ± k·σ envelope. Closes near the upper band are stretched; closes near the lower band are exhausted.

You'll see all of these in the Charts tab and as numbers in the Market Report.

### Fundamentals (Fundamentals Analyst lens)

Comes from yfinance `.info` + financial statements:

- **P/E ratio** = price / earnings-per-share. ≈ how many years of current earnings the market is paying for the stock. ≈ 15-20 is typical for the S&P 500; >30 means the market expects strong growth.
- **P/B ratio** = price / book value. Useful for banks (book value is the regulated capital base).
- **Margin** (gross / operating / net) = how much of each rupee of revenue survives to profit.
- **Free cash flow (FCF)** = operating cash flow − capex. The cash the company actually has left to return to shareholders.
- **Debt-to-equity** = leverage. High D/E means earnings are sensitive to interest rates.

### Macro lens — why the country matters

A stock's price is the sum of company-specific news AND the *macro regime* around it. For a US name the macro basket the Macro Analyst reads is:

- **10-year US Treasury yield (`^TNX`)** — the risk-free rate. When it goes up, future cash flows discount more aggressively → high-growth (low-current-earnings) names fall harder.
- **DXY / DX-Y.NYB** — the US dollar index against a basket of major currencies. Strong dollar = headwind for US exporters, tailwind for emerging-market borrowers.
- **VIX (`^VIX`)** — implied volatility of the S&P 500 over the next 30 days. High VIX = traders are pricing in big moves = "fear gauge".
- **WTI crude (`CL=F`)** and **gold (`GC=F`)** — input cost / safe-haven proxies.

For an **Indian** name the basket is completely different and that's the bug we just fixed:

- **Nifty 50 (`^NSEI`)** and **Sensex (`^BSESN`)** — India's S&P 500 / Dow equivalents.
- **India VIX (`^INDIAVIX`)** — Indian implied volatility.
- **USD/INR (`INR=X`)** — the exchange rate. A stronger dollar (weaker rupee) makes imported crude more expensive for India.
- **Brent (`BZ=F`)** — India imports ~85% of its oil; Brent is the relevant benchmark (not WTI).
- News themes: RBI repo rate, Union Budget, FII / DII flows, India CPI / WPI.

### Geopolitical lens (industry-specific)

A trade idea in a US semiconductor name is dominated by Taiwan strait risk + chip-export controls + TSMC fab capacity. A trade idea in an **Indian shipbuilder** like MAZDOCK is dominated by:

- **Defence budget allocation** — Indian Union Budget defence capex (Make-in-India).
- **Naval procurement** — submarines, frigates, destroyers ordered by the Indian Navy.
- **Strait of Hormuz / Red Sea / Suez Canal shipping disruption** — when these chokepoints are blocked, ship insurance war-risk premia spike and shipbuilders' order books fatten.
- **Ship insurance war-risk premium** — premiums on hull / cargo policies; a leading indicator for shipping demand.

The Geopolitical Analyst now derives these themes from `get_market_context` automatically based on the ticker's industry, so the right risks are surfaced for each name.

### Catalysts (News + Quant Analyst lens)

- **Earnings event** — companies publish results quarterly. The Trader checks `days_to_event` via `get_event_proximity` so it doesn't size a 5-day swing trade through a 3-day-out earnings print without flagging it.
- **Implied move** = the ATM straddle price / spot ≈ the move the options market is pricing in for the next expiry. A 7% implied move into earnings means: option buyers are paying for a ±7% move; if the stock moves less than that, options sellers win.
- **Put/Call ratio** — total put open interest / total call open interest. >1 = bearish positioning; <0.5 = bullish positioning.
- **Analyst recommendations** — the sell-side (Wall Street + Indian brokers) target prices. They're not gospel but extreme dispersion is itself a signal.

### Sizing the position (Risk + PM lens)

Once the team has decided "BUY with 0.65 confidence" (or, more emphatically, "STRONG_BUY with 0.78 confidence"), the deterministic sizer (paper §7.2) decides *how big*. Three independent caps, the **smallest binds**:

1. **Fractional Kelly**: `f* = (expected_return − risk_free_rate) / variance`, multiplied by `kelly_fraction=0.25`. Full Kelly is mathematically optimal for log-wealth growth but empirically too aggressive on noisy estimates — quarter-Kelly is the practitioner default.
2. **Vol target**: cap the position so it contributes at most `vol_target=10%` to portfolio annualised volatility. A high-vol stock gets a smaller weight than a low-vol one.
3. **CVaR clamp**: cap so the position's 95% tail loss (CVaR — expected loss in the worst 5% of days) is below 2% of NAV.

All three are in `trading_crew/agentic/risk/sizing.py`. The L3 grid search sweeps these knobs.

### Risk gates (Layer A — deterministic, can override the LLM)

Before any fill, six predicates in `trading_crew/agentic/risk/gates.py` block bad trades:

1. **Concentration** — |weight| ≤ `max_position_weight=20%`.
2. **Leverage** — gross exposure / NAV ≤ `max_leverage=1.5x`.
3. **Drawdown kill-switch** — if the book is already down 20% peak-to-trough, no new positions until manual reset. Catastrophic-loss circuit breaker.
4. **Single-position CVaR** — tail loss of THIS one trade ≤ 2% NAV.
5. **Stale data** — last bar must be within 5 days of decision. Refuses to trade on month-old prices.
6. **Cash sufficiency** — long buys must have cash ≥ notional + estimated fees (no implicit borrowing).

All six are evaluated (not short-circuited at the first failure) so the UI shows *every* breach, not just the first.

### Execution simulator (Layer A — what a fill actually costs)

`trading_crew/agentic/execution/simulator.py` models:

- **Next-bar fill semantics**: orders submitted at the close are filled at the *next* bar's open — not at the closing price we already saw. This is the single most important guardrail against "implicit execution" cheating in backtests (paper §6.1).
- **Costs**: fees + half-spread + Almgren-Chriss square-root impact (impact ∝ √(order_size / ADV)). Trading 5% of average daily volume costs noticeably more than trading 0.5%.
- **Partial fills**: orders exceeding 5% ADV get partially filled — the remainder is reported as `PARTIAL_FILL`, not silently transacted at the same price.

### Walk-forward backtest vs `backtest_setup` (Backtest tab)

There are **two different "backtests"** in this codebase and they're solving different problems:

- **`backtest_setup` tool** (per-trade base-rate lookup): "Out of 1225 historical times this ticker was trading like this, what % hit my target before my stop?" Runs once per Trader task. Helps calibrate confidence on the *current* idea.
- **Walk-forward backtest** (Backtest tab): replay *all* the proposals you've ever logged for a ticker through the M5 risk gates + M2 simulator, under strict train/embargo/test folds. Produces an equity curve + Deflated Sharpe (Bailey & López de Prado 2014 — adjusts raw Sharpe for the selection-bias inherent in trying many configurations). Reveals whether the *strategy* actually has edge over many trades.

### Memory + reflection (L1 + L2)

Every run writes an **Episode** to `~/.trading_crew/memory/episodes.jsonl`: state summary, action proposal, regime tag, decision_ts, outcome_ts (= decision_ts + horizon_days). The Episodes tab (and `GET /api/memory/retrieve`) is the human tape. Prediction desks do **not** retrieve past episodes as a live tool.

When you click **Resolve outcomes & reflect** in the Episodes tab, the system:

1. Pulls OHLCV between `decision_ts` and `outcome_ts` (or the 30-day review clock), computes realised return (signed by side), alpha vs SPY, max drawdown.
2. Always writes deterministic `desk_lessons` (which desk leaned against the tape). If an LLM is configured it may call web/news/filings/forwards tools to name why the tape moved, then writes `market_reason`, `unhelpful_tools`, `overemphasized`, and an optional playbook vote.
3. Hermes scores the 30-day book and opens **one** bounded experiment (desk prompt or graph topology). That is how the system learns without touching LLM weights.

### Why the Reflective Critic exists

LLMs hallucinate. The Critic (`trading_crew/critic.py`) runs after the PM. By default it is **single-pass** (`critic_samples=1`); raise `critic_samples` to run the *same* prompt N times at different temperatures for a consistency vote (if the votes disagree past the threshold the Critic forces **ABSTAIN**). Each sample returns a typed `CritiqueResponse` with booleans per stage (intent, evidence, risk). If a numeric claim in the rationale doesn't carry an inline `[source: <id>]` tag whose `<id>` is in the decision's `sources` list, the Evidence check fails — that's why every tool appends a `Source: …` footer and the PM surfaces a deduplicated `sources` list. The whole critic can be turned off with `critic_enabled=false`. Note the deterministic `sanitize_citations()` pass already rewrites fabricated `example.com` `[source: …]` tags to `[analyst judgement]` before the critic runs, so a model-invented URL no longer downgrades a sound thesis.

### How to read the final Workflow output (in order)

1. **Cascade controller badge** — regime detected from 252 bars of OHLCV. `CRISIS` short-circuits the whole debate.
2. **Per-analyst Reports** (5 desks: Market/positioning, News/sentiment, Fundamentals/sector, Reports/filings-guidance, Macro/geopolitical) — each numeric claim carries `[source: …]`.
3. **Bull / Bear debate** — adversarial; the Research Manager picks a side (does NOT split the difference).
4. **Trader Plan** — entry, stop, target, horizon, sizing intent, driven by the CHARACTER of the forward thesis (backtest was removed from the trader toolset/prompt; it lives on the independent M6 Backtest tab).
5. **Risk debate** — Aggressive vs Conservative argue in parallel, Neutral synthesises.
6. **Quality Reviewer** (optional, advisory) — flags every uncited or weakly-supported claim; a below-4 score advises the PM to right-size (it does NOT gate the trader and never forces a HOLD by itself). A `[analyst judgement]` label counts as supported-qualitative.
7. **Compliance Officer** (optional, advisory) — `CLEAR / FLAGGED / BLOCKED`. Only a concrete BLOCKED hit maps to `HOLD`.
8. **PM Decision** — typed `PortfolioDecision` JSON with sources list. The PM emits the retail 5-tier action (`STRONG_BUY` / `BUY` / `HOLD` / `SELL` / `STRONG_SELL`), sees the raw analyst reports directly (no dilution), and weights the multi-year compounding picture at least as heavily as the near-term signal. `STRONG_BUY` / `STRONG_SELL` are reserved for one-sided evidence across the mid-/long-term horizons; `HOLD` is reserved for genuinely contradictory evidence.
9. **Reflective Critic** — single-pass by default (raise `critic_samples` for the consistency vote). May revise or ABSTAIN.
10. **M1 ActionProposal** — typed contract for the deterministic layer.
11. **M5 Sizer** — Kelly / vol / CVaR caps; which one bound is shown.
12. **M5 Risk Gates** — every breach reported.
13. **M2 Simulator** — next-bar fill, fees, slippage, impact.
14. **M3 Episode** — appended to memory; will be resolved later when `outcome_ts` passes.

Every step has a UI panel in the Workflow tab.

---

## Finance basics — terms used throughout the dashboard

This section is the source of truth for the inline glossary highlighter in the **Reports** tab and the PM rationale. Hover any highlighted term in the UI to see the same definition.

### Price & technicals

- **OHLCV** — Open / High / Low / Close / Volume bars. The basic time-series unit.
- **SMA / EMA** — Simple / Exponential moving averages. EMA weights recent bars heavier.
- **RSI (14)** — Relative Strength Index over a 14-bar window. Scale 0–100; > 70 = "overbought", < 30 = "oversold".
- **MACD** — Moving-Average Convergence/Divergence: \(\text{EMA}_{12} − \text{EMA}_{26}\). Sign + slope used as a trend filter.
- **Bollinger %b** — Position of the current close within the 2-σ Bollinger band; 0 = lower band, 1 = upper band.
- **ATR** — Average True Range; bar-by-bar volatility estimate used by traders to set stops.

### Valuation

- **P/E** — Price/Earnings ratio. Higher = market expects more growth (or is overpaying).
- **EPS** — Earnings per share, usually trailing-twelve-months.
- **EV/EBITDA** — Enterprise-value / EBITDA. Capital-structure-neutral valuation multiple.

### Risk-adjusted returns

- **Sharpe** — \((r − r_f) / σ\). Annualised mean excess return per unit of volatility.
- **Sortino** — Sharpe variant where the denominator is the *downside* deviation only.
- **Calmar** — \(\text{CAGR} / |\text{max drawdown}|\). Penalises long drawdowns more harshly than Sharpe.
- **Deflated Sharpe** — Bailey & López de Prado 2014. Penalises Sharpe for selection-bias inflation when many configs were tried on the same history. Default ranking metric in the L3 grid search.

### Sizing & risk

- **Kelly criterion** — Optimal sizing for log-wealth growth: \(f^* = (E[r] − r_f) / σ^2\). We use **fractional Kelly** (e.g. 0.25 × f*) because pure Kelly is brutally aggressive on real strategies.
- **CVaR** (Conditional Value-at-Risk) — Expected loss *given* the loss exceeds the VaR threshold. The risk gate caps every position so its CVaR contribution stays inside a budget.
- **Vol target** — Annualised volatility budget per position. Size is back-solved so realised vol ≈ this target.
- **Max position weight** — Hard cap on \(|w_i|\) as a fraction of NAV.
- **Max leverage** — Cap on \(\sum_i |w_i|\) — total gross exposure as a multiple of NAV.
- **Drawdown kill switch** — Flat-the-book trigger when running drawdown exceeds a configured fraction of peak NAV.

### Market regimes

- **TREND** — Directional persistence at moderate vol.
- **RANGE** — Mean-reverting at moderate vol.
- **HIGH_VOL_TREND / HIGH_VOL_RANGE** — Elevated vol, split by trend filter (Phase 2E).
- **CRISIS** — Extreme vol *and* > 10% short-window drawdown. Triggers the cascade override.

### Portfolio construction

- **HRP** — Hierarchical Risk Parity (López de Prado 2016). Diversification via correlation-distance clustering.
- **MV** — Mean-variance (Markowitz). Maximises expected return for a given variance.
- **EQR** — Equal Risk Contribution. Each name contributes the same marginal volatility.
- **Ledoit-Wolf shrinkage** — Closed-form optimal interpolation between sample covariance and a diagonal target. Stabilises the covariance estimate when N >> T or T is small.

### Microstructure / execution

- **ADV** — Average daily volume. Slippage scales with order size / ADV.
- **Bid-ask spread** — Difference between best bid and best ask; the round-trip cost of trading at market.
- **Open interest** (futures / options) — Number of open contracts; liquidity proxy.
- **Contango / backwardation** — Futures curve in *contango* when far-month > spot, *backwardation* when spot > far-month. Roll-yield flips sign.
- **Put/Call ratio** — Volume of puts traded ÷ volume of calls. Sentiment indicator.

### Episodic memory & training

- **Outcome embargo** — An episode is invisible to retrieval until `outcome_ts + embargo_days` has passed. Prevents future-known returns leaking into "past" retrieval.
- **Walk-forward / CPCV** — Out-of-sample backtest schemes. Walk-forward slides chronologically; CPCV enumerates every k-of-N group combination with per-side embargoes.
- **CVaR-PPO** — Risk-sensitive PPO variant where steps in the lower CVaR-α tail of realised rewards get extra advantage weight. Mirrors the M5 sizing mindset *inside* the policy gradient.
