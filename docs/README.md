# TradingCrew — Full Documentation

> Complete reference for TradingCrew. For the high-level overview, see the [project README](../README.md).

> Multi-agent **stock + commodity** research workflow built on [CrewAI](https://docs.crewai.com), with a deterministic post-LLM pipeline (sizing, risk gates, execution simulator, episodic memory, walk-forward backtest, and optional PPO training).

![Workflow diagram](images/workflow.png)

A lean team of specialist LLM agents debates a trade idea; the verdict is then fed through 7 deterministic modules (M1–M7) so the *narrative* comes from an LLM but the *sizing, risk, and execution* are pure code you can read.

> **Lean workflow (2024 refactor).** The analyst team was consolidated from **8 narrow desks to 4 wider ones** (each a merged multi-lens desk), the Quality Reviewer + Compliance Officer became **optional advisory reviewers** (default off, run async), memory defaults **off**, and the reflective critic defaults to **single-pass and advisory** (it audits + flags but no longer overrides the PM's action/size unless compliance is BLOCKED or a source is fabricated). Together these cut the per-run LLM-call count by well over half while keeping the multi-round bull/bear debate that drives reasoning quality. See [Workflow & LLM-call budget](#workflow--llm-call-budget).

---

## Why this exists

LLM-only trading agents are great at producing a confident-sounding rationale and terrible at producing a trade you'd actually let touch a book. TradingCrew puts an auditable, LangChain-free deterministic layer **after** the LLM debate so every decision lands as a typed `ActionProposal` that goes through:

| Stage | What it does |
| --- | --- |
| **M1** Bridge | Maps `PortfolioDecision` (LLM) → typed `ActionProposal` |
| **M2** Simulator | Next-bar fills + fees + half-spread + √-impact + partial fills |
| **M3** Episodic memory | Outcome-embargoed retrieval of past trades (no future leakage) |
| **M4** Reflective critic | 5-stage critique (**advisory by default**) → flags issues but keeps the PM's action/size unless a hard integrity failure (compliance BLOCKED / fabricated source) |
| **M5** Sizer + risk gates | Fractional Kelly × vol-target × CVaR clamp, then 6 hard gates |
| **M6** Walk-forward backtest | Embargoed train/test folds, **Deflated Sharpe** ranking |
| **M7** Allocator | HRP / Mean-Variance / Equal-Risk across all tickers |

See [Detailed design](#detailed-design) below for the full design, every endpoint, and the L2/L3 training loops.

## Features at a glance

- **Two dashboards on one backend** — `/stock` (lean equities crew) and `/commodity` (futures crew with curve/COT/seasonality tools). Both run the same M1–M7 deterministic layer; only the cost-model preset differs.
- **Live workflow visualisation** — left-to-right SVG flow with idle / running (animated cyan dash) / done / degraded edge states; per-tool chip strip under each agent.
- **LLM + embedder picker in the UI** — switch between hosted vLLM, local vLLM, OpenAI, or Anthropic per run; the closed-source options grey out until the matching API key is set.
- **Multi-region tickers** — `AAPL`, `RELIANCE.NS`, `0700.HK`, `7203.T`, `HSBA.L`, `CL=F`, … resolved automatically with the right macro basket + news themes per country.
- **India-first data path** — `.NS` / `.BO` tickers optionally route through **screener.in** (Ind-AS fundamentals, promoter/FII/DII/public shareholding, SEBI/BSE announcements) and **upstox v3** (exchange-clean OHLCV, NSE F&O option chain) instead of yfinance, closing the corporate-action-adjustment and shareholding-visibility gaps. Toggled via `TRADINGCREW_SCREENER_ENABLED` and `UPSTOX_ANALYTICS_TOKEN`; disabled = the workflow falls back to the standard yfinance path.
- **Retail 5-tier decisions** — the Portfolio Manager emits `STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL` (successor to the legacy `OVERWEIGHT / NEUTRAL / UNDERWEIGHT` triplet, silently coerced on load for backwards compatibility).
- **Three-tier horizon vocabulary** — every decision carries an explicit `horizon`: `SHORT` (≤ 3 months, catalyst plays), `MEDIUM` (a few months to a year — the desk's default), or `LONG` (1–10 years, compounder / value thesis). The `backtest_setup` panel now spans SHORT / MEDIUM / LONG windows out to `2520d ≈ 10y`, and `get_long_term_thesis` aggregates revenue / EPS CAGR + ROCE trend + shareholding drift from screener.in so the PM can anchor a genuine 5–10 year call on multi-year fundamentals rather than a chart pattern.
- **Swarm intelligence + probability-edge backtest** — a seeded **particle-swarm optimizer** converges on robust sizing/gate parameters (Grid/PSO toggle + convergence sparkline), a **consensus ensemble** fuses the top-K configs and reports swarm agreement/dispersion, and a deterministic **crowd-positioning signal** (options/insider/analyst-dispersion) drives an optional fade-the-crowd tilt. The backtest engine adds **CPCV** paths, **PBO**, bootstrap Sharpe CIs, probability-edge metrics (win-rate / expectancy / payoff / profit-factor), benchmark alpha/beta, and a **multi-asset portfolio replay** through the M7 allocator.
- **Evidence-based conviction (M1, live by default)** — the PM's raw self-reported `confidence` empirically collapses into a narrow, overconfident band: across 824 production episodes **99% of scores fell in [0.60, 0.80]** and **0.5% were ever LOW**, so M5 sizing barely differentiated a weak call from a strong one and same-day re-runs could flip BUY↔SELL at *identical* conviction. The bridge (`trading_crew/agentic/bridge.py`) now derives `conviction_score` deterministically from the evidence the PM already emits — **action strength** (STRONG_BUY/STRONG_SELL vs plain, which the 4-value `ActionSide` otherwise discards), **evidence balance** (`key_drivers` vs `key_risks + falsifiers`), **capital committed** (`size_pct_of_book`), and a de-weighted, floor-stretched raw confidence — so it spans the full range (on a synthetic replay of the production mix the 0.60–0.80 pile-up drops 84%→24%, LOW/MEDIUM/HIGH goes 0/40/60% → 5/78/18%, and HIGH is *earned* by strength + support + size). Toggle with `TRADINGCREW_CONVICTION_MODE=evidence|raw` (`raw` = legacy confidence pass-through for A/B / rollback).
- **Data completeness + decision determinism (workflow-level, live by default)** — a study of 824 production decisions found two leaks that made the same evidence yield different calls: (1) the deterministic **reports-desk guidance signal** (`NET_GUIDANCE_BIAS` + trend card) was computed for **~96%** of runs but survived into the PM's actual reasoning only **~23%** of the time — it was being dropped in the free-text hop from the reports analyst to the PM — and that dropped lens is the one that leans *long*, so its omission fed the short bias; (2) the aggregation/decision desks ran at the same exploration temperature as the debate, so ~identical evidence flipped BUY↔SELL **~32%** of the time. Two fixes, no hardcoded side: the runner now appends the **machine-parsed guidance ledger** straight onto the PM prompt (`report_signals.render_pm_ledger`) so the signal *always* reaches the decision and must be cited; and the **decision/synthesis desks** (`research_manager`, `trader`, `risk_neutral`, `portfolio_manager`) run near-deterministic (`TRADINGCREW_DECISION_TEMPERATURE=0.0`) while analyst/debate desks keep exploring (`TRADINGCREW_DEBATE_TEMPERATURE=0.3`). The **intra-day flip rate** and **per-lens coverage** are now standing diagnostics in `scripts/scale_report.py` (a "same data → same call?" health check, tracked not optimised away).
- **ReAct planning on the decision desks + direct-signal injection (workflow-level, live by default)** — building on the determinism fix, the three synthesis/decision desks (`research_manager`, `trader`, `portfolio_manager`) *can* run CrewAI's **plan → act** loop (`planning=True` + `PlanningConfig`), gated by `TRADINGCREW_DECISION_PLANNING`. Planning is *internal to the agent*, so it deepens reasoning on the desks that weigh the assembled evidence **without enlarging any downstream agent's context**. The tool-heavy analysts (already ReAct via their tool loops) and the bull/bear/risk debaters (valued for argument *diversity*) are deliberately left untouched. **It is DISABLED by default on this deployment (`TRADINGCREW_DECISION_PLANNING=0`)** because CrewAI 1.15.2's planning does not integrate with our `gpt-oss` model — *both* phases fail: (a) *observe* — the model can't emit the nested `StepObservation` schema (`suggested_refinements` must be `[{step_number, new_description}]` objects; it returns bare strings/`{step, action}` dicts), so every observation fails validation and is retried 2–3×; (b) *act* — the StepExecutor produces an **empty result for every plan step**, so CrewAI hits its last-resort finalize fallback (`experimental/agent_executor.py`) and emits the raw plan JSON + `Step N: (no result)` lines *as the desk's report* — a core synthesis desk returning garbage. Net: broken output, strictly worse than the plain (already-ReAct-via-tools) agent loop. Re-enable only on a model verified to produce structured plans **and** non-empty step results (`TRADINGCREW_DECISION_PLANNING=1`; the workflow's ReAct tile badge, driven by `/api/options#react_agents`, then reappears on those three desks). The **direct-signal injection below is independent of planning and stays live.** Alongside this, **direct signals** are handed straight to the PM at the decision point (never broadcast mid-pipeline): the machine-parsed guidance ledger (above), the **deterministic market regime** (M4 cascade, injected by `runner._inject_pm_regime`, skipped when `UNKNOWN`), and a **per-desk bias board** (`runner._inject_pm_bias_board`, `TRADINGCREW_PM_DESK_BIAS_BOARD=1`) — a compact "every lens at a glance" table of each analyst desk's directional lean, dropped onto the PM prompt the instant its node is announced (all desks have finished by then). To make capture reliable (the free-text TLDR `Bias:` line varies too much run-to-run on the OSS model — a live A/B showed it populating only 1–2 of 5 desks), each analyst task now emits a **canonical `DESK_BIAS: BULLISH|BEARISH|NEUTRAL` machine tag** as its last line — the same structured-output discipline the reports desk uses for `NET_GUIDANCE_BIAS` — which `_parse_desk_bias` reads deterministically (falling through to the prose line only as a secondary read; a desk that declares neither simply doesn't appear — we never guess a lean). **In the UI** the board is emitted as a `desk_bias_board` event and rendered in the workflow's **Decision-node panel** (colour-coded chips per desk, persisted on the run record so it rehydrates post-run), and every tile whose desk runs the plan→act→observe loop carries a **`ReAct` badge** driven by `/api/options#react_agents` — so the badge reflects the live `TRADINGCREW_DECISION_PLANNING` setting rather than a hardcoded assumption (empty ⇒ no badges). The board is the structural counter to the dropped-lens leak: the PM literally sees every desk's directional lean, so a lens it ignores is a lens it *chose* to ignore, not one that silently evaporated in the prose hop. It's **non-directive** (a coverage aid, not a vote) and adds **no** extra LLM/tool/network call — a pure regex over TLDRs already in hand. The *diluted* prose signals keep flowing through the normal analyst→manager→PM pipeline. Split-brain by design: compact structured signals for the final call, rich prose for the debate. **A/B it** with `scripts/ab_pm_signals.py` (`--feature board` | `planning` | `all`), which reruns the real workflow with a flag flipped and reports wall-clock delta vs. non-yfinance lens-coverage lift so each addition has to earn its cost.
- **Deterministic price-structure signals (workflow-level, live by default)** — a 6-year / 198-name / 22k-sample point-in-time probe (`scripts/price_structure_probe.py`) measured which price-derived signals actually condition forward outcomes at the workflow's medium-term horizon before any were wired. The verdict shaped the design: **trend/momentum direction has NO edge** on this liquid universe (Hurst buckets all ~0.50 60d hit-rate) — so `HURST` is emitted as *descriptive context only*, never a vote; **realised-vol percentile predicts forward vol/drawdown** (monotonic: low/mid/high → 0.264/0.285/0.304 fwd-60d vol) — a *sizing* signal; **BOCPD regime age** flags fresh breaks (younger regime → higher forward vol, weaker momentum) — a mild "don't chase / size down"; and **dip + oversold is a real mean-reversion edge** (a sharp 20d drop <-8% while in the bottom quartile of the 52w range returned +7.5% fwd-60d at a 66% win-rate vs +3.6%/58% baseline) — the "buy a temporary sentiment dip" setup. `trading_crew/price_structure.py` computes these deterministically from OHLCV (DFA-Hurst, rolling-vol percentile, ATR%, position-in-range, and a variance-scaled BOCPD run-length) and emits them as the same canonical `SIGNAL | NAME | DIRECTION | conf | rationale [source]` grammar the reports desk uses — so the existing parser lifts them. They reach the decision two ways, **never as a directional vote**: (1) `get_indicators` appends a `## PRICE STRUCTURE SIGNALS` card so the Market Analyst reasons over them; (2) the runner computes them straight from OHLCV and prepends an **authoritative, non-directive PM ledger** (`runner._inject_pm_price_structure`, gated by `TRADINGCREW_PM_PRICE_STRUCTURE=1`) the instant the PM node is announced — so the vol-state (SIZE) and any `DIP_BUY` setup *always* reach the call and must be cited. On the sizing side, `sizing.vol_state_to_risk_mult` folds the vol-percentile / fresh-regime penalty into the M5 pipeline's risk multiplier (**only ever reduces** size, composed with the debate multiplier). **In the UI** it's a `price_signals` event rendered as a violet **Price structure** card on the Workflow tab (vol-state, regime age, S/R, a green **DIP-BUY setup** chip, and an amber **size ×N** chip), persisted on the run record + prediction DB so it rehydrates post-run. **A/B it** with `scripts/ab_pm_signals.py --feature price` (a validated AAPL run showed a stable BUY→BUY decision with a wall-clock delta in the noise — the ~1s BOCPD is dwarfed by LLM latency). Flip `TRADINGCREW_PM_PRICE_STRUCTURE=0` to roll back.
- **Conviction calibration harness** — `trading_crew/agentic/calibration.py` scores whether the PM's stated `conviction_score` is *honest*: it turns resolved directional episodes into a **Brier score**, **reliability bins** (predicted conviction vs realised hit-rate), and a signed **over/under-confidence gap** (positive = the desk systematically oversizes its losers). Measurement-only today (deterministic, stdlib, same cohort as the learning loops via `compute_calibration`); once enough episodes resolve the gap can feed a *post-hoc* conviction-recalibration multiplier on top of the evidence score above. Surfaced in the dashboard's **Episodes** tab as a *Conviction calibration* panel (scorecards + reliability bars, with basis/cohort/this-ticker toggles) via `GET /api/calibration`.
- **Hermes prediction harness (one Resolve step)** — after resolve, the tape is labelled WIN/LOSS vs the predicted side (`trading_crew/agentic/outcome.py`) and Hermes opens **one** experiment from that evidence (`trading_crew/agentic/harness.py`: desk prompt, split-board topology, size, or more debate). Mass 1PO/2TO/`promote_graph.py` remain **legacy**. There is no separate L2 backstory prepend and no RLMF few-shot / LoRA path. The next run uses `apply_harness` + the graph router. Autonomous stock runs **round-1 bull and bear in parallel** as independent first thoughts (not a bear rebuttal of a bull turn). Later rounds are sequential rebuttals. The **risk wings use the same contract**: round 1 is two unanchored size views in parallel; later rounds are sequential Aggressive → Conservative rebuttals. Deterministic CrewAI keeps the **bull/bear** pair sync after the async analyst batch (marking them async raises ValidationError) but the same round-1 prompts are independent openings. Round-1 risk wings sit after the sync trader, so they can fan out. They share a compact scratch pad — they do not redump the five full analyst reports. Each turn may `research_web` / `get_news` with a new query, or `request_evidence` (cap 2) from an owning analyst desk.
- **Harness + a 30-day self-learn clock** — L3 walk-forward grid search ranked by Deflated Sharpe stays on the Training tab (sizing, not LLM weights) and **composes with** an open Hermes `size_scale` experiment. L4 PPO is **not** on the live prediction path. Every PENDING episode is also scored at **30 calendar days** against the editable [`trading_crew/config/desk_strategy.yaml`](../trading_crew/config/desk_strategy.yaml) success/failure bars (`trading_crew/agentic/objectives.py` + `self_learn.py` + `harness.py`). That review does **not** replace the MEDIUM/LONG thesis horizon — it is the learning signal so the book can improve before a 180-day episode resolves. The loop changes one experiment at a time from Resolve evidence and never promotes from a single LOSS.

- **Direction-aware backtest** — `backtest_setup(ticker, horizon_days, target_pct, stop_pct, direction)` now accepts `direction="long"` or `direction="short"`. Before this fix, the tool silently simulated the LONG side for every trade, which meant SHORT plans on positive-drift names were shown a spuriously positive expectancy (the "+1.10% per trade" convergence pattern the desk was seeing across unrelated tickers). The trader task prompt now requires the LLM to pass `direction` explicitly on every call; SHORT-side backtests return returns signed by the short side so expectancy is directly comparable to the LONG case.
- **Deterministic stop + target sanity (volatility + horizon aware)** — the PM's `entry_price` / `stop_loss` / `target_price` used to be free-form LLM numbers with no code check, so the model would attach a ~2% technical stop (often "justified" by an invented `ATR252`) to a multi-month/multi-year horizon — a stop that gets hit by one ordinary day of noise. `trading_crew/agentic/risk/stops.py` now computes a real ATR(14) + annualised vol from the same OHLCV the pipeline uses and, before the M1 bridge freezes the levels: (a) widens any implausibly tight / wrong-side / missing **stop** to a horizon-scaled floor, and (b) validates the **target** — a missing / wrong-side / degenerate target is set to a horizon-scaled default, and a moonshot target (beyond ~3× the full-horizon sigma band, or a >90% gain on a short) is capped. `expected_return_pct` is recomputed to `(target−entry)/entry×100` whenever the target moves, and `final_decision` is re-emitted so the report card shows the grounded levels and honest reward:risk. A plausible target (e.g. a +119% target on a 5-year hold) is left untouched. When volatility can't be measured it declines to assess rather than fabricate a level. **Staged targets (T1/T2/T3):** every directional plan now also ships **scale-out tiers** (`stops.compute_staged_targets`) placed at 50% / 75% / 100% of the validated final-target move with `[0.40, 0.35, 0.25]` exit weights and per-tier reward:risk — so the desk can bank partial profit and de-risk instead of an all-or-nothing target. They're attached to `PortfolioDecision.staged_targets` and emitted on the `final_decision` event (empty for HOLD / when vol is unmeasurable — nothing fabricated), and rendered as a *Staged targets (scale-out plan)* strip on the decision card.
- **Quality Review is safe to enable** — the optional Quality Reviewer audits the manager's thesis and treats any inline `[source: <identifier>]` tag (a real `http(s)` URL, `yfinance` / `screener.in` / `get_report_insights` tool id, or a desk TLDR) and any `[analyst judgement]` label as SUPPORTED. It must **not** require that URL to reappear in the truncated analyst pack, and must **not** tell the PM to drop tagged facts. Only an untagged bare number or a fabricated `example.com`-style URL is UNSUPPORTED. Tools stamp `[source: …]` on briefing / news / price figures the LLM left bare (and keep a real URL when a tag mixes it with `example.com`) so valid data is not lost as untagged. A single missing tag never forces a size-down. The score is advisory: below 4 may right-size, never HOLD by itself. The graph also stuffs a **citation index** of every `[source: …]` seen in the run so those tags stay visible.
- **Forward-looking consensus (`get_forward_estimates`)** — the deliberate counterweight to the trailing `get_long_term_thesis`. Wired to the fundamentals desk, it surfaces analyst price target + implied upside, the **forward-vs-trailing P/E and EPS trajectory**, consensus earnings/revenue growth for the current and next fiscal year, EPS-revision momentum, and the rating distribution — so the desk weighs where the business is *heading*, not only where it has been. The fundamentals prompt now requires an explicit **reconcile past-vs-future** step. (On NTNX this flipped a fabricated-data `STRONG_SELL` into a data-grounded `BUY`: real revenue CAGR is +17%, forward P/E 25 vs trailing 58, EPS 0.95→2.19, zero sell ratings.) Sparse-coverage names get an honest "no forward consensus" line rather than invented estimates.
- **Honest long-term thesis tool** — `get_long_term_thesis` (yfinance path) now labels CAGR lines with the **real fiscal-year window** (e.g. `FY2022→FY2025`) instead of a bare period count, marks ROE/ROA as a *latest snapshot (not a trend)*, and prints an explicit **data-availability contract** stating that ROCE / multi-year ROE trends / out-of-window years are NOT available from this path. The reflective critic now flags "fabricated precision" (a real tool's `[source: …]` tag on figures it never produced, e.g. a `2019–2023` ROCE trend) the same way it flags fabricated `example.com` URLs.
- **`long_term_view` is required on every decision** — even when the desk picks SHORT or MEDIUM, the PM's structured output carries a 1-2 sentence "multi-year thesis is intact / broken / N/A" line so the LONG-tier evidence is never silently dropped from the final report. LONG-tier decisions still carry the full 2-5 sentence narrative. The old field name `long_term_thesis` is accepted as an alias so persisted `RunRecord` JSONs keep loading.
- **Virtual portfolio (₹10L growth-weighted paper book)** — a **Portfolio** tab (global) that, per run date, notionally deploys **₹10,00,000** across that run's every `BUY` / `STRONG_BUY`, **splitting the capital in proportion to each name's predicted growth** (`expected_return_pct`), then marks each holding to the **latest** price we hold for that ticker. Because every daily run re-fetches prices, opening an older run's book after a newer run automatically revalues it — no separate quote fetch. The tab shows portfolio totals (invested / current value / return) and a **click-to-sort** per-name table (weight, invested, entry, current, value, % change) — each row links to that stock's workflow, the run-date selector is kept in lock-step with the left-pane **History** selection, and a **market filter (Both / India / US)** re-deploys the full ₹10L *within* the chosen market (weights re-normalise, so each view is a self-contained book) via `?market=IN|US|all`. It's a **pure read-side derivation** of the `predictions` table (no new table, no write path) served by `web/backend/vportfolio.py` via `GET /api/portfolio/virtual[/{run_date}]`. A holding's ₹ value rides its own native price move (entry → latest run price) **and, for US names, a USD→INR overlay** — converted at the run-date rate on entry and the latest rate now (`INR=X` daily series from yfinance, cached hourly), so the book reflects rupee moves too. The FX rate + as-of dates used are surfaced under the table.
- **Full audit trail** — every run is persisted to `~/.trading_crew/runs/{TICKER}/{ts}.json`. Open the **Recent runs** panel in the sidebar to re-load any prior run into the dashboard without re-running.

## Screenshots

The same four shots as the [project README](../README.md).

|  |  |
| --- | --- |
| Prediction universe | Virtual portfolio |
| ![Prediction](images/prediction.png) | ![Portfolio](images/portfolio.png) |
| Per-ticker workbench | Training |
| ![Workflow](images/workflow-run.png) | ![Training](images/training.png) |

## Workflow & LLM-call budget

The equities crew defines **14 agent personas**; a **lean default run uses 11 agents** across 5 stages:

```
5 analyst desks (parallel)  →  Bull + Bear opening (parallel)  →  rebuttal rounds  →  Research Manager
    →  Trader  →  Aggressive ‖ Conservative opening → (later risk rebuttals) → Neutral  →  Portfolio Manager → Decision
```

**The analyst desks** (four merged from the original eight to cut redundant tool-loops and information dilution, plus a dedicated **Reports** desk added back so the company's own primary filings are actually read):

| Desk | Merged remit | Tools |
| --- | --- | --- |
| **Market** | price / technicals **+ options / positioning** (was Market + Quant) | `get_stock_data`, `get_indicators`, `get_options_summary`, `get_analyst_recommendations` |
| **News** | catalysts / corporate actions **+ sentiment / insider** (was News + Social) | `get_news`, `get_global_news`, `get_corporate_actions`, `get_insider_transactions` |
| **Fundamentals** | financials + long-term thesis **+ sector / peers** (was Fundamentals + Sector) | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_shareholding_pattern`, `get_long_term_thesis`, `get_sector_peers` |
| **Reports** | the company's **own primary filings** — concall / earnings-call transcripts, investor decks, annual-report MD&A, credit-rating rationales — read for **management guidance + target-vs-actual delivery** | `get_report_insights`, `get_market_context`, `get_forward_estimates` |
| **Macro** | macro regime **+ geopolitical / supply-chain** (was Macro + Geopolitical) | `get_macro_data`, `get_global_news`, `get_geopolitical_news`, `get_supply_chain_risk` |

> **Reports desk (guidance / target-vs-actual).** Nothing else in the workflow read the *primary* filings where management states explicit forward TARGETS (revenue / margin / order-book / capex guidance) and where a prior target can be checked against what was actually DELIVERED — the "target / actual-achieved" signal. `get_report_insights` is **market-aware**: for **Indian (NSE/BSE)** names it pulls screener.in's dated `#documents` feed (concalls, annual reports, credit ratings); for **US (SEC-registered)** names it pulls the official, free, keyless **SEC EDGAR** submissions feed — 10-K / 10-Q annual & quarterly reports (MD&A / outlook) and earnings 8-Ks (item 2.02 — results + guidance press releases, whose numbers live in the Exhibit 99.1 the tool resolves to before crawling). Either way it crawls the (large) documents and has a dedicated `REPORT_PROCESS_*` LLM **summarise each one before it reaches the agent** (so the desk never ingests a 200-page report). Reprocess is **change-gated**: listing metadata (url + title/date + kind) is compared to the last snapshot; if titles drift, the first ~3 pages are hashed. Unchanged filings reuse the digest; only new/changed PDFs are mapped again. It is **point-in-time-safe** — filings are date-filtered to on/before the decision cutoff (EDGAR filing dates are exact) and the summariser is told to ignore anything dated after it. EDGAR needs no key but requires a descriptive contact `User-Agent` (`SEC_EDGAR_USER_AGENT`, toggle `TRADINGCREW_EDGAR_ENABLED`); note concall transcripts / credit ratings aren't on EDGAR, so those two surfaces are US-empty (the web-search desk still finds transcripts). For a name in **neither** feed (e.g. a London / HK listing) the tool returns a clear "not available" note and the desk defers to `get_forward_estimates` rather than fabricating a target.
>
> **Firefly-style trend signal card.** The digest LEADS with a compact, machine-readable card of **directional, confidence-scored signals** derived only from the filings and — where the sources allow — compared latest-vs-prior period so they read as *trends*, not snapshots: `OUTLOOK_TONE` (bullish/cautious/defensive), `REVENUE_GUIDANCE` (raised/maintained/cut), `MARGIN_GUIDANCE`, `DELIVERY_TRACK_RECORD` (beat/in-line/miss + streak), `DEMAND_MOMENTUM`, `CREDIT_TRAJECTORY`, and a synthesised `NET_GUIDANCE_BIAS` (positive/neutral/negative). Each is one line — `SIGNAL | NAME | DIRECTION | conf=0.00-1.00 | rationale [source: url]` — so the Reports analyst and PM get an actionable headline read, and a later step can lift them into the decision schema / fusion / dashboard. A signal the filings don't support is emitted `INSUFFICIENT_DATA, conf=0.0` (never fabricated).
>
> **Deterministic financial-trend signals.** Alongside the LLM-read filing signals, the digest appends a `## COMPUTED FINANCIAL TREND SIGNALS` block computed from **hard quarterly numbers** (screener.in's `#quarters` table for Indian names — reused from the same fetch; yfinance's quarterly income statement otherwise): `REVENUE_TREND` / `EARNINGS_TREND` (year-over-year growth **and** its acceleration — "is revenue up vs last year, and is that growth speeding up or slowing?") and `MARGIN_TREND` (net-margin latest-quarter vs year-ago). No LLM, no fabrication: if the numbers aren't there the signal is omitted. It is PIT-safe — the computed block is skipped under a backfill (current financials aren't point-in-time), and it makes the tool useful even for a name with no filing feed. Code: `trading_crew/report_signals.py`.
>
> **Structured + persisted.** The whole card (filing + computed) is parsed by `report_signals.parse_signal_card` into a machine-readable `ReportSignals` object with a single confidence-weighted `guidance_bias_score` in `[-1, 1]`. The runner emits it as a `report_signals` event; it's persisted on the `RunRecord` and in the prediction DB (`signals_json`), rendered on the Workflow tab, and surfaced as **advisory context** next to the optional signal-fusion panel. (Fusion itself is left unchanged — it's opt-in + advisory and not yet validated to move sizing, so the guidance bias is shown beside it rather than folded into its math.)
>
> **Big documents aren't truncated.** Chunk size is derived from `REPORT_PROCESS_CONTEXT_TOKENS` (128k or 256k) so a filing is split only when it will not fit the report-process model — not into many small slices. **Map** keeps guidance **and** the broader briefing (reported actuals, order book, MD&A, named risks, operations); **reduce** consolidates that. The **newest annual report** is enough (prior years already sit in its comparatives / MD&A); concalls still keep multiple vintages. The doc set is **de-duped by URL and diversified across kinds**. Tunables: `TRADINGCREW_REPORT_MAX_DOCS` (default 8), `TRADINGCREW_REPORT_MAX_PER_KIND` (default 3, concalls / earnings / credit), `TRADINGCREW_REPORT_MAX_ANNUAL_REPORTS` (default 1), `REPORT_PROCESS_CONTEXT_TOKENS` (chunk size), optional `TRADINGCREW_REPORT_DOC_CHARS` to force a char size (tests).

**Toggles** (sidebar → *Run* section, or `POST /api/sessions` body):

| Toggle | Default | Effect |
| --- | --- | --- |
| `run_mode` | **autonomous** | Stock: LangGraph + autonomous prompts, sequential + deterministic prompts, or harness conductor + replacement remits. Commodity is always sequential/deterministic. Harness is a **long-horizon** observe/dispatch loop (`max_steps` is a ceiling, not a rush signal). Loop / session / planner / desk skills compose from `harness_plugins.yaml`; a spawn planner writes a full child instruction from the live coverage board and that contract leads the child task (the YAML one-liner remit is standing text, not the assignment). Finish is gated on market + news + fundamentals + macro + filings (`get_report_insights`) + a **bundled debate** (`desk=debate`) + a **bundled risk debate** (`desk=risk`: aggressive, conservative, and neutral write opinions then debate). Separate bull/bear or risk-wing spawns remain available after those bundles. Exhausted step budget runs `exhaust_steps` closing conductor turns (no new spawns; finish from the desks that already ran — HOLD when they conflict) and **still returns a PM ticket** with that reasoning, not a canned size-0 stub. Context overflow **compacts** the transcript and retries instead of failing the run. `research_web` prefers Gemini Google Search when `GEMINI_WEB_SEARCH=1`, then SearXNG. |
| `quality_review` | **off** | Advisory thesis audit. When on it runs **async** and feeds the PM (right-sizes the position; never forces a HOLD). |
| `compliance` | **off** | Sanctions / restricted-list / ESG gate. Runs **async** alongside Quality Review; advisory to the PM. |
| `critic_enabled` | **on** | Single-pass reflective self-audit of the final decision (fabricated-source + intent checks). |
| `critic_advisory` | **on** | **Advisory by default:** the critic's audit is surfaced in the UI but it does **not** override the PM's action/size — except on a *hard integrity failure* (compliance BLOCKED, or a fabricated/placeholder source). The critic is a one-directional operator (only ever holds / downgrades / abstains), so overriding it flipped decisions at the last moment and biased the book toward HOLD/SELL. Set `false` to restore the legacy always-override behaviour. |
| `critic_samples` | **1** | Raise to run the multi-temperature consistency vote (each extra sample = 1 extra LLM call). In advisory mode the vote is shown but does not flip the decision. |
| `debate_rounds` | graph / Hermes on autonomous; slider on deterministic; omitted on harness | Round 1 is a parallel independent opening; later rounds are Bull ⇄ Bear rebuttals. Autonomous and harness do not take an operator count. |
| `risk_rounds` | graph / Hermes on autonomous; slider on deterministic; omitted on harness | Size/execution debate (not direction). Round 1 is a parallel independent opening; later rounds are Aggressive → Conservative rebuttals, then Neutral. |

**What runs in parallel.** CrewAI's sequential process drains pending async futures at the next *synchronous* task, so parallelism is exploited where the DAG allows it:

- **Analyst fan-out** — all 4 desks run concurrently (`async_execution`).
- **Risk debate** — Aggressive + Conservative write independent first thoughts in parallel on **round 1**; later rounds are sequential rebuttals (same as bull/bear). Neutral synthesises a book-defensible size. This is a size/execution stress test of the trader's plan, not a second direction vote.
- **Advisory reviewers** — when *both* Quality Review and Compliance are enabled they run as a concurrent async pair right before the PM, so their critical-path cost is `max(QR, compliance)` rather than the sum. They are **not** wired into the trader (they'd block it), which is why they advise the PM instead of gating the pipeline.

**MIXED is not an automatic HOLD.** A true split is bullish count == bearish count, both > 0. Neutrals do not create a split (3 bullish / 2 neutral is a bullish majority). `STANCE: MIXED` is a locked thesis with named evidence on both sides; the PM sits out only on a true split, a majority-opposed directional vote, contradictory evidence, or BLOCKED.

**Fabricated-citation fix.** LLMs sometimes invent an `example.com` URL in their own prose to satisfy the `[source: …]` provenance rule; a downstream reviewer then flags the (otherwise sound) claim as unsupported and downgrades the whole thesis. `sanitize_citations()` runs deterministically at the runner boundary and rewrites any `[source: <placeholder-url>]` tag to the honest `[analyst judgement]` label, leaving real URLs and real tool identifiers untouched.

## Quick start

```bash
git clone <your-fork-url> tradingcrew
cd tradingcrew/agents/crewai/agentic_workflow

python -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env       # fill in VLLM_LLM_* + TAVILY_API_KEY at minimum

./run_web.sh               # binds 0.0.0.0:8001 (foreground)
```

For a **background** server (survives the shell/SSH session, logs to a file):

```bash
./start.sh                 # nohup run_web.sh, PID → .server.pid, logs → logs/server/
PORT=9001 ./start.sh       # custom port/host (HOST=… PORT=… honoured)
./stop.sh                  # graceful SIGTERM (then SIGKILL) using .server.pid
```

`start.sh` is idempotent (won't launch a second instance), waits for uvicorn
to report readiness, and symlinks the active log to `logs/server/current.log`
for later analysis. `run_web.sh`'s Xvfb setup runs inside it, so the same
google-headless websearch path works.

Open <http://localhost:8001> → pick a ticker → **Run analysis**.

For a single-shot CLI run (no UI):

```bash
.venv/bin/python main.py NTNX --debate-rounds 2 --risk-rounds 1
```

See [Detailed design → Quick start](#quick-start-1) below for the full configuration options, every env var, the LLM / embedder presets, the per-run call estimator, and the parallel memory save pool.

### Multi-run sessions (background execution)

Every "Run analysis" click creates a **background session** owned by the server, not the tab. Sessions survive page reloads, tab closes, and network blips — the frontend is a viewer, not the run itself. Practical implications:

* **Queue multiple tickers.** Kick off ZENTEC.NS, then RELIANCE.NS, then TCS.NS — they show up in the sidebar "Sessions" panel and run in parallel up to `TRADINGCREW_MAX_CONCURRENT_RUNS` (default `4`). Anything beyond the cap sits in `queued` and promotes to `running` when a slot frees up.
* **Refresh-safe.** Close and reopen the tab, or hit F5 mid-run — the browser reconnects to the same session by id (persisted in `localStorage`) and replays the full event buffer so the diagram + reports rebuild exactly where they were.
* **Batching = throughput.** With 3-4 concurrent kickoffs against a single vLLM server the decode-batch fills past `1`, roughly tripling aggregate tokens/sec on Qwen3-32B versus a purely sequential queue. `--enable-auto-tool-choice` in `scripts/vllm_serve.sh` is already batch-aware.
* **Cancel any session.** The row-level "Cancel" button targets *that* session's `run_id`; you don't need to attach to it first.

Programmatic Prediction API (same process as the UI, port **8002**). Knobs other
than ticker / debate / risk rounds come from `trading_crew/config/api_run.yaml`.
OpenAPI: <http://localhost:8002/docs>.

```bash
curl -sS http://localhost:8002/api/v1/health
curl -sS -X POST http://localhost:8002/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"NVDA","debate_rounds":2,"risk_rounds":1}'
curl -sS http://localhost:8002/api/v1/runs/<run_id>
curl -sS -X POST http://localhost:8002/api/v1/runs/<run_id>/stop
```

The dashboard still uses `/api/sessions` + `/ws/session/{run_id}` for live tiles.
See the project [README — Prediction API](../README.md#prediction-api).

Env-vars:

| Var | Default | Purpose |
| --- | --- | --- |
| `TRADINGCREW_MAX_CONCURRENT_RUNS` | `4` | How many CrewAI kickoffs may execute at once. Tune to your vLLM decode-batch sweet spot (usually 3-4 on 1×H100 for Qwen3-32B). |
| `TRADINGCREW_MAX_SESSIONS_RETAINED` | `20` | Cap on the in-memory session store; oldest terminal sessions are evicted first. |
| `LLM_MAX_TOKENS` | `0` (uncapped) | Optional per-request generation ceiling. Left **off** by default: capping mid-`<think>` on Qwen3 truncates responses and leaks raw reasoning into the content. Set to a positive int (e.g. `20000`) only as runaway insurance while diagnosing a stuck turn. The proper throughput lever is concurrency (above), not this cap. |
| `LLM_TASK_RETRIES` | `2` | Extra attempts on a **transient upstream gateway failure** (HTTP 502/503/504 — the hosted endpoint wraps its 504 as `InternalServerError: upstream request timeout`) before an agent's task degrades to a placeholder. Only *upstream* gateway errors are retried — a client-side read-timeout is **not** (a fresh attempt would leave a vLLM "zombie request"; see `LLM_MAX_RETRIES=0` in `_common.get_llm`). Set `0` to restore the old degrade-immediately behaviour. |
| `LLM_TASK_RETRY_BACKOFF` | `5` | Base seconds for exponential backoff (`base·2^attempt` + jitter) between the task retries above. |

## Prediction harness (Hermes; Mass is legacy)

Strategy knobs, desk prompts, and graph topology are selected **by Hermes** after Resolve (`trading_crew/agentic/harness.py`).

1. **Generate feedback.** Run predictions, then **Resolve** (Episodes tab or `POST /api/memory/resolve`). The 30-day review always writes deterministic `desk_lessons`. When an LLM is configured it may **investigate** (web, news, filings, forwards — at most three named tools) and write `market_reason`, `unhelpful_tools`, `overemphasized`, and a `suggested_playbook` vote. Prediction desks do **not** retrieve past episodes.
2. **Hermes diagnosis.** Score the 30-day directional book against `desk_strategy.yaml`. On FAILURE with `n ≥ min_reviews_to_change`, open **one** experiment from that descriptive evidence (Mass-style: text names the miss; Hermes picks a bounded graph/prompt action): dominant hurt desk with `recommended_change` or `what_they_got_wrong` → `desk_prompt`; reports/filings blamed → `skip_reports`; `request_evidence` / `research_web` blamed → `evidence_budget_0`; majority of LOSSes on split/MIXED boards → `extra_debate_on_split`; hit-rate above the failure bar but alpha/DD failed → `size_scale_down`; hit-rate failed → `debate_rounds_2`; trader reopen blamed → `no_trader_reopen`; majority `suggested_playbook` vote (must already be a YAML/`PLAYBOOK_ACTIONS` id) → that knob; else the next unused YAML catalog knob.
3. **Apply.** `apply_harness` prepends strategy + the open experiment (desk addendum only on a `desk_prompt`). A playbook experiment is live topology via `resolve_playbook_action` (debate-round, evidence-budget, skip-reports, no-reopen, and risk-round knobs). Mass promote is not required. **Settings** (`/settings`) edits the live strategy file, overlay agent goal/backstory, and the bulk run snapshot. Training → Hermes is the 30-day scoreboard; Resolve opens the next knob.
4. **Legacy Mass.** 1PO/2TO/3PO and `scripts/promote_graph.py` remain in the tree for later A/B. They are not the live training step. An existing `promoted.json` still applies.

**How it tests.** Hermes is not an LLM search and not a value sweep. After Resolve it scores the 30-day book, then opens **one** named knob with **one** fixed value from the diagnosis rules or the YAML catalog (`debate_rounds_2` is exactly two rounds; `size_scale_down` is exactly 0.75). The next live predictions use that setting. If the book is still FAILURE at `n ≥ min_reviews_to_change`, the knob is rejected and the next unused one can open. The LLM writes descriptive lessons; it does not choose or score knobs. A `suggested_playbook` is a vote, not an install.

Hermes and L3 share the Training tab. One knob at a time on the live book (`one_variable_at_a_time`). Do not replay the same historical tape through every knob offline. L4 PPO is not on the live prediction path. There is no RLMF few-shot prefix or LoRA serve path.

## Trust & point-in-time backfill (Phase 0 / 1)

Two foundations that let the desk generate **trustworthy** training labels — including offline, as-of-a-past-date runs — without leaking the future.

### Phase 0 — workflow-version stamping + cohort filtering

Every episode is stamped at write time with its provenance so a drifting/experimental workflow's proposals can't silently pollute evaluation:

- `workflow_version` (`{git8}-{prompts8}`), `prompts_hash` (SHA-256 of `config/agents.yaml` + `config/tasks.yaml`), `code_git_sha`, and `source` (`live` | `forward` | `backfill`) — see `trading_crew/agentic/versioning.py`. `forward` is the trustworthy at-scale cohort written by `python -m trading_crew.agentic.scale collect` (no PIT leakage — see below); `backfill` is historical and PIT-guarded but only as clean as the least-PIT-aware tool.
- Backtest / grid-search / RLMF can filter to a single cohort. The `/api/backtest` and `/api/grid_search` endpoints accept `workflow_version=` and `source=` query params; `rlmf.build_dataset(...)` accepts the same. A concrete cohort value **excludes unstamped legacy episodes** (that's the point — untrusted history is quarantined, not trusted by default).

### Phase 1 — point-in-time (PIT) data layer

`trading_crew/data_sources/pit.py` provides an `AsOfContext` (via `contextvars`) so a decision dated *T* sees only information public on or before *T*:

```python
from trading_crew.data_sources import pit
with pit.as_of_scope(as_of_date, mode="backfill", strict=True):
    run_crew(...)   # every data tool inside is gated to as_of
```

- **Fail-closed leakage guard** — `assert_before_cutoff` / `filter_before_cutoff` drop (or raise on) any datum whose publish date is after the cutoff; in strict backfill an *unknown* date is also rejected. Live mode (`as_of=None`) makes every guard a no-op, so normal runs are unchanged.
- **OHLCV truncation** — `_load_price_history` (crew tools) and `charts._fetch_ohlcv` clamp to the cutoff in backfill.
- **News** (`get_news` / `get_global_news`) — merged across **Tavily + SerpAPI + Firecrawl**, PIT-filtered by publish date. Tavily is additionally bounded server-side via `end_date`.
- **Fundamentals** — the *restated* screener.in / yfinance snapshot is **withheld in backfill** (it reflects today's restated figures, not the as-of view). PIT-safe **dated filings/announcements** remain (`get_corporate_actions`), filtered to `filing_date ≤ as_of`.
- **screener.in snapshot cache + politeness** — each company page is captured once to an on-disk snapshot (`~/.trading_crew/pit_snapshots/`) and reused across every backfill cell for that symbol; requests are rate-limited (`SCREENER_MIN_REQUEST_GAP_SECONDS`, jittered) with automatic 429 backoff.

### Multi-provider news key rotation

Each news provider accepts **multiple legitimately-held keys** in round-robin rotation to spread quota. Set the plural env var (comma-separated) or the singular fallback:

- `TAVILY_API_KEYS` / `TAVILY_API_KEY`
- `SERPAPI_API_KEYS` / `SERPAPI_API_KEY`
- `FIRECRAWL_API_KEYS` / `FIRECRAWL_API_KEY`

Rotation reads env fresh each call. On a per-call failure (rate-limit / invalid key) **all three providers roll over through the remaining keys** in round-robin order before giving up (`news_providers.ordered_keys` + the Tavily/SerpAPI builders in `tools.py`).

### News tiers — custom crawl (free) first, then the paid APIs

News / global / geopolitical / supply-chain headlines (`_web_news` in `tools.py`) are fetched in **two clearly-separated tiers**, and by default the **free tier is tried first** to conserve limited Tavily/SerpAPI/Firecrawl credits:

1. **Custom crawl (FREE)** — the `search → crawl → LLM-summarize` pipeline below: free link search + **keyless HTTP crawl** (no Firecrawl) + one LLM summarize. No paid credits.
2. **News APIs (PAID)** — a fallback SEQUENCE **Tavily → SerpAPI → Firecrawl** (each with its own key rotation), only reached when the free tier comes back empty. Rows are PIT-filtered (`published <= as_of`, fail-closed on unknown dates) and de-duplicated within the winning provider, and carry publish dates (the strong PIT source for backfill).

`TRADINGCREW_PREFER_CUSTOM_CRAWL` (default `1`) controls the order; set it to `0` to restore the old API-first order (better publish-date PIT for historical backfill). `get_corporate_actions` **also honours this toggle** — at scale its per-theme searches (buyback / dividend / M&A / board changes / …) would otherwise burn API quota, so custom-first it does a free keyless link search per theme and only falls through to the paid APIs when the free tier is empty (the free rows carry no publish date, so they are fail-closed in strict backfill and kept in live mode). Empirically the free custom search returns **more entity-targeted links than Tavily** (which is also easy to exhaust), so custom-first rarely degrades quality for live runs.

### Open-web research pipeline (`research_web` + news fallback)

The **search → crawl → LLM-summarize** pipeline (`trading_crew/data_sources/web_research.py`) is both the News desk's `research_web` tool and the final step of the news fallback sequence above:

1. **Search for links** — *free backends only, no paid search API*: the `websearch.sh` script **vendored in-repo at `scripts/websearch/`** (the default `WEBSEARCH_SH_PATH`; it chains google-headless → searxng → google-cse → brave → mojeek → ddg) followed by inline keyless scrapers (DuckDuckGo / Brave / Mojeek). Override the order with `WEBSEARCH_BACKENDS`. The script and its `playwright-google/` helper are copied into the repo so the tool has no external-folder dependency.

   > **Which free backend actually scales — use SearXNG.** The keyless HTML scrapers (`brave`, `mojeek`, `ddg`) hit the public search pages directly and are bot-throttled: brave soft-blocks (429 challenge) after ~7 rapid queries, mojeek 403s after ~5, ddg after ~2–5. They're fine for the odd query but collapse under a 200-stock/day campaign (each name fires several theme queries). `google-headless` needs Chromium **and** clears a CAPTCHA on datacenter IPs only via a one-time interactive `PW_HEADLESS=0` solve; `google-cse` needs API keys and is capped at 100/day. **The one free backend that scales is a self-hosted [SearXNG](https://docs.searxng.org/)** — it fans each query out to ~70 upstream engines so no single upstream ever sees enough of your traffic to rate-limit. A local instance is running here (Docker, `~/searxng`, port **18080**, JSON format enabled + limiter off, `--restart unless-stopped`); point the tools at it with `SEARXNG_URL=http://localhost:18080` in `.env`. With it up, `websearch.sh auto` returns SearXNG results right after google-headless and never has to reach the throttled scrapers. One-time setup instructions: `bash scripts/websearch/websearch.sh --setup-searxng`.

   > **One node per query — a unified circular ring with retry (default).** The original design *fanned* every query out to all of `SEARXNG_ENGINES` at once and merged/deduped the results ("don't lose info"). That maximised recall but meant **every upstream saw 100% of the batch's traffic**, so under a 200-stock campaign `brave`/`duckduckgo`/`yahoo` 429-throttled almost continuously (a real run showed brave 257/258 throttled, ddg 252/258, yahoo 146/258, google 258/258 empty — only `bing`/`startpage` survived). The fix (`TRADINGCREW_SEARXNG_SINGLE_ENGINE=1`, default): build **one flat circular ring** of *nodes* — `searxng:<engine>` for each entry of `SEARXNG_ENGINES`, plus `google` (the real-browser google-headless scraper) appended last when it's enabled — and serve **exactly one node per query**, popping it off the head and pushing it to the tail on every call. So N nodes split the load and each sees only **~1/N** of the traffic. The default engine mix puts the engines that actually returned on-topic equity links (`duckduckgo web`, `privacywall`, `infospace`, `abcnyheter`, `bing news`, `google cse`, `duckduckgo news`, `gmx`) ahead of the older `bing` / `startpage` / `google` / `yahoo` nodes. **A SearXNG hop is kept only when the workflow LLM marks those titles/URLs as about the query** (`links_relevant_to_query`). An off-topic dump (same-site dealer nav, weather tags, …) is treated as empty: the ring pays the same search wait again and retries the next node. Throttled or empty nodes also rotate. There is still **no fan-out** (one node contacted at a time, first *relevant* success wins). Combined with the Kafka-paced request stream (bounded concurrency + launch stagger, below) this keeps every node well under its per-IP rate limit. `_next_provider()` is thread-safe, so K concurrent workers draw K *different* nodes off the ring. Knobs: `SEARXNG_ENGINES` is the engine pool; `TRADINGCREW_SEARCH_RING` overrides the whole ring verbatim; `TRADINGCREW_SEARCH_MAX_ATTEMPTS` caps retries per query (0 = one full rotation). Set `TRADINGCREW_SEARXNG_SINGLE_ENGINE=0` to restore the legacy fan-out-and-merge behaviour.

   > **`google-headless` is folded into the same ring.** With the fan-out gone, Google (the best ranking, the one path SearXNG can't replicate) is worth re-enabling: as one node in the ring it serves only ~1/N of queries, is serialized machine-wide (one Chromium at a time) and cools down 30 min on a CAPTCHA — so its volume is far below what got it CAPTCHA-locked before. If its profile isn't primed it fails soft (exit 4 → `_SearchRateLimited` → cooldown → skipped) and the ring simply retries the `searxng:*` nodes, so nothing breaks. Enable/disable with `TRADINGCREW_GOOGLE_HEADLESS_ENABLED` (needs Xvfb `:99` + a one-time `GH_INTERACTIVE=1` profile prime).
2. **Crawl the raw page** — keyless `requests`+BeautifulSoup extractor (`crawler="http"`, the default for the news tier so it spends no credits); Firecrawl `/v1/scrape` (`crawler="firecrawl"`/`"auto"`) remains available for the standalone `research_web` tool and big-PDF report crawling. To resist the few bot-protected sites (e.g. business-standard 403s the keyless crawler), the pipeline **over-fetches candidates and crawls down the list until it has the target number of successful crawls**, so a couple of blocked pages don't shrink the source set.
3. **LLM-summarize** — the workflow LLM consolidates the crawled text into a concise briefing where **every claim carries an inline `[source: <URL>]` tag**.

**Point-in-time handling for websearch.** The web search itself is date-less, so PIT-awareness is added in two places when an `AsOfContext` cutoff is active:

- **Search stage** — the query is bounded with a Google-style `before:<cutoff+1d>` operator (`_pit_bounded_query`) so only results published on/before the decision date are returned (honoured by google-headless; ignored harmlessly by engines that don't parse it).
- **Summarize stage** — the summarizer is told to treat the cutoff as "today" and to ignore anything dated after it.

*Residual caveat:* a crawled page's **body** is still fetched live and could contain post-decision edits (e.g. "last updated" articles / wikis), which cannot be fully eliminated for arbitrary pages. So for strict training labels the dated news APIs (Tavily `end_date`, SerpAPI / Firecrawl publish dates) remain the primary PIT source and the websearch pipeline is the last-resort supplement. The standalone `research_web` analyst tool (free-form queries) additionally **fails closed** — returns a "skipped in point-in-time mode" note — under a backfill scope.

### Phase 2 — offline decision-label backfill

Generate labeled decisions *now* instead of waiting months for live ones to resolve. The harness (`trading_crew/agentic/backfill/`) enumerates `(ticker, as_of_date)` cells over a window, runs the full crew for each under a strict PIT scope, records a `source="backfill"` episode, and resolves the outcome immediately (the horizon is already in the past) — producing an alpha-labeled row for L2/L5.

- **Resumable** — a deterministic job id (hash of tickers/window/spacing/horizon) keys a checkpoint at `~/.trading_crew/backfill/state/<job_id>.json`; `done` cells are skipped on resume and episodes upsert by id, so **Ctrl-C is safe and re-running continues**.
- **Rate-limited** — `--cell-gap-seconds` between cells on top of the screener throttle.
- **Fast LLM** — uses the `VLLM_LLM_*` endpoint from `.env` by default.
- **Leak-safe** — live desks do not retrieve past episodes. Resolve investigation tools are skipped under a backfill PIT scope (`pit.is_backfill()`); resolution itself runs *outside* that scope so it can legitimately see the post-decision tape.
- **Dated screener surfaces** — announcements, **annual reports, concall transcripts, and credit-rating actions** are categorized and PIT-filtered (`screener.render_documents_report`), giving the analysts point-in-time-safe primary sources even while the restated ratio snapshot is withheld.

CLI (resumable):

```bash
scripts/backfill.sh --tickers RELIANCE.NS,TCS.NS --start 2023-01-01 --end 2024-06-01 \
    --spacing-days 30 --horizon-days 90 --cell-gap-seconds 8
```

Or drive it from the API as a managed background job: `POST /api/backfill/start`, `POST /api/backfill/stop`, `GET /api/backfill/status`.

### Phase 2b — forward collection at scale (the RL data engine)

Historical backfill is only as clean as the least-PIT-aware tool (e.g. `get_event_proximity` returns *today's* forward earnings calendar even under an as-of-2023 scope), so its labels aren't fully trustworthy for RL. The durable path is **forward collection**: run the crew *live* on a broad universe every day and let the decisions resolve as their horizons elapse. `trading_crew/agentic/scale.py` is the headless driver (no web server needed):

```bash
# 1) COLLECT — run the crew live over a universe today; record source="forward" PENDING episodes
python -m trading_crew.agentic.scale collect --universe universe.txt        # one ticker per line
python -m trading_crew.agentic.scale collect --tickers RELIANCE.NS,AAPL --workers 2

# 2) RESOLVE — daily sweep; prices every episode whose horizon elapsed into an alpha-labelled RL row
python -m trading_crew.agentic.scale resolve --reflect

# 3) STATUS — track progress toward the RL training goal
python -m trading_crew.agentic.scale status --goal 20000
```

- **Clean cohort** — unlike the backfill harness it runs with **no `as_of` scope** (nothing to leak today) and stamps `source="forward"`, so you can train exclusively on the trustworthy cohort: `rlmf.build_dataset(source="forward")`, or `source=forward` on the backtest / grid-search endpoints.
- **Resumable & idempotent** — episode ids are date-based (`TICKER-YYYY-MM-DD`, one decision per name per day) and a per-day checkpoint in the DB (`prediction.db` → `collect_state` table, keyed by date — replaces the old `~/.trading_crew/forward/collect-<day>.json` file) skips names already done, so Ctrl-C and re-run continues; `--workers N` parallelises crew runs (mind LLM/web rate limits), `--gap-seconds` throttles the serial path.
- **The 200×100 plan** — collect ~200 names/day and run `resolve` daily; short-horizon calls become labelled within weeks while medium/long ones mature over the campaign. The live trainers that consume that tape are **Hermes** (one experiment from Resolve evidence) and **L3** grid/PSO (logged proposals + prices). L4 PPO stays an offline library only. Cron it: `scale collect` in the morning, `scale resolve --reflect` at night.

#### Daily 200-stock runner (dashboard-driven)

Forward collection is **triggered from the web dashboard's Prediction tab** — the batch card's **Start / Pause / Resume / Kill** buttons (see `scale_runner.py` below). The batch runs the crew LIVE over the curated **200-name universe** (100 Indian `.NS` + 100 US, sector-diversified) shipped in `universe/` (`in100.txt`, `us100.txt`, `all200.txt`):

- takes a **single-flight `flock`** on `~/.trading_crew/forward/collect.lock` so two dashboard collectors never collect at once and race the episode store;
- runs `collect` at the chosen worker count — at the observed ~10-12 min/name, `workers=6` finishes the 200 in **~4-5h** (the run is resumable per-day, so a slow day just continues on the next run);
- **logs everything** under `logs/scale/<UTC-date>/`: `collect.jsonl` (JSONL progress), `run.log` (crew `logging` — search-backend / screener / login lines), `report.md` + `report.json`.

There is **no cron**: collection happens only when you hit the button on the dashboard. Run `scripts/scale_report.py --day <UTC-date>` after a batch to generate the post-run report.

**Search at scale.** Open-web research now leads with a first-class **SearXNG** backend (`web_research.py`, `SEARXNG_URL`), and the `websearch.sh` fallback is called with `--disable google-headless` — so a 200-stock batch never stalls on Google soft-CAPTCHAs or the keyless-scraper soft-blocks.

#### Memory hygiene at scale (RSS creep → OOM)

Everything runs in **one Python process** (FastAPI + the Kafka workflow consumers + up to `WORKERS` crews for the web path; the `scale collect` CLI), so a multi-hour 200-name batch is unforgiving about per-run leaks — an unbounded creep OOM-kills the VM (redpanda + the server die; the tell-tale is a redpanda `NodeNotReadyError` reconnect storm with no in-app traceback, i.e. an external SIGKILL). The root cause was **per-run CrewAI `Memory` not being released**: `crew.py` builds a fresh LanceDB-backed `Memory` *and* swaps in an 8-worker `memory-save` `ThreadPoolExecutor` for every run, and neither was torn down — so each name leaked ~8 live threads (each pinning its submitted analyze work + results) plus a LanceDB handle. Over 200 names × `WORKERS=6` that's ~1600 leaked threads and steady RSS growth. The fixes:

- **Release the memory per run.** `memutil.close_crew_memory(crew)` shuts the `memory-save` pool (`cancel_futures`) and calls `Memory.close()` (releases the LanceDB handle); it's called in the `finally` of both `scale._collect_one` (collect) and `runner._run_blocking` (web), followed by `memutil.release_memory()` (`gc.collect()` + glibc `malloc_trim(0)`) to hand freed pages back to the OS. Validated: after a completed run the process drops from ~86 threads to **13** and RSS settles into a flat ~1 GB band across runs instead of the previous monotonic ~1.5 GB/min climb. Note the LanceDB **tokio** runtime (~45 worker threads) + embedding model are a **one-time ~0.7 GB** resident cost — created once and reused, *not* per-run, so they're expected baseline, not a leak.
- **Cap heap fragmentation.** `run_web.sh` exports `MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_`/`MALLOC_MMAP_THRESHOLD_=131072` so glibc's per-thread arenas can't fragment the heap under the crews' thread pools (overridable from the environment).
- **Bound the tool caches.** `tools._tool_cache` (TTL-only before) now has a hard entry cap + purge-expired-on-write (`TRADINGCREW_TOOL_CACHE_MAX_ENTRIES`, default 512), and `screener_source._CACHE` got the same treatment — so a 200-symbol batch can't accumulate every large tool result forever.
- **Diagnostic.** `scripts/mem_leak_probe.py TICKER…` runs the exact per-name path sequentially under `tracemalloc` and reports RSS + thread count + top allocation sites after each run — the tool used to pin the leak (LanceDB `_lancedb.abi3.so` → tokio/jemalloc threads) when `py-spy`/`dmesg` needed root.

#### Post-run analysis (`scripts/scale_report.py`)

The report (printed + `report.json`) answers, per run: **did it scale?** (done/failed/skipped, per-name avg/median/p95 latency, wall clock, runs/hour); **what failed?** (per-error-class histogram + failed-name list); **are predictions sane / unbiased?** (BUY/SELL/HOLD distribution overall, by market, and by sector, with directional-bias and sector-unanimity flags, plus conviction + size distributions); **is scraping healthy?** (screener 429 / login-fail / search-backend-failure counts parsed from `run.log`). Run it standalone for any day: `python scripts/scale_report.py --day 2026-07-20`.

#### Session tracing + per-session search telemetry (no log-grepping)

Every run carries a **`trace_id`** that groups all its work for analysis (`trading_crew/observability/trace.py`). A batch tags *all* its tickers with the stream's `job_id`; a standalone interactive run traces to its own `run_id`. The id is a `contextvars.ContextVar` set once in `SessionManager._drive` and it rides down for free — `asyncio.create_task` / `asyncio.to_thread` copy the context, and CrewAI's `Task.execute_async` runs each parallel analyst via `contextvars.copy_context().run(...)`, so a web-search made deep inside a parallel analyst tool still reports the right session (no process-global, no fallback).

- **Logs grouped by session.** The root formatter prints `[%(trace_id)s]` on every line (`web/backend/logging_setup.py` + `TraceLogFilter`). Group-by is a plain filter: `rg "\[scale-2026-…\]" logs/server/current.log`.
- **Web-search counters in the DB.** `search_links` records each call's outcome — `success` / `empty` / `throttled` / `error` — per **backend** (google / searxng / yahoo / …) against the active `trace_id`, in the `search_stats` table of `prediction.db` (`bump_search_stat` / `get_search_stats` / `list_search_trace_ids`). It's a no-op outside a session and best-effort (telemetry never breaks a search).
- **Dashboard + API.** The **Prediction** tab's batch-status card shows a live **"Web search backends"** table (success/empty/throttled/error/total per backend) for the running session. The table scrolls (`max-h-60`) with a sticky header and total row so the longer SearXNG engine mix stays readable. API: `GET /api/search_stats` (recent sessions) and `GET /api/search_stats/{trace_id}` (per-backend breakdown).

#### Tuning google-headless — the concurrency + break experiment

The best-ranked backend (`google`) was under-performing in batch (≈18/105 useful calls). A dedicated experiment (`scripts/experiments/google_search_experiment.py`, which drives the real driver and sweeps `--gap`/`--jitter`/`--concurrency`, logging every attempt to a JSONL) isolated why:

- **Concurrency, not rate, is the dominant failure.** All batch workers share **one** Chromium profile dir (`~/.cache/playwright-google`); Chromium's SingletonLock lets only one process own it, so concurrent launches collide and all-but-one exit 1. Measured: **4 concurrent → 17% success** (10/12 `launch_error`) vs **serialized → 83%**.
- **Micro-gaps between queries don't buy CAPTCHA relief.** CAPTCHAs track *cumulative volume per rolling window*, not the per-query gap — bigger gaps just add latency while the IP still trips. Worse, once this datacenter IP crossed Google's budget it stayed CAPTCHA-locked for a **long** time: a 5-minute break didn't recover it, and light probing every 2 min for 16 min never recovered (probing itself resets the clock).

**Chosen config (implemented in `web_research.py`):**
- **Serialize google machine-wide** — an in-process lock **plus** a POSIX file lock (`~/.cache/playwright-google.singleton.lock`) around the subprocess, so at most one Chromium touches the profile at a time (web-server included); HTTP backends (searxng/yahoo) stay concurrent. Knobs: `TRADINGCREW_GOOGLE_HEADLESS_SERIALIZE=1` (default), `GOOGLE_HEADLESS_LOCK_WAIT_SECONDS=60` (fall through to the next backend if the lock can't be taken in time), `GOOGLE_HEADLESS_LOCKFILE`.
- **Long CAPTCHA cooldown** — because recovery is slow, a google CAPTCHA cools it down for `GOOGLE_HEADLESS_COOLDOWN_SECONDS=1800` (30 min, via the per-backend `_BACKEND_COOLDOWN_OVERRIDE`) instead of the generic 15 min, so we stop hammering it and let the IP recover. **SearXNG remains the workhorse** and google is used as a serialized minority in the rotation.
- **Yahoo 5xx/429/999 → throttled.** `_search_yahoo` now maps those HTTP statuses to `_SearchRateLimited`, so Yahoo's soft-rate-limit (mostly `500`) is counted as `throttled` and cooled down, not retried as a hard `error`.

#### Batch status is "running" only while work is in flight

The Prediction tab's batch card (and its Pause/Kill buttons + the search-backends panel) is driven by `GET /api/scale/status`. A pub-sub stream has no hard "batch complete" event, so status is **derived**: it reads `running` only while a name is mid-run (`active_runs`) or work is still pending (`processed < total`); once drained it reads `completed` (something ran) or `idle` (nothing did) — never a stuck "running" (`scale_runner._ScaleJob.derived_status`). This is independent of `total` alone, so an inconsistent old in-memory job (e.g. `total=0, processed>0`) no longer shows a phantom **Pause** button. Additionally, a **previous-day** job that has drained is not surfaced as *today's* collection — `get_scale_status` reports `idle` for it (job_id `null`), which also clears the stale per-session search-backends panel — while a batch still genuinely running across midnight is preserved.

#### Run it from the dashboard — hierarchical multi-page UI

The UI is a **hierarchical, URL-driven** app served from one Alpine SPA shell (`web/frontend/index.html`):

- **`/`** — landing chooser (`homepage.html`): just **Stock Trading** → `/stocks` and **Commodity Trading** → `/commodity`.
- **`/stocks`** — the **live** stock dashboard: the prediction universe + global tabs (**Prediction · Training · Episodes**) with the History **left pane**. Header **Home** returns to `/`; the **gear** opens Settings. (Renamed from `/predict`; `/predict`, `/stock`, and `/predict/history/{date}` **307-redirect** here / to `/stocks/history/{date}` for back-compat.)
- **`/stocks/history/{date}`** — a **prediction-run** review (not the Prediction tab): **Names** (stocks that ran that day) and **Portfolio** (the ₹10L virtual book for that date). Click a name (or a holding) to open `/run/{ticker}?date={date}`. **← Stocks** returns to `/stocks`.
- **`/commodity`** — the **high-level** commodity dashboard, mirroring `/stocks`: a **Contracts** picker (the futures catalog grouped by energy/metals/grains/softs/livestock) + **Training · Episodes** global tabs, with the left pane. Header **Home** returns to `/`; the **gear** opens Settings. (Prediction / Names / Portfolio are stock-only — commodity has no batch forward-collection or INR virtual book — so they're hidden here.)
- **`/settings`** — operator console (`?book=commodity` for futures). Header **← Stocks / ← Commodities** returns to `/stocks` or `/commodity`.
- **`/run/{ticker}`** and **`/commodity/run/{contract}`** — the **focused per-instrument** workbench: instrument tabs only (**Workflow · Charts · Backtest**, + **Futures** for commodities), **no left pane**, and a **← Stocks / ← Commodities** back button plus the header gear. Opening a tile (or a custom Settings run) pushes this URL; browser back returns to the high-level page. To keep this page's first paint lean it skips the universe/history/enrichment loads (they lazy-load on Back). The left-pane collapse toggle lives at the **bottom-left** of the sidebar. (`{contract}` uses a `:path` converter since futures symbols contain `=`, e.g. `CL=F`.)

The dashboard is backed by a self-contained SQLite DB at `$TRADINGCREW_CACHE_DIR/prediction.db` (`web/backend/prediction_db.py`). That DB is now the single source of truth for **all** app state — one file, multiple tables: the prediction universe/results (`companies`/`categories`/`predictions`/`run_days`), the per-day `collect_state` counter, the full UI **run history** (`runs`, replacing `runs/*.json` + `index.jsonl`), the point-in-time scrape cache (`pit_snapshots`, replacing `pit_snapshots/*/*.json`), **episodic memory** (`episodes`, replacing `memory/episodes.jsonl`), and the **per-session web-search counters** (`search_stats`, keyed by `trace_id`). Each legacy filesystem store is imported once on first access and its old file/tree renamed `*.migrated`(`-<ts>`). The layout:

- **Left sidebar (`/stocks` only) — History.** Live sessions (queued / running only) share the pane evenly with **Prediction runs** when anything is in flight. **Live · today** returns to `/stocks` (today's Prediction index). **Review names →** (when anything has completed today) and each **past** date open that day's prediction-run review (`/stocks/history/{date}`: Names + Portfolio). Today's date is not duplicated in the dated list. Clicking a live session opens `/run/{ticker}` and streams that kickoff — it does not replay a cached completed HOLD on top of a still-running job. Settings is the header gear, not this pane. The pane is **hidden entirely on `/run/{ticker}`**, and its collapse toggle sits at the **bottom-left**.
- **Tabs by page.** `/stocks` shows **Prediction** (default) · **Training** · **Episodes**. `/stocks/history/{date}` shows **Names** (default) · **Portfolio**. Commodity `/commodity` shows **Contracts** (default) · **Training** · **Episodes**. The focused `/run` page shows **only** the instrument tabs — **Workflow · Charts · Backtest** (+ **Futures** for commodities) — preceded by **← {date}** when you came from a prediction run, otherwise **← Stocks / ← Commodities**. Tabs carry a `scope` (`global`/`stock`), an optional `asset` (`stock`/`commodity`), and an optional `desk` (`live`/`review`) so both dashboards drive off the same array.
- **Decision card.** The standalone **Sources** list at the bottom of the final-decision card was removed — every quantitative claim in the rationale/drivers/risks already carries an inline `[source: <id>]` citation (rendered as a colored link/badge), so the deduplicated list was redundant. The PM still emits `sources` for the Reflective Critic's evidence check.
- **Prediction tab.** **Bulk | Single**, same Settings snapshot. The 100 IN + 100 US names (seeded from `universe/in100.txt` / `us100.txt`) render as a two-market grid of collapsible **sector** cards. Expand a sector to see **company tiles** — each shows the resolved **company name** with **hover links** to the homepage (`⌂ site`) and Wikipedia (`W`), resolved lazily from Yahoo Finance (`company_meta.py`) and cached in the DB. The **market selector** (All / India / US) filters both the grid and the batch scope. **Search-to-add** resolves a query to real, runnable tickers — with an **India** mode that constructs and validates the NSE listing (`infosys → INFY.NS`, `reliance → RELIANCE.NS`) since Yahoo otherwise returns the US ADR / Frankfurt line and appends `.NS` automatically; `✕` on a tile removes it.
- **Same-day cache.** A completed **single** or **bulk** name is written to the Prediction index for the local calendar **today** (`predictions` + `collect_state` + `run_days`). **Live · today** is that bucket; the dated sidebar list is past days only. The next Single or Bulk attempt on that name returns the cached row (no LLM) unless **Force rerun** is checked. A `failed` cell is not a cache hit. Commodity sessions never use this index.
- **Batch run + tile colours.** Start / Pause / Resume / Kill the collect over the visible universe with a worker count (`scale_runner.py`, single-flight `flock`). The UI batch drives every name through the **same `SessionManager` that powers a manual "Run analysis"** — so each in-flight name is a real, subscribable session: clicking an **orange** tile opens its **Workflow diagram and streams it live**, node-by-node (green = completed, no colour = not yet executed, `⚠` = failed). Batch episodes are tagged `source="forward"` for the RL cohort, and each completed name is persisted to `predictions` (action / conviction / size / expected-return + full decision JSON + rendered report). Default `workers=5`, launches staggered ~2 s apart; overall LLM concurrency is `min(workers, TRADINGCREW_MAX_CONCURRENT_RUNS)` (default cap 4) — raise the cap to run more in parallel, but a single 17-agent crew is ~12 min, so over-parallelising one LLM endpoint just triggers gateway timeouts.
- **Open a name.** Clicking a live tile opens `/run/{ticker}` (Workflow + stored decision). Clicking a name on a prediction-run page opens `/run/{ticker}?date={date}` and replays **that day's** workflow; Back returns to the run.

API surface (all under `/api/predict/*`): `universe`, `search` (`?market=IN|US`), `company` (POST add / DELETE remove), `enrich`, `prediction/{ticker}`, `history`, `history/{date}`; the batch reuses `POST /api/scale/{start,pause,resume,stop}` + `GET /api/scale/status`. **Pause** stops cooperatively (in-flight names finish, then the pool drains; every completed name is checkpointed) and **Resume** starts a fresh pass that skips the `done` names; **Kill** is terminal. Because the lock is single-flight, only one collect runs at a time.

### Phase 3 — harness feedback + PnL attribution

**Prediction harness (one loop).** After each episode resolves, the tape is labelled WIN/LOSS vs the predicted side and Hermes opens one experiment from that evidence (`harness.step_harness` / `harness.propose_experiment`). Resolve is how you generate that feedback. There is no separate per-ticker L2 backstory prepend. Mass ingest is legacy and not part of this step.

**Backtest attribution (decision vs sizing vs cost).** A Sharpe tells you *whether* a strategy lost, not *why*. `trading_crew/agentic/backtest/attribution.py` decomposes each decision's realised PnL against a unit reference exposure:

- **decision** = `direction × ref_weight × asset_return` — was the directional call right? (analysts/PM)
- **sizing** = `gross − decision` — right call, wrong size? (M5 risk gate / L3 config)
- **cost** = `−|sized_weight| × round_trip_cost` — from the real `CostModel` (fees + half-spread)

The `/api/backtest` walk-forward response now carries an `attribution` block with per-leg totals, hit rate, and a one-line `diagnosis` telling the desk which layer to fix.

### Phase 4 — live signal fusion

`trading_crew/agentic/signals/fusion.py` fuses the desk's four independent readings of a name into one guardrailed prediction:

- **LLM decision** (required primary) — direction + conviction from the crew. If the LLM abstains/holds, fusion does **not** open a trade on the quant signals alone (`require_llm`).
- **L4 RL policy** — **not assembled live.** Fusion still accepts a `policy=` argument in unit tests; `assemble_fusion_inputs` always sets `policy=None`.
- **L3 tuned expectancy** — a *confidence gate*: no positive per-trade edge ⇒ size is dampened. By default this is the **PSO-tuned** M5 config's per-trade expectancy: `signals.pso_tuned_expectancy` runs a particle-swarm search (`swarm/pso.run_swarm_search`) over sizing/gate params against the ticker's logged proposals and reads the winning config's `expectancy` — a stronger edge estimate than a raw mean of realised returns.
- **Crowd positioning** — a *contrarian fade*: siding with an already-extreme crowd trims size; being contrarian does not.

Guardrails fire explicitly (each returned in `guardrails`): LLM/policy disagreement halves confidence, an edgeless L3 config dampens size, a crowded-aligned trade is faded, and a sub-threshold fused confidence stands aside (`ABSTAIN`). Target weight is bounded by `max_position_weight`.

Fusion runs as an **optional in-workflow step** — enable **Signal fusion (advisory)** in the run options and the workflow fuses *this run's* decision with the L3 edge / crowd view after the PM decides, surfacing a guardrailed `FusedSignal` card on the **Workflow** tab (advisory only — it never changes the PM's action/size). The inline step uses the fast `proxy` expectancy so it stays lightweight. The same assembly is also exposed as a live endpoint: `GET /api/signals/fusion?ticker=...&include_crowd=false&expectancy_source=pso` returns the fused signal, the sources used, per-source contributions, and the guardrails that fired; `expectancy_source` selects the L3 edge: `pso` (swarm-tuned, default), `proxy` (mean resolved-episode edge), or `none` (disable the gate). The fusion weights/thresholds are hyperparameters on `FusionConfig`.

### Phase 5 — trustworthy prediction: canonical units, fractional-fill backtests & a one-command validation harness

Phase 5 makes the *whole* stack runnable and measurable as one unit, and fixes the two silent defects that made backfilled predictions un-trustworthy.

**Canonical `expected_return_pct` convention (units + sign).** The field was consumed as a **decimal** (`0.05 = +5%`) by the M5 Kelly sizer and the M7 allocator, but produced as **percent points** (`×100`) by `risk/stops.py` and passed through the M1 bridge unconverted — so a `+20%` thesis landed in a decimal field as `20.0` (= 2000%), and shorts sometimes carried a *positive* expected return. Kelly and mean-variance inputs were quietly corrupted. Phase 5 makes **decimal the single convention everywhere** (documented in the `PortfolioDecision` schema and the PM prompt; `stops.py` now emits `(target−entry)/entry` without the `×100`; the frontend scales for display), and the bridge enforces **sign-consistency** via one helper (`_signed_expected_return`): a BUY carries `+|er|`, a SELL `−|er|`, HOLD/ABSTAIN `0`.

**Fractional-share backtests.** `proposal_to_order(..., allow_fractional=False)` rounded every order to whole shares and dropped any sub-1-share delta. On high-priced names a valid 1–2 % target weight rounded to zero, so an entire book of small positions collapsed to `NO_DELTA` and the backtest reported an all-zero equity curve — un-measurable, not flat. The M6 engine now defaults to `allow_fractional_shares=True` (live order compilation is unchanged), so small-but-valid decisions actually express their exposure.

**Reusable evaluation service + validation harness.** The `episodes → proposals → OHLCV → backtest → attribution → fusion` orchestration used to live *inside* the FastAPI handlers (and was re-implemented in throwaway scripts). `trading_crew/agentic/evaluation.py` extracts it into one library path (`load_cohort`, `proposals_from_episodes`, `evaluate_ticker`, `latest_fused_signal`; OHLCV provider is injectable) that both the web app (`/api/backtest` now calls it) and a CLI share. Run the whole stack on a cohort with one command:

```bash
# Every ticker in the store, or a specific cohort:
python -m trading_crew.agentic.validate --tickers TCS.NS
python -m trading_crew.agentic.validate --workflow-version <sha> --source backfill --validation cpcv
```

It prints, per ticker, the proposal side mix, the walk-forward/CPCV metrics, fill-status counts (so `SIZED_TO_ZERO`/`NO_FUTURE_BAR` are visible, not hidden behind a zero Sharpe), the decision-vs-sizing-vs-cost attribution + one-line diagnosis, the fused live signal + guardrails, and whether the RL policy/torch is available. Exit code is non-zero if no ticker produced a measurable backtest, so CI can gate on it.

> **Data volume, not code, is now the limit.** A freshly backfilled 5-decision cohort validates the fixes end-to-end (decimal, correctly-signed `expected_return_pct`; zero spurious ABSTAINs; measurable fractional fills), but 5 decisions can't yield a statistically meaningful Sharpe — the harness surfaces that honestly (`filled=1, folds=2`). Trustworthy metrics need a larger stamped cohort from `python -m trading_crew.agentic.backfill`. A separate residual PIT gap also remains for training data: the screener financial-statement tools (balance sheet / income / forward estimates) still return current columns during a historical backfill, so backfilled *features* can see the future even though the price-based *label* is PIT-safe.

## Project layout (top level)

```
agentic_workflow/
├── main.py               # CLI entry point: python main.py NTNX
├── run_web.sh            # launches the FastAPI UI on :8001
├── trading_crew/         # lean equities crew + M1–M7 deterministic core
│   ├── crew.py           # @CrewBase, agents/tasks via @agent / @task
│   ├── tools.py          # @tool functions (yfinance + Tavily/SerpAPI) + sanitize_citations
│   ├── critic.py         # M4 reflective critic
│   └── agentic/          # asset-class-agnostic deterministic Layer-A
│       ├── portfolio/    # state, allocator (HRP / MV / EQR)
│       ├── execution/    # cost models, next-bar fill simulator
│       ├── memory/       # episodic / semantic / regime
│       ├── risk/         # VaR/CVaR, sizing, gates
│       ├── swarm/        # PSO, consensus ensemble, crowd signal
│       ├── backtest/     # walk-forward + CPCV engine, Deflated Sharpe, PBO
│       └── rl/           # L4 PPO trainer (env, networks, storage)
├── commodity_crew/       # futures crew (curve, COT, seasonality)
├── web/                  # FastAPI backend + Alpine.js frontend
│   ├── backend/          # app.py + runner/session/scale_runner + prediction_db.py, company_meta.py
│   └── frontend/         # homepage.html · index.html (SPA: / , /stocks, /commodity, /run/{ticker})
├── universe/             # curated ticker lists: in100 / us100 / all200 (daily scale run)
├── scripts/              # scale_report.py, sector_bias_basket, backfill.sh, ...
├── tests/                # 730+ tests covering everything above
├── docs/                 # detailed design + screenshots
└── requirements.txt
```

## Tests

```bash
.venv/bin/pip install pytest pytest-asyncio
.venv/bin/python -m pytest tests/
```

760+ tests across every M1–M7 module, the swarm-intelligence layer (PSO / consensus / crowd signal), CPCV + PBO + probability-edge metrics, the multi-asset portfolio backtest, both crews, the FastAPI endpoints, the WebSocket runner, the L2/L3/L4 training loops, the degraded-output detection, the citation sanitizer, the lean-workflow call budget, and the ticker resolver.

## Inspiration & references

- **TauricResearch/TradingAgents** — the high-level multi-agent debate topology (analyst team → bull/bear → trader → 3-way risk → PM) is inspired by this Apache-2.0 project. <https://github.com/TauricResearch/TradingAgents>
- **Xiao et al., "TradingAgents: Multi-Agents LLM Financial Trading Framework"** — arXiv:[2412.20138](https://arxiv.org/abs/2412.20138) (2025).
- **Deterministic Layer-A design** — the M1–M7 contract (bridge, simulator, memory, critic, sizer + gates, walk-forward, allocator) draws on arXiv:[2605.19337](https://arxiv.org/pdf/2605.19337).

TradingCrew is an independent implementation: it is built on **CrewAI** (not LangGraph), adds a futures dashboard with cost-model presets, ships its own FastAPI/WebSocket web UI, and contributes the deterministic M1–M7 post-LLM pipeline + L2/L3/L4 training loops. No source code is copied from the upstream project. See [`NOTICE`](../NOTICE) for the full attribution.

## Disclaimer

This project is for **research and educational purposes only**. The deterministic risk layer is intentionally conservative, but LLM outputs are non-deterministic, market data is delayed, and back-tested performance is not a guarantee of future returns. **Nothing in this repository is investment advice.**

## License

Apache License 2.0 — see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE).

---

# Detailed design

> The following is the in-depth companion to the overview above: every module, endpoint, the M1–M7 deterministic pipeline, the L2/L3/L4/L5 training loops, the commodity dashboard, and the finance vocabulary.

Multi-agent **stock + commodity** research workflow built with [CrewAI](https://docs.crewai.com).

Two dashboards share the same homepage and the same deterministic backend:

- **`/stock`** — lean equities crew (5 analyst desks: 4 merged from the original 8-desk `agentic_workflow.ipynb` lift + a dedicated Reports/filings desk). Portfolio Manager emits a typed `PortfolioDecision`.
- **`/commodity`** — futures crew with futures-aware tools (curve, COT, seasonality, geopolitics). Same debate topology as the equity crew — analysts → bull/bear debate → research manager → futures trader → 3-way risk debate → Portfolio Manager (with optional Quality Review + Compliance advisory reviewers). The Portfolio Manager emits a typed `FuturesDecision` that is bridged to a `PortfolioDecision` so it can flow through the same M1–M7 pipeline.

The Portfolio Manager (or its commodity equivalent) emits a typed decision JSON gated by a confidence guardrail. After the LLM debate completes, the decision is handed off to a **deterministic post-LLM pipeline (M1–M7)** that compiles a typed `ActionProposal`, runs risk gates + a fill simulator, books the position, and records an episode for time-aware retrieval on the next run.

> **Asset-class agnostic core.** The deterministic Layer-A pipeline (`trading_crew/agentic/*`) is identical for both dashboards. Only the LLM debate layer and the cost-model preset (`futures_standard` vs `standard`) differ.

> **Market-aware CRISIS routing.** The `CRISIS_OVERRIDE` short-circuit (skip the 18-agent debate, force an ABSTAIN) is reserved for *systemic* risk-off. `runner._run_cascade_controller` now cross-checks the **single name's** regime against its **broad-market benchmark** (`market_context.benchmark_index_for` → `^NSEI` / `^GSPC` / `^FTSE` / …): it only overrides when the benchmark is *also* CRISIS/HIGH_VOL. A lone name in CRISIS while the index is calm is an **idiosyncratic shock** — exactly the oversold-dip case we want evaluated — so it's downgraded to `RISK_ONLY` (the full desk runs, risk-first, instead of auto-abstaining). Either way the deterministic **price-structure card is emitted on the override path too** so the Workflow tab / portfolio isn't blank for these names. Validated: IBM (name CRISIS, S&P 500 RANGE) reroutes CRISIS_OVERRIDE → `RISK_ONLY` and produces a real decision (SELL @ 1.3% with VOL_STATE=ELEVATED / DIP_BUY=SETUP / size ×0.72 populated) instead of the previous instant HOLD-0.

```
[M4] Cascaded Controller (regime detector)
   ├── CRISIS (name) + CRISIS/HIGH_VOL (benchmark) → systemic → short-circuit, skip debate, route to ABSTAIN
   ├── CRISIS (name) + calm benchmark             → idiosyncratic shock → RISK_ONLY (full debate, don't auto-abstain)
   └── else                                       → continue
   ↓
5 analyst desks (parallel via async_execution)
  [M3] Episode is stored after the run; desks do not retrieve past episodes
  -> DEBATE_ROUNDS x (Bull -> Bear)
    -> Research Manager synthesis
        -> Trader (uses get_event_proximity)
          -> RISK_ROUNDS x (Aggressive || Conservative -> Neutral)
              -> Portfolio Manager (output_pydantic = PortfolioDecision)
                ↓
   (optional, async, advisory to the PM: Quality Review audit +
    Compliance Officer CLEAR / FLAGGED / BLOCKED — default OFF)
                ↓
[M4] Reflective Critic (single-pass, ADVISORY by default; audits + flags but keeps the PM's action/size unless a hard integrity failure: compliance BLOCKED / fabricated source)
   ↓
[M1] Bridge: PortfolioDecision -> ActionProposal (deterministic, no LLM)
   ↓
[M5] Sizer (Fractional Kelly + vol-target + CVaR clamp + risk-debate multiplier)
   ↓
[M5] Risk Gates (concentration, leverage, drawdown stop, kill switch, ...)
   ↓
[M2] Execution Simulator (next-bar fill, partial fills, fees + half-spread + impact)
   ↓
[M1] PortfolioState update (atomic JSON store, audit trail)
   ↓
[M3] Episode written (state, action, outcome_ts, regime tag, embargo)
   ↓
[Phase E] Full run record persisted to ~/.trading_crew/runs/{TICKER}/{ts}.json
```

## Project layout

```
agentic_workflow/
├── pyproject.toml             # setuptools, picks up requirements.txt
├── requirements.txt           # crewai, yfinance, tavily, fastapi, …
├── .env.example               # LOCAL_LLM_*, TAVILY_API_KEY, OPENAI_API_KEY
├── main.py                    # CLI: `python main.py NTNX --debate-rounds 2`
├── run_web.sh                 # launches the FastAPI UI on :8001
├── trading_crew/              # ← equities crew
│   ├── __init__.py            # exposes TradingCrew, PortfolioDecision
│   ├── crew.py                # @CrewBase class; agents/tasks via @agent / @task
│   ├── tools.py               # 18 @tool functions (yfinance + Tavily)
│   ├── schemas.py             # PortfolioDecision Pydantic schema
│   ├── guardrails.py          # confidence_guardrail (4 rejection branches)
│   ├── _common.py             # get_llm + get_embedder_config factories
│   ├── critic.py              # M4 reflective critic
│   ├── agentic/               # ← asset-class agnostic deterministic core (M1–M7)
│   │   ├── reflection.py      #   L2 training — outcome resolution + LLM reflection
│   │   ├── grid_search.py     #   L3 training — walk-forward hyper-param sweep
│   │   ├── rl/                #   L4 training — PPO on the M2 simulator
│   │   │   ├── state.py       #     OHLCV → fixed-size feature vector
│   │   │   ├── env.py         #     Gym-style env wrapping the M2 simulator
│   │   │   ├── networks.py    #     Actor-critic MLP (PyTorch)
│   │   │   ├── ppo.py         #     PPO trainer (clipped surrogate + GAE + entropy)
│   │   │   ├── storage.py     #     Run records + checkpoints + promotion
│   │   │   └── inference.py   #     Load + score a promoted policy
│   │   ├── swarm/            #   swarm intelligence — PSO, consensus ensemble, crowd signal
│   │   ├── portfolio/         #   state, allocator (HRP / MV / EQR)
│   │   ├── execution/         #   cost (equity + futures presets), simulator, pipeline
│   │   ├── memory/            #   episodic / semantic / regime
│   │   ├── risk/              #   VaR/CVaR, sizing, gates
│   │   └── backtest/          #   walk-forward + CPCV engine, probability-edge metrics, PBO, portfolio replay
│   └── config/
│       ├── agents.yaml        # 18 personas
│       └── tasks.yaml         # all task descriptions and expected_outputs
├── commodity_crew/            # ← 17-agent futures crew (full debate pipeline)
│   ├── __init__.py            # exposes CommodityCrew, FuturesDecision
│   ├── crew.py                # @CrewBase 17-agent debate + risk + PM workflow
│   ├── tools.py               # futures-aware tools (curve, COT, seasonality, …)
│   ├── schemas.py             # FuturesDecision Pydantic schema (contract_month, curve_view, roll_yield, …)
│   ├── bridge.py              # FuturesDecision → PortfolioDecision adapter
│   └── config/
│       ├── agents.yaml        # 8 commodity personas
│       └── tasks.yaml         # 8 tasks culminating in a FuturesDecision
└── web/
    ├── backend/
    │   ├── app.py             # FastAPI: /, /stock, /commodity, /api/options, /api/chart,
    │   │                      #          /api/memory/resolve, /api/grid_search,
    │   │                      #          /api/training/rl/* (410 — L4 not live),
    │   │                      #          /api/commodity/{curve,cot,seasonality}, /ws/analyze
    │   ├── runner.py          # CrewAI step/task callbacks + asset-class routing
    │   ├── rl_runner.py       # L4 PPO worker (library; HTTP surface is 410)
    │   ├── charts.py          # OHLCV + indicator series (yfinance + stockstats)
    │   └── events.py          # event taxonomy + role→kind map for the diagram
    └── frontend/
        ├── homepage.html      # landing page — cards for Stock vs Commodity
        ├── index.html         # shared dashboard shell (asset-class aware)
        ├── diagram.js         # SVG workflow with positioned nodes + highlighted edges
        ├── charts.js          # Chart.js — price/RSI/MACD + futures curve/COT/seasonality
        ├── app.js             # Alpine controller (asset_class routing, WS, training, …)
        └── style.css          # dark theme + flow / tool-chip / sidebar widgets
```

## Quick start

### 1. Install

The project reuses the workspace `.venv` at `../.venv`. If you don't have one yet:

```bash
cd agentic_workflow
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in:

```ini
# Preferred — any OpenAI-compatible hosted vLLM-style endpoint. Pick a
# long-context model (~131K) if you plan to use the full 4-desk analyst
# debate flow — the bull / bear / research-mgr / PM prompts include every
# analyst report and can easily reach 30-50K tokens before the LLM
# responds.
VLLM_LLM_BASE_URL=https://your-vllm-host.example.com/v1
VLLM_LLM_MODEL=your-long-context-oss-model
VLLM_LLM_API_KEY=...

# Fallback — local vLLM (used only when the VLLM_LLM_* vars above are unset).
LOCAL_LLM_BASE_URL=http://localhost:8081/v1
LOCAL_LLM_MODEL=your-local-model
OPENAI_API_KEY=dummy

LLM_PROVIDER_PREFIX=hosted_vllm
TAVILY_API_KEY=...
```

`get_llm()` (in `trading_crew/_common.py`) prefers `VLLM_LLM_*` over `LOCAL_LLM_*`. All agents are constructed with `respect_context_window=True`, so if a prompt would still overflow the window CrewAI auto-summarises prior messages instead of hanging the request. Tavily is used by all news / supply-chain / geopolitical tools.

#### LLM picker in the UI

Desk vs report is two env jobs:

* `TRADINGCREW_LLM_PRESET` — desks / planner / critic. `local-vllm` reads
  `LOCAL_LLM_*`; `hosted-vllm` reads `VLLM_LLM_*`. The UI dropdown and
  `POST /api/v1/runs` both use this (sidebar can override one run).
* `REPORT_PROCESS_BASE_URL` / `MODEL` / `API_KEY` — `get_report_insights`
  map-reduce only. Ignores the desk preset.
* `LLM_PROVIDER_PREFIX` — LiteLLM wire prefix (`hosted_vllm`), not a model.

Closed-source dropdown rows stay grey until `OPENAI_PROD_API_KEY` /
`ANTHROPIC_API_KEY` are set.

#### Memory embedder (with truncation guard)

CrewAI's native short-term / long-term / entity memory is wired up by
`trading_crew/_common.py::get_embedder_config()`, which points at any
OpenAI-compatible embedding server:

```ini
VLLM_EMBEDDING_BASE_URL=https://your-embedding-host.example.com
VLLM_EMBEDDING_MODEL=embeddinggemma-300m   # or any OpenAI-compatible model
VLLM_EMBEDDING_API_KEY=...
TRADINGCREW_EMBEDDER_MAX_CHARS=6000        # default; ~1500 tokens
```

Settings no longer exposes an embedding-model picker (`retrieve_past_episodes`
is not a live desk tool). Sequential memory still uses
`trading_crew/embedding_presets.py` from `.env` (`VLLM_EMBEDDING_*`).
`embedding_preset` is stripped from `v3/bulk_settings.json` on load/save.

The crew partitions the on-disk LanceDB store by preset id
(`/crew/<crew>/emb/<preset>`) so a 768-d store and a 1536-d store
coexist side-by-side instead of crashing the next `search_memory`
with *"query dim doesn't match the column vector dim"*.

Small embedders like `embeddinggemma-300m` cap inputs at **2048
tokens**. The Portfolio Manager's output (action + rationale +
falsifiers + analyst summaries) can exceed that and surface as a
confusing `400 - maximum context length is 2048 tokens` *right after
PM finishes*. To prevent this, `get_embedder_config()` returns a
**custom** CrewAI provider (`trading_crew/embedder.py::TruncatingOpenAIEmbedder`)
that pre-truncates every document to `TRADINGCREW_EMBEDDER_MAX_CHARS`
characters before forwarding to chromadb's OpenAI embedding function.
The default 6000 chars (~1500 tokens) keeps a comfortable margin under
the 2048-token cap; bump it only if you switch to a larger-window
embedder.

Two implementation details worth knowing:

1. **Multi-inheritance.** `TruncatingOpenAIEmbedder` inherits from
   both `crewai.rag.embeddings.providers.custom.embedding_callable.CustomEmbeddingFunction`
   *and* `chromadb.api.types.EmbeddingFunction`. CrewAI's pydantic
   `EmbedderConfig` validates the custom callable against the chromadb
   protocol via `issubclass()`, which (for `runtime_checkable`
   Protocols) still requires *explicit* inheritance.
2. **Config from env vars + active preset, not kwargs.** CrewAI's
   `CustomProviderConfig` TypedDict only declares `embedding_callable`
   — pydantic strips every other key (`api_key`, `model_name`,
   `api_base`, `max_chars`) during validation. So
   `get_embedder_config()` returns `{"provider": "custom", "config":
   {"embedding_callable": …}}` *only*, and the embedder reads its
   credentials and budget from `VLLM_EMBEDDING_*` /
   `TRADINGCREW_EMBEDDER_MAX_CHARS` (or a leftover thread-local preset
   if an older session still sent `embedding_preset`) at construction time.

The crew constructs `Memory(llm=get_llm(), embedder=…, root_scope=…)`
explicitly (instead of `memory=True`) so CrewAI's analyze subsystem
runs against **our** local LLM rather than defaulting to
`gpt-4o-mini` against the placeholder `OPENAI_API_KEY=dummy` the vLLM
client needs.  Settings no longer exposes this toggle (it was not
`retrieve_past_episodes`). The UI/runner leave Memory off; scripts may
still pass `memory=True` into `TradingCrew`.

#### Parallel save pool

CrewAI's default `Memory._save_pool` is a
`ThreadPoolExecutor(max_workers=1)`, which serializes every
`remember()` call.  On a single ticker run there are roughly **100
memory-analyze calls** (≈ 28 tasks × 3.5 analyze calls + ~12
`search_memory` recall analyses); at ~10 s each against a typical
hosted vLLM endpoint that's *17 minutes* of LLM time if they all queue
sequentially.  This is exactly why memory-on used to balloon a 5-min
run to ~30 min.

After constructing `Memory`, we shut the default pool down and swap
it for `ThreadPoolExecutor(max_workers=8, …)` (configurable via the
`TRADINGCREW_MEMORY_SAVE_WORKERS` env var).  Combined with the
existing 10-worker pool inside `EncodingFlow` (for batch-internal
items), concurrent saves fan out and we can saturate the local-LLM
60-req/min throughput budget — bringing memory-on runs back to ~5
min while keeping the full LLM-enriched scope/category/entity
metadata that the analyze pass produces.

#### Per-run call estimate (shown in the UI)

The sidebar displays an estimate next to the memory toggle (e.g.
*"~108 LLM analysis calls ≈ 1.8 min"*) so users see the cost before
launching a run.  The formula:

```
tasks ≈ 5 analysts                          # analyst grid (1 task each)
      + (2 × debate_rounds)                  # bull + bear per round
      + 2                                    # research-mgr + trader
      + (quality_review ? 1 : 0)             # optional advisory reviewer
      + (3 × risk_rounds)                    # 3 risk personas per round
      + (compliance ? 1 : 0)                 # optional advisory reviewer
      + 1                                    # portfolio manager

calls ≈ tasks × 3.5 + 12                     # ~3.5 analyze calls/task + ~12 recall calls (memory ON only)
minutes ≈ calls / 60                         # local LLM rate limit = 60 req/min
```

With memory **off** (the default) there are no analyze/recall calls at
all — the estimate is just the agent tasks themselves.  These
*analysis* calls (memory on) run in parallel with agent work, so the
*effective* added wall-clock is closer to half the figure above on
typical runs (around 1 min for default sliders).

### 3. Run from the CLI

```bash
python main.py NTNX --debate-rounds 2 --risk-rounds 1
```

End of run prints the typed `PortfolioDecision` JSON.

#### Sector-bias basket runner

`scripts/sector_bias_basket.py` runs the full crew across a diverse US + India + multi-sector basket and writes a markdown report flagging any country / sector concentration in the decision mix.

```bash
../.venv/bin/python scripts/sector_bias_basket.py
# or pick your own subset:
../.venv/bin/python scripts/sector_bias_basket.py --tickers NVDA,AAPL,RELIANCE,MAZDOCK
```

Reports land in `reports/sector_bias_<timestamp>.{json,md}`. The companion static check `tests/test_sector_bias.py` runs in CI and catches hardcoded ticker / sector priors in prompt files.

#### Decision-impact worked example (backtest + walk-forward)

`scripts/decision_impact_demo.py` is the deterministic worked example for
"how much did the multi-horizon `backtest_setup` panel + the M6
walk-forward actually move the PM's decision?". It replays the
deterministic levers without calling an LLM, so it runs in seconds and
produces a markdown + JSON report:

```bash
../.venv/bin/python scripts/decision_impact_demo.py RELIANCE.NS --horizon 20 --target-pct 8 --stop-pct 3
../.venv/bin/python scripts/decision_impact_demo.py NTNX --horizon 20 --target-pct 6 --stop-pct 3
```

Reports land in `reports/decision_impact_<TICKER>_<ts>.{md,json}`.

For a real run on 2026-06-26, the two cases below illustrate the
complementary roles of the two halves:

| Ticker | OLD rules (pre-2F) | NEW rules (current) | Δ size | Δ confidence | Walk-forward says |
|---|---|---|---:|---:|---|
| RELIANCE.NS | NEUTRAL, 0% size, conf 0.55 | **OVERWEIGHT**, 1.12% size, conf 0.81 | **+1.12%** | **+0.26** | _silent_ (no proposal history yet) |
| NTNX | OVERWEIGHT, 1.5% size, conf 0.60 | OVERWEIGHT, 1.5% size, conf **0.79** | 0 | **+0.19** | Sharpe **−7.94** out-of-sample — temper conviction |

The reports in `reports/decision_impact_<TICKER>_<ts>.{md,json}` capture
both cases with the per-horizon bar charts, side-by-side PM decisions,
and the walk-forward overlay.

### 4. Run the web UI

```bash
./run_web.sh                     # binds 0.0.0.0:8001 (foreground)
PORT=9001 ./run_web.sh           # custom port
HOST=127.0.0.1 ./run_web.sh

./start.sh                       # background: nohup + logs/server/ + .server.pid
./stop.sh                        # graceful stop of the backgrounded server
```

Open http://localhost:8001. The **homepage** shows two cards:

- **Stock Trading** → `/stock`  (NVDA, AAPL, MSFT, …; lean 4-desk equity crew)
- **Commodity Trading** → `/commodity`  (CL=F, GC=F, ZC=F, …; 17-agent futures crew)

Both dashboards share the same UI shell. The page detects the asset class from the URL and:

- swaps the ticker preset grid (equities vs continuous-futures contracts),
- swaps the agent / tool catalog rendered in the sidebar,
- shows an extra **Futures** tab on `/commodity` (curve, COT, seasonality charts),
- selects the `standard` vs `futures_standard` cost-model preset when running M2.

Common tabs:

| Tab        | What it shows |
| ---------- | ------------- |
| Workflow   | Strict left-to-right SVG diagram, one column per pipeline stage: 5 analyst desks → bull/bear (parallel) → research mgr → trader → risk team (3 stacked, parallel) → portfolio mgr → decision, with the optional Quality Review + Compliance reviewers advising the PM. Parallel agents stack vertically inside a single column; sequential hand-offs each get their own column. Edge animation mirrors the React-Flow / Argo / GitLab CI convention: **idle** = thin gray, **running** = cyan dashed line that *flows* toward its destination via `stroke-dashoffset` animation, **done** = solid green. Each agent with tools shows a chip strip below its node — chips light yellow on tool start and green on success. Click any node to inspect its activity. Toolbar in the top-right has `−` / `+` / `Fit` / `100%` buttons (plus **Ctrl + scroll** to zoom at the cursor and drag-to-pan in the surrounding viewport). |
| Charts     | 180-day price chart with toggleable EMA / SMA / Bollinger overlays, plus dedicated RSI and MACD panels. Data comes from `/api/chart` (yfinance + stockstats). |
| Futures    | *(commodity dashboard only)* Term-structure chart for the front 6 contracts, weekly CFTC Commitments-of-Traders chart, and 5-year monthly seasonality bars. |
| *(Reports — removed)* | Agent output is now **tile-specific**: click an agent node in the **Workflow** diagram to read that agent's markdown output in the inspector below the diagram (click another tile to switch) — live and after completion. The report **trend-signal card** (filing + computed) and the final-decision card (action / sizing / drivers / risks / falsifiers / compliance) render just below the diagram. |
| Episodes   | The episodic-memory store (replaces the old *Memory* tab): embargo-aware top-k **retrieval** for this ticker *and* the full **episode-store table** (filterable, Kaggle-style). **Resolve outcomes & reflect** triggers `/api/memory/resolve` to compute realised PnL / α / max-DD and run the Hermes harness (one evidence-selected experiment). |
| Backtest   | **Walk forward** — single-name, out-of-sample fold replay of recorded proposals (no new LLM calls; latest workflow version only): equity curve, PnL attribution, CPCV/PBO robustness. Plus **Portfolio replay** — multi-ticker coherent-book replay (HRP/MV/ERC, absorbs the old *Portfolio* tab) — and the crowd-positioning view. |
| Training   | **Hermes** 30-day book scoreboard plus **L3** Apply as the per-ticker size base. Resolve opens the next knob. Strategy, prompts, and bulk run snapshot live on **Settings** (header gear → `/settings`). L4 PPO is not on the live path. |
| *(Logs — removed)* | The activity log was removed entirely — the diagram's edge colours (cyan = running, green = done, amber = degraded) already show live progress, and per-agent output is read by clicking a tile. |

The sidebar exposes:

- **History** on `/stocks` is live sessions + prediction-run dates; on `/commodity` it is live sessions + recent contract runs. Each list scrolls inside the left pane (sessions stay capped; dates/contracts fill the rest). Click a session to open `/run/{ticker}` or `/commodity/run/{c}`. Settings is the header gear → `/settings` (left pane = LLM/tools/dialogue; right pane = live strategy + prompt overlays + Save all). Back on Settings returns to `/stocks` or `/commodity`; Home on those dashboards returns to `/`.
- **LLM** picker (dropdown + model/`via` line; no embedding picker)
- **Run** (same level as LLM — no Advanced wrapper): autonomous vs deterministic (different packaged prompts), debate/risk sliders only for deterministic/commodity, advisory reviewers. No CrewAI-memory toggle.
  - **Tools** — per-agent enable/disable checkboxes. Disabling a tool hands the
    agent a shorter tool list at construction time.
- Single prediction Run lives on the Prediction tab; the focused `/run/{ticker}` page has Re-run. Both write today's Prediction-index row and reuse it unless **Force rerun** is checked.

WebSocket protocol: client sends a single JSON config frame on `/ws/analyze`, then receives JSON events of types `run_started → node_started → tool_call → tool_result → node_completed → final_decision → run_completed`.

### Streaming-gap fix

CrewAI's `step_callback` only fires once per task with the local LLM (native function calling lumps all tool iterations into one `AgentFinish`). For tool-less sequential agents (Bull / Bear / Research Manager / Quality Reviewer / risk team / Compliance / PM) that means the UI used to sit completely silent for 30–60 s after the analyst phase while the next single LLM call ran — looking like the workflow was "stuck after analyst team".

Two fixes in `runner.py`:

1. The runner pre-computes `expected_role_order = [t.agent.role for t in crew.tasks]` and synthesizes `node_started` for the next-up agent as soon as the previous task completes — so the diagram lights the next node immediately.
2. Per-tool `tool_call` / `tool_result` events come from the CrewAI **event bus** (`ToolUsageStartedEvent` / `ToolUsageFinishedEvent`) rather than `step_callback`. This is required because native-tool LLMs bypass `step_callback` for individual tool iterations.

## What was inspired by `TauricResearch/TradingAgents`

- **Project shape** (`web/backend` + `web/frontend` + `run_web.sh`) — same topology.
- **Schemas in their own module** — mirrors `tradingagents/agents/schemas.py`.
- **Sidebar UX** — History on `/stocks` and `/commodity`; Settings left pane is LLM / tools / dialogue; Prediction tab has Bulk vs Single.
- **SVG workflow diagram** with positioned nodes, group rectangles and curved Bezier edges that highlight on activity.
- **Charts tab** powered by `stockstats` (Bollinger / RSI / MACD / EMA / SMA / ATR) + Chart.js, sourced from a thin `/api/chart` endpoint.
- **Streaming pump** — synchronous worker thread → `queue.Queue` → asyncio queue → WebSocket — so CrewAI's blocking `kickoff()` plays nicely with FastAPI.

## What's CrewAI-specific (different from TradingAgents/LangGraph)

- The crew is built with the **`@CrewBase` / `@agent` / `@task` / `@crew`** decorators that load `config/agents.yaml` and `config/tasks.yaml` automatically.
- Streaming is hooked via `Crew(step_callback=…, task_callback=…)` plus the `crewai_event_bus` for per-tool events (LangGraph's `astream` / `chunk` interface doesn't apply).
- `Task(output_pydantic=PortfolioDecision, guardrail=confidence_guardrail, guardrail_max_retries=2)` is the only path to a typed final answer + auto-retry.
- Deterministic stock matches v2: the 5 analyst tasks fan out (`async_execution=True`) on `Process.sequential`. Autonomous LangGraph fans out the five analysts, the **round-1 bull/bear openings**, and the **round-1 risk-wing openings** (later debate and risk rounds stay sequential rebuttals). Deterministic CrewAI also fans out the round-1 risk wings (they sit after the sync trader, so they do not hit the analyst-batch ValidationError). The optional Quality Review + Compliance reviewers use `async_execution` as a concurrent pair right before the PM.
- Per-agent tool toggles are passed into `TradingCrew(tools_enabled={...})` and resolved at agent-construction time via `_tools_for(agent_key)`.

## Files you'll usually edit

| To change… | Edit |
| --- | --- |
| Any agent's persona | `trading_crew/config/agents.yaml` |
| Any task's prompt   | `trading_crew/config/tasks.yaml` |
| Tool wiring or new tools | `trading_crew/tools.py` + `crew.py` |
| Guardrail rules     | `trading_crew/guardrails.py` |
| Final-decision schema | `trading_crew/schemas.py` |
| ActionProposal bridge | `trading_crew/agentic/bridge.py` |
| Risk / sizing / gates | `trading_crew/agentic/risk/{var,sizing,gates}.py` |
| Cost model + simulator | `trading_crew/agentic/execution/{cost,simulator}.py` |
| Episodic memory       | `trading_crew/agentic/memory/{episodic,semantic,regime}.py` |
| Allocator (HRP / MV)  | `trading_crew/agentic/portfolio/allocator.py` |
| Backtest engine       | `trading_crew/agentic/backtest/{engine,walk_forward,metrics,manifest}.py` |
| UI lanes / styling  | `web/frontend/{index.html,app.js,style.css}` |

## What is "backtest"?  (and why two places say "backtest")

There are **two different backtests** in this codebase. They share a name only because the LLM literature uses the same word for both. Knowing which is which removes a lot of confusion.

### 1. `backtest_setup` tool — *single-trade base-rate lookup*

Lives in `trading_crew/tools.py`. Runs automatically inside the LLM Trader's task. Given a ticker + a candidate (target%, stop%, horizon-days, **direction**) trade idea, it scans the historical price tape and computes:

- N samples that ever started a position under those entry conditions,
- win rate (target hit first), stop-out rate, timeout rate,
- avg / median realised return,
- payoff ratio and per-trade expectancy across a SHORT / MEDIUM / LONG multi-horizon panel out to `2520d ≈ 10y`.

It's a **decision-time aide** for sizing one specific trade idea — "out of 1225 historical setups like this, 20% hit target". It runs on every Trader task and you'll see its output in the Reports tab inside the Trade Plan card.

**Direction-aware (regression fix, 2026):** the tool now takes an explicit `direction="long"` or `direction="short"`. Before this fix `_simulate_horizon` was LONG-only — for a SHORT trade the trader passed positive target/stop magnitudes and the tool computed "did the price go UP target% or DOWN stop%", which is the wrong distribution. On most surviving liquid names (which have positive long-run drift) that silently over-reported SHORT-trade expectancy and produced a suspicious "~+1 to +2% per trade" convergence across unrelated tickers. `direction="short"` inverts the WIN / LOSS conditions (WIN when price falls target%, LOSS when it rises stop%) and signs the returns so expectancy is directly comparable to the LONG case. The trader task prompt now requires the LLM to pass `direction` on every call.

### 2. Walk-forward backtest engine — *strategy-level replay*

Lives in `trading_crew/agentic/backtest/`. Replays **all `ActionProposal`s you have ever logged** for a ticker through the same M1+M2+M5 deterministic pipeline (risk gates + cost model + simulator) under embargoed train/test folds, then reports:

- equity curve **plus a buy-and-hold benchmark overlay** (so you see *alpha*, not just absolute return),
- Sharpe / Sortino / Calmar / MaxDD / **Deflated Sharpe** (selection-bias corrected),
- **probability-edge metrics** — win rate, expectancy, payoff ratio, profit factor (the Firefly "a system can win with sub-50% accuracy if the average win dwarfs the average loss" idea),
- **benchmark alpha / beta** vs buy-and-hold,
- a **bootstrap 95% confidence band on the Sharpe** (stationary bootstrap — the number is a range, not a point estimate),
- per-fold metrics.

This is the **Backtest tab** in the UI and the `GET /api/backtest` endpoint.

**Validation modes** (`validation=walk_forward|cpcv`):

- **Walk-forward** — one contiguous embargoed out-of-sample path (the default).
- **CPCV** (Combinatorial Purged Cross-Validation, López de Prado AFML ch. 12) — `C(n_groups, k_test)` distinct out-of-sample paths, so the headline is a *distribution* of OOS Sharpes (mean / median / σ) rather than a single number that is trivially overfit. Tune with `cpcv_groups` / `cpcv_k`.

**PBO (Probability of Backtest Overfitting).** When you run a sweep (grid or swarm), the search reports a PBO via the CSCV algorithm (Bailey, Borwein, López de Prado & Zhu 2017): the probability that the config which looked best in-sample lands in the *bottom half* out-of-sample. A PBO near 0 means the selection generalises; > 0.5 means the "winner" is likely luck.

> **Latest workflow version only (default).** The **Walk forward** panel (and `GET /api/backtest`) restricts to proposals stamped with the *latest* `workflow_version` so the replay measures the *current* workflow, not a blend of older, incomparable versions. Older-version proposals are excluded and the count is shown in the cohort note. Pass `all_versions=1` on the endpoint to replay across every version.

> **Why it sometimes shows "need N more proposals":** the walk-forward engine needs ≥ `train + embargo + test` proposals *from the latest version* to form even one fold. The UI defaults to `train=3 / embargo=1 / test=1` (5 proposals minimum). Run analyses on a ticker on the current workflow enough times to clear that threshold and the equity curve appears.

### 3. Multi-asset portfolio backtest — *coherent-book replay*

`run_portfolio_backtest` (endpoint `GET /api/portfolio_backtest?tickers=AAA,BBB,CCC`) replays the logged proposals of *several* tickers as one book: at each rebalance date every symbol's proposal is handed to the **M7 allocator** (HRP / Mean-Variance / Equal-Risk), which produces a single cross-sectional weight vector respecting the gross-risk budget and per-name caps, before execution against **one shared `PortfolioState`**. The benchmark is an equal-weight buy-and-hold of the same tickers. This is the **Portfolio replay** panel on the Backtest tab.

## Swarm intelligence

> **Status: diagnostic-only (frozen).** The PSO optimizer, the consensus ensemble, and Phase-4 signal fusion (with the L4 PPO policy) are kept as **research/diagnostic** tools and are **not** on the live decision's critical path. In practice they re-process the *same* LLM signal (PSO is a cheaper tuner for the same objective as the grid; fusion echoes the PM decision back with guardrail text) rather than adding orthogonal information, and they are data-hungry. Effort is instead focused on the loops that actually shape and learn the decision: the **agent debate** (the real "swarm of bots from different angles"), **L2 structured reflection**, **L5 RLMF**, **cross-desk feedback**, and **conviction calibration**. The one swarm piece with genuinely independent information — the **crowd-positioning signal** — stays on the roadmap to A/B once enough episodes resolve.

Inspired by the Firefly / Fintrens platforms' "swarm intelligence + probability-edge" framing, `trading_crew/agentic/swarm/` adds three deterministic, seeded, LLM-free capabilities:

- **Particle Swarm Optimizer** (`swarm/pso.py`) — a swarm of candidate sizing/gate configurations converges on the robust region of the parameter space, ranked by **Deflated Sharpe**, in far fewer backtests than the `5^3` brute-force fine grid. Choose it in the **Backtest → Auto-tune** panel via `optimizer=pso` (particles / iterations / seed are exposed), and watch the per-iteration global-best converge in the **swarm-convergence sparkline**. Fully reproducible for a given seed.
- **Consensus ensemble** (`swarm/consensus.py`) — the "strategy swarm": replay the top-K configs, fuse their per-decision target weights (weighted by each config's Deflated Sharpe) into one consensus book, and report the swarm's **directional agreement** and **weight dispersion** (how tightly the strategies agree = a robustness signal). Enable with `consensus=true` on `/api/grid_search`.
- **Crowd-positioning signal** (`swarm/crowd.py` + the `get_crowd_positioning` tool) — fuses options put/call & IV skew, insider net buying, and analyst rating dispersion into a single crowding score in `[-1, +1]` (`+` crowd long, `-` crowd short). Surfaced as a chip on the Backtest tab (`GET /api/crowd_positioning`) and available to the Market desk. It powers an **optional, off-by-default** "fade-the-crowd" sizing tilt (`CrowdConfig.enabled`) — trim size when siding with an over-crowded consensus, size up when contrarian — and, with `TRADINGCREW_CROWD_SIGNAL_ENABLED=1`, stamps each episode with its crowd regime (score **plus the per-component breakdown**) so backtests can segment by it.
- **Data-driven crowd weights** (`swarm/crowd_fit.py`) — the base signal fuses its components with equal, hand-set weights (`w_* = 1.0`). Once episodes carry a per-component breakdown and a resolved `alpha_return`, `python -m trading_crew.agentic.swarm.crowd_fit` ridge-regresses the components against realised alpha and rewrites each component's fusion weight by its predictive strength (persisted to `crowd_weights.json`; signed coefficients kept as a contrarian/momentum diagnostic). `load_crowd_config()` picks up the fitted weights automatically; **until a fit exists the equal weights are retained — nothing is invented.** Enable crowd logging (`TRADINGCREW_CROWD_SIGNAL_ENABLED=1`), let the corpus accumulate and outcomes resolve, then run the fitter (`--dry-run` to preview, `--min-samples` to set the corpus floor).

## Agentic training on historical data (harness + L3)

Beyond the inference-time M3 episodic memory ("L1"), the live loops are Hermes (Episodes → Resolve) and L3 sizing.

| Level | What it learns | Live effect (what actually changes) |
| ----- | -------------- | ----------------------------------- |
| **L1** | Nothing on the live desk — episode store + human Episodes tab / `GET /api/memory/retrieve` | Prediction desks do not retrieve past episodes. Hermes trains from Resolve lessons. |
| **Harness** | 30-day book vs `desk_strategy.yaml`; Hermes picks one experiment (desk / topology / size / debate) | Trader / RM / PM get strategy + the open experiment. A `desk_prompt` prepends only that desk. A playbook experiment is **live topology** on the graph router (`skip_reports`, debate/evidence/risk rounds, `no_trader_reopen`; debate-round knobs also override the sequential slider). Mass promote is not required. An open `size_scale` multiplies live Kelly and max position. Scoreboard: **Training** tab. Edits: **Settings** (`/settings`). |
| **L3** | Best `kelly_fraction` / `vol_target` / `max_position_weight` via walk-forward grid or PSO, ranked by Deflated Sharpe | **Only if you Apply.** Writes `tuned_sizing.json`. Live `run_pipeline` and walk-forward compose: L3 tune is the per-ticker base (or built-in defaults if none); an open Hermes `size_scale` multiplies Kelly + max position on top. |
| **L4** | Offline PPO library on the M2 simulator (OHLCV only) | **Not on the live prediction path.** Does not change the PM action or size. Crew tools and `/api/training/rl/*` are unwired (HTTP 410). Fancy algorithm, no Resolve/desk/filing evidence — complexity without prediction value. |


### Resolve — outcome resolution + harness step

Module: `trading_crew/agentic/reflection.py`. Endpoint: `POST /api/memory/resolve`.

Two clocks, one episode:

1. **Learning review (30 calendar days)** — if `review_ts` has elapsed, price the 30-day path, label WIN/LOSS, write `Episode.review` (always including deterministic `desk_lessons`), and score the book against `desk_strategy.yaml`. Status `REVIEW_RESOLVED`. The thesis stays `PENDING`. This is the Hermes training signal.
2. **Thesis outcome (`horizon_days`, often 90–180d MEDIUM or longer LONG)** — when `outcome_ts` elapses, the original L2 path still runs:

   1. Re-fetch OHLCV between `decision_ts` and `outcome_ts`.
   2. Compute **realised return** (signed by side), **alpha vs SPY benchmark**, **max drawdown**.
   3. If an LLM is configured (`skip_llm: false`), it first **plans an investigation** (`resolve_investigate.py`): whether to call `research_web` / `get_news` / `get_global_news` / `get_report_insights` / `get_forward_estimates` / `get_commodity_news` (max 3; skipped in backfill). Then it writes a **structured reflection**: free-text lesson plus `desks`, `error_class`, `recommended_change`, `desk_lessons`, `market_reason`, `unhelpful_tools`, `overemphasized`, and an optional `suggested_playbook` vote. Unknown tags and invented playbook ids are dropped.
   4. Patch the episode in place. Hermes reads the 30-day review (and thesis meta) on the next Resolve step — prediction desks do not retrieve this as a live tool.

Existing PENDING episodes that were recorded with a 180-day thesis and no `review_ts` get the 30-day clock backfilled on the next Resolve sweep.

> **Why structured routing matters for the data campaign.** The `desks`/`error_class` tags make every resolved episode route accurately to the right desk and segment cleanly by failure mode — so a 20k-episode forward-collection corpus becomes far more valuable to the cross-desk feedback + calibration layers than free-text lessons matched by keyword.

UI: **Episodes tab → Resolve outcomes & reflect** button.

```bash
curl -X POST localhost:8001/api/memory/resolve \
  -H 'content-type: application/json' \
  -d '{"ticker": "NTNX", "skip_llm": false}'
```

Status codes per episode: `REVIEW_RESOLVED` (30-day learning clock only), `RESOLVED` (thesis horizon), `SKIPPED_PENDING` (neither clock elapsed), `SKIPPED_NO_DATA`, `ABANDONED` (older than the configurable cutoff).

### L3 — walk-forward grid search on sizing & risk

Module: `trading_crew/agentic/grid_search.py`. Endpoint: `POST /api/grid_search`.

Sweeps a grid of `kelly_fraction × vol_target × max_position_weight` over your logged proposals using the M6 walk-forward engine. **Ranking is by Deflated Sharpe**, which corrects for selection bias when you test many configurations against the same history (Bailey & Lopez de Prado, 2014). The "best" configuration the UI surfaces is the one that survived this correction, not the highest raw Sharpe.

Set `optimizer=pso` to replace the brute-force grid with the **particle-swarm optimizer** (`swarm/pso.py`) — a seeded swarm that converges on the robust region of the sizing/gate space in far fewer backtests. `validation=cpcv` swaps the single walk-forward path for Combinatorial Purged CV, and the search reports a **PBO** (probability of backtest overfitting). Set `consensus=true` to also fuse the top-K configs into a robustness-weighted **consensus ensemble** (`swarm/consensus.py`).

**Closing the loop (L3 + Hermes → prediction).** By default L3 is *diagnostic* — it shows the better config but doesn't change anything. Pass `apply=1` (UI: the **"Apply to live + walk-forward"** checkbox next to Auto-tune) to persist the winning config as this ticker's tuned `SizingConfig`/`GateConfig` in `tuned_sizing.json` (`trading_crew/agentic/tuning.py`). Live `run_pipeline` and Walk forward then call `resolve_live_sizing`: the L3 tune is the per-ticker base (or built-in defaults if none); an open Hermes `size_scale` multiplies Kelly and max position. The scale comes from the journal stamp, or from the YAML catalog entry for the same knob id if the journal predates that field. A knob without `size_scale` does not invent a number. The `execution_result` event carries `tuned_config_applied` and `hermes_size_scale`. Only apply a tune you trust: check the Deflated Sharpe and PBO first, since a grid/PSO winner fit on few proposals overfits easily.

UI: **Training tab → Size — L3** panel (with a Grid/PSO toggle + swarm-convergence sparkline). Strategy / prompts: **Settings**.

```bash
# Deterministic particle-swarm search, CPCV-validated, with a consensus ensemble
curl -X POST localhost:8001/api/grid_search \
  -H 'content-type: application/json' \
  -d '{"ticker": "NTNX", "optimizer": "pso", "swarm_size": 12, "swarm_iterations": 8,
       "swarm_seed": 7, "validation": "cpcv", "consensus": true, "rank_by": "deflated_sharpe"}'
```

### L4 — removed from the live prediction path

Module: `trading_crew/agentic/rl/` (offline library + `tests/test_rl.py`). HTTP: `/api/training/rl/*` returns **410 Gone**. UI: Training tab is Hermes scoreboard + L3 Apply. Strategy / prompts: **Settings**.

L4 was judged on whether it improves the **live prediction**, not on algorithm novelty. It does not:

- It learns ~21 OHLCV features and 7 weight buckets. No filings, guidance, desk debate, or Resolve lessons.
- It never changed the PM action (advisory tool + optional fusion). Fusion already did not load it as a decision override.
- Complexity (PPO, promote, two crew tools, Training UI, APIs) exceeded the value add.

The trainer stays in-tree for later A/B. Live sizing is **L3 Apply × Hermes `size_scale`**. Bookmarks that hit `/api/training/rl/*` fail honestly (410), not a silent 404.

What the library still is (offline only):

```
                                gradients
       ┌─────────────────────────────────────────────────┐
       │                                                 ▼
+------+------+    state    +---------------+   action   +----------------+
|             | ──────────▶ | Actor-Critic  | ─────────▶ |  M2 Simulator  |
|   Feature   |             |  (PyTorch     |            |  (next-bar     |
|  extractor  |             |   MLP, ~8K    |            |   fill + fees  |
|             |             |   params)     |            |   + impact)    |
+-------------+             +---------------+            +-------+--------+
       ▲                          ▲                              │
       │      observation         │ value                        │ reward
       │      ←────────────────── │ ←────────────────────────────┘
       │
       │
   PortfolioState (mark-to-market every bar — drawdown gate, NAV)
```

What it is:

- **Algorithm**: Proximal Policy Optimization (PPO) — clipped surrogate, GAE-λ advantages, value-function clipping, entropy bonus, gradient clipping. Standard PPO tricks; no stable-baselines3 dependency, ~200 lines of in-tree code.
- **Network**: 2-hidden-layer MLP (64 units, tanh, orthogonal init). Tiny on purpose — daily OHLCV gives you ~1000 bars per ticker and bigger nets just overfit.
- **State (21 features)**: past-10 log-returns, realised vol (z-scored), 5/20-day momentum, RSI(14), MACD, Bollinger %b, current position weight + unrealised PnL + bars-in-trade, day-of-week sin/cos.
- **Action space**: 7 discrete target-weight buckets — `-20%, -10%, -5%, 0%, +5%, +10%, +20%`. The same grammar the LLM uses (`SHORT / NEUTRAL / LONG` + size tier).
- **Reward**: per-bar Δ NAV / starting NAV after the M2 simulator deducts fees + slippage + impact + carry. Optional shaping terms: turnover penalty, drawdown penalty, hard drawdown-kill termination.
- **Asset class**: agnostic. The runner selects the `futures_*` cost preset automatically when `asset_class == "commodity"`.

Live use is disabled. The HTTP surface answers 410:

```bash
curl -X POST localhost:8002/api/training/rl/start \
  -H 'content-type: application/json' \
  -d '{"ticker": "SPY"}'
# {"detail":"L4 PPO is removed from the live prediction path. ..."}
```

Persistence layout if you train the library offline (under `$TRADINGCREW_CACHE_DIR/rl_runs/`):

```
rl_runs/
  SPY/
    20260611T170611-825/
      record.json     # config + metrics + eval result
      policy.pt       # torch checkpoint (load with PPOTrainer.load_model)
      metrics.jsonl   # one TrainingMetrics per line (streams to the UI)
  promoted/
    SPY.json          # { ticker, run_id, asset_class, promoted_ts, summary }
```

Worth keeping in mind:

- A run with `total_steps=20_000` and the default 512-step rollouts trains in **~60-120 s on CPU** for a single ticker.  The default MLP is intentionally small (~8K params) — bigger nets overfit on a single ticker's history.
- The eval window is **strictly held out** from training.  The runner slices the most recent `eval_window_days` bars off the end, trains only on what's before that, and reports `eval_result` (PnL%, drawdown, action frequency, vs buy-and-hold) on the held-out segment.
- **No future leakage**: feature extraction uses only bars up to and including `t`, and the simulator always fills at `t+1` open.  This is the same guarantee M2 enforces for live runs.
- Stopping a run is **cooperative** (the trainer checks a `should_stop` flag between rollouts). The live UI no longer starts or stops L4 jobs.

### Prediction harness (Hermes; Mass is legacy)

Module: `trading_crew/agentic/harness.py`. Wired from Resolve (no env gate).

The live crew prepends **desk strategy + one open experiment** (trader / RM / PM). A `desk_prompt` experiment also prepends that desk's Resolve `recommended_change`. Legacy Mass `promoted.json` still applies if present. Permanent Mass install is still `scripts/promote_graph.py`.

```bash
# After real reviews are due
curl -X POST http://127.0.0.1:8002/api/memory/resolve

# Or exercise the same step with synthetic 30-day reviews
curl -X POST http://127.0.0.1:8002/api/harness/sample/seed -H 'Content-Type: application/json' -d '{"count":12}'
curl -X POST http://127.0.0.1:8002/api/harness/sample/clear
```

Promote a confirmed experiment:

```bash
.venv/bin/python scripts/promote_graph.py --held-out-after 2026-06-01
```

## Horizon tiers (SHORT / MEDIUM / LONG)

Every `PortfolioDecision` carries an explicit `horizon` field alongside the numeric `horizon_days`:

| Tier | Days | Typical use case |
| --- | --- | --- |
| `SHORT` | 5 – 60 | Catalyst / earnings play, technical breakout, buyback close.  The exception, not the rule. |
| `MEDIUM` | 61 – 252 | The desk's DEFAULT.  Ind-AS quarterly-print thesis, multi-quarter earnings trend, mid-term relative-strength. |
| `LONG` | 253 – 2520 (up to 10 years) | Compounder / value thesis — requires the `get_long_term_thesis` tool AND positive expectancy on the 5y row of the backtest panel. |

The `backtest_setup` tool now emits a **tier-labelled multi-horizon panel** covering 60d / 126d / 189d / 252d / 504d / 1260d / 2520d (5y / 10y windows drop automatically when history can't cover them, and the panel says which ones dropped so the LLM does not silently treat a missing row as neutral evidence).  A "Per-tier best expectancy" summary line gives the PM the single strongest expectancy at each tier so it can compare short vs medium vs long edge without eyeballing.

The `get_long_term_thesis` tool aggregates screener.in's 10-year Ind-AS tables (falls back to yfinance's 3-4y annuals for non-Indian names) into a single markdown block:

- Revenue / Net Profit / EPS CAGR over the longest available window
- ROCE / ROE trend — the "quality of compounding" tell (>= 15% avg ROCE ⇒ compounder; < 10% ⇒ value-trap risk)
- Debt trajectory over the window
- Shareholding stability (promoter / FII drift on Indian names)
- Screener's own structural pros/cons

For the accounting side: because the outcome-resolution loop can't wait 10 years, LONG-tier decisions have their `ActionProposal.horizon_days` **clamped to 5 years (1260 trading days)** — the narrative 5-10 year thesis lives in `PortfolioDecision.long_term_view` (the field was previously named `long_term_thesis` and the schema accepts the old key as an alias) and the RLMF loop uses the 5y realised return as the reward proxy.

`long_term_view` is now **required on every decision**, not just LONG-tier calls.  For horizon=LONG it holds the 2-5 sentence multi-year narrative; for horizon=SHORT / MEDIUM it holds a 1-2 sentence "the multi-year compounding thesis is intact / broken / not applicable" line so the LONG-panel context is preserved in the PM's final report even when the desk chose a shorter primary window.  Historically the trader reasoned over the LONG row of the backtest panel but that context was silently dropped once the PM's JSON was emitted — the field rename + always-populate rule closes that gap.

## Commodity dashboard

Open `/commodity` instead of `/stock` and the UI swaps in:

- **8 futures-specialist agents** (Market Analyst, Curve Analyst, Inventory & Demand Analyst, Geopolitical Analyst, News Analyst, Bullish / Bearish Researchers, Futures Trader) defined in `commodity_crew/config/{agents,tasks}.yaml`.
- **Futures-aware tools** in `commodity_crew/tools.py`:
  - `get_commodity_ohlcv` / `get_commodity_indicators` — front-month OHLCV + stockstats indicators.
  - `get_futures_curve` — front 6 contracts pulled live to compute CONTANGO / BACKWARDATION + roll yield %.
  - `get_cot_report` — weekly CFTC Commitments-of-Traders (downloaded + cached).
  - `get_seasonality` — N-year monthly mean/std returns from front-month series.
  - `get_commodity_news` / `get_commodity_geopolitical` — Tavily web search scoped to commodity keywords + OPEC / Black Sea / Suez / Hormuz triggers.
  - Commodity desks do not retrieve past episodes. Resolve investigation may call `get_commodity_news` when the miss is a futures name.
- **`FuturesDecision` schema** (`commodity_crew/schemas.py`) — adds `contract_month`, `contract_size`, `curve_view`, `roll_yield_pct_annualised` on top of the equity decision fields.
- **`futures_decision_to_portfolio_decision` bridge** (`commodity_crew/bridge.py`) — adapts the futures decision to `PortfolioDecision` so the existing M1–M7 deterministic pipeline (sizer, risk gates, simulator, memory) handles both asset classes unchanged.
- **`futures_low / futures_standard / futures_high` cost-model presets** (`trading_crew/agentic/execution/cost.py`) — tighter spreads + exchange fees + lower fixed cost than the equity presets. The runner auto-selects the `futures_*` preset when `asset_class == "commodity"`.
- **`roll_yield_carry_cost` helper** (same file) — converts an annualised roll-yield % + holding days into a P&L drag/credit for futures positions.

New commodity-specific endpoints:

| Endpoint | Purpose |
| -------- | ------- |
| `GET /api/commodity/curve?ticker=CL=F&n_months=6` | Front-N contracts + structure classification |
| `GET /api/commodity/cot?ticker=GC=F&weeks=8` | Last N weekly CFTC reports for the commodity |
| `GET /api/commodity/seasonality?ticker=NG=F&years=5` | N-year monthly mean/std returns |

## Agentic-Trading Roadmap (M1–M7)

The deterministic backend draws on the design described in [arxiv:2605.19337](https://arxiv.org/pdf/2605.19337). Everything lives in `trading_crew/agentic/` and is **LangChain-free** — the LLM stays in CrewAI, the deterministic Layer-A pieces run on top.

| Milestone | Module | What it does | UI surface |
| --------- | ------ | ------------ | ---------- |
| **M1** Auditable state + I/O contract | `agentic/portfolio/state.py`, `agentic/execution/contracts.py`, `agentic/bridge.py` | Append-only `PortfolioState` (atomic JSON), typed `ActionProposal` schema, deterministic `PortfolioDecision → ActionProposal` bridge | "Order ticket" card in Workflow tab |
| **M2** Execution + cost layer | `agentic/execution/{cost,simulator,pipeline}.py` | Cost model (fees + half-spread + √impact), next-bar fill simulator, sensitivity sweep | "Execution" card in Workflow tab |
| **M3** Audit-grade memory v2 | `agentic/memory/{episodic,semantic,regime}.py` | Outcome-embargoed `EpisodicMemory`, time-decay retrieval, regime tags, `SemanticKnowledgeBase` with provenance | **Episodes** tab + episode banner |
| **M4** Reflection + cascaded controller | `trading_crew/critic.py`, `web/backend/runner.py::_run_cascade_controller` + `_crisis_override_decision` | **Cascaded controller** detects regime *before* the crew kicks off and short-circuits to ABSTAIN on CRISIS (saving ~30k tokens). **Reflective Critic** runs after the PM with 5-stage protocol (intent / evidence / counterfactual / risk / verdict), bounded reflection budget (2 iterations), and 3-temperature consistency vote (mode < 2/3 → forced ABSTAIN). | Cascade badge + Reflective-critic panel with per-sample checklist |
| **M5** Risk + sizing layer | `agentic/risk/{var,sizing,gates}.py` | Historical + parametric VaR/CVaR, Fractional Kelly × vol-target × CVaR clamp sizing, hard gates (concentration, leverage, drawdown stop, kill switch) | "Sizing" + "Risk gates" panels |
| **M6** Walk-forward backtest + manifest | `agentic/backtest/{walk_forward,metrics,manifest,engine}.py` | Embargoed train/test folds **or CPCV paths**, Sharpe / Sortino / Calmar / MaxDD / **Deflated Sharpe**, probability-edge metrics (win-rate / expectancy / payoff / profit-factor), benchmark alpha/beta, bootstrap Sharpe CI, **PBO**, run manifest with git-sha + prompts hash + data hashes | **Backtest** tab (equity + benchmark curve, per-fold + probability-edge cards) |
| **M6+** Swarm intelligence | `agentic/swarm/{pso,consensus,crowd}.py` | Seeded particle-swarm optimizer over the sizing/gate space, robustness-weighted consensus ensemble (agreement/dispersion), deterministic crowd-positioning signal + optional fade-the-crowd tilt | Grid/PSO toggle + convergence sparkline, consensus panel, crowd chip |
| **M7** Multi-ticker allocator | `agentic/portfolio/allocator.py` | HRP / Mean-Variance / Equal-Risk allocator over the most recent `ActionProposal` per ticker, **replayable as a coherent-book portfolio backtest** (`run_portfolio_backtest`) | **Backtest** tab — Portfolio replay panel |

### How a single run flows through M1–M7

1. **UI → WS `/ws/analyze`** with `{ticker, debate_rounds, risk_rounds, tools_enabled, critic_iterations, critic_samples, ...}`.
2. **Runner** opens a `RunRecord`, loads ~252 bars of OHLCV, calls **`detect_regime()` (M4)** and emits `cascade_status`.
3. **If regime == CRISIS** → the runner skips `crew.kickoff()` entirely and builds a synthetic ABSTAIN `PortfolioDecision` (paper §5.3 emergency stop). The deterministic M1→M5→M3 pipeline still runs so the audit trail is identical.
4. **Otherwise** `crew.kickoff()` (or the stock LangGraph scheduler) runs the lean flow. Past-episode retrieval is not a live desk tool — Hermes applies the open experiment from Resolve. UI streams `node_started` / `agent_step` / `tool_call` / `tool_result` / `node_completed` / `final_decision`.
5. **Reflective Critic (M4)** — once the Portfolio Manager returns its typed `PortfolioDecision`, the runner runs the 5-stage critique (up to 2 iterations) then, if `critic_samples > 1`, a multi-temperature consistency vote. **Advisory by default** (`critic_advisory=on`): the runner always emits `reflection_records` so the audit renders in the UI, but it **keeps the PM's action/size** and only re-emits a `final_decision` with `source="critic"` on a *hard integrity failure* (compliance BLOCKED, or a fabricated/placeholder source). Set `critic_advisory=false` to restore the legacy always-override behaviour (critic may revise/ABSTAIN freely).
6. **Bridge (M1)** — the post-critic `PortfolioDecision` is mapped to a typed `ActionProposal` via `portfolio_decision_to_action_proposal()` and emitted as `action_proposal`.
7. **Pipeline (M2+M5)** — `run_pipeline()` computes VaR/CVaR from the OHLCV, runs the **sizer** (Fractional Kelly × vol-target × CVaR clamp × debate-derived risk multiplier), runs the **risk gates** (concentration, leverage, DD stop, kill switch, …), and runs the **simulator** for a next-bar fill. The runner emits `execution_result` with sizing breakdown, risk-gate outcomes, and a low/standard/high cost-sensitivity sweep.
8. **Episode (M3)** — a PENDING `Episode` (state, action, outcome_ts = decision_ts + horizon, regime tag, embargo) is appended to episodic memory and `episode_recorded` is emitted.
9. **Persistence (Phase E)** — every event since step 1 is captured into the `RunRecord` and written atomically to `~/.trading_crew/runs/{TICKER}/{ts}.json` plus an index line so the UI sidebar can list it.

### New API endpoints

| Endpoint | Purpose |
| -------- | ------- |
| `GET /api/memory/retrieve?ticker=&as_of=&query=&k=` | Top-k embargo-aware retrieval of past episodes |
| `GET /api/portfolio` | Book snapshot + HRP allocator preview over the latest proposal per ticker |
| `GET /api/backtest?ticker=&train_size=&embargo_size=&test_size=&validation=&all_versions=` | Walk-forward **or CPCV** backtest of logged proposals (probability-edge metrics, benchmark alpha/beta, bootstrap Sharpe CI). Defaults to the **latest workflow version only**; `all_versions=1` replays every version |
| `GET /api/portfolio_backtest?tickers=&alloc_method=&gross_budget=` | Multi-asset coherent-book replay through the M7 allocator against one shared `PortfolioState` |
| `GET /api/crowd_positioning?ticker=` | Deterministic crowd-positioning score fused from options / insider / analyst-dispersion inputs |
| `GET /api/runs/recent?limit=&ticker=` | List the most recent UI runs across all tickers (or one) |
| `GET /api/runs/{ticker}/latest` | Full record of the most recent run for a ticker |
| `GET /api/runs/{ticker}/{run_id}` | Full record of a specific past run |
| `POST /api/memory/resolve` | **L2 training** — resolve PENDING episodes into realised outcomes + LLM reflections |
| `POST /api/grid_search` | **L3 training** — sizing & risk search ranked by Deflated Sharpe. `optimizer=grid\|pso` (particle-swarm), `validation=walk_forward\|cpcv`, optional `consensus` ensemble; reports **PBO** |
| `GET /settings` | Operator Settings SPA (left: run config; right: strategy + prompts). Optional `?book=commodity`. |
| `GET /api/settings?book=` | One document: live strategy, prompt overlays, knob catalog, bulk run snapshot (`bulk` is `null` if none saved). |
| `PUT /api/settings?book=` | Persist only keys sent: `strategy`, `prompts`, `unset_prompts`, `revert_strategy`, `bulk`. Existing `/api/strategy*` stay. |
| `POST /api/training/rl/*` and `GET /api/training/rl/*` | **410 Gone** — L4 removed from the live prediction path. Sizing is L3 + Hermes `size_scale`. |
| `GET /api/commodity/curve` | Futures term structure (CONTANGO / BACKWARDATION) |
| `GET /api/commodity/cot` | Weekly CFTC Commitments-of-Traders report |
| `GET /api/commodity/seasonality` | Monthly seasonality from front-month series |

### New CrewAI tools

- `get_crowd_positioning(ticker)` — deterministic crowd-positioning score in `[-1, +1]` fused from options put/call & IV skew, insider net buying, and analyst rating dispersion. Wired into `market_analyst`. Past-episode retrieval is not a live desk tool.

### New WebSocket event types

```
cascade_status      { regime, route, reason }                                // M4 cascade controller
reflection_records  { records[5-stage per sample], final_action, revised }   // M4 reflective critic
action_proposal     { proposal, markdown }                                   // M1
execution_result    { order, fill, sizing, risk_gate, cost_scenarios, ... }  // M2+M5
episode_recorded    { episode_id, regime, decision_ts, outcome_ts }          // M3
```

### How to use the new features

1. **Run an analysis**: `./run_web.sh`, open the UI, pick a ticker, click *Run analysis*.
2. Watch the **Workflow tab**:
   - The **Cascade-status badge** at the top shows the regime detected from OHLCV. If `CRISIS`, the agent debate is skipped entirely and the run goes straight to an ABSTAIN proposal.
   - Once the Portfolio Manager finishes, the **Reflective Critic** panel renders the 5-stage checklist (intent / evidence / risk per sample) and the consistency-vote outcome — REVISED / RATIFY / ABSTAIN.
   - The **Order ticket**, **Execution**, **Sizing**, **Cost sensitivity sweep**, and **Risk-Gate** panels populate from the post-critic deterministic pipeline.
3. Open the **Recent runs** section in the sidebar — every kickoff is saved to `~/.trading_crew/runs/`. Click any past run to re-load its full state (diagram, per-tile agent reports, trend signals, decision) into the Workflow tab without re-running.
4. Run the same ticker on a few more dates to build an **episode history**. The Episodes tab is the human tape (optional embargo-aware retrieval for operators). After **Resolve**, Hermes opens one experiment from the 30-day lessons — prediction desks do not retrieve those episodes as a live tool.
5. Trade a few different tickers, then open the **Backtest** tab's Portfolio-backtest panel. With ≥2 tickers' recent proposals on file, the HRP/MV/ERC allocator replays them as one coherent book.
6. Click **Backtest** tab to walk-forward replay your recorded proposals through the deterministic pipeline. The equity curve + per-fold metrics use the same risk gates and cost model as live runs.

### Where the M1-M7 modules live (and what's *not* duplicated)

- The `trading_crew/agentic/` package is the **LangChain-free deterministic core** — `{portfolio,execution,memory,risk,backtest}` modules implementing the M1–M7 contracts from the paper. No source is copied from the upstream `TauricResearch/TradingAgents` project; only the high-level debate topology is shared.
- **M1**'s Action Compiler in the LangGraph project is reduced to the deterministic `bridge.py` mapping from CrewAI's existing `PortfolioDecision` (so the CrewAI crew remains the single LLM driver for the *narrative*; the bridge is pure Python).
- **M4** is now implemented as a full LLM-driven Reflective Critic (`trading_crew/critic.py`) that calls the LLM directly via `crewai.LLM.call(..., response_model=CritiqueResponse)` for clean structured output without spinning up a mini-crew per sample. The cascaded-controller half of M4 (regime routing) lives in `runner.py::_run_cascade_controller` + `_crisis_override_decision`.
- **Cache directories**: all app state is consolidated into a single SQLite DB at `$TRADINGCREW_CACHE_DIR/prediction.db` (default `~/.trading_crew`) — episodes, run history, PIT scrape cache, `collect_state`, and prediction data all live there as separate tables. Portfolio state goes to `$TRADINGAGENTS_PORTFOLIO_DIR` (default `~/.tradingagents/portfolios`).

### Tests

```bash
.venv/bin/pip install pytest pytest-asyncio
.venv/bin/python -m pytest tests/
```

**290 tests** covering all M1–M7 modules + the `PortfolioDecision → ActionProposal` bridge + the end-to-end deterministic pipeline + the Reflective Critic (5-stage protocol, RATIFY/REVISE/ABSTAIN paths, consistency vote, abstain on split, abstain on provider failure) + **L2 reflection** (`tests/test_reflection.py`) + **L3 grid search** (`tests/test_grid_search.py`) + **commodity crew bits** (`tests/test_commodity_crew.py`) + **L4 PPO/RL** (`tests/test_rl.py`) — feature extractor (output shape, clipping, position features), env (action space, step return, trim-reduce delta semantics, drawdown-kill termination, close-out), actor-critic network (forward / act / probs), PPO trainer (metrics streaming, eval action frequencies, cooperative stop, checkpoint round trip), storage (save/load round trip, JSONL append, promotion requires checkpoint), the inference client (no-policy returns None, recommendation shapes, short-history rejection), and **market context** (`tests/test_market_context.py`) — ticker resolution (US bare / Indian `.NS` auto-suffix / already-suffixed pass-through / cache hits / total miss), country routing from `.info` or exchange fallback, India macro basket + RBI/INR themes for an Indian shipbuilder, US basket for a US semiconductor, dynamic country / industry / supply-chain query builders, and peer-basket lookup.

## Multi-region support (US + India + UK + HK + JP)

The crew now reads tickers from **multiple exchanges**, not just the NYSE/Nasdaq.

### Type any of these in the Ticker box

- `AAPL`, `NVDA`, `SPY` → US (passes through unchanged).
- `MAZDOCK` → resolves to **`MAZDOCK.NS`** (Mazagon Dock Shipbuilders, NSE).
- `RELIANCE` → `RELIANCE.NS`, `INFY` → `INFY.NS`, `HDFCBANK` → `HDFCBANK.NS`.
- `0700` → `0700.HK` (Tencent), `7203` → `7203.T` (Toyota), `HSBA` → `HSBA.L`.

The resolver (`trading_crew/market_context.py::resolve_ticker`) probes the common exchange suffixes in order — `.NS, .BO, .L, .HK, .T, .SI, .TO, .AX, .DE, .PA` — and caches the first match. Already-suffixed inputs and commodity tickers (`CL=F`) pass through.

### Ticker-aware research context (no more "US treasury yields" on every name)

`get_market_context(ticker)` is a new tool every analyst calls FIRST. It returns **facts** — country, currency, sector, industry, market cap, and the **macro basket** (yfinance symbols for the country's rates / FX / vol / commodities). Search prose is not stored here: the analyst writes `query=` from those facts + their desk remit + any blackboard gap. Peers are not a ticker dictionary — `get_sector_peers` screens Yahoo's live industry universe in the listing's country (and accepts `peers=` when filings named better comparables).

Example brief for `MAZDOCK.NS`:

```
## Market context for MAZDOCK.NS
- Name: Mazagon Dock Shipbuilders Limited
- Country / currency / exchange: India / INR / NSI
- Sector / industry: Industrials / Aerospace & Defense
- Market cap: …
- Benchmark index: ^NSEI
- Peer basket: live Yahoo industry + country screen via get_sector_peers
- Macro basket (yfinance symbols):
    nifty_50: ^NSEI       india_vix: ^INDIAVIX
    usd_inr: INR=X        sensex: ^BSESN
    brent_crude: BZ=F     gold: GC=F
- Query contract: YOU write the 6-12 word search query. No canned theme list.
```

Same call for `AAPL` returns the US macro basket (10y UST, DXY, VIX, WTI, gold). The US flow is unchanged — *the system now picks the right instruments for the ticker* instead of hard-coding one phrasebook.

### What the agents actually do now

| Tool | Before | Now |
| ---- | ------ | --- |
| `get_news(ticker)` | `"{ticker} stock news sentiment last week"` | Analyst writes `query=` from the brief; default is the company name only |
| `get_global_news(ticker)` | `"global markets macro economy fed rates"` always | Analyst writes `query=`; default is `{country} central bank inflation currency` |
| `get_macro_data()` | US 10y UST + DXY + VIX + WTI + gold | Country-specific basket (`get_macro_data(ticker)`) — India = Nifty + India VIX + INR/USD + Sensex + Brent |
| `get_geopolitical_news(ticker)` | `"{ticker} tariffs export controls sanctions China Taiwan"` | Analyst writes `query=` from remit + gap; default is name + country + industry + lens verbs |
| `get_supply_chain_risk(ticker)` | `"TSMC China dependency 10-K customer"` | Analyst writes `query=`; default is name + country + industry + supply-chain verbs |
| `get_sector_peers(ticker)` | Per-ticker dictionary (`NTNX→DELL/HPE`, `MAZDOCK→COCHINSHIP/…`) | Live Yahoo industry + country screen; always pass the run ticker (`{}` previously failed validation). If omitted, the tool binds the kickoff symbol from the run board — it does not invent a name. Desk may pass `peers=` from filings |

The task prompts (`trading_crew/config/tasks.yaml`) now explicitly require analysts to call `get_market_context` first and ground their searches in the company's home country — no implicit US default.

---

## Finance primer

New to the finance vocabulary the system uses? The concept walk-through and the dashboard glossary live in the **[finance primer](finance/README.md)**.
