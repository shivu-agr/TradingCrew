/* TradingCrew in-app help content.
 *
 * Single source of truth for every tooltip, info-icon, agent hover-card,
 * Help-drawer entry, and glossary term in the UI.  Mirrors the content
 * in README.md (Finance basics + How-it-works sections) so a casual
 * user never has to leave the dashboard to understand what they're
 * looking at.
 *
 * Structure
 * ---------
 *   HELP_CONTENT[anchor_id] = {
 *     title:     short label (drawer heading + tooltip title),
 *     short:     1-2 sentence tooltip-friendly summary (≤200 chars ideal),
 *     long:      full markdown body rendered in the Help drawer,
 *     section:   one of "tabs" / "panels" / "agents" / "sidebar" / "glossary"
 *                (controls drawer grouping + search bucket),
 *     keywords:  optional array of additional search terms,
 *   }
 *
 * Anchor IDs are dotted strings:
 *   tab.<id>           - one per tab (workflow / charts / reports / memory / portfolio / backtest / training / logs / futures)
 *   sidebar.<field>    - one per sidebar control
 *   panel.<id>         - one per workflow / report panel
 *   agent.<role_id>    - one per agent persona
 *   glossary.<term>    - one per finance term (P/E, RSI, MACD, Kelly, …)
 *
 * The Alpine helper in app.js exposes:
 *   helpEntry(anchorId)              → object | null
 *   openHelp(anchorId)               → opens the drawer at that anchor
 *   helpSearch(query)                → filtered list
 *
 * Add new entries here; the audit test (`tests/test_help_content.py`)
 * fails if any data-help attribute in index.html lacks a definition.
 */

window.HELP_CONTENT = {

  /* =========================================================== TABS === */

  "tab.workflow": {
    section: "tabs",
    title: "Workflow",
    short: "One view per name: live agent diagram + deterministic post-LLM pipeline. Click any agent tile to read that agent's output; the decision + trend-signal card sit below — all replayed from the saved run on reopen.",
    long: `
The **Workflow** tab is the orchestration view. Stock and commodity both use a two-band flowchart (research on top, decision below) so Fit stays readable. Parallel desks (Bull/Bear, the three risk analysts) sit as a pair or stack; sequential hand-offs read left to right on the decision band.

\`\`\`
stock (autonomous):     Playbook → analysts fan-out → blackboard → bull + bear opening (parallel) → rebuttal cycles → RM → trader → risk → PM
stock (deterministic):  v2 LTR pipeline — Analyst team → Researcher team → Trader → Risk → Compliance + PM → Decision
commodity:              analysts → bull / bear → RM → quality reviewer → trader → risk → compliance → PM
\`\`\`

Wrapped around the LLM debate are three deterministic gates:

1. **Cascaded Controller** (before the debate) — detects the market regime from 252 bars of OHLCV. If \`CRISIS\`, skips the entire 18-agent debate and emits a synthetic ABSTAIN decision.
2. **Reflective Critic** (after the PM) — runs the same critique prompt at three temperatures; if 2/3 votes don't agree, the decision is forced to ABSTAIN.
3. **Deterministic execution pipeline** (M1→M5→M2→M3) — sizer (fractional Kelly + vol target + CVaR), risk gates (concentration / leverage / drawdown), next-bar fill simulator with fees/spread/impact, episode write.

**Edge animation** (mirrors React Flow / Argo / GitLab CI conventions):

- **idle** — thin static gray line
- **running** — **cyan dashed line that flows** toward the destination (the stroke animation makes direction obvious)
- **done** — solid green

Tool chips overlay each agent tile and light yellow on tool start and green on success.

**Zoom + pan.** The toolbar in the top-right has \`−\` / \`+\` / \`Fit\` / \`100%\` and a live **elapsed timer** for this run (ticks while queued/running; freezes at the final duration when the run ends). You can also hold **Ctrl + scroll** anywhere inside the diagram to zoom at the cursor; drag the scrollbars to pan when zoomed in. The current zoom level is shown as a percentage between \`−\` and \`+\`.
`.trim(),
    keywords: ["agents", "pipeline", "diagram", "zoom", "animated", "timer", "elapsed"],
  },

  "tab.charts": {
    section: "tabs",
    title: "Charts",
    short: "180-day OHLCV price chart with toggleable EMA / SMA / Bollinger overlays + dedicated RSI and MACD panels.",
    long: `
180-day price chart powered by [Chart.js](https://www.chartjs.org/) and indicators computed server-side with [stockstats](https://github.com/jealous/stockstats).

The chart end date is **today**. The agent crew uses live data (analyst tools call yfinance at run time).

Toggleable overlays: 10-EMA, 50-SMA, 200-SMA, Bollinger Mid/Upper/Lower. Dedicated subplots: RSI(14), MACD + MACD Signal + MACD Histogram, ATR.
`.trim(),
    keywords: ["price", "candlestick", "indicators"],
  },

  "tab.futures": {
    section: "tabs",
    title: "Futures",
    short: "Commodity-only tab: term-structure (CONTANGO / BACKWARDATION), CFTC Commitments-of-Traders, and 5-year monthly seasonality.",
    long: `
Only visible on the \`/commodity\` dashboard. Three views:

- **Term structure** — front-6 contracts on the curve, classified as CONTANGO (back > front, holders pay roll cost) or BACKWARDATION (back < front, holders earn roll yield). The annualised roll yield is reported alongside.
- **CFTC COT** — weekly Commitments-of-Traders report. Managed-money + commercial long/short/net positioning history. Useful for spotting positioning extremes.
- **Seasonality** — N-year monthly mean ± stdev returns from the front-month series. Identifies recurring seasonal patterns (e.g. natural-gas winter rallies, gasoline summer demand).
`.trim(),
    keywords: ["contango", "backwardation", "COT", "seasonality"],
  },

  "tab.reports": {
    section: "tabs",
    title: "Reports (now per-tile in Workflow)",
    short: "The Reports tab was removed. Click an agent tile in the Workflow diagram to read THAT agent's output; click another tile to switch. Works live and after completion.",
    long: `
The standalone Reports tab has been removed. Agent output is now **tile-specific**: click any agent node in the **Workflow** diagram and its full markdown report opens in the inspector below the diagram; click another tile and the inspector switches to that agent. This replaces the old single scrollable list of every report stacked together.

This works both while the run streams (you also see the agent's intermediate steps + tool calls live) and after completion (the saved run replays each agent's final report(s) into the inspector). Every numeric claim carries an inline \`[source: <identifier>]\` tag — the **Reflective Critic** validates these tags against the PM's \`sources\` list to distinguish real data from hallucinated numbers. The final-decision card and the Firefly-style **trend-signal card** render just below the diagram.

The bottom of the tab shows the **final-decision card**: action on the retail 5-tier scale (**STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL**), size, confidence, key drivers, key risks, falsifiers, compliance status, and the deduplicated sources list.  Older runs may still show the legacy 3-tier action strings (OVERWEIGHT / NEUTRAL / UNDERWEIGHT) — the UI collapses them to BUY / HOLD / SELL for display.

### "Degraded" badge
A report card tagged with an amber **DEGRADED** badge means the LLM emitted a *tool-call list* as its final answer for that task instead of synthesising the expected markdown. This is a known limitation of some open-source LLMs (the \`hosted-vllm-oss\` preset is the usual offender) on tool-heavy tasks like News, Macro, Social, and Geopolitical — the model fires a tool call and then fails to compose a closing assistant message, so the agent loop terminates with the tool-call objects in the final-answer slot. The defensive patch in \`trading_crew/_patches.py\` catches the failure, prints a clear placeholder showing which tool calls were attempted, and lets the rest of the crew run. Mitigations: switch to a closed-source LLM via the LLM picker, re-run the workflow (the failure is non-deterministic), or shorten that analyst's tool chain.
`.trim(),
    keywords: ["analyst", "rationale", "sources", "provenance", "degraded", "empty report"],
  },

  "tab.episodes": {
    section: "tabs",
    title: "Episodes",
    short: "The tape: episode store + Resolve. Hermes training lives on the Training tab.",
    long: `
Episodic memory stores \`(state, action, outcome, timestamp)\` tuples for every past run.

1. **Store / table** — the human tape for this ticker (optional embargo-aware retrieval for operators). Prediction desks do not retrieve past episodes as a live tool.
2. **Resolve** — labels 30-day reviews and elapsed theses. The LLM may investigate why the call was wrong (web, news, filings). That tape is what Hermes reads.

Hermes (prompts / topology / size) and L3 sizing live on the **Training** tab. Mass promote is legacy.
`.trim(),
    keywords: ["episode", "embargo", "reflection", "L2", "training", "memory", "store"],
  },

  "tab.backtest": {
    section: "tabs",
    title: "Backtest",
    short: "Walk-forward replay of logged proposals with embargo. Reports Sharpe / Sortino / Calmar / MaxDD / Deflated Sharpe per fold.",
    long: `
**Strategy-level walk-forward** evaluation. Replays *all* logged \`ActionProposal\`s for a ticker through the M1 + M2 + M5 deterministic pipeline (risk gates + cost model + simulator) under strict train / embargo / test folds.

**Why "backtest" appears in two places**: the *tool* \`backtest_setup\` is a decision-time base-rate lookup ("out of 1225 historical setups like this, 20% hit target"). The *tab* is the post-hoc strategy-level harness.

The **L3 auto-tune** (grid/PSO search over \`kelly_fraction × vol_target × max_position_weight\`, ranked by **Deflated Sharpe**) lives on the **Training** tab and composes with an open Hermes \`size_scale\`. This tab keeps the walk-forward replay, PnL attribution, CPCV/PBO robustness, the multi-ticker portfolio backtest, and the crowd-positioning view.

If you need ≥ \`train+embargo+test\` proposals (default 3+1+1 = 5), use the **Seed N synthetic proposals** button to bulk-generate proposals from existing PM decisions without re-running the LLM crew.
`.trim(),
    keywords: ["walk-forward", "CPCV", "Deflated Sharpe", "attribution", "portfolio"],
  },

  "tab.training": {
    section: "tabs",
    title: "Training",
    short: "Hermes: one open experiment on the live book. L3: per-ticker size base.",
    long: `
Prediction writes decisions. Episodes → Resolve labels the 30-day tape and runs Hermes. This tab is the **scoreboard**: the 30-day book name, SUCCESS/FAILURE vs the strategy bars, n / hit-rate / α / DD, the live experiment Hermes already opened, and used knobs.

Hermes picks the next unused knob on Resolve — one at a time. There is no dropdown here. Mandate, prompts, and enabled knobs are edited on **Settings**.

Hermes still picks **one named knob with one fixed value** from rules + the live strategy (not an LLM search, not a value sweep) when you Resolve. If the book is still FAILURE after enough reviews, that knob is rejected and the next unused enabled one can open.

L3 grid/PSO is separate: it searches Kelly / vol / cap. Apply writes the base; an open Hermes \`size_scale\` multiplies on top.
`.trim(),
    keywords: ["L3", "Hermes", "topology", "size_scale", "PSO", "auto-tune", "harness", "settings"],
  },

  "tab.prediction": {
    section: "tabs",
    title: "Prediction",
    short: "Bulk or single. Completions land on today's Prediction run and are reused unless you force a rerun.",
    long: `
The **Prediction** tab is the live forward book on \`/stocks\`. **Bulk** and **Single** use the same Settings snapshot.

A completed single or bulk name is written to the Prediction index for **today**. **Live · today** on the left returns here. The dated list below it opens that day's **prediction run** (\`/stocks/history/{date}\`) — Names + Portfolio, not this tab.

The next single or bulk attempt on the same name **returns the cached result**. A failed cell is not a cache hit. Check **Force rerun** to ignore today's cache and run the crew again. Commodity sessions never use this index.
`.trim(),
    keywords: ["prediction", "cache", "force rerun", "bulk", "single", "today"],
  },

  "tab.run-names": {
    section: "tabs",
    title: "Names",
    short: "Stocks that ran on this prediction-run date. Click a name to open that day's workflow.",
    long: `
A **prediction run** is a calendar day's completed (and failed) names — not the live universe on \`/stocks\`.

Click a sidebar date or **Review names →** to open \`/stocks/history/{date}\`. This tab lists every name that ran that day. Click a row to open \`/run/{ticker}?date={date}\` and replay that day's workflow. **Back** on the workbench returns here; **Stocks** returns to \`/stocks\`.
`.trim(),
    keywords: ["names", "prediction run", "history", "workflow"],
  },

  "tab.portfolio": {
    section: "tabs",
    title: "Portfolio",
    short: "₹10L virtual book for this prediction-run date, marked to the latest close.",
    long: `
The **Portfolio** tab lives on the prediction-run page (\`/stocks/history/{date}\`), next to **Names**. It notionally deploys ₹10,00,000 across that day's BUY / STRONG_BUY names, weighted by predicted growth, then marks each holding to the latest known price.

Switch dates from the left pane (or the run-date dropdown). Click a holding to open that name's workflow for this run date. Market Both / India / US re-deploys the full book inside the selection.
`.trim(),
    keywords: ["portfolio", "virtual", "10 lakh", "history", "prediction run"],
  },

  "page.settings": {
    section: "sidebar",
    title: "Settings",
    short: "Run config on the left; live strategy, prompts, and bulk snapshot on the right. One save.",
    long: `
The **Settings** page (\`/settings\`, or \`/settings?book=commodity\`) is the operator console. Open it from the gear on \`/stocks\` or \`/commodity\` (or a focused run). Back on Settings returns to that book. Home on the stock/commodity dashboards returns to \`/\`. History stays on \`/stocks\` and \`/commodity\`.

**Left pane** is run config only: LLM, Run (mode + optional sliders + reviewers), tools. There is no Advanced wrapper, no embedding picker, and no CrewAI-memory toggle. Ticker, analysis date, sessions, and recent runs are not here — ticker lives on Prediction (single), date is always today, sessions live on History.

**Right pane** is desk strategy and per-desk goal/backstory overlays. **Save all settings** writes strategy + the selected prompt overlay + the current run snapshot through \`PUT /api/settings\`. Single and bulk prediction both use that same snapshot. Revert deletes the live strategy file. Unset restores one packaged persona.

Writes go to the cache (\`v3/desk_strategy.yaml\`, \`v3/prompt_overlay.yaml\`, \`v3/bulk_settings.json\`), not the git yaml. \`TRADINGCREW_STRATEGY_PATH\` still wins if set.

LLM API keys and vLLM URLs stay in \`.env\`. There is no profile or account menu.
`.trim(),
    keywords: ["settings", "strategy", "prompts", "bulk", "overlay"],
  },

  "tab.logs": {
    section: "tabs",
    title: "Logs (removed)",
    short: "The activity log was removed — the diagram's edge colours already show live progress, and per-agent output is read by clicking a tile.",
    long: `
The standalone activity log has been removed entirely. Progress is shown by the **Workflow** diagram itself: edges animate cyan while an agent runs and turn solid green (or amber if degraded) when it completes. To read what an agent produced, click its tile — the output opens in the inspector below the diagram.
`.trim(),
    keywords: ["events", "trace", "activity log", "removed"],
  },

  /* ========================================================= SIDEBAR === */

  "sidebar.prediction_runs": {
    section: "sidebar",
    title: "Prediction runs",
    short: "Live · today is the /stocks universe. Dated rows open that day's Names + Portfolio.",
    long: `
The left pane splits **Live sessions** (queued / running kickoffs) and **Prediction runs** evenly when anything is in flight. Live sessions is only the active queue — finished kickoffs leave this list.

**Live · today** returns to \`/stocks\` (the live Prediction universe). **Review names →** opens that same live collection day — the server calendar date from the universe API, which can differ from the browser's local date around midnight.

The scrollable list under it is **past** run dates only. Today's date is not listed twice. Click a past date for that day's ran names and virtual book (\`/stocks/history/{date}\`). Click a name there to open \`/run/{ticker}?date=...\`. Back from the workflow returns to the run; **Stocks** on the run page returns to \`/stocks\`.

Opening a live session on `/run/{ticker}` streams that kickoff. It does not replay yesterday's cached HOLD on top of a still-running job.

A name already \`done\` today is reused on the next single or bulk run unless **Force rerun** is checked. Failed cells retry without Force.
`.trim(),
    keywords: ["history", "prediction runs", "today", "cache"],
  },

  "sidebar.force_rerun": {
    section: "sidebar",
    title: "Force rerun",
    short: "Ignore today's cached prediction and run the crew again. Off by default.",
    long: `
Same-day **done** predictions are cached: Single → Run analysis and Bulk → Start return that row instead of calling the LLM.

Check **Force rerun** when you want a new kickoff for names already completed today. The new result overwrites the day's Prediction-index row. Failed cells are not cached, so they retry without this box.

Commodity runs never read or write this cache.
`.trim(),
    keywords: ["force", "rerun", "cache", "skip"],
  },

  "sidebar.ticker": {
    section: "sidebar",
    title: "Ticker",
    short: "Stock symbol to analyse. Type bare (MAZDOCK → MAZDOCK.NS auto-resolves) or use an exchange suffix.",
    long: `
The crew runs on a single ticker per kickoff. The text input is uppercased and the runner uses a **ticker resolver** that probes common exchange suffixes if the symbol has no data:

- US (no suffix): \`AAPL\`, \`NVDA\`, \`SPY\`
- India: \`.NS\` (NSE), \`.BO\` (BSE) — \`MAZDOCK\` → \`MAZDOCK.NS\`
- UK: \`.L\` (LSE) — \`HSBA\` → \`HSBA.L\`
- Hong Kong: \`.HK\` — \`0700\` → \`0700.HK\` (Tencent)
- Japan: \`.T\` — \`7203\` → \`7203.T\` (Toyota)

The system also routes the **macro basket** + **news themes** by country: an Indian ticker gets Nifty + India VIX + INR/USD + RBI / Union Budget news themes; a US ticker gets 10y UST + DXY + VIX + Fed / CPI themes.
`.trim(),
    keywords: ["symbol", "NSE", "BSE", "exchange"],
  },

  "sidebar.date": {
    section: "sidebar",
    title: "Analysis date",
    short: "Drives the chart end-date only. The crew itself always uses live data.",
    long: `
Affects the Charts tab end date and the \`as_of\` parameter passed to memory retrieval (so the embargo respects the chosen date — useful for "what would the crew have known on date X?" what-if analysis).

Does NOT replay historical news from that date — the analyst tools always call live news APIs.
`.trim(),
    keywords: ["date", "as-of", "embargo"],
  },

  "sidebar.llm": {
    section: "sidebar",
    title: "LLM picker",
    short: "Pick the LLM for the next run. Every desk and tool call in that run uses this endpoint.",
    long: `
One dropdown. The selection is persisted to \`localStorage\` and sent as \`llm_preset\`. Default comes from \`TRADINGCREW_LLM_PRESET\` in \`.env\` (\`local-vllm\` → \`LOCAL_LLM_*\`, \`hosted-vllm\` → \`VLLM_LLM_*\`). Desks, planner, critic, and tool summarize use that catalog. \`get_report_insights\` map-reduce is separate: it uses \`REPORT_PROCESS_*\` only.
`.trim(),
    keywords: ["model", "vllm", "openai", "anthropic", "preset", "picker"],
  },

  "sidebar.embedding": {
    section: "sidebar",
    title: "Embedding model (removed)",
    short: "The Settings embedding picker is gone. Sequential memory still uses the .env embedder.",
    long: `
Live \`retrieve_past_episodes\` is no longer a desk tool, so Settings no longer offers an embedding preset. \`embedding_preset\` is dropped from \`v3/bulk_settings.json\` on the next Settings load/save.

If a script still constructs CrewAI Memory, the embedder comes from \`VLLM_EMBEDDING_*\` in \`.env\` via \`trading_crew/embedding_presets.py\`.
`.trim(),
    keywords: ["embedding", "embedder", "memory", "removed"],
  },

  "sidebar.run_mode": {
    section: "sidebar",
    title: "Run mode (stock)",
    short: "Autonomous LangGraph, deterministic CrewAI, or harness conductor loop. Applies to Run Analysis and Prediction batch.",
    long: `
Stock runs on v3 can use one of three orchestrators. The same Settings radios are sent on **Run Analysis** and on **Prediction → Start batch**.

- **Autonomous (multi-agent)** — default. LangGraph canvas plus **autonomous desk/task prompts** (\`run_mode_prompts.yaml\`): desks must lock a lean or MIXED-with-conviction; debate does not treat the printed round count as a stop rule; extra turn on a split / MIXED board. Hermes can still open \`debate_rounds_1/2\` or \`risk_rounds_2\`.
- **Deterministic** — sequential v2 crew plus **deterministic prompts**: each desk fills its assigned slot and stops. Debate and risk-round sliders are honoured.
- **Harness** — a conductor loop (not Hermes). It is a **long-horizon** investigation: the step budget is a ceiling, not a reason to finish early. It observes results, picks tools or parallel sub-agents, and must cover **market**, **news**, **fundamentals**, **macro**, **filings** (\`get_report_insights\`), a **bundled debate** (\`desk=debate\`: bull and bear in one crew; each side may \`research_web\` / news / fundamentals mid-exchange), and a **bundled risk debate** (\`desk=risk\`: aggressive, conservative, and neutral in one crew — first each writes an independent opinion, then they debate each other). A **spawn planner** expands each desk brief into a full child instruction (assignment, constraints, priors to re-check, verify, return-to-parent) from \`harness_plugins.yaml\` skills plus the live coverage board. Separate bull/bear or risk-wing spawns remain available after those bundles. If the step budget runs out, the loop takes \`exhaust_steps\` closing turns (no new spawns) so the PM can write a ticket from the desks that already ran — HOLD when they conflict, with that conflict in the rationale — instead of a canned size-0 stub. If the transcript overflows the model window, the loop **compacts** earlier observations and continues instead of failing. Spawned desks use **harness remits** in \`run_mode_prompts.yaml\` (they pick their own tools; no tool-order or round-N script). Packaged standing rules live in \`harness.yaml\`. \`research_web\` uses Gemini Google Search first when \`GEMINI_WEB_SEARCH=1\`, then SearXNG.

Commodity is always sequential and uses the deterministic prompt contract. M1–M7 after the LLM (stops, critic, Kelly/vol, gates) run on all stock modes.
`.trim(),
    keywords: ["run mode", "autonomous", "deterministic", "harness", "langgraph", "sequential", "batch"],
  },

  "sidebar.dialogue": {
    section: "sidebar",
    title: "Run mode + debate / risk",
    short: "Autonomous and harness stock use different prompts and no debate/risk sliders. Deterministic and commodity still expose sliders.",
    long: `
- **Run mode (stock)** — **Autonomous** uses the LangGraph scheduler plus autonomous prompts (continue until a lean or MIXED-with-conviction). **Deterministic** is the v2 CrewAI pipeline plus one-slot-and-stop prompts. **Harness** is a conductor loop plus replacement remits (no tool-order or round-N scripts). Both **Run Analysis** and **Prediction batch** read this radio. Personas live in \`agents.yaml\`; the mode contract is \`run_mode_prompts.yaml\`.
- **Autonomous / harness debate / risk** — no operator sliders. Autonomous: opening debate is one bundled crew (bull opens with tools, bear rebuts with tools, bull closes); the graph may take an extra turn on a split or MIXED board; Hermes can open \`debate_rounds_1/2\` or \`risk_rounds_2\`. Harness: finish is refused until a bundled \`debate\` spawn (and a risk desk) have written; extra bull/bear one-shots stay available. Those keys are omitted from the Settings bulk document.
- **Debate / risk sliders** — shown only for **Deterministic** stock and commodity (sequential). More rounds = more adversarial turns and linearly more LLM cost.
- **CrewAI memory toggle** — removed. Live \`retrieve_past_episodes\` is gone; CrewAI Memory was a different, expensive per-task analyze pass and is no longer an operator control. The Episodes tab / Resolve tape is unchanged.
`.trim(),
    keywords: ["debate", "rounds", "run mode", "autonomous"],
  },

  "sidebar.memory": {
    section: "sidebar",
    title: "LLM-enriched memory (removed)",
    short: "Settings no longer offers CrewAI Memory. It was not retrieve_past_episodes.",
    long: `
CrewAI Memory was **not** \`retrieve_past_episodes\`. It auto-saved each sequential task through extra LLM analyze calls (~60+/run) and did not read the 30-day tape.

Live desks no longer retrieve past episodes. Settings therefore dropped this toggle; \`memory\` is stripped from \`v3/bulk_settings.json\`. The Episodes tab and Resolve still use the episodic store. Scripts may still pass \`memory=True\` into \`TradingCrew\` for tests.
`.trim(),
    keywords: ["memory", "removed", "crewai"],
  },

  "sidebar.tools": {
    section: "sidebar",
    title: "Tools",
    short: "Per-agent enable/disable checkboxes. Disabling a tool hands the agent a shorter tool list at construction time.",
    long: `
Per-agent tool toggles. Each analyst has a curated list of tools they can call (see \`trading_crew/tools.py::DEFAULT_AGENT_TOOLS\`). Disabling a tool removes it from that agent's tool list before the crew is constructed — the agent will fall back to its training knowledge for that information rather than fetching live data. Useful when:

- A specific tool is timing out (e.g. Tavily rate-limited).
- You want to test the crew on cached/offline data.
- You want to A/B compare an agent's reasoning with vs without a particular tool.
`.trim(),
    keywords: ["tools", "agents", "yfinance", "tavily"],
  },

  "sidebar.runs": {
    section: "sidebar",
    title: "Recent runs",
    short: "Click any past kickoff to re-load its full Workflow / Reports state into the dashboard without re-running.",
    long: `
Every kickoff is saved atomically to \`~/.trading_crew/runs/{TICKER}/{ts}.json\`. The sidebar lists the most recent runs across all tickers; clicking one rehydrates the entire UI state (workflow diagram, per-task reports, final decision, execution result) without any LLM cost.

**Refresh** re-reads the on-disk index — useful if a run completed in another tab.

**Clear all** moves the entire \`runs/\` tree into a sibling \`runs.trash-<ts>/\` folder under \`$TRADINGCREW_CACHE_DIR\` and starts a fresh empty index. The deletion is *not* permanent: if you hit the button by accident you can recover the history by renaming the trash folder back to \`runs/\`. The button only clears UI run records — it does **not** touch the LanceDB episodic memory under \`crewai_storage/\` (use \`rm -rf\` for that, after confirming you really mean it).
`.trim(),
    keywords: ["history", "runs", "clear", "delete"],
  },

  "sidebar.book": {
    section: "sidebar",
    title: "Book (paper / prod)",
    short: "Which portfolio book your runs flow into. paper is sandboxed; prod is the live book.",
    long: `
\`PortfolioState\` files are partitioned by **book** so you can iterate on a strategy in \`paper\` without polluting the \`prod\` audit trail. Defaults to \`paper\`. The Portfolio tab respects the same selector; switching books re-fetches the snapshot.
`.trim(),
    keywords: ["paper", "prod", "book"],
  },

  /* ============================================== WORKFLOW PANELS === */

  "panel.cascade": {
    section: "panels",
    title: "Cascade-status badge",
    short: "Market regime detected before the debate starts. CRISIS short-circuits the 18-agent crew and goes straight to ABSTAIN.",
    long: `
The **Cascaded Controller** (paper §5.3) runs before any agent fires:

1. Pulls the last 252 bars of OHLCV.
2. Classifies the regime: \`TREND / RANGE / HIGH_VOL_TREND / HIGH_VOL_RANGE / CRISIS / UNKNOWN\`.
3. Picks a route:
   - \`FULL_DEBATE\` (default) — all 18 agents run.
   - \`RISK_HEAVY\` (HIGH_VOL) — debate runs, risk team gets priority.
   - \`CRISIS_OVERRIDE\` — skip the 18-agent debate, emit a synthetic ABSTAIN \`PortfolioDecision\`, run the deterministic risk gate.

Saves ~30K tokens on CRISIS runs while preserving an identical audit trail.
`.trim(),
    keywords: ["regime", "controller", "crisis"],
  },

  "panel.critic": {
    section: "panels",
    title: "Reflective Critic",
    short: "Runs after the PM. 5-stage protocol + 3-temperature consistency vote. Mode <2/3 forces ABSTAIN.",
    long: `
Not a CrewAI Agent — calls the LLM directly so it can run **N samples at different temperatures** (default 3 samples at \`temperature=0\`, plus 2 stochastic ones). Each sample returns a typed \`CritiqueResponse\`:

- \`intent_ok\` — does the action follow from the rationale?
- \`evidence_ok\` — every numeric claim must carry a \`[source: <id>]\` tag whose \`<id>\` is in the decision's \`sources\` list. Pure provenance check, not a re-fetch.
- \`risk_ok\` — macro / liquidity / regime / compliance risks honestly enumerated?
- \`counterfactual_flip_evidence\` — the single fact that would flip the action.
- \`verdict ∈ {RATIFY, REVISE, ABSTAIN}\` with optional revised action / size / confidence.

REVISE loops up to \`max_iterations\` times. If the 3 votes don't agree by ≥2/3 → forced ABSTAIN.
`.trim(),
    keywords: ["critic", "reflective", "consistency"],
  },

  "panel.sources": {
    section: "panels",
    title: "Sources",
    short: "Deduplicated list of provenance identifiers cited by the PM. Every quantitative claim in the rationale must trace back to one of these.",
    long: `
Every tool's response ends with a \`Source: <identifier> · retrieved <UTC ISO timestamp>\` line. Analysts copy the identifier into inline \`[source: <id>]\` tags next to each quantitative claim. The PM emits the deduplicated set of identifiers in the \`sources\` field of \`PortfolioDecision\`.

The Reflective Critic uses this list to validate that every number in the rationale traces back to a real fetch — claims without a matching source tag are flagged as fabricated.
`.trim(),
    keywords: ["provenance", "citations", "sources"],
  },

  "panel.order_ticket": {
    section: "panels",
    title: "Order ticket",
    short: "Typed ActionProposal — side, target weight, conviction tier — bridged from PortfolioDecision by the M1 layer.",
    long: `
The M1 bridge (\`trading_crew/agentic/bridge.py\`) converts the PM's \`PortfolioDecision\` into a typed \`ActionProposal\`:

\`\`\`
ActionProposal {
  symbol, side (LONG/SHORT/FLAT), target_weight,
  conviction (LOW/MED/HIGH), horizon_days,
  entry/stop/target, valid_until
}
\`\`\`

Pure deterministic — no LLM. The schema is what the M5 sizer + M2 simulator consume.
`.trim(),
    keywords: ["action", "proposal", "M1"],
  },

  "panel.sizing": {
    section: "panels",
    title: "Sizing breakdown",
    short: "Three caps applied to target weight: Fractional Kelly, vol target, CVaR clamp. The smallest binds.",
    long: `
The M5 sizer (\`trading_crew/agentic/risk/sizing.py\`) applies three independent caps and uses the **smallest**:

1. **Fractional Kelly** — \`f* = (E[r] − rf) / σ²\` scaled by \`kelly_fraction=0.25\`. Full Kelly is mathematically optimal for log-wealth growth but empirically too aggressive on noisy estimates; quarter-Kelly is the practitioner default.
2. **Vol target** — cap so the position contributes ≤ \`vol_target=10%\` to portfolio annualised volatility. High-vol stocks get smaller weights.
3. **CVaR clamp** — cap so the position's 95% CVaR (expected loss in the worst 5% of days) is ≤ 2% of NAV.

Then multiplied by a \`[0.5, 1.0]\` confidence / risk-debate multiplier and clamped to \`max_position_weight\` (default 20%).

The panel surfaces *which* cap bound the final weight — that's the binding constraint to relax or tighten.
`.trim(),
    keywords: ["Kelly", "CVaR", "vol", "sizing"],
  },

  "panel.risk_gates": {
    section: "panels",
    title: "Risk gates",
    short: "Six hard predicates. All evaluated (not short-circuited) so every breach is reported.",
    long: `
Six predicates evaluated in \`trading_crew/agentic/risk/gates.py\` — every breach is reported, not just the first:

1. **Concentration** — |weight| ≤ \`max_position_weight=20%\`.
2. **Leverage** — gross exposure / NAV ≤ \`max_leverage=1.5x\`.
3. **Drawdown kill-switch** — if the book is already down 20% peak-to-trough, no new positions until manual reset.
4. **Single-position CVaR** — tail loss of this one trade ≤ 2% NAV.
5. **Stale data** — last bar must be within 5 days of decision. Refuses to trade on month-old prices.
6. **Cash sufficiency** — long buys must have cash ≥ notional + estimated fees (no implicit borrowing).
`.trim(),
    keywords: ["gates", "risk", "concentration", "leverage"],
  },

  "panel.execution": {
    section: "panels",
    title: "Execution result",
    short: "Next-bar fill simulator (M2). Fees + half-spread + Almgren-Chriss √impact + partial fills.",
    long: `
\`trading_crew/agentic/execution/simulator.py\` models:

- **Next-bar fill semantics** — orders submitted at the close fill at the *next* bar's open. Single most important guardrail against "implicit execution" cheating in backtests (paper §6.1).
- **Costs** — fees + half-spread + Almgren-Chriss square-root impact (impact ∝ √(order_size / ADV)). Trading 5% of average daily volume costs noticeably more than trading 0.5%.
- **Partial fills** — orders exceeding 5% ADV partially fill; remainder reported as \`PARTIAL_FILL\` rather than silently transacted at the same price.

The simulator is the same code path used by Backtest and RL training — that's deliberate (paper §6.1 "no implicit execution").
`.trim(),
    keywords: ["fill", "slippage", "impact", "M2"],
  },

  "panel.cost_sensitivity": {
    section: "panels",
    title: "Cost sensitivity sweep",
    short: "Same trade simulated under low / standard / high cost regimes. Shows fragility to friction.",
    long: `
Re-runs the M2 simulator with three cost presets — \`low\`, \`standard\`, \`high\` (or \`futures_low / standard / high\` for commodities) — so you can see whether the trade's edge survives realistic transaction costs.

A wide spread between low and high P&L is a robustness flag — the strategy depends on a friction regime that may not hold in stressed markets.
`.trim(),
    keywords: ["costs", "slippage", "robustness"],
  },

  "panel.episode": {
    section: "panels",
    title: "Episode recorded",
    short: "Memory write — (state, action, regime, decision_ts, outcome_ts) appended to JSONL. Resolved later when the horizon elapses.",
    long: `
Every run appends an \`Episode\` to \`~/.trading_crew/memory/episodes.jsonl\`. The episode is \`PENDING\` until \`outcome_ts = decision_ts + horizon_days\` elapses, at which point you can click **Resolve outcomes & reflect** in the Memory tab to materialise the realised return + alpha + max drawdown + LLM reflection.

The Episodes table is the human tape. Resolve writes 30-day lessons for Hermes. Prediction desks do not retrieve past episodes as a live tool.
`.trim(),
    keywords: ["episode", "memory", "embargo"],
  },

  "panel.stagedTargets": {
    section: "panels",
    title: "Staged targets (scale-out plan)",
    short: "Deterministic T1/T2/T3 profit tiers the risk layer derives from the validated final target — a partial-exit plan, not a single all-or-nothing exit.",
    long: `
The LLM emits one \`target_price\`. The risk layer (\`stops.compute_staged_targets\`) turns that into a **scale-out ladder**: three tiers between entry and the validated final target, each with the fraction of the position to exit there (\`weight\`, summing to ~1.0).

Each tier shows:
- **price** — the level to trim at.
- **%** — signed move from entry (profit-direction positive).
- **R:R** — reward-to-risk at that tier (distance to tier ÷ distance to stop).
- **exit** — fraction of the position to scale out at the tier.

This is deterministic and additive: it never overrides the PM's final target, it just structures how you'd realise it. Tiers only appear for directional (BUY/SELL) decisions with a valid entry / stop / target.
`.trim(),
    keywords: ["staged", "targets", "scale-out", "partial exit", "T1", "T2", "T3", "reward risk"],
  },

  "panel.calibration": {
    section: "panels",
    title: "Conviction calibration",
    short: "Is the desk's stated confidence honest? Brier score + reliability bins + over/under-confidence gap over resolved directional episodes.",
    long: `
The PM attaches a \`conviction_score\` in [0,1] to every call, and M5 sizing scales risk by it. This panel measures whether that number is **calibrated**: when the desk says "0.8 conviction", does it win ~80% of the time?

- **Brier** — mean squared error between conviction and the realised win/loss. Lower is better; **0.25** is the "always guess 0.5" baseline.
- **Hit-rate vs conviction** — overall realised win-rate next to the mean predicted conviction. Big gap ⇒ miscalibrated.
- **Confidence gap** — \`mean(predicted − realised)\`. **Positive = overconfident** (oversizing losers); **negative = underconfident** (leaving edge on the table); near zero = well-calibrated.
- **Reliability bins** — conviction bucketed on [0,1]; the grey bar is mean predicted conviction, the coloured edge is realised hit-rate. Aligned bars ⇒ calibrated.

Scope: only **RESOLVED, directional (BUY/SELL)** episodes count (HOLD/ABSTAIN carry no bet). **Basis** toggles raw return vs benchmark-relative alpha; **cohort** restricts to the current workflow version (default) or all history. Measurement-only — it does not change sizing today.
`.trim(),
    keywords: ["calibration", "brier", "conviction", "reliability", "overconfident", "hit rate"],
  },

  "panel.backtest_extra_axes": {
    section: "panels",
    title: "Extra grid axes",
    short: "Optionally sweep max_leverage and drawdown_kill_threshold alongside the default 3 axes.",
    long: `
Adds **opt-in** axes to the L3 grid search:

- \`max_leverage ∈ {1.0, 1.5, 2.0}\` — risk-gate cap on total gross exposure as a multiple of NAV.
- \`drawdown_kill_threshold ∈ {0.10, 0.20, 0.30}\` — fraction of peak NAV at which the kill-switch flattens the book.

Both default to **off** because they multiply the grid size (3× per axis). The defaults from \`SizingConfig\` / \`GateConfig\` are used when the axis is disabled.
`.trim(),
    keywords: ["max_leverage", "drawdown", "kill-switch", "grid"],
  },

  "panel.backtest_cost_sweep": {
    section: "panels",
    title: "Cost-model sweep",
    short: "Re-runs the entire grid under low/standard/high cost presets and picks the best (config, cost) pair.",
    long: `
A *robustness* axis. When enabled, the L3 search runs once per cost preset:

- For stocks: \`low\` → \`standard\` → \`high\` (commission + slippage + impact ramped up).
- For futures: \`futures_low\` → \`futures_standard\` → \`futures_high\` (futures-specific friction model with notional × tick-size + spread).

The "winner" must be the best across the **union** of all points — a strategy that only wins under low-friction is fragile and is automatically deprioritised.
`.trim(),
    keywords: ["costs", "robustness", "futures"],
  },

  "panel.backtest_validation": {
    section: "panels",
    title: "Validation scheme",
    short: "Walk-forward (default) or Combinatorial Purged CV (López de Prado, AFML ch.12).",
    long: `
**Walk-forward** rolls through the dataset chronologically: \`[train | embargo | test]\` and slides forward by \`test_size\`. Simple, no information leakage, but you only get one fold per slide.

**CPCV** (Combinatorial Purged Cross-Validation) splits observations into \`N\` contiguous groups and enumerates every \`N-choose-k\` combination as a test set, with **per-side embargoes** around each test group to prevent label leakage from adjacent train neighbours. This yields \`C(N, k)\` folds from the same data — far more out-of-sample evaluations for the same compute, at the cost of slightly biased (positively correlated) fold statistics.
`.trim(),
    keywords: ["CPCV", "walk-forward", "validation", "purging"],
  },

  "panel.backtest_history": {
    section: "panels",
    title: "Backtest OHLCV history",
    short: "How many years of daily OHLCV to feed the engine. Defaults to 5y so long-term regimes are represented.",
    long: `
Controls the size of the historical window passed to the walk-forward / CPCV engine.

- **1y** — short-term only; useful for fast iteration but the equity curve barely covers one regime (e.g. the most recent leg of a bull or bear).
- **3y** — covers one typical earnings cycle plus a mild regime change.
- **5y** (default) — covers one full bull-and-bear macro cycle for most large-cap names; this is also what \`backtest_setup\` uses for hit-rate / expectancy estimation.
- **7-10y** — full multi-cycle stress; recommended when grid-searching robustness toggles (max_leverage, drawdown_kill) because rare tail events are sparse in shorter windows.

Longer history doesn't change the simulator — it just gives the proposals more out-of-sample bars to fill against, which usually improves the deflated Sharpe stability across folds.
`.trim(),
    keywords: ["history", "lookback", "5y", "10y", "backtest"],
  },

  "panel.rl_horizon_mode": {
    section: "panels",
    title: "RL horizon mode",
    short: "Short-term / Balanced / Long-term presets that auto-fill the train + eval windows.",
    long: `
RL training and the backtest tab now share the same horizon vocabulary:

- **Short-term** — \`train=252d\` (~1y), \`eval=30d\` (~6 weeks). Fast iteration; only the current regime is in the training set.
- **Balanced** (default) — \`train=750d\` (~3y), \`eval=60d\`. Covers one earnings cycle and a mild regime change.
- **Long-term** — \`train=1260d\` (~5y), \`eval=252d\` (~1y). Full bull/bear cycle in the train set; the eval window is a held-out year for honest out-of-sample reporting.
- **Custom** — lets you type the two windows by hand without snapping to a preset.

Why both: the multi-horizon \`backtest_setup\` already feeds the PM short + long expectancy. Picking a long-term RL mode here keeps the L4 policy honest on the same horizon — short-term trained policies overfit to recent regimes and look great in-sample but lose out-of-sample, exactly the failure mode \`eval_window_days\` is meant to catch.
`.trim(),
    keywords: ["RL", "horizon", "short-term", "long-term", "train_window", "eval_window"],
  },

  "panel.rl_algorithm": {
    section: "panels",
    title: "RL algorithm",
    short: "Which trainer to run. PPO is the default; CVaR-PPO is risk-sensitive; CQL/C51/Decision Transformer are alternative trainers.",
    long: `
- **PPO** — vanilla Proximal Policy Optimisation (Schulman 2017).
- **CVaR-PPO** — risk-sensitive variant. Lower-CVaR-tail steps get extra advantage weight, so the policy is *more* eager to avoid worst-case bars.
- **CQL** — Conservative Q-Learning, offline RL on the stored episode log without re-running the env.
- **C51 / QR-DQN** — distributional RL. The critic models the *full distribution* of returns; useful for risk-aware sizing.
- **Decision Transformer** — frames trading as conditional sequence modelling on offline trajectories.

All five share the same env + checkpoint format. Promote works identically across algorithms.
`.trim(),
    keywords: ["RL", "PPO", "CVaR", "CQL", "C51", "Decision Transformer"],
  },

  "panel.rl_risk_aversion": {
    section: "panels",
    title: "Risk aversion (CVaR-PPO)",
    short: "Extra weight on the bottom CVaR α-quantile of returns. 0 = vanilla PPO; 0.5 = +50% weight on tail-loss steps.",
    long: `
Multiplies the advantage of every step whose realised reward falls in the **bottom \`cvar_alpha\`** quantile of the rollout by \`1 + risk_aversion\`.

Interpretation: at \`risk_aversion = 0.5\` the policy pays 1.5× attention to tail-loss bars when computing gradients — equivalent to "I'd give up half my expected return to avoid worst-case draws". Mirrors the M5 fractional-Kelly mindset *inside* the policy gradient itself.

Set to \`0\` to recover vanilla PPO.
`.trim(),
    keywords: ["CVaR", "risk", "tail", "PPO"],
  },

  "panel.rl_cvar_alpha": {
    section: "panels",
    title: "CVaR quantile (α)",
    short: "Quantile cutoff for the CVaR re-weighting. 0.20 = re-weight the worst 20% of bar-level rewards.",
    long: `
Sets the empirical quantile threshold for the CVaR re-weighting:
every step whose reward falls *below* the \`cvar_alpha\` percentile of the rollout's reward histogram is treated as part of the tail.

Smaller α → narrower tail → more concentrated punishment for the rare bad days. \`0.20\` is a standard institutional choice; \`0.05\` is aggressive.
`.trim(),
    keywords: ["CVaR", "alpha", "quantile"],
  },

  "panel.rl_turnover_penalty": {
    section: "panels",
    title: "Turnover penalty (bps)",
    short: "Extra reward penalty per unit of |Δweight| in basis points. Encourages lower-turnover policies.",
    long: `
The M2 simulator already deducts fees + spread + slippage from the per-bar reward. **Turnover penalty** is an *additional* shaping term equal to \`penalty_bps × 10⁻⁴ × |Δweight|\` charged on top.

Used to push the policy toward "trade only when you have edge" — necessary when the underlying cost model is too forgiving and PPO learns to scalp.
`.trim(),
    keywords: ["turnover", "shaping", "costs"],
  },

  "panel.rl_drawdown_penalty": {
    section: "panels",
    title: "Drawdown shaping",
    short: "Adds `coef × Δdrawdown` to the reward whenever drawdown deepens.",
    long: `
When drawdown grows within an episode, the policy receives an extra negative reward equal to \`coef × (new_dd − old_dd)\`. \`0\` disables the shaping; higher values produce a more risk-averse policy that exits early.
`.trim(),
    keywords: ["drawdown", "shaping", "risk"],
  },

  "panel.rl_curriculum": {
    section: "panels",
    title: "Volatility curriculum",
    short: "Train on low-vol regimes first; gradually expand to the full volatility distribution as training progresses.",
    long: `
When enabled, the env sorts every valid start bar by 20-bar rolling realised vol. At training progress = 0 only the **bottom 10% of vol-sorted bars** are eligible as episode starts; at progress = 1 the full distribution is sampled. The pool size grows linearly with progress.

Speeds up + stabilises early learning — the value head can find traction on calm regimes before being asked to handle vol shocks.
`.trim(),
    keywords: ["curriculum", "volatility", "training"],
  },

  "panel.rl_universe": {
    section: "panels",
    title: "Policy universe (multi-ticker)",
    short: "Comma-separated tickers. When set, the state vector grows a one-hot tail identifying which ticker the bar belongs to.",
    long: `
Single-ticker policies are the default. Setting **policy universe** trains one network shared across multiple tickers: the state vector grows a one-hot tail of length \`|universe|\` identifying which ticker the current bar belongs to.

Useful when you have many correlated assets (sector ETFs, factor portfolios, futures legs) and want one policy that's seen all of them rather than N independent overfit nets.

Empty string → single-ticker policy (default).
`.trim(),
    keywords: ["universe", "multi-ticker", "policy"],
  },

  "panel.workflow_parallel_risk": {
    section: "panels",
    title: "Parallel risk debate",
    short: "Aggressive + Conservative agents now run in parallel instead of sequentially; Neutral waits for both.",
    long: `
Phase 2E — Aggressive and Conservative argue *independently* from the same prior context. Both tasks are marked \`async_execution=True\` and the Neutral synthesiser lists both in its \`context\` so it still waits for both to complete before producing the round's verdict.

Saves roughly one round-trip-time per risk round vs the previous sequential rebuttal flow.
`.trim(),
    keywords: ["risk", "parallel", "async"],
  },

  "panel.workflow_per_agent_llm": {
    section: "panels",
    title: "Per-agent LLM selection",
    short: "Configure a different model per agent via the LLM_PER_AGENT env var (JSON).",
    long: `
\`LLM_PER_AGENT\` is a JSON map from \`agent_key\` to override dict. Example::

    LLM_PER_AGENT='{
      "social_analyst": {"model": "small-cheap-model", "temperature": 0.5},
      "research_manager": {"model": "long-context-model-131k"}
    }'

Each override may set \`model\`, \`base_url\`, \`api_key\`, \`provider\`, and/or \`temperature\`. Agents without an entry use the global default. Cheap agents (Social, Macro) can use a smaller/cheaper model; long-context ones (Research Manager, Bull/Bear) can pin the 131K hosted endpoint.
`.trim(),
    keywords: ["LLM", "per-agent", "model"],
  },

  "panel.workflow_analyst_cache": {
    section: "panels",
    title: "Analyst output cache",
    short: "Caches each analyst's markdown payload, keyed by (ticker, tools-hash, prompt-hash, trade-date).",
    long: `
Cache lives at \`~/.trading_crew/cache/analyst/<sha1>.json\`. Each entry stores the markdown output of one analyst task. The cache key combines ticker, trade date, the user's tool-enable flags (normalised), and the task id.

Re-runs with identical config can therefore skip the analyst LLM calls. Debate / risk / PM still re-run because their outputs depend on the *interaction* between agents and on the current portfolio state — neither is captured by the cache key.

Tunable via the \`TRADINGCREW_CACHE_DIR\` env var.
`.trim(),
    keywords: ["cache", "analyst", "performance"],
  },

  "panel.workflow_tool_retry": {
    section: "panels",
    title: "Tool retry w/ exponential backoff",
    short: "Tavily + yfinance wrappers retry 3 times with jittered exponential backoff on transient errors.",
    long: `
\`_with_retry\` is a tiny dependency-free helper (no \`tenacity\`). It wraps any callable and retries up to \`attempts\` times with delay \`base_delay × 2^i + uniform(0, base_delay)\` between attempts, capped at \`max_delay\`.

Applied to:

- Tavily search (news / global news / insider tools)
- yfinance \`Ticker.history\` (OHLCV + indicator paths)

A transient 502 or socket-timeout on a single call no longer blows up the analyst.
`.trim(),
    keywords: ["retry", "backoff", "Tavily", "yfinance"],
  },

  "panel.workflow_regime_split": {
    section: "panels",
    title: "HIGH_VOL → TREND / RANGE split",
    short: "HIGH_VOL_TREND keeps the full 18-agent debate. HIGH_VOL_RANGE routes to a risk-only mini-crew.",
    long: `
Previously HIGH_VOL was a single regime. Phase 2E splits it into two by checking the trend filter even when realised vol is above the high-vol threshold:

- **HIGH_VOL_TREND** — vol-heavy *with* directional drift. Full debate runs (the trend can be ridden with proper sizing).
- **HIGH_VOL_RANGE** — vol-heavy *without* direction. The cascaded controller skips the 18-agent analyst fan-out and routes to a risk-only mini-crew, because the analyst layer rarely uncovers actionable signal in this regime.

Surfaced via the cascade-status badge on the Workflow tab.
`.trim(),
    keywords: ["regime", "HIGH_VOL", "cascade"],
  },

  "panel.backtest_seed": {
    section: "panels",
    title: "Synthetic proposal seeder",
    short: "Re-runs M1→M5→M2→M3 over historical bars to bulk-generate N proposals without invoking the LLM.",
    long: `
The L3 grid search needs at least \`train + embargo + test\` proposals per ticker (default = 5). For fresh tickers with only a handful of real runs, this button bulk-creates **synthetic** \`ActionProposal\`s by:

1. Sampling existing episodes for the ticker.
2. Perturbing horizon and expected-return / target-weight inside a small jitter envelope.
3. Persisting each as a \`PENDING\` \`Episode\` (so the harness sees them as logged proposals).

No LLM cost — purely deterministic perturbation of saved past PM decisions.
`.trim(),
    keywords: ["seed", "synthetic", "M3", "backtest"],
  },

  "panel.action_proposal": {
    section: "panels",
    title: "Action Proposal",
    short: "Post-M1-bridge typed contract that the deterministic M5 sizer + M2 simulator consume.",
    long: `
Bridges the LLM's narrative intent (\`PortfolioDecision\`) into the strict deterministic action contract (\`ActionProposal\`). All downstream layers (M5 sizing, M5 risk gates, M2 simulator, M3 memory) consume this — never the raw \`PortfolioDecision\`. That separation is what makes the audit trail tractable: the LLM's prose lives in \`rationale\`, the *action* lives in a typed Pydantic schema.
`.trim(),
    keywords: ["bridge", "M1", "ActionProposal"],
  },

  /* ================================================== AGENT HOVER === */

  "agent.market_analyst": {
    section: "agents",
    title: "Market Analyst",
    short: "Reads price action + technical indicators (SMA20/50, RSI14, 60d return). Calls the trend regime.",
    long: `
Veteran technical analyst. Speaks in moving averages, RSI, trend regime and risk/reward — never mental arithmetic, always tool calls for the actual numbers.

**Tools**: \`get_stock_data\`, \`get_indicators\`.

**Output**: MARKET REPORT with trend regime, last close, SMA20/SMA50, RSI14, momentum read, and a 1-line bias. Every number carries an inline \`[source: <identifier>]\` tag.
`.trim(),
    keywords: ["technical", "indicators", "trend"],
  },

  "agent.social_analyst": {
    section: "agents",
    title: "Social Analyst",
    short: "Surfaces sentiment + crowd-positioning signals. Scans social-style headlines for tone, dispersion, narrative shifts.",
    long: `
Sentiment desk lead. Scans recent news and social-style headlines for tone, dispersion, and any narrative shifts that could move the stock.

**Tools**: \`get_market_context\`, \`get_news\`.

**Output**: SOCIAL REPORT — overall sentiment, top 3 narrative themes (each with a \`[source: <url>]\` tag), 1-line bias.
`.trim(),
    keywords: ["sentiment", "social", "news"],
  },

  "agent.news_analyst": {
    section: "agents",
    title: "News Analyst",
    short: "Identifies material catalysts — company news, macro events, insider moves. Ticker-aware (country-specific themes).",
    long: `
Newswire-trained analyst. Separates signal from noise and ties every headline to a likely impact on price.

**Tools**: \`get_market_context\`, \`get_news\`, \`get_global_news\`, \`get_insider_transactions\`.

The global-news search is grounded in the company's home country: an Indian name gets RBI / Union Budget / INR headlines, a US name gets Fed / CPI / DXY headlines. No more "US treasury yields" on every ticker.
`.trim(),
    keywords: ["catalyst", "news", "insider"],
  },

  "agent.fundamentals_analyst": {
    section: "agents",
    title: "Fundamentals Analyst",
    short: "Evaluates financial health — P/E, EPS, margins, FCF, leverage. Reads statements as prose.",
    long: `
CFA-charter-holder. Reads income statements like prose, anchors every claim in the filings, never guesses a multiple.

**Tools**: \`get_fundamentals\`, \`get_balance_sheet\`, \`get_cashflow\`, \`get_income_statement\`, \`get_long_term_thesis\` (trailing multi-year), \`get_forward_estimates\` (forward consensus — targets, fwd P/E, next-FY growth, revisions), \`get_sector_peers\` (always pass the run ticker; live industry + country screen; pass \`peers=\` to replace a wrong set). If the desk omits ticker, the tool binds the kickoff symbol — it does not invent a name.

**Output**: FUNDAMENTALS REPORT — growth, profitability, leverage/cash, valuation snapshot (P/E, P/B), a forward-consensus read (implied upside + fwd-vs-trailing EPS), and 1-line bias.
`.trim(),
    keywords: ["fundamentals", "valuation", "earnings"],
  },

  "agent.reports_analyst": {
    section: "agents",
    title: "Reports Analyst",
    short: "Reads the company's OWN primary filings — concall transcripts, investor decks, annual-report MD&A, credit ratings — for management guidance + target-vs-actual delivery.",
    long: `
Buy-side analyst who lives in the primary documents. Reads what management COMMITTED to (revenue / margin / order-book / capex guidance) and checks it against what was DELIVERED — beat, miss or in-line.

**Tools**: \`get_report_insights\` (market-aware: **Indian** names pull screener.in concall / annual-report / credit-rating PDFs over HTTP — no Firecrawl — and extract text + tables from every page, with the selected filings extracted in parallel; the **newest annual report** is enough (prior years already sit in its comparatives / MD&A) while several concalls are kept for guidance vintages; **US** names pull the SEC EDGAR feed — 10-K / 10-Q reports + earnings 8-Ks resolved to their Exhibit 99.1 press release — then a dedicated \`REPORT_PROCESS_*\` LLM map-reduces the whole document in **context-sized chunks** (128k or 256k) BEFORE it reaches the agent, point-in-time-safe). Unchanged filings (listing date/title + first-page hash) reuse the last digest instead of a same-day timer. \`get_market_context\`, \`get_forward_estimates\` (the forward-consensus fallback for names with no primary-filing feed).

**Output**: REPORTS & GUIDANCE briefing that LEADS with a Firefly-style **trend signal card** — directional, confidence-scored signals derived from the filings (OUTLOOK_TONE, REVENUE_GUIDANCE raised/maintained/cut, MARGIN_GUIDANCE, DELIVERY_TRACK_RECORD beat/in-line/miss, DEMAND_MOMENTUM, CREDIT_TRAJECTORY, and a synthesised NET_GUIDANCE_BIAS) — plus a **COMPUTED FINANCIAL TREND SIGNALS** block (deterministic, from hard quarterly numbers: REVENUE_TREND / EARNINGS_TREND with YoY growth + acceleration, and MARGIN_TREND). Below the card, the supporting evidence: guidance/targets, target-vs-actual, **reported actuals & segments**, **order book / operations**, **MD&A and named risks**, and any credit-rating action. Broader briefing is kept (not dropped as NONE). Never fabricates a target or a signal direction: marks a signal INSUFFICIENT_DATA when the filings/numbers don't support it, and says so plainly when a name is in neither the Indian nor the US filing feed.

The whole card is parsed into a structured **ReportSignals** object (with a single numeric \`guidance_bias_score\` in [-1, 1]), emitted as a \`report_signals\` event, rendered on the Workflow tab, and persisted with the run + prediction so it's queryable downstream (and shown alongside the optional signal-fusion panel as advisory context).
`.trim(),
    keywords: ["reports", "filings", "guidance", "concall", "annual report", "target", "transcript"],
  },

  "agent.macro_analyst": {
    section: "agents",
    title: "Macro Analyst",
    short: "Reads the macro regime (rates, FX, vol, oil, gold) for the COMPANY'S HOME COUNTRY. Translates into a tilt on the position.",
    long: `
Top-down macro PM. Translates yields, dollar moves and VIX regime into a tilt on the position.

**Tools**: \`get_market_context\`, \`get_macro_data\`, \`get_global_news\`.

The macro basket is country-aware: US → 10y UST + DXY + VIX + WTI + gold; India → Nifty + India VIX + INR/USD + Sensex + Brent; UK → 10y gilt + GBP/USD + FTSE; etc.
`.trim(),
    keywords: ["macro", "rates", "FX"],
  },

  "agent.geopolitical_analyst": {
    section: "agents",
    title: "Geopolitical Analyst",
    short: "Identifies geopolitical / regulatory / trade-policy catalysts. The analyst writes the search query from the brief + remit + any gap.",
    long: `
Ex-policy-desk analyst. Tracks tariffs, export controls, sanctions, election risk and regional conflict for their direct impact on revenue and supply chain. Quantifies exposure: "X% of revenue is at risk if rule Y passes".

**Tools**: \`get_market_context\`, \`get_geopolitical_news\`, \`get_global_news\`, \`get_supply_chain_risk\`.

\`get_market_context\` is facts only (country, industry, name). You pass \`query=\` on the news tools — there is no canned Hormuz / Taiwan / OPEC phrasebook.
`.trim(),
    keywords: ["geopolitical", "sanctions", "regulation"],
  },

  "agent.sector_analyst": {
    section: "agents",
    title: "Sector / Peer Analyst",
    short: "Compares the ticker to its sector peers across YTD return / market cap / P/E. Calls alpha vs the median peer.",
    long: `
Sector specialist. Only commits to a name if it's better than its closest 4–5 peers on momentum, valuation or growth.

**Tools**: \`get_market_context\`, \`get_sector_peers\`, \`get_indicators\`.

Peer baskets are a live Yahoo industry + country screen, not a ticker dictionary. The desk may pass \`peers=\` when filings named better comparables. Quote the tool's printed alpha vs the median.
`.trim(),
    keywords: ["sector", "peers", "alpha"],
  },

  "agent.quant_analyst": {
    section: "agents",
    title: "Quant / Options Analyst",
    short: "Reads positioning + option-implied expectations. ATM IV, put/call ratio, sell-side consensus.",
    long: `
Volatility-desk quant. Reads IV term-structure, skew and put/call ratio for a positioning signal, and weights it against sell-side consensus and price targets.

**Tools**: \`get_options_summary\`, \`get_analyst_recommendations\`.

**Output**: QUANT REPORT — implied move, put/call read, sell-side consensus, 1-line tilt.
`.trim(),
    keywords: ["options", "IV", "skew"],
  },

  "agent.bull_researcher": {
    section: "agents",
    title: "Bullish Researcher",
    short: "Builds the strongest evidence-based case for going long. Cites the analyst report each argument leans on — no vibes.",
    long: `
Optimistic but disciplined buy-side researcher. In the bundled debate they share one crew with the Bear (open → rebut → close) and may call \`research_web\` / \`get_news\` / \`get_fundamentals\` / \`get_stock_data\` mid-exchange to prove a point. They also share a scratch pad (compact analyst TLDRs + prior turns). Open-web search uses Gemini Google Search first when \`GEMINI_WEB_SEARCH=1\`, then rotates SearXNG engines (duckduckgo web, privacywall, infospace, …) and keeps a hop only when the run LLM marks the URLs as about the query.

Round 1 is an independent opening run in parallel with the Bear (not a rebuttal). Later rounds alternate: each turn must explicitly rebut the most recent Bear claim on the pad.
`.trim(),
    keywords: ["bull", "debate", "researcher"],
  },

  "agent.bear_researcher": {
    section: "agents",
    title: "Bearish Researcher",
    short: "Builds the strongest evidence-based case against going long. Treats every bull thesis as a hypothesis to falsify.",
    long: `
Skeptical short-seller. In the bundled debate they share one crew with the Bull and may call \`research_web\` / \`get_news\` / \`get_fundamentals\` / \`get_stock_data\` mid-exchange to falsify a claim. They also share a scratch pad. Extra one-shot turns after the bundle rebut the most recent bull claim.
`.trim(),
    keywords: ["bear", "debate", "researcher"],
  },

  "agent.research_manager": {
    section: "agents",
    title: "Research Manager",
    short: "Moderates the bull/bear debate. Picks the side with stronger evidence — does NOT split the difference.",
    long: `
Head of research. Reads the shared scratch pad plus the last bull/bear turns (not the five full analyst reports). Doesn't pick the side they like — picks the side with stronger evidence, and explicitly notes 1–2 things that would change their mind.

\`STANCE: MIXED\` is a locked thesis with named evidence on both sides — not an instruction for the PM to sit out. HOLD is only for a true split (bullish count == bearish count, both > 0) or a majority of directional desks against the lean. Neutrals do not create a split.

**Output**: RESEARCH THESIS — STANCE (BULL/BEAR/MIXED), 3 key drivers, 3 key risks, 1–2 falsifiers. Every line that contains a number ends with an inline \`[source: …]\` tag inherited from the analyst report it cites.
`.trim(),
    keywords: ["manager", "synthesis", "thesis"],
  },

  "agent.quality_reviewer": {
    section: "agents",
    title: "Quality Reviewer",
    short: "Audits the thesis. A real [source: …] tag is SUPPORTED. Advisory to the PM — does not drop tagged facts.",
    long: `
Internal-audit lead for the research desk. For every claim in the manager's thesis, marks SUPPORTED (the claim already has \`[source: <identifier>]\`), SUPPORTED-qualitative (\`[analyst judgement]\`), or UNSUPPORTED (untagged bare number or an example.com placeholder), then grades the thesis 0–10.

The tag is the citation. A news URL or tool id (\`screener.in\`, \`yfinance\`, \`get_report_insights\`) is SUPPORTED even if that URL does not reappear in the truncated analyst pack. Tools stamp \`[source: …]\` on valid figures that left the extract untagged, so those numbers are not marked UNSUPPORTED. Quality Review must not tell the PM to omit or down-weight tagged claims. Score below 4 may right-size; it never forces a HOLD by itself.
`.trim(),
    keywords: ["audit", "quality", "review"],
  },

  "agent.trader": {
    section: "agents",
    title: "Trader",
    short: "Turns the thesis into a concrete trade plan: entry, stop, target, horizon, sizing. Validates against historical base rates.",
    long: `
Buy-side trader. Translates views into entries, stops, targets, time horizons and rough sizing.

**Tools**: \`get_event_proximity\` (next earnings + days-to-event + EPS estimates).

If \`days_to_event\` is less than the intended horizon, the trader either shortens the horizon or explicitly sizes for the catalyst. If the historical hit-rate from \`backtest_setup\` disagrees with intuitive confidence (<40% hit rate but 70% confidence wanted), the plan is tightened or the size shrunk.

MIXED-with-conviction on a directional majority is a smaller or hedged plan on that majority — not an automatic sit-out. Sit out only on a true split (bullish count == bearish count, both > 0) or a majority-opposed board.
`.trim(),
    keywords: ["trader", "plan", "sizing"],
  },

  "agent.risk_aggressive": {
    section: "agents",
    title: "Aggressive Risk Analyst",
    short: "Argues for higher conviction sizing when the edge justifies it. Pushes back when EV is being left on the table.",
    long: `
Risk analyst with a high-conviction bias. Pushes back when the team is leaving expected value on the table. This debate is about **size and execution**, not direction — the trader already has a plan.

In harness, the bundled \`desk=risk\` crew runs two rounds: each wing writes an independent opinion, then they debate. Extra one-shot turns after that remain available. Call \`research_web\` / news / fundamentals / price mid-exchange when a size claim needs a source.
`.trim(),
    keywords: ["aggressive", "risk", "sizing"],
  },

  "agent.risk_conservative": {
    section: "agents",
    title: "Conservative Risk Analyst",
    short: "Argues for tight stops, smaller size, or skipping the trade. Highlights tail risks and correlated exposures.",
    long: `
Capital-preservation specialist. Highlights tail risks, correlated exposures, and reasons the trade could quietly fail.

In the bundled risk crew, round 1 is an independent size-down / skip case — not a rebuttal of a turn that has not happened. Round 2 debates the other openings. Extra one-shot turns after the bundle remain available.
`.trim(),
    keywords: ["conservative", "risk", "tail"],
  },

  "agent.risk_neutral": {
    section: "agents",
    title: "Neutral Risk Analyst",
    short: "Synthesises the aggressive + conservative arguments into a balanced sizing/execution plan.",
    long: `
Risk analyst who weighs aggressive and conservative arguments and proposes a middle path defensible to the PM. Does not re-decide side.

In the bundled risk crew, Neutral first writes their own size opinion, then debates the other two and locks one size. Extra one-shot turns after the bundle remain available. On the sequential graph, Neutral can still emit \`REOPEN_TRADER\` when the plan's size is indefensible.
`.trim(),
    keywords: ["neutral", "risk", "synthesis"],
  },

  "agent.compliance_officer": {
    section: "agents",
    title: "Compliance Officer",
    short: "Flags regulatory / sanctions / ESG / conflict-of-interest issues. Status: CLEAR / FLAGGED / BLOCKED.",
    long: `
Buy-side compliance. Vetoes trades that touch sanctions or restricted lists (e.g. chip-export-controlled entities), and downgrades trades with material ESG or conflict-of-interest concerns.

Hard rule: if status is BLOCKED, the PM's action MUST be HOLD and size 0. FLAGGED caps size at 1.0% of book.
`.trim(),
    keywords: ["compliance", "sanctions", "ESG"],
  },

  "agent.portfolio_manager": {
    section: "agents",
    title: "Portfolio Manager",
    short: "Final decision maker. Emits typed PortfolioDecision (action, horizon tier, size, confidence, drivers, risks, sources).",
    long: `
Final decision maker. Reads the trader's plan, the risk team's debate, and the fundamentals analyst's long-term thesis, then commits to a single sized position + horizon tier with a one-paragraph rationale.

**Output**: typed \`PortfolioDecision\` JSON (gated by a confidence guardrail) with:
- **5-tier retail action** (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL). Older records persisted with the legacy 3-tier alphabet (OVERWEIGHT / NEUTRAL / UNDERWEIGHT) are coerced to BUY / HOLD / SELL when they're re-loaded.
- **3-tier horizon** (SHORT ≤ 3 months / MEDIUM ≈ few months to 1 year / LONG = 1-10 year compounder / value thesis). The desk defaults to MEDIUM; escalation to LONG requires the fundamentals analyst's \`get_long_term_thesis\` tool output (revenue / EPS CAGR, ROCE trend, debt trajectory).
- \`horizon_days\` — the numeric review window (SHORT: 10-60d; MEDIUM: 90-252d; LONG: 504-1260d representative windows, with the 5-10 year narrative in the \`long_term_view\` free-form field).
- \`long_term_view\` — REQUIRED on every decision, not just LONG.  For LONG-tier calls it's the full 2-5 sentence multi-year narrative (CAGR + ROCE + debt + falsifier).  For SHORT / MEDIUM calls it's a 1-2 sentence "the multi-year thesis is intact / broken / N/A" line, so the LONG-panel context is never silently dropped from the final report.  Older records may carry this under the legacy field name \`long_term_thesis\`; the schema accepts either.
- size_pct_of_book, confidence ∈ [0,1], key_drivers, key_risks, falsifiers, compliance_status, geopolitical_flags, and the deduplicated \`sources\` list of provenance identifiers.

Hard rules:
- Never claim >0.85 confidence on a single-name trade.
- If backtest hit-rate <40% AND every horizon has expectancy ≤ 0%, cap confidence at 0.6 and shrink size accordingly.
- If compliance is BLOCKED, action MUST be HOLD.
- STRONG_BUY / STRONG_SELL are reserved for one-sided evidence across the chosen tier's row + the adjacent tier.
- HOLD requires size_pct_of_book == 0; directional tiers require size_pct_of_book > 0.
- HOLD is for a true split (bullish count == bearish count, both > 0), a majority-opposed board, contradictory evidence, or BLOCKED. Neutrals do not create a split. MIXED-with-conviction on a 3–0 or 3–1 board is not an automatic HOLD.
- \`horizon == LONG\` requires \`long_term_view\` to be a full 2-5 sentence narrative AND requires a \`get_long_term_thesis\` citation.  For SHORT / MEDIUM decisions, \`long_term_view\` still has to be a non-empty 1-2 sentence context line ("multi-year thesis is intact/broken/N/A") — the desk shows it in every final report so the LONG-panel evidence is always visible.
`.trim(),
    keywords: ["PM", "decision", "PortfolioDecision", "horizon", "SHORT", "MEDIUM", "LONG"],
  },

  "concept.horizon_tiers": {
    section: "glossary",
    title: "Horizon tiers (SHORT / MEDIUM / LONG)",
    short: "3-tier investment-horizon alphabet: SHORT (≤3 months), MEDIUM (~few months), LONG (1-10 year compounder / value thesis).",
    long: `
Every PortfolioDecision now carries an explicit **horizon tier** alongside \`horizon_days\`:

- **SHORT** — ≤ 60 trading days (~3 months).  Catalyst / earnings plays, technical breakouts.  The exception.
- **MEDIUM** — 61-252 days (~few months to a year).  The desk's DEFAULT.  Ind-AS quarterly-print theses, multi-quarter earnings trends, mid-term relative-strength setups.
- **LONG** — 253+ days (1-10 years).  A genuine compounder / value thesis anchored on multi-year revenue / EPS CAGR + stable-or-improving ROCE.  Requires the fundamentals analyst's \`get_long_term_thesis\` tool to have run (revenue / EPS CAGR, debt trajectory, shareholding stability) and requires the backtest_setup 5y / 10y row to show positive expectancy.

The backtest_setup tool surfaces all three tiers in one panel (with rows tagged [SHORT] / [MEDIUM] / [LONG] and human-readable windows like "1260d ≈ 5y"), plus a **Per-tier best expectancy** summary the PM anchors on.

Because outcome resolution can't wait a decade, LONG-tier decisions have their **accounting horizon** clamped to 5 years (1260d) — the *narrative* 5-10 year thesis lives in the \`long_term_view\` free-form field (accepted under the legacy name \`long_term_thesis\` too). The 30-day harness review still scores the short path while this thesis stays pending.
`.trim(),
    keywords: ["horizon", "SHORT", "MEDIUM", "LONG", "5-year", "10-year", "compounder"],
  },

  "concept.harness": {
    section: "glossary",
    title: "Prediction harness — Hermes (Mass is legacy)",
    short: "One loop after Resolve: score the 30-day book, open one evidence-backed experiment. Mass promote stays for later A/B.",
    long: `
The prediction is made by **strategy knobs + agent prompts + graph topology**. Hermes selects the next change from Resolve evidence.

**Generate feedback.** Run predictions, then **Resolve** (Episodes tab or \`POST /api/memory/resolve\`). A 30-day review scores the path. Hurt desks write \`recommended_change\` only when the LLM reflection ran — skip-LLM reviews often have empty change text, so Hermes will not invent a desk prompt.

**Hermes diagnosis.** On FAILURE with \`n ≥ min_reviews_to_change\`, open **one** experiment from Resolve's descriptive lessons (which desk misread, which tools hurt, what was over-weighted): dominant hurt desk → desk prompt; reports/filings blamed → skip reports; debate evidence/web blamed → close the evidence drain; split boards → extra debate; hit-rate OK but alpha/DD failed → size down; hit-rate failed → two debate rounds; trader reopen blamed → no reopen; majority playbook vote (must already be a YAML action) → that knob; else the next unused YAML catalog knob. SUCCESS does not auto-install. One LOSS never rewrites production. The LLM does not pick knobs.

**Apply.** Trader / research manager / PM get strategy + the open experiment. A desk-prompt experiment prepends only that desk. A playbook experiment is live on the graph router (\`skip_reports\`, debate/evidence/risk rounds, \`no_trader_reopen\`; debate-round knobs also override the sequential slider). Analyst YAML files are not rewritten. Mass promote is not required.

**Hermes UI.** The Training tab is the 30-day book scoreboard (verdict, n, hit-rate, the experiment Resolve already opened). Strategy, prompts, and bulk run snapshot live on **Settings** (header gear → \`/settings\`). Save writes \`$TRADINGCREW_CACHE_DIR/v3/desk_strategy.yaml\`, \`prompt_overlay.yaml\`, and \`bulk_settings.json\` — not the git yaml, unless you revert. \`TRADINGCREW_STRATEGY_PATH\` still wins if set.

**Legacy Mass.** 1PO/2TO/3PO writes and \`scripts/promote_graph.py\` remain in the tree so Mass-alone can still be A/B'd. They are not the live training step. An existing \`promoted.json\` still applies.

Hermes does not sweep values of a knob and does not use an LLM to pick the winner. One named treatment is live at a time; the 30-day book score keeps or rejects it.
`.trim(),
    keywords: ["harness", "Hermes", "Mass", "self-learn", "promote", "knob", "desk addenda", "legacy"],
  },

  /* ================================================ GLOSSARY TERMS === */

  "glossary.ohlcv": {
    section: "glossary", title: "OHLCV",
    short: "Open / High / Low / Close / Volume — the five numbers reported for each trading day.",
    long: "The standard daily price record. Open = first traded price, High/Low = day's extremes, Close = last traded price, Volume = number of shares/contracts traded.",
    keywords: ["open", "high", "low", "close", "volume"],
  },
  "glossary.sma": {
    section: "glossary", title: "SMA / EMA",
    short: "Simple/Exponential Moving Average — a smoothed line through the closing prices. SMA20 = 20-day average.",
    long: "Smoothes price noise. Price > SMA20 > SMA50 = uptrend; price < SMA20 < SMA50 = downtrend. The EMA weights recent prices more heavily.",
    keywords: ["moving average", "EMA", "SMA"],
  },
  "glossary.rsi": {
    section: "glossary", title: "RSI",
    short: "Relative Strength Index over 14 days, bounded 0–100. >70 = overbought; <30 = oversold.",
    long: "Momentum oscillator that tracks the ratio of recent gains to recent losses. Over 70 usually signals a stretched rally (mean-reversion likely); under 30 signals an exhausted decline.",
    keywords: ["RSI", "overbought", "oversold"],
  },
  "glossary.macd": {
    section: "glossary", title: "MACD",
    short: "Moving Average Convergence Divergence — the difference of two EMAs. Crossings above the signal line are typically read as bullish.",
    long: "MACD = 12-EMA − 26-EMA. The signal line is a 9-EMA of MACD. The histogram is MACD − signal. Bullish crossover (MACD crosses above signal) is a common momentum trigger.",
    keywords: ["MACD", "momentum"],
  },
  "glossary.bollinger": {
    section: "glossary", title: "Bollinger Bands",
    short: "Moving average ± k·σ envelope. Closes near the upper band are stretched; closes near the lower band are exhausted.",
    long: "Default = 20-SMA ± 2 stdev. Width expands in high-vol regimes and contracts in calm ones. Used for volatility breakout and mean-reversion setups.",
    keywords: ["bollinger", "bands", "volatility"],
  },
  "glossary.atr": {
    section: "glossary", title: "ATR",
    short: "Average True Range — typical daily price-range size. Used to set volatility-aware stops.",
    long: "ATR(14) = 14-day average of max(High−Low, |High−PrevClose|, |Low−PrevClose|). A 2×ATR stop is a common volatility-normalised stop placement.",
    keywords: ["ATR", "stop", "volatility"],
  },
  "glossary.pe": {
    section: "glossary", title: "P/E ratio",
    short: "Price / Earnings-per-share. Roughly the number of years of current earnings the market is paying for the stock.",
    long: "S&P 500 typically trades at 15–20. Above 30 means the market expects strong growth. Below 10 often signals deep value or distress.",
    keywords: ["PE", "valuation", "earnings"],
  },
  "glossary.pb": {
    section: "glossary", title: "P/B ratio",
    short: "Price / Book value. Especially useful for banks — book value is the regulated capital base.",
    long: "Price-to-book < 1 often signals distress (the market values the firm at less than its accounting equity). >5 typical for asset-light tech.",
    keywords: ["PB", "book", "valuation"],
  },
  "glossary.fcf": {
    section: "glossary", title: "Free Cash Flow",
    short: "Operating cash flow − capex. The cash actually available to return to shareholders.",
    long: "Distinct from earnings — earnings include depreciation and other non-cash items. FCF is what funds dividends, buybacks, debt paydown.",
    keywords: ["FCF", "cash flow"],
  },
  "glossary.kelly": {
    section: "glossary", title: "Kelly criterion",
    short: "Optimal sizing formula for log-wealth growth: f* = (E[r] − rf) / σ². We use fractional Kelly (¼ × Kelly) — full Kelly is too aggressive on noisy estimates.",
    long: "John Kelly Jr. (1956). Full Kelly maximises geometric growth but is empirically too volatile because edge estimates are noisy. Practitioners use ¼ or ½ Kelly. This system uses 0.25.",
    keywords: ["Kelly", "sizing", "growth"],
  },
  "glossary.cvar": {
    section: "glossary", title: "CVaR",
    short: "Conditional Value-at-Risk — expected loss in the worst 5% of days. Tail-risk measure.",
    long: "CVaR(α) = E[loss | loss > VaR(α)]. More conservative than VaR (which only reports the threshold, not the average breach). The M5 sizer clamps positions so CVaR ≤ 2% of NAV.",
    keywords: ["CVaR", "tail", "VaR"],
  },
  "glossary.sharpe": {
    section: "glossary", title: "Sharpe ratio",
    short: "(Annualised excess return) / (annualised volatility). >1 is decent for a single strategy.",
    long: "Reward per unit of risk. Sharpe = (Rp − Rf) / σp. Typical benchmarks: >1 decent, >2 great, >3 likely a fitting artefact or insider edge.",
    keywords: ["Sharpe", "performance"],
  },
  "glossary.sortino": {
    section: "glossary", title: "Sortino ratio",
    short: "Like Sharpe but only penalises downside volatility. (Excess return) / (downside σ).",
    long: "Argues that upside volatility shouldn't be penalised — only downside. Sortino > Sharpe when the return distribution is positively skewed.",
    keywords: ["Sortino"],
  },
  "glossary.calmar": {
    section: "glossary", title: "Calmar ratio",
    short: "Annualised return / max drawdown. Penalises strategies that hit deep peak-to-trough holes.",
    long: "A return-to-pain ratio. Useful complement to Sharpe because Sharpe is path-blind — it doesn't care whether returns were smooth or came in one big run after a near-bankruptcy.",
    keywords: ["Calmar", "drawdown"],
  },
  "glossary.deflated_sharpe": {
    section: "glossary", title: "Deflated Sharpe ratio",
    short: "Bailey & López de Prado (2014). Corrects raw Sharpe for selection-bias when many configurations are tested against the same history.",
    long: "Estimates the probability that an observed Sharpe could be reached by a no-skill strategy after N trials. The L3 grid search ranks configurations by Deflated Sharpe rather than raw Sharpe to avoid picking the winner of a 27-way coin-flip tournament.",
    keywords: ["DSR", "Deflated", "overfit"],
  },
  "glossary.contango": {
    section: "glossary", title: "Contango",
    short: "Futures curve where back-month prices > front-month. Long-holders pay roll cost.",
    long: "Common in commodities where storage costs + interest dominate (e.g. WTI when oversupplied). The opposite is backwardation, where back-month < front-month and holders earn roll yield.",
    keywords: ["contango", "futures", "curve"],
  },
  "glossary.backwardation": {
    section: "glossary", title: "Backwardation",
    short: "Futures curve where back-month prices < front-month. Long-holders earn positive roll yield.",
    long: "Typical of supply-constrained commodities (e.g. natural gas in winter, crude during geopolitical tightness). Holders rolling each month earn the curve slope.",
    keywords: ["backwardation", "futures", "roll yield"],
  },
  "glossary.open_interest": {
    section: "glossary", title: "Open interest",
    short: "Total number of open options/futures contracts. Rising OI = new money entering; falling OI = closing.",
    long: "Distinct from volume — volume counts every transaction, OI tracks net contracts outstanding. Sharp OI rise + price rise = strong directional positioning.",
    keywords: ["OI", "open interest"],
  },
  "glossary.put_call_ratio": {
    section: "glossary", title: "Put/Call ratio",
    short: "Total put OI / total call OI. >1 = bearish positioning; <0.5 = bullish.",
    long: "Sentiment + positioning gauge. Extremes are often contrarian — very high put/call = excess fear = potential bottom; very low = complacency = potential top.",
    keywords: ["put", "call", "ratio"],
  },
  "glossary.iv": {
    section: "glossary", title: "Implied Volatility (IV)",
    short: "Vol the options market is pricing in, back-solved from option premia using Black-Scholes.",
    long: "Distinct from realised vol. Earnings + macro events typically spike IV before the event and crush it after (\"vol crush\"). The implied move = ATM straddle / spot ≈ the move the market expects by expiry.",
    keywords: ["IV", "implied", "volatility"],
  },
  "glossary.adv": {
    section: "glossary", title: "ADV",
    short: "Average Daily Volume. The M2 simulator caps fills at 5% of ADV per bar.",
    long: "Used to compute market impact (Almgren-Chriss model: impact ∝ √(order_size / ADV)). Big positions in thin names are expensive; the simulator surfaces that cost so the LLM doesn't size into illiquid trades.",
    keywords: ["ADV", "liquidity", "impact"],
  },
  "glossary.drawdown": {
    section: "glossary", title: "Max drawdown",
    short: "Largest peak-to-trough loss over the equity curve. Triggers the kill switch if it exceeds 20%.",
    long: "MDD = min((NAV_t − peak_NAV_t) / peak_NAV_t). The M5 drawdown kill-switch blocks new positions once the live book has lost 20% peak-to-trough — circuit breaker against catastrophic loss.",
    keywords: ["drawdown", "MDD"],
  },
  "glossary.alpha": {
    section: "glossary", title: "Alpha",
    short: "Return in excess of the benchmark. Realised α = (strategy return) − (benchmark return) over the same period.",
    long: "L2 outcome resolution scores α = realised − SPY benchmark per episode. Positive α = strategy outperformed the market for that holding period.",
    keywords: ["alpha", "excess"],
  },
  "glossary.beta": {
    section: "glossary", title: "Beta",
    short: "Stock's sensitivity to market moves. β=1 ⇒ moves 1:1 with the index; β=2 ⇒ amplifies moves 2x.",
    long: "Regression coefficient of stock returns on benchmark returns. High-β names amplify market moves; low-β names (utilities, staples) dampen them.",
    keywords: ["beta", "market"],
  },
  "glossary.z_score": {
    section: "glossary", title: "Z-score",
    short: "Standardised distance from the mean: (x − μ) / σ. >2 = 2 standard deviations above the mean.",
    long: "Lets you compare apples to oranges by normalising. The RL feature extractor z-scores realised vol so the policy network sees a comparable signal across high- and low-vol regimes.",
    keywords: ["z-score", "standardise"],
  },
  "glossary.spread": {
    section: "glossary", title: "Bid-ask spread",
    short: "Difference between the highest buyer's bid and the lowest seller's ask. Half the spread is paid on each round-trip.",
    long: "Major implicit transaction cost. The M2 simulator deducts half-spread per fill. Tight spreads (large caps) cost ~1 bp; wide spreads (thin names, emerging markets) can cost 20+ bps.",
    keywords: ["spread", "bid", "ask"],
  },

  /* ============================================ HELP DRAWER INTRO === */

  "_intro": {
    section: "intro",
    title: "TradingCrew — in-app help",
    short: "Hover any (i) icon for a quick definition, or click it to deep-link this drawer to the matching section. Press Cmd/Ctrl+K to search.",
    long: `
Welcome. This drawer is the in-app version of the project README — searchable, anchored, and accessible from every (i) icon in the UI.

**Quick navigation**

- Press \`Cmd/Ctrl+K\` to open the search.
- Click any (i) icon in the dashboard to jump straight to that section.
- Click an anchor in the Tabs / Workflow panels / Agents / Glossary lists below.

**No finance background?** Skim the Glossary first — P/E, RSI, MACD, Kelly, CVaR, Sharpe, Deflated Sharpe, contango/backwardation, etc.
`.trim(),
  },
};
