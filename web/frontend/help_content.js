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
    short: "Live SVG diagram of every agent + the deterministic post-LLM pipeline (sizing, risk gates, simulator, episode write).",
    long: `
The **Workflow** tab is the orchestration view. It shows the 18-agent (stock) or 17-agent (commodity) debate as a strict left-to-right pipeline, with one column per pipeline stage. Genuinely parallel agents (Bull/Bear debate, 3-analyst risk team) stack vertically inside a single column; every sequential hand-off gets its own column:

\`\`\`
analysts (4×2 grid) → bullish / bearish → research mgr → quality reviewer → trader → risk team (3 stacked) → compliance → portfolio mgr → decision
\`\`\`

Wrapped around the LLM debate are three deterministic gates:

1. **Cascaded Controller** (before the debate) — detects the market regime from 252 bars of OHLCV. If \`CRISIS\`, skips the entire 18-agent debate and emits a synthetic ABSTAIN decision.
2. **Reflective Critic** (after the PM) — runs the same critique prompt at three temperatures; if 2/3 votes don't agree, the decision is forced to ABSTAIN.
3. **Deterministic execution pipeline** (M1→M5→M2→M3) — sizer (fractional Kelly + vol target + CVaR), risk gates (concentration / leverage / drawdown), next-bar fill simulator with fees/spread/impact, episode write.

**Edge animation** (mirrors React Flow / Argo / GitLab CI conventions):

- **idle** — thin static gray line
- **running** — **cyan dashed line that flows** toward the destination (the stroke animation makes direction obvious)
- **done** — solid green

Tool chips below each agent light yellow on tool start and green on success.

**Zoom + pan.** The toolbar in the top-right has \`−\` / \`+\` / \`Fit\` / \`100%\` buttons. You can also hold **Ctrl + scroll** anywhere inside the diagram to zoom at the cursor; drag the scrollbars to pan when zoomed in. The current zoom level is shown as a percentage between \`−\` and \`+\`.
`.trim(),
    keywords: ["agents", "pipeline", "diagram", "zoom", "animated"],
  },

  "tab.charts": {
    section: "tabs",
    title: "Charts",
    short: "180-day OHLCV price chart with toggleable EMA / SMA / Bollinger overlays + dedicated RSI and MACD panels.",
    long: `
180-day price chart powered by [Chart.js](https://www.chartjs.org/) and indicators computed server-side with [stockstats](https://github.com/jealous/stockstats).

The chart's end date is driven by **Analysis date** in the sidebar — useful for back-dating an analysis to see what the indicators looked like on a specific day. The agent crew itself always uses *live* data (the analyst tools call yfinance at run time).

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
    title: "Reports",
    short: "Each completed task's full markdown output in execution order, plus the final-decision card.",
    long: `
Renders every CrewAI task output as the run progresses. Each card shows the agent role, the rendered markdown, and an inline source-attribution list. Every numeric claim in a report carries an inline \`[source: <identifier>]\` tag — the **Reflective Critic** validates these tags against the PM's \`sources\` list to distinguish real data from hallucinated numbers.

The bottom of the tab shows the **final-decision card**: action (OVERWEIGHT / NEUTRAL / UNDERWEIGHT), size, confidence, key drivers, key risks, falsifiers, compliance status, and the deduplicated sources list.

### "Degraded" badge
A report card tagged with an amber **DEGRADED** badge means the LLM emitted a *tool-call list* as its final answer for that task instead of synthesising the expected markdown. This is a known limitation of some open-source LLMs (the \`hosted-vllm-oss\` preset is the usual offender) on tool-heavy tasks like News, Macro, Social, and Geopolitical — the model fires a tool call and then fails to compose a closing assistant message, so the agent loop terminates with the tool-call objects in the final-answer slot. The defensive patch in \`trading_crew/_patches.py\` catches the failure, prints a clear placeholder showing which tool calls were attempted, and lets the rest of the crew run. Mitigations: switch to a closed-source LLM via the LLM picker, re-run the workflow (the failure is non-deterministic), or shorten that analyst's tool chain.
`.trim(),
    keywords: ["analyst", "rationale", "sources", "provenance", "degraded", "empty report"],
  },

  "tab.memory": {
    section: "tabs",
    title: "Memory",
    short: "Audit-grade episodic memory (paper §M3). Embargo-aware retrieval + L2 outcome resolution + LLM reflection.",
    long: `
Episodic memory stores \`(state, action, outcome, timestamp)\` tuples for every past run on this ticker.

**Embargo-aware retrieval**: an episode is invisible until its outcome materialises (\`outcome_ts = decision_ts + horizon_days\`). Without this, future-known returns leak into "past" retrieval and silently inflate reported edge.

**Resolve outcomes & reflect** (L2 training): sweeps episodes whose horizon has elapsed; prices the outcome from yfinance for \`[decision_ts, outcome_ts]\`; scores realised return + alpha vs SPY + max drawdown; asks the LLM for a constrained one-paragraph lesson (≤600 chars). Reflections become visible to future runs via the \`retrieve_past_episodes\` tool.

The Market Analyst + Trader both call this tool automatically — that's how the system **learns** between runs without touching LLM weights.
`.trim(),
    keywords: ["episode", "embargo", "reflection", "L2", "training"],
  },

  "tab.portfolio": {
    section: "tabs",
    title: "Portfolio",
    short: "Book snapshot (cash, positions, P&L) + HRP / Mean-Variance / Equal-Risk allocator preview across recent proposals.",
    long: `
Two distinct components live behind this tab:

1. **PortfolioState book** — single JSON file per book (\`paper\` / \`prod\`), atomic-write via \`<path>.tmp\` + \`os.replace\`. Cash, open positions (symbol, qty, avg_cost, last_price, opened_ts), realised P&L, peak NAV, max drawdown.
2. **Allocator preview** (triggered when ≥2 tickers have recent proposals). Three methods are switchable in the UI:
   - **HRP** (Hierarchical Risk Parity, López de Prado 2016) — cluster by correlation distance, recursively split the risk budget. Robust to ill-conditioned covariance. **Default for thin universes.**
   - **MEAN_VARIANCE** (Markowitz with long-only sum-to-budget constraint).
   - **EQUAL_RISK** (1/N then vol-scale).

The allocator only **redistributes** within the gross budget (default 0.80 = 80% of NAV invested, 20% cash). It cannot increase gross exposure.
`.trim(),
    keywords: ["allocator", "HRP", "Markowitz", "book"],
  },

  "tab.backtest": {
    section: "tabs",
    title: "Backtest",
    short: "Walk-forward replay of logged proposals with embargo. Reports Sharpe / Sortino / Calmar / MaxDD / Deflated Sharpe per fold.",
    long: `
**Strategy-level walk-forward** evaluation. Replays *all* logged \`ActionProposal\`s for a ticker through the M1 + M2 + M5 deterministic pipeline (risk gates + cost model + simulator) under strict train / embargo / test folds.

**Why "backtest" appears in two places**: the *tool* \`backtest_setup\` is a decision-time base-rate lookup ("out of 1225 historical setups like this, 20% hit target"). The *tab* is the post-hoc strategy-level harness.

The **Auto-tune sizing & risk gates** panel runs an L3 grid search across \`kelly_fraction × vol_target × max_position_weight\` (optionally extending to \`max_leverage\`, \`drawdown_kill_threshold\`, \`cost_model\`). Ranked by **Deflated Sharpe** (Bailey & López de Prado 2014), which penalises selection-bias from testing many configurations against the same history.

If you need ≥ \`train+embargo+test\` proposals (default 3+1+1 = 5), use the **Seed N synthetic proposals** button to bulk-generate proposals from existing PM decisions without re-running the LLM crew.
`.trim(),
    keywords: ["walk-forward", "CPCV", "Deflated Sharpe", "L3"],
  },

  "tab.training": {
    section: "tabs",
    title: "RL Training",
    short: "L4 — full reinforcement learning. Agents act, the M2 simulator pays out, gradients flow into the policy. Promote a run to expose it to the LLM crew.",
    long: `
The closed-loop reinforcement-learning workbench. Default algorithm is **PPO** (Proximal Policy Optimization — Schulman 2017, clipped surrogate + GAE-λ advantages + value clipping + entropy bonus + gradient clipping). Optional alternatives:

- **CVaR-PPO** — risk-sensitive variant; weights the advantage by the lower-CVaR quantile of returns.
- **CQL** (Conservative Q-Learning) — offline RL; trains off the stored episode log without re-running the env.
- **C51** (Distributional RL) — models the *distribution* of returns; surfaces the lower-quantile estimate to the M5 risk gate.
- **Decision Transformer** — frames trading as conditional sequence modelling on offline trajectories.

State: 21 features (10 lagged log-returns, vol_20, mom_5/20, RSI(14), MACD, Bollinger %b, position weight/unrealised/age, day-of-week sin/cos). Action: 7 discrete weight buckets (-20%, -10%, -5%, 0, +5%, +10%, +20%). Reward: per-bar ΔNAV after the M2 simulator deducts fees + slippage + impact + carry.

**Promote** writes a single pointer in \`~/.trading_crew/rl_runs/promoted/<TICKER>.json\`. Until promoted, the policy stays invisible to the LLM crew (no implicit "use the latest run" fallback).
`.trim(),
    keywords: ["PPO", "CVaR", "CQL", "Decision Transformer", "L4"],
  },

  "tab.logs": {
    section: "tabs",
    title: "Logs",
    short: "Live event-by-event console of every node_started / tool_call / tool_result / node_completed event with timestamps.",
    long: `
Raw event stream from the WebSocket: \`run_started → node_started → tool_call → tool_result → node_completed → final_decision → reflection_records → action_proposal → execution_result → episode_recorded → run_completed\`.

The badge next to the Logs tab counts pending log entries (the events you haven't viewed yet on this tab). Click into the tab to clear the badge.
`.trim(),
    keywords: ["events", "websocket", "trace"],
  },

  /* ========================================================= SIDEBAR === */

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
    short: "Pick the LLM for the next run — open-source local / hosted vLLM or closed-source OpenAI / Anthropic.",
    long: `
Each run can use a different LLM. Pick a **preset** from the dropdown — your selection is persisted to \`localStorage\` and sent with the next \`/ws/analyze\` payload as \`llm_preset\`.

**Open-source presets**
- \`hosted-vllm-oss\` — any OpenAI-compatible vLLM-style endpoint you point \`VLLM_LLM_BASE_URL\` / \`VLLM_LLM_MODEL\` / \`VLLM_LLM_API_KEY\` at in \`.env\`. Recommended default for debate rounds that ship all 8 analyst reports — pick a long-context (131K-ish) model so prompts don't overflow.
- \`local-vllm\` — your local \`vllm serve <model> --port 8081\` (configurable via \`LOCAL_LLM_*\`). Tool calling enabled. Watch the context window so debate rounds stay within the model's limit.

**Closed-source presets**
- \`openai-gpt-4o-mini\` / \`openai-gpt-4o\` — needs \`OPENAI_PROD_API_KEY\` in \`.env\` (kept separate from \`OPENAI_API_KEY=dummy\` used by the local vLLM client).
- \`anthropic-claude-sonnet\` — needs \`ANTHROPIC_API_KEY\` in \`.env\`.

The dropdown greys out closed-source options whose API key isn't configured. Adding more presets: edit \`trading_crew/llm_presets.py::BUILTIN_PRESETS\`.

**Resolution order** inside \`get_llm()\`:
1. \`LOCAL_LLM_*\` env (fallback)
2. \`VLLM_LLM_*\` env (default)
3. \`LLM_PER_AGENT\` JSON (per-agent override)
4. **Active UI preset** (thread-local — what this dropdown sets)
5. Direct kwargs

Per-agent and UI-preset overrides compose: e.g. you can pick \`hosted-vllm-oss\` as the default and still pin the Reflective Critic to \`openai-gpt-4o\` via \`LLM_PER_AGENT\` in \`.env\`.
`.trim(),
    keywords: ["model", "vllm", "openai", "anthropic", "preset", "picker"],
  },

  "sidebar.embedding": {
    section: "sidebar",
    title: "Embedding model picker",
    short: "Pick the embedder used by LLM-enriched memory. Only relevant when memory is ON.",
    long: `
The embedder turns task outputs (PM rationale, analyst summaries, debate synthesis) into vectors so CrewAI's memory can retrieve relevant context on later runs. It's only used when the **LLM-enriched memory** toggle is ON.

**Open-source preset**
- \`vllm-embedding\` — any OpenAI-compatible embedding server you point \`VLLM_EMBEDDING_BASE_URL\` / \`VLLM_EMBEDDING_MODEL\` / \`VLLM_EMBEDDING_API_KEY\` at in \`.env\`. Default deployments use \`embeddinggemma-300m\` (768-d, 2048-token input cap); the \`TruncatingOpenAIEmbedder\` wrapper clips inputs to \`TRADINGCREW_EMBEDDER_MAX_CHARS\` (default 6000 ≈ 1500 tokens) so memory writes never trip the upstream 2048-token limit.

**Closed-source presets** (need \`OPENAI_PROD_API_KEY\` in \`.env\`)
- \`openai-text-embedding-3-small\` — 1536-d, 8191-token input cap.
- \`openai-text-embedding-3-large\` — 3072-d, 8191-token input cap. Better retrieval quality at higher cost.

**Why partition memory by preset?** Embedding dim is locked into the LanceDB schema the first time a store is written. Switching from a 768-d embedder to a 1536-d embedder on the same scope would crash the next \`search_memory\` with *"query dim doesn't match the column vector dim"*. The crew folds the active preset id into the memory \`root_scope\` (\`/crew/<crew>/emb/<preset-id>\`) so switching presets just starts a fresh store side-by-side with the old one.

**Resolution order** inside \`TruncatingOpenAIEmbedder.__init__\`:
1. \`VLLM_EMBEDDING_*\` env (fallback)
2. **Active UI preset** (thread-local — what this dropdown sets)
3. Direct kwargs (tests / explicit construction)

Adding more presets: edit \`trading_crew/embedding_presets.py::BUILTIN_PRESETS\`.
`.trim(),
    keywords: ["embedding", "embedder", "memory", "vector", "dim", "preset"],
  },

  "sidebar.dialogue": {
    section: "sidebar",
    title: "Debate / Risk rounds + memory",
    short: "Sliders for bull/bear debate rounds (1–10) + risk-team rounds (1–10) + shared-memory toggle.",
    long: `
- **Debate rounds** — how many turns the bull and bear researchers take before the Research Manager synthesises. More rounds = more thorough adversarial argumentation but linearly more LLM cost.
- **Risk rounds** — same idea for the Aggressive / Conservative / Neutral risk debate.
- **LLM-enriched memory** — toggles CrewAI's unified short-term / long-term / entity memory. When **ON**, every task save is enriched by an LLM pass (\`analyze_for_save\` / \`extract_memories_from_content\` / \`analyze_for_consolidation\`) and every \`search_memory\` recall runs \`analyze_query\`. The displayed estimate (e.g. *"~108 LLM analysis calls ≈ 1.8 min"*) shows what to expect — it scales with debate / risk rounds because every extra round adds tasks, and each task adds ~3.5 analysis calls on top of the agent's own LLM call. When **OFF**, memory is disabled entirely and agents start each task fresh. Memory embeddings always use the vLLM \`embeddinggemma-300m\` model configured in \`.env\` (truncation-guarded via \`TruncatingOpenAIEmbedder\`).

  *Why this matters for runtime.* The local LLM endpoint caps at ~60 req/min, so the analysis pass adds roughly *count ÷ 60 min* of latency on top of the agent work. We override CrewAI's default 1-worker save pool to 8 workers (configurable via \`TRADINGCREW_MEMORY_SAVE_WORKERS\`) so concurrent saves actually fan out to that budget instead of serializing. The pool size pairs with the existing 10-worker pool inside \`EncodingFlow\` for batch-internal items.
`.trim(),
    keywords: ["debate", "rounds", "memory"],
  },

  "sidebar.memory": {
    section: "sidebar",
    title: "LLM-enriched memory",
    short: "Toggles CrewAI's unified memory. When ON, every save / recall is enriched by an LLM analysis pass.",
    long: `
When **ON**, every CrewAI task save runs an LLM analysis pass that classifies the content into a scope (e.g. \`/crew/trading-crew-NTNX/decisions\`), assigns an importance score, and extracts entities / dates / topics. Every \`search_memory\` tool call also runs an LLM query-analysis step to pick keywords and target scopes.

**Per-run cost.** The sidebar displays an estimate like *"~108 LLM analysis calls ≈ 1.8 min"*. The math:

\`\`\`
tasks ≈ (8 analysts × 2 tasks)              # analyst grid
      + (2 × debate_rounds)                  # bull + bear per round
      + 3                                    # research-mgr / quality-reviewer / trader
      + (3 × risk_rounds)                    # 3 risk personas per round
      + 2                                    # compliance + portfolio manager

calls ≈ tasks × 3.5 + 12                     # ~3.5 analyze calls/task + ~12 recall calls
minutes ≈ calls / 60                         # local LLM rate limit = 60 req/min
\`\`\`

For the default debate=2 / risk=1 sliders that's *~28 tasks × 3.5 + 12 = ~110 calls ≈ 1.8 min*. Cranking debate=10 / risk=10 climbs to *~63 tasks × 3.5 + 12 = ~233 calls ≈ 3.9 min*.

**Parallelism override.** CrewAI's default \`Memory._save_pool\` is a 1-worker \`ThreadPoolExecutor\`, which serializes all saves. We override it to 8 workers (configurable via \`TRADINGCREW_MEMORY_SAVE_WORKERS\`) so concurrent \`remember()\` calls fan out. Combined with the existing 10-worker pool inside \`EncodingFlow\` (for batch-internal items), the analyze calls can actually saturate the local-LLM 60-req/min budget instead of leaving ~90 % of throughput on the table. Without this override, a single ticker run took ~30 min; with it, the same run is back to ~5 min.

When **OFF**, no \`Memory\` object is constructed — every task starts fresh, zero extra LLM calls. Memory embeddings (vLLM \`embeddinggemma-300m\` via \`TruncatingOpenAIEmbedder\`) are still configured in \`.env\` because the \`search_memory\` tool needs them when memory is on; they have no effect when memory is off.
`.trim(),
    keywords: ["memory", "llm", "parallel", "rate-limit", "embedding"],
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

The outcome embargo is enforced server-side: an episode is invisible to \`retrieve_past_episodes\` until its \`outcome_ts + embargo_days\` has passed — that's how the system prevents future-known returns from leaking into "past" retrieval.
`.trim(),
    keywords: ["episode", "memory", "embargo"],
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

**Tools**: \`get_stock_data\`, \`get_indicators\`, \`retrieve_past_episodes\`, \`rl_policy_recommendation\`.

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

**Tools**: \`get_fundamentals\`, \`get_balance_sheet\`, \`get_cashflow\`, \`get_income_statement\`.

**Output**: FUNDAMENTALS REPORT — growth, profitability, leverage/cash, valuation snapshot (P/E, P/B), and 1-line bias.
`.trim(),
    keywords: ["fundamentals", "valuation", "earnings"],
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
    short: "Identifies geopolitical / regulatory / trade-policy catalysts. Industry-aware (shipbuilder → Hormuz; semis → Taiwan).",
    long: `
Ex-policy-desk analyst. Tracks tariffs, export controls, sanctions, election risk and regional conflict for their direct impact on revenue and supply chain. Quantifies exposure: "X% of revenue is at risk if rule Y passes".

**Tools**: \`get_market_context\`, \`get_geopolitical_news\`, \`get_global_news\`, \`get_supply_chain_risk\`.

Industry-aware: a shipbuilder gets Strait of Hormuz / Red Sea / naval procurement; semis get Taiwan + chip-export controls; oil & gas get OPEC quota + Red Sea shipping.
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

Peer baskets are country-aware: MAZDOCK.NS → COCHINSHIP.NS / GRSE.NS / BEL.NS / HAL.NS; NVDA → AMD / AVGO / INTC / TSM / QCOM.
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
Optimistic but disciplined buy-side researcher. Always cites the analyst report each argument leans on — no vibes-based bullishness.

Runs \`debate_rounds\` turns alternating with the Bear; each round sees prior rounds + the 8 analyst reports in context, and must explicitly rebut the most recent Bear turn.
`.trim(),
    keywords: ["bull", "debate", "researcher"],
  },

  "agent.bear_researcher": {
    section: "agents",
    title: "Bearish Researcher",
    short: "Builds the strongest evidence-based case against going long. Treats every bull thesis as a hypothesis to falsify.",
    long: `
Skeptical short-seller. Treats every bull thesis as a hypothesis to falsify, and looks for cracks in numbers, narrative, and positioning.

Runs \`debate_rounds\` turns alternating with the Bull; each round must explicitly rebut the most recent Bull turn.
`.trim(),
    keywords: ["bear", "debate", "researcher"],
  },

  "agent.research_manager": {
    section: "agents",
    title: "Research Manager",
    short: "Moderates the bull/bear debate. Picks the side with stronger evidence — does NOT split the difference.",
    long: `
Head of research. Doesn't pick the side they like — picks the side with stronger evidence, and explicitly notes 1–2 things that would change their mind.

**Output**: RESEARCH THESIS — STANCE (BULL/BEAR), 3 key drivers, 3 key risks, 1–2 falsifiers. Every line that contains a number ends with an inline \`[source: …]\` tag inherited from the analyst report it cites.
`.trim(),
    keywords: ["manager", "synthesis", "thesis"],
  },

  "agent.quality_reviewer": {
    section: "agents",
    title: "Quality Reviewer",
    short: "Audits the thesis. Marks each claim SUPPORTED (with tag) or UNSUPPORTED. Below-5 score halves the trader's default size.",
    long: `
Internal-audit lead for the research desk. For every claim in the manager's thesis, marks SUPPORTED (with the analyst report it cites) or UNSUPPORTED, then grades the thesis 0–10.

A claim counts as SUPPORTED only if it carries an inline \`[source: <identifier>]\` tag. A bare number with no tag is automatic UNSUPPORTED — flagged explicitly so the trader can omit it.
`.trim(),
    keywords: ["audit", "quality", "review"],
  },

  "agent.trader": {
    section: "agents",
    title: "Trader",
    short: "Turns the thesis into a concrete trade plan: entry, stop, target, horizon, sizing. Validates against historical base rates.",
    long: `
Buy-side trader. Translates views into entries, stops, targets, time horizons and rough sizing.

**Tools**: \`get_event_proximity\` (next earnings + days-to-event + EPS estimates), \`backtest_setup\` (per-trade base-rate lookup), \`retrieve_past_episodes\`, \`rl_policy_recommendation\`.

If \`days_to_event\` is less than the intended horizon, the trader either shortens the horizon or explicitly sizes for the catalyst. If the historical hit-rate from \`backtest_setup\` disagrees with intuitive confidence (<40% hit rate but 70% confidence wanted), the plan is tightened or the size shrunk.
`.trim(),
    keywords: ["trader", "plan", "sizing"],
  },

  "agent.risk_aggressive": {
    section: "agents",
    title: "Aggressive Risk Analyst",
    short: "Argues for higher conviction sizing when the edge justifies it. Pushes back when EV is being left on the table.",
    long: `
Risk analyst with a high-conviction bias. Pushes back when the team is leaving expected value on the table.

Runs in parallel with the Conservative analyst; the Neutral analyst then synthesises both into a balanced sizing/execution recommendation.
`.trim(),
    keywords: ["aggressive", "risk", "sizing"],
  },

  "agent.risk_conservative": {
    section: "agents",
    title: "Conservative Risk Analyst",
    short: "Argues for tight stops, smaller size, or skipping the trade. Highlights tail risks and correlated exposures.",
    long: `
Capital-preservation specialist. Highlights tail risks, correlated exposures, and reasons the trade could quietly fail.

Runs in parallel with the Aggressive analyst.
`.trim(),
    keywords: ["conservative", "risk", "tail"],
  },

  "agent.risk_neutral": {
    section: "agents",
    title: "Neutral Risk Analyst",
    short: "Synthesises the aggressive + conservative arguments into a balanced sizing/execution plan.",
    long: `
Risk analyst who weighs aggressive and conservative arguments and proposes a middle path defensible to the PM.
`.trim(),
    keywords: ["neutral", "risk", "synthesis"],
  },

  "agent.compliance_officer": {
    section: "agents",
    title: "Compliance Officer",
    short: "Flags regulatory / sanctions / ESG / conflict-of-interest issues. Status: CLEAR / FLAGGED / BLOCKED.",
    long: `
Buy-side compliance. Vetoes trades that touch sanctions or restricted lists (e.g. chip-export-controlled entities), and downgrades trades with material ESG or conflict-of-interest concerns.

Hard rule: if status is BLOCKED, the PM's action MUST be NEUTRAL and size 0. FLAGGED caps size at 1.0% of book.
`.trim(),
    keywords: ["compliance", "sanctions", "ESG"],
  },

  "agent.portfolio_manager": {
    section: "agents",
    title: "Portfolio Manager",
    short: "Final decision maker. Emits typed PortfolioDecision (action, size, confidence, drivers, risks, sources).",
    long: `
Final decision maker. Reads the trader's plan and the risk team's debate and commits to a single, sized position with a one-paragraph rationale.

**Output**: typed \`PortfolioDecision\` JSON (gated by a confidence guardrail) with action (OVERWEIGHT / NEUTRAL / UNDERWEIGHT), size_pct_of_book, confidence ∈ [0,1], key_drivers, key_risks, falsifiers, compliance_status, geopolitical_flags, and the deduplicated \`sources\` list of provenance identifiers.

Hard rules: never claim >0.85 confidence on a single-name trade; if backtest hit-rate <40%, cap confidence at 0.6 and shrink size accordingly; if compliance is BLOCKED, action MUST be NEUTRAL.
`.trim(),
    keywords: ["PM", "decision", "PortfolioDecision"],
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
