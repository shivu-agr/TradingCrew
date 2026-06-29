# TradingCrew — Detailed Design

> This is the deep-dive companion to the project [README](../README.md). It documents every module, every endpoint, the M1–M7 deterministic pipeline, the L2/L3/L4 training loops, the commodity dashboard, and the finance vocabulary used across the UI.

Multi-agent **stock + commodity** research workflow built with [CrewAI](https://docs.crewai.com).

Two dashboards share the same homepage and the same deterministic backend:

- **`/stock`** — 18-agent equities crew (the original `agentic_workflow.ipynb` lift). Portfolio Manager emits a typed `PortfolioDecision`.
- **`/commodity`** — 17-agent futures crew with futures-aware tools (curve, COT, seasonality, geopolitics). Same debate topology as the equity crew — analysts → bull/bear debate → research manager → quality reviewer → futures trader → 3-way risk debate → compliance → Portfolio Manager. The Portfolio Manager emits a typed `FuturesDecision` that is bridged to a `PortfolioDecision` so it can flow through the same M1–M7 pipeline.

The Portfolio Manager (or its commodity equivalent) emits a typed decision JSON gated by a confidence guardrail. After the LLM debate completes, the decision is handed off to a **deterministic post-LLM pipeline (M1–M7)** that compiles a typed `ActionProposal`, runs risk gates + a fill simulator, books the position, and records an episode for time-aware retrieval on the next run.

> **Asset-class agnostic core.** The deterministic Layer-A pipeline (`trading_crew/agentic/*`) is identical for both dashboards. Only the LLM debate layer and the cost-model preset (`futures_standard` vs `standard`) differ.

```
[M4] Cascaded Controller (regime detector)
   ├── CRISIS  → short-circuit, skip the 18-agent debate, route to ABSTAIN
   └── else    → continue
   ↓
8 analysts (parallel via async_execution)
  [M3] Market Analyst + Trader can call retrieve_past_episodes
  -> DEBATE_ROUNDS x (Bull -> Bear)
    -> Research Manager synthesis
      -> Quality Reviewer audit
        -> Trader (uses get_event_proximity + backtest_setup + retrieve_past_episodes)
          -> RISK_ROUNDS x (Aggressive -> Conservative -> Neutral)
            -> Compliance Officer (CLEAR / FLAGGED / BLOCKED)
              -> Portfolio Manager (output_pydantic = PortfolioDecision)
                ↓
[M4] Reflective Critic (5-stage protocol + 3-temp consistency vote → may revise or ABSTAIN)
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
│   │   ├── portfolio/         #   state, allocator (HRP / MV / EQR)
│   │   ├── execution/         #   cost (equity + futures presets), simulator, pipeline
│   │   ├── memory/            #   episodic / semantic / regime
│   │   ├── risk/              #   VaR/CVaR, sizing, gates
│   │   └── backtest/          #   walk-forward engine, metrics, manifest
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
    │   │                      #          /api/training/rl/{start,status,stop,runs,promote,recommend},
    │   │                      #          /api/commodity/{curve,cot,seasonality}, /ws/analyze
    │   ├── runner.py          # CrewAI step/task callbacks + asset-class routing
    │   ├── rl_runner.py       # L4 PPO background training worker + status tracker
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
# long-context model (~131K) if you plan to use the full 8-analyst
# debate flow — the bull / bear / research-mgr prompts include every
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

The sidebar exposes an **LLM** dropdown listing every preset in
`trading_crew/llm_presets.py::BUILTIN_PRESETS`. The user can pick a
different model per run without restarting the server:

* Open-source: `hosted-vllm-oss` (env-driven via `VLLM_LLM_*`),
  `local-vllm` (env-driven via `LOCAL_LLM_*`).
* Closed-source: OpenAI `gpt-4o-mini` / `gpt-4o`, Anthropic
  `claude-3-5-sonnet`. Closed-source options are greyed out until
  `OPENAI_PROD_API_KEY` / `ANTHROPIC_API_KEY` are set in `.env`.

Resolution order in `get_llm()`: env defaults → `LLM_PER_AGENT` JSON →
**active UI preset (thread-local)** → direct kwargs. Per-agent and UI
preset overrides compose, so you can pin the Reflective Critic to a
closed-source model via `LLM_PER_AGENT` while keeping the rest of the
crew on the UI's selected open-source preset.

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

The UI also exposes an **embedding model** dropdown right under the LLM
picker (only relevant when memory is on).  Presets live in
`trading_crew/embedding_presets.py::BUILTIN_PRESETS`:

* `vllm-embedding` — env-driven, points at whatever `VLLM_EMBEDDING_*`
  resolves to (typical: `embeddinggemma-300m`, 768-d, 2048-token cap).
* `openai-text-embedding-3-small` — 1536-d, 8191-token cap. Closed-source,
  needs `OPENAI_PROD_API_KEY`.
* `openai-text-embedding-3-large` — 3072-d, 8191-token cap. Same key.

Switching presets is safe: the crew partitions the on-disk LanceDB store
by preset id (`/crew/<crew>/emb/<preset>`) so a 768-d store and a 1536-d
store coexist side-by-side instead of crashing the next `search_memory`
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
   credentials and budget from the **active UI embedding preset**
   (thread-local) or the `VLLM_EMBEDDING_*` /
   `TRADINGCREW_EMBEDDER_MAX_CHARS` env vars at construction time.

The crew constructs `Memory(llm=get_llm(), embedder=…, root_scope=…)`
explicitly (instead of `memory=True`) so CrewAI's analyze subsystem
runs against **our** local LLM rather than defaulting to
`gpt-4o-mini` against the placeholder `OPENAI_API_KEY=dummy` the vLLM
client needs.  Memory is gated by the **LLM-enriched memory** toggle
in the sidebar — when off, no `Memory` object is constructed at all
and every task starts fresh.

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
tasks ≈ (8 analysts × 2 tasks)              # analyst grid
      + (2 × debate_rounds)                  # bull + bear per round
      + 3                                    # research-mgr / quality-reviewer / trader
      + (3 × risk_rounds)                    # 3 risk personas per round
      + 2                                    # compliance + portfolio manager

calls ≈ tasks × 3.5 + 12                     # ~3.5 analyze calls/task + ~12 recall calls
minutes ≈ calls / 60                         # local LLM rate limit = 60 req/min
```

Defaults (debate=2, risk=1): ~110 calls ≈ 1.8 min added.  Max sliders
(debate=10, risk=10): ~233 calls ≈ 3.9 min added.  These are LLM
*analysis* calls — they run in parallel with agent work, so the
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
./run_web.sh                     # binds 0.0.0.0:8001
PORT=9001 ./run_web.sh           # custom port
HOST=127.0.0.1 ./run_web.sh
```

Open http://localhost:8001. The **homepage** shows two cards:

- **Stock Trading** → `/stock`  (NVDA, AAPL, MSFT, …; 18-agent equity crew)
- **Commodity Trading** → `/commodity`  (CL=F, GC=F, ZC=F, …; 17-agent futures crew)

Both dashboards share the same UI shell. The page detects the asset class from the URL and:

- swaps the ticker preset grid (equities vs continuous-futures contracts),
- swaps the agent / tool catalog rendered in the sidebar,
- shows an extra **Futures** tab on `/commodity` (curve, COT, seasonality charts),
- selects the `standard` vs `futures_standard` cost-model preset when running M2.

Common tabs:

| Tab        | What it shows |
| ---------- | ------------- |
| Workflow   | Strict left-to-right SVG diagram, one column per pipeline stage: 2-column analyst grid → bull/bear (parallel) → research mgr → quality reviewer → trader → risk team (3 stacked, parallel) → compliance → portfolio mgr → decision. Parallel agents stack vertically inside a single column; sequential hand-offs each get their own column. Edge animation mirrors the React-Flow / Argo / GitLab CI convention: **idle** = thin gray, **running** = cyan dashed line that *flows* toward its destination via `stroke-dashoffset` animation, **done** = solid green. Each agent with tools shows a chip strip below its node — chips light yellow on tool start and green on success. Click any node to inspect its activity. Toolbar in the top-right has `−` / `+` / `Fit` / `100%` buttons (plus **Ctrl + scroll** to zoom at the cursor and drag-to-pan in the surrounding viewport). |
| Charts     | 180-day price chart with toggleable EMA / SMA / Bollinger overlays, plus dedicated RSI and MACD panels. Data comes from `/api/chart` (yfinance + stockstats). |
| Futures    | *(commodity dashboard only)* Term-structure chart for the front 6 contracts, weekly CFTC Commitments-of-Traders chart, and 5-year monthly seasonality bars. |
| Reports    | Each completed task's full markdown output in execution order, plus the final decision card (action / sizing / drivers / risks / falsifiers / compliance). |
| Memory     | Embargo-aware retrieval of past episodes for this ticker. **L2 training** button **Resolve outcomes & reflect** triggers `/api/memory/resolve` to compute realised PnL / α / max-DD for any episode whose `outcome_ts` has elapsed and (optionally) generate an LLM reflection per episode. |
| Backtest   | Walk-forward replay of recorded proposals (no new LLM calls). Includes the **L3 Auto-tune sizing & risk gates** panel — kicks `/api/grid_search` which sweeps `kelly_fraction × vol_target × max_position_weight`, ranks by Deflated Sharpe to control for selection bias, and reports the best configuration. |
| Portfolio  | Book snapshot + HRP / MV / Equal-Risk allocator preview over the most recent proposal per ticker. |
| Logs       | Live event-by-event console of every `node_started` / `tool_call` / `tool_result` / `node_completed` etc. with timestamps and elapsed times. |

The sidebar exposes:

- **Ticker** preset grid + custom input
- **Analysis date** (drives the chart end-date)
- **LLM** read-only summary of the configured `LOCAL_LLM_*` env values
- **Advanced settings** (collapsed by default):
  - Multi-turn dialogue: debate rounds (1–4), risk rounds (1–3), shared-memory toggle.
  - **Tools** — per-agent enable/disable checkboxes. Disabling a tool hands the
    agent a shorter tool list at construction time.
- **▶ Run analysis** stays sticky at the bottom of the sidebar.

WebSocket protocol: client sends a single JSON config frame on `/ws/analyze`, then receives JSON events of types `run_started → node_started → tool_call → tool_result → node_completed → final_decision → run_completed`.

### Streaming-gap fix

CrewAI's `step_callback` only fires once per task with the local LLM (native function calling lumps all tool iterations into one `AgentFinish`). For tool-less sequential agents (Bull / Bear / Research Manager / Quality Reviewer / risk team / Compliance / PM) that means the UI used to sit completely silent for 30–60 s after the analyst phase while the next single LLM call ran — looking like the workflow was "stuck after analyst team".

Two fixes in `runner.py`:

1. The runner pre-computes `expected_role_order = [t.agent.role for t in crew.tasks]` and synthesizes `node_started` for the next-up agent as soon as the previous task completes — so the diagram lights the next node immediately.
2. Per-tool `tool_call` / `tool_result` events come from the CrewAI **event bus** (`ToolUsageStartedEvent` / `ToolUsageFinishedEvent`) rather than `step_callback`. This is required because native-tool LLMs bypass `step_callback` for individual tool iterations.

## What was inspired by `TauricResearch/TradingAgents`

- **Project shape** (`web/backend` + `web/frontend` + `run_web.sh`) — same topology.
- **Schemas in their own module** — mirrors `tradingagents/agents/schemas.py`.
- **Sidebar UX** — Ticker preset grid, collapsible sections, advanced-settings drawer, sticky Run button at the bottom, per-agent tool checkboxes.
- **SVG workflow diagram** with positioned nodes, group rectangles and curved Bezier edges that highlight on activity.
- **Charts tab** powered by `stockstats` (Bollinger / RSI / MACD / EMA / SMA / ATR) + Chart.js, sourced from a thin `/api/chart` endpoint.
- **Streaming pump** — synchronous worker thread → `queue.Queue` → asyncio queue → WebSocket — so CrewAI's blocking `kickoff()` plays nicely with FastAPI.

## What's CrewAI-specific (different from TradingAgents/LangGraph)

- The crew is built with the **`@CrewBase` / `@agent` / `@task` / `@crew`** decorators that load `config/agents.yaml` and `config/tasks.yaml` automatically.
- Streaming is hooked via `Crew(step_callback=…, task_callback=…)` plus the `crewai_event_bus` for per-tool events (LangGraph's `astream` / `chunk` interface doesn't apply).
- `Task(output_pydantic=PortfolioDecision, guardrail=confidence_guardrail, guardrail_max_retries=2)` is the only path to a typed final answer + auto-retry.
- The 8 analyst tasks fan out concurrently via `async_execution=True` on `Process.sequential`; the next non-async task (Bull round 1) gathers them.
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

Lives in `trading_crew/tools.py`. Runs automatically inside the LLM Trader's task. Given a ticker + a candidate (target%, stop%, horizon-days) trade idea, it scans the historical price tape and computes:

- N samples that ever started a position under those entry conditions,
- win rate (target hit first), stop-out rate, timeout rate,
- avg / median realised return.

It's a **decision-time aide** for sizing one specific trade idea — "out of 1225 historical setups like this, 20% hit target". It runs on every Trader task and you'll see its output in the Reports tab inside the Trade Plan card.

### 2. Walk-forward backtest engine — *strategy-level replay*

Lives in `trading_crew/agentic/backtest/`. Replays **all `ActionProposal`s you have ever logged** for a ticker through the same M1+M2+M5 deterministic pipeline (risk gates + cost model + simulator) under embargoed train/test folds, then reports:

- equity curve,
- Sharpe / Sortino / Calmar / MaxDD / **Deflated Sharpe** (selection-bias corrected),
- per-fold metrics.

This is the **Backtest tab** in the UI and the `GET /api/backtest` endpoint.

> **Why it sometimes shows "need N more proposals":** the walk-forward engine needs ≥ `train + embargo + test` proposals to form even one fold. The UI defaults to `train=3 / embargo=1 / test=1` (5 proposals minimum). Run analyses on a ticker enough times to clear that threshold and the equity curve appears.

## Agentic training on historical data (L2 + L3 + L4)

Beyond the inference-time M3 episodic memory ("L1"), the system has three offline training loops you can drive from the UI or the API.

| Level | What it learns | What it updates |
| ----- | -------------- | --------------- |
| **L1** | (no learning — retrieval only) | Episodic memory query results at inference time |
| **L2** | Per-episode reflections from realised outcomes | Episode reflection field consumed by `retrieve_past_episodes` next run |
| **L3** | Best sizing & risk-gate hyperparameters via walk-forward grid search | M5 config that the runner sends into `run_pipeline` |
| **L4** | A *parametric policy network* end-to-end via PPO | Torch checkpoint surfaced to the LLM through the `rl_policy_recommendation` tool |


### L2 — outcome resolution + LLM reflection

Module: `trading_crew/agentic/reflection.py`. Endpoint: `POST /api/memory/resolve`.

For every `Episode` whose `outcome_ts` has elapsed but whose actual outcome was never written back:

1. Re-fetch OHLCV between `decision_ts` and `outcome_ts`.
2. Compute **realised return** (signed by side), **alpha vs SPY benchmark**, **max drawdown**.
3. If an LLM is configured (`skip_llm: false`), ask it for a structured reflection: *what worked, what didn't, what to do differently*.
4. Patch the episode in place — future runs of the `retrieve_past_episodes` tool now surface this reflection verbatim to the Market Analyst + Trader.

UI: **Memory tab → Resolve outcomes & reflect** button.

```bash
curl -X POST localhost:8001/api/memory/resolve \
  -H 'content-type: application/json' \
  -d '{"ticker": "NTNX", "skip_llm": false}'
```

Status codes per episode: `RESOLVED`, `SKIPPED_PENDING` (horizon not elapsed), `SKIPPED_NO_DATA`, `ABANDONED` (older than the configurable cutoff).

### L3 — walk-forward grid search on sizing & risk

Module: `trading_crew/agentic/grid_search.py`. Endpoint: `POST /api/grid_search`.

Sweeps a grid of `kelly_fraction × vol_target × max_position_weight` over your logged proposals using the M6 walk-forward engine. **Ranking is by Deflated Sharpe**, which corrects for selection bias when you test many configurations against the same history (Bailey & Lopez de Prado, 2014). The "best" configuration the UI surfaces is the one that survived this correction, not the highest raw Sharpe.

UI: **Backtest tab → Auto-tune sizing & risk gates** panel.

```bash
curl -X POST localhost:8001/api/grid_search \
  -H 'content-type: application/json' \
  -d '{"ticker": "NTNX", "size": 5, "rank_by": "deflated_sharpe"}'
```

### L4 — Full RL with the M2 simulator as the environment

Module: `trading_crew/agentic/rl/`. Endpoints: `/api/training/rl/*`. UI: **RL Training** tab.

This is the closed-loop training the rest of M1–M7 was setting up for. Agents *act*, the M2 simulator *pays out*, gradients flow back into the policy network.

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

How to use it:

1. Open the **RL Training** tab on either `/stock` or `/commodity`.
2. Pick a ticker, tweak `total_steps`, `learning_rate`, `entropy_coef`, `train_window_days`, `eval_window_days`, `drawdown_kill_pct`, `cost_model_name`.
3. Click **Start training**. The live chart streams `mean_episode_pnl_pct` / `entropy` / `sharpe_per_step` from each rollout; the bottom strip shows the final action distribution + per-rollout PPO diagnostics (policy loss, value loss, approx KL, clip fraction).
4. When the run finishes, the **Run history** table at the bottom shows eval PnL%, eval vs buy-and-hold, eval max drawdown, and a **Promote** button.
5. Promote the run — that writes a single pointer in `~/.trading_crew/rl_runs/promoted/<TICKER>.json`. **Until you promote, the policy stays invisible to the LLM crew** (no implicit "use the latest run" fallback — half-trained policies should not bleed into live decisions).
6. Click **Preview promoted-policy action** to see what the active policy would do *today* on the current OHLCV — direction + confidence + full action distribution.
7. Run a regular analysis. The Market Analyst + Trader now have an extra tool, `rl_policy_recommendation` (or `rl_policy_recommendation_commodity`), which returns the same recommendation as a markdown block. The LLM is free to follow it, discount it, or override it — it's an **advisory** prior, not a hard override.

CLI example end-to-end:

```bash
# 1. Kick off training (runs in a background worker, returns the record immediately).
curl -X POST localhost:8001/api/training/rl/start \
  -H 'content-type: application/json' \
  -d '{
    "ticker": "SPY",
    "asset_class": "stock",
    "train_window_days": 750,
    "eval_window_days": 120,
    "ppo_config": {"total_steps": 20000, "entropy_coef": 0.01}
  }'

# 2. Poll progress.
curl 'localhost:8001/api/training/rl/status?ticker=SPY' | jq

# 3. After it transitions to status="completed", list runs + promote the winner.
curl 'localhost:8001/api/training/rl/runs?ticker=SPY' | jq '.runs[0]'
curl -X POST localhost:8001/api/training/rl/promote \
  -H 'content-type: application/json' \
  -d '{"ticker": "SPY", "run_id": "20260611T170611-825"}'

# 4. Score the live state through the promoted policy.
curl -X POST localhost:8001/api/training/rl/recommend \
  -H 'content-type: application/json' \
  -d '{"ticker": "SPY"}' | jq '.recommendation'
```

Persistence layout (under `$TRADINGCREW_CACHE_DIR/rl_runs/`):

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
- Stopping a run is **cooperative** (the trainer checks a `should_stop` flag between rollouts).  Hitting Stop in the UI calls `/api/training/rl/stop` which sets the flag and joins the worker.

## Commodity dashboard

Open `/commodity` instead of `/stock` and the UI swaps in:

- **8 futures-specialist agents** (Market Analyst, Curve Analyst, Inventory & Demand Analyst, Geopolitical Analyst, News Analyst, Bullish / Bearish Researchers, Futures Trader) defined in `commodity_crew/config/{agents,tasks}.yaml`.
- **Futures-aware tools** in `commodity_crew/tools.py`:
  - `get_commodity_ohlcv` / `get_commodity_indicators` — front-month OHLCV + stockstats indicators.
  - `get_futures_curve` — front 6 contracts pulled live to compute CONTANGO / BACKWARDATION + roll yield %.
  - `get_cot_report` — weekly CFTC Commitments-of-Traders (downloaded + cached).
  - `get_seasonality` — N-year monthly mean/std returns from front-month series.
  - `get_commodity_news` / `get_commodity_geopolitical` — Tavily web search scoped to commodity keywords + OPEC / Black Sea / Suez / Hormuz triggers.
  - `retrieve_past_episodes_commodity` — M3 memory, scoped to the commodity universe.
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
| **M3** Audit-grade memory v2 | `agentic/memory/{episodic,semantic,regime}.py` | Outcome-embargoed `EpisodicMemory`, time-decay retrieval, regime tags, `SemanticKnowledgeBase` with provenance | **Memory** tab + episode banner |
| **M4** Reflection + cascaded controller | `trading_crew/critic.py`, `web/backend/runner.py::_run_cascade_controller` + `_crisis_override_decision` | **Cascaded controller** detects regime *before* the crew kicks off and short-circuits to ABSTAIN on CRISIS (saving ~30k tokens). **Reflective Critic** runs after the PM with 5-stage protocol (intent / evidence / counterfactual / risk / verdict), bounded reflection budget (2 iterations), and 3-temperature consistency vote (mode < 2/3 → forced ABSTAIN). | Cascade badge + Reflective-critic panel with per-sample checklist |
| **M5** Risk + sizing layer | `agentic/risk/{var,sizing,gates}.py` | Historical + parametric VaR/CVaR, Fractional Kelly × vol-target × CVaR clamp sizing, hard gates (concentration, leverage, drawdown stop, kill switch) | "Sizing" + "Risk gates" panels |
| **M6** Walk-forward backtest + manifest | `agentic/backtest/{walk_forward,metrics,manifest,engine}.py` | Embargoed train/test folds, Sharpe / Sortino / Calmar / MaxDD / **Deflated Sharpe**, run manifest with git-sha + prompts hash + data hashes | **Backtest** tab (equity curve + per-fold metrics) |
| **M7** Multi-ticker allocator | `agentic/portfolio/allocator.py` | HRP / Mean-Variance / Equal-Risk allocator over the most recent `ActionProposal` per ticker | **Portfolio** tab |

### How a single run flows through M1–M7

1. **UI → WS `/ws/analyze`** with `{ticker, debate_rounds, risk_rounds, tools_enabled, critic_iterations, critic_samples, ...}`.
2. **Runner** opens a `RunRecord`, loads ~252 bars of OHLCV, calls **`detect_regime()` (M4)** and emits `cascade_status`.
3. **If regime == CRISIS** → the runner skips `crew.kickoff()` entirely and builds a synthetic ABSTAIN `PortfolioDecision` (paper §5.3 emergency stop). The deterministic M1→M5→M3 pipeline still runs so the audit trail is identical.
4. **Otherwise** `crew.kickoff()` runs the 18-agent CrewAI flow. The Market Analyst and Trader can call the new `retrieve_past_episodes` tool to ground their analysis in prior outcomes (M3, embargo-enforced). UI streams `node_started` / `agent_step` / `tool_call` / `tool_result` / `node_completed` / `final_decision`.
5. **Reflective Critic (M4)** — once the Portfolio Manager returns its typed `PortfolioDecision`, the runner runs the 5-stage critique (up to 2 iterations) then a 3-temperature consistency vote. If the modal verdict has < 2/3 share, the decision is forced to ABSTAIN. The runner emits `reflection_records` and (if revised) a second `final_decision` with `source="critic"`.
6. **Bridge (M1)** — the post-critic `PortfolioDecision` is mapped to a typed `ActionProposal` via `portfolio_decision_to_action_proposal()` and emitted as `action_proposal`.
7. **Pipeline (M2+M5)** — `run_pipeline()` computes VaR/CVaR from the OHLCV, runs the **sizer** (Fractional Kelly × vol-target × CVaR clamp × debate-derived risk multiplier), runs the **risk gates** (concentration, leverage, DD stop, kill switch, …), and runs the **simulator** for a next-bar fill. The runner emits `execution_result` with sizing breakdown, risk-gate outcomes, and a low/standard/high cost-sensitivity sweep.
8. **Episode (M3)** — a PENDING `Episode` (state, action, outcome_ts = decision_ts + horizon, regime tag, embargo) is appended to episodic memory and `episode_recorded` is emitted.
9. **Persistence (Phase E)** — every event since step 1 is captured into the `RunRecord` and written atomically to `~/.trading_crew/runs/{TICKER}/{ts}.json` plus an index line so the UI sidebar can list it.

### New API endpoints

| Endpoint | Purpose |
| -------- | ------- |
| `GET /api/memory/retrieve?ticker=&as_of=&query=&k=` | Top-k embargo-aware retrieval of past episodes |
| `GET /api/portfolio` | Book snapshot + HRP allocator preview over the latest proposal per ticker |
| `GET /api/backtest?ticker=&train_size=&embargo_size=&test_size=` | Walk-forward backtest of logged proposals |
| `GET /api/runs/recent?limit=&ticker=` | List the most recent UI runs across all tickers (or one) |
| `GET /api/runs/{ticker}/latest` | Full record of the most recent run for a ticker |
| `GET /api/runs/{ticker}/{run_id}` | Full record of a specific past run |
| `POST /api/memory/resolve` | **L2 training** — resolve PENDING episodes into realised outcomes + LLM reflections |
| `POST /api/grid_search` | **L3 training** — walk-forward grid search on sizing & risk, ranked by Deflated Sharpe |
| `POST /api/training/rl/start` | **L4 training** — launch a PPO-on-M2 background training run |
| `GET  /api/training/rl/status?ticker=` | Poll the active run's metrics (streamed from `metrics.jsonl`) |
| `POST /api/training/rl/stop` | Cooperative stop (ticker or run_id) |
| `GET  /api/training/rl/runs[?ticker=]` | Leaderboard of past + current runs |
| `GET  /api/training/rl/runs/{ticker}/{run_id}` | Full per-run snapshot (record + metrics) |
| `POST /api/training/rl/promote` | Promote a completed run so it becomes the active policy |
| `GET  /api/training/rl/promoted` | List every promoted policy (one per ticker) |
| `POST /api/training/rl/recommend` | Score live OHLCV through the promoted policy (UI preview) |
| `GET /api/commodity/curve` | Futures term structure (CONTANGO / BACKWARDATION) |
| `GET /api/commodity/cot` | Weekly CFTC Commitments-of-Traders report |
| `GET /api/commodity/seasonality` | Monthly seasonality from front-month series |

### New CrewAI tool

`retrieve_past_episodes(ticker, as_of, k=3)` — exposes M3 episodic memory to agents.  Embargo-aware: any episode whose outcome timestamp is on or after `as_of` is excluded so analysts can never lean on future information.  Wired into `market_analyst` and `trader` defaults.

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
   - The **Cascade-status badge** at the top shows the regime detected from OHLCV. If `CRISIS`, the 18-agent debate is skipped entirely and the run goes straight to an ABSTAIN proposal.
   - Once the Portfolio Manager finishes, the **Reflective Critic** panel renders the 5-stage checklist (intent / evidence / risk per sample) and the consistency-vote outcome — REVISED / RATIFY / ABSTAIN.
   - The **Order ticket**, **Execution**, **Sizing**, **Cost sensitivity sweep**, and **Risk-Gate** panels populate from the post-critic deterministic pipeline.
3. Open the **Recent runs** section in the sidebar — every kickoff is saved to `~/.trading_crew/runs/`. Click any past run to re-load its full state into the Workflow / Reports tabs without re-running.
4. Run the same ticker on a few more dates to build an **episode history**. The Memory tab will show retrieved past episodes ranked by time-decayed similarity (with the outcome embargo enforced server-side). The Market Analyst and Trader will *also* call `retrieve_past_episodes` automatically and reference prior outcomes in their reports.
5. Trade a few different tickers to populate the **Portfolio tab**. With ≥2 tickers' recent proposals on file, the HRP allocator preview kicks in.
6. Click **Backtest** tab to walk-forward replay your recorded proposals through the deterministic pipeline. The equity curve + per-fold metrics use the same risk gates and cost model as live runs.

### Where the M1-M7 modules live (and what's *not* duplicated)

- The `trading_crew/agentic/` package is the **LangChain-free deterministic core** — `{portfolio,execution,memory,risk,backtest}` modules implementing the M1–M7 contracts from the paper. No source is copied from the upstream `TauricResearch/TradingAgents` project; only the high-level debate topology is shared.
- **M1**'s Action Compiler in the LangGraph project is reduced to the deterministic `bridge.py` mapping from CrewAI's existing `PortfolioDecision` (so the CrewAI crew remains the single LLM driver for the *narrative*; the bridge is pure Python).
- **M4** is now implemented as a full LLM-driven Reflective Critic (`trading_crew/critic.py`) that calls the LLM directly via `crewai.LLM.call(..., response_model=CritiqueResponse)` for clean structured output without spinning up a mini-crew per sample. The cascaded-controller half of M4 (regime routing) lives in `runner.py::_run_cascade_controller` + `_crisis_override_decision`.
- **Cache directories**: episodes go to `$TRADINGCREW_CACHE_DIR/memory/episodes.jsonl`, run history goes to `$TRADINGCREW_CACHE_DIR/runs/` (both default to `~/.trading_crew`). Portfolio state goes to `$TRADINGAGENTS_PORTFOLIO_DIR` (default `~/.tradingagents/portfolios`).

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

`get_market_context(ticker)` is a new tool every analyst calls FIRST. It returns the country, currency, sector, peer basket, **macro basket** (yfinance symbols for the country's rates / FX / vol / commodities), and **news themes** (country macro + industry seeds). The downstream tools then use these to phrase their searches.

Example brief for `MAZDOCK.NS`:

```
## Market context for MAZDOCK.NS
- Name: Mazagon Dock Shipbuilders Limited
- Country / currency / exchange: India / INR / NSI
- Sector / industry: Industrials / Aerospace & Defense
- Benchmark index: ^NSEI
- Peer basket: MAZDOCK.NS, COCHINSHIP.NS, GRSE.NS, BEL.NS, HAL.NS
- Macro basket (yfinance symbols):
    nifty_50: ^NSEI       india_vix: ^INDIAVIX
    usd_inr: INR=X        sensex: ^BSESN
    brent_crude: BZ=F     gold: GC=F
- Country news themes: RBI repo rate, India CPI, INR USD,
                       Union Budget capex, FII/DII flows, crude import bill
- Industry news themes: shipbuilding order book, naval defence contract,
                        submarine destroyer frigate procurement,
                        ship insurance war-risk premium,
                        Strait of Hormuz shipping disruption,
                        Red Sea / Suez Canal shipping,
                        defence budget allocation, defence export contract
```

Same call for `AAPL` returns the US macro basket (10y UST, DXY, VIX, WTI, gold) and US news themes (FOMC, CPI, DXY, US-China trade). The US flow is unchanged — *the system now picks the right basket for the ticker* instead of hard-coding one.

### What the agents actually do now

| Tool | Before | Now |
| ---- | ------ | --- |
| `get_news(ticker)` | `"{ticker} stock news sentiment last week"` | Uses the company's *long name* + ticker — no more ambiguous bare-symbol misses |
| `get_global_news(ticker)` | `"global markets macro economy fed rates"` always | Calls `country_news_query()` — RBI / Union Budget for India; FOMC / CPI for US |
| `get_macro_data()` | US 10y UST + DXY + VIX + WTI + gold | Country-specific basket (`get_macro_data(ticker)`) — India = Nifty + India VIX + INR/USD + Sensex + Brent |
| `get_geopolitical_news(ticker)` | `"{ticker} tariffs export controls sanctions China Taiwan"` | Country + industry-driven (shipbuilder → Hormuz + Red Sea + naval procurement; semis still gets Taiwan + chip-export) |
| `get_supply_chain_risk(ticker)` | `"TSMC China dependency 10-K customer"` | Industry-aware — semis keeps TSMC; shipbuilder gets ship insurance + war-risk premium |
| `get_sector_peers(ticker)` | US-only (`NVDA/AMD/AVGO/INTC...`) | Adds Indian baskets (`MAZDOCK.NS, COCHINSHIP.NS, GRSE.NS, BEL.NS, HAL.NS`, Indian banks, Indian IT, etc.) |

The task prompts (`trading_crew/config/tasks.yaml`) now explicitly require analysts to call `get_market_context` first and ground their searches in the company's home country — no implicit US default.

## Finance basics (for non-finance readers)

If you've come at this from the engineering side and want the finance vocabulary the system uses, this section walks through every concept in order and links each one back to a file you can read.

### Stocks vs commodities vs futures

A **stock** is a claim on a company's future cash flows (Apple → Apple's earnings). A **commodity** is a physical good (oil, copper, wheat). You don't usually trade physical commodities — you trade **futures contracts**, which are standardised agreements to buy/sell a commodity at a fixed price on a fixed future date. `CL=F` (the yfinance symbol) is the next-expiry WTI crude oil futures contract; `MAZDOCK.NS` is a stock.

This repo runs two dashboards on the same backbone: `/stock` for equities and `/commodity` for futures.

### Buying long vs selling short

- **Long**: you own the asset and profit when it goes up. `+10%` weight = "10% of my portfolio is long this stock".
- **Short**: you borrow the asset, sell it now, hope to buy it back cheaper later. `-10%` weight = "10% of my portfolio is short this stock". You profit when it goes down.
- **Flat / Neutral**: zero weight, no position.

The PM's `PortfolioDecision.action` is `OVERWEIGHT` (long), `NEUTRAL` (flat), or `UNDERWEIGHT` (short or simply less than the benchmark weight).

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

Once the team has decided "OVERWEIGHT with 0.65 confidence", the deterministic sizer (paper §7.2) decides *how big*. Three independent caps, the **smallest binds**:

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

Every run writes an **Episode** to `~/.trading_crew/memory/episodes.jsonl`: state summary, action proposal, regime tag, decision_ts, outcome_ts (= decision_ts + horizon_days). The Market Analyst can retrieve the top-k most similar past episodes via `retrieve_past_episodes`, **outcome-embargoed**: an episode is invisible until its outcome materialises, otherwise future-known returns leak into "past" retrieval and inflate reported edge.

When you click **Resolve outcomes & reflect** in the Memory tab (L2 training), the system:

1. Pulls OHLCV between `decision_ts` and `outcome_ts`, computes realised return (signed by side), alpha vs SPY, max drawdown.
2. Asks the LLM for a one-paragraph reflection: *what worked, what didn't, what to do differently*.
3. Writes the reflection back to the episode. Next run, when the Market Analyst calls `retrieve_past_episodes`, this lesson surfaces verbatim — that's how the system *learns* without touching LLM weights.

### Why the Reflective Critic exists

LLMs hallucinate. The Critic (`trading_crew/critic.py`) runs after the PM and runs the *same* prompt 3 times at different temperatures. Each sample returns a typed `CritiqueResponse` with booleans per stage (intent, evidence, risk). If 2/3 votes disagree, the Critic forces **ABSTAIN**. If a numeric claim in the rationale doesn't carry an inline `[source: <id>]` tag whose `<id>` is in the decision's `sources` list, the Evidence check fails — that's why every tool now appends a `Source: …` footer and the PM is required to surface a deduplicated `sources` list.

### How to read the final Workflow output (in order)

1. **Cascade controller badge** — regime detected from 252 bars of OHLCV. `CRISIS` short-circuits the whole debate.
2. **Per-analyst Reports** (8 of them) — each numeric claim carries `[source: …]`.
3. **Bull / Bear debate** — adversarial; the Research Manager picks a side (does NOT split the difference).
4. **Quality Reviewer** — flags every uncited or weakly-supported claim. Below-4 score recommends size reduction (scores in [4, 6] are "actionable with caveats", not a size cut).
5. **Trader Plan** — entry, stop, target, horizon, sizing intent. The trader now sees a **multi-horizon backtest** (trader-chosen window + 60d / 120d / 252d) with per-horizon **hit-rate, expectancy, payoff ratio**. A sub-40% hit-rate with positive expectancy and payoff ≥ 1.5 is still actionable — confidence is no longer auto-capped at 0.6 in that case.
6. **Risk debate** — Aggressive vs Conservative argue, Neutral synthesises.
7. **Compliance Officer** — `CLEAR / FLAGGED / BLOCKED`. BLOCKED → action MUST be NEUTRAL.
8. **PM Decision** — typed `PortfolioDecision` JSON with sources list. The PM is now required to reason about BOTH the short-term horizon AND the 120d/252d long-term horizon before defaulting to NEUTRAL, and the "hit-rate < 40% ⇒ confidence ≤ 0.6" cap only fires when EVERY horizon has expectancy ≤ 0%.
9. **Reflective Critic** — 5-stage protocol + 3-temp consistency vote. May revise or ABSTAIN.
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
