/* TradingCrew — Alpine.js component wiring sidebar + workflow + charts + logs.
 *
 * Responsibilities:
 *   1. Load /api/options on boot, seed defaults.
 *   2. Build the workflow diagram inside #workflow-diagram.
 *   3. Open a WebSocket to /ws/analyze, send config, render events live.
 *   4. Load /api/chart on ticker / date change and draw price + RSI + MACD.
 *   5. Track per-node activity for the inspector + per-task reports.
 */

function tradingCrewApp() {
  return {
    /* ------------------------------------------------ dashboard identity */
    // Asset class is read from the URL once at boot ("/stock" -> "stock",
    // "/commodity" -> "commodity").  All API and WebSocket calls thread
    // this value through so the backend routes to the right crew.
    assetClass: (() => {
      const p = (typeof window !== "undefined" && window.location && window.location.pathname) || "";
      if (p.startsWith("/commodity")) return "commodity";
      if (p === "/settings" || p.startsWith("/settings/")) {
        const book = new URLSearchParams(window.location.search).get("book") || "";
        if (book === "commodity") return "commodity";
      }
      return "stock";
    })(),

    /* ---------------------------------------------------------- catalogs */
    tickers: [],
    toolsInfo: {},
    toolOwners: [],     // [{ key, role, kind, tools: [...] }]
    indicatorLabels: {},
    llm: {},
    llmPresets: [],         // [{ id, label, kind, provider, model, base_url, description, api_key_configured, ... }]
    selectedLlmPreset: "",  // id of the preset to use on the next run; persisted to localStorage
    tavily: false,
    // Surface bootstrap failures so the user sees a clear retry affordance
    // instead of an empty sidebar (e.g. when /api/options is unreachable
    // because the backend died, or an extension blocked the fetch).
    optionsLoaded: false,
    optionsError: "",
    optionsRetrying: false,

    /* ---------------------------------------------------------- form */
    form: {
      ticker: "NTNX",
      trade_date: (() => {
        const n = new Date();
        const m = String(n.getMonth() + 1).padStart(2, "0");
        const d = String(n.getDate()).padStart(2, "0");
        return `${n.getFullYear()}-${m}-${d}`;
      })(),
      max_debate_rounds: 2,
      max_risk_rounds: 1,
      // Stock-only: autonomous = LangGraph; deterministic = sequential CrewAI;
      // harness = observe/dispatch conductor loop.
      // Prediction batch reads the same field via _bulkCfgFromForm.
      run_mode: "autonomous",
      // Optional advisory reviewers (async, feed the PM).  Default OFF
      // for a lean run.
      quality_review: false,
      compliance: false,
      // Reflective critic — on by default but single-pass (the 3-temp
      // consistency vote is skipped unless critic_samples is raised).
      critic_enabled: true,
      // Optional signal fusion (folded in from the old Signals tab).  When
      // on, the workflow fuses its decision with the L4 policy / L3 edge /
      // crowd view and surfaces a guardrailed FusedSignal in the Workflow
      // tab.  Advisory only — never changes the PM's action/size.
      fusion_enabled: false,
      fusion_include_crowd: true,
      tools_enabled: {},   // agent_key -> [tool_name, ...]
      // Phase 2B.3 — partition the on-disk portfolio between a
      // sandboxed "paper" book and a live "prod" book.  Defaults to
      // paper so casual experimentation can't corrupt prod state.
      book: "paper",
    },

    /* ---------------------------------------------------------- collapsible state */
    sidebarCollapsed: false,
    openSections: {
      ticker: true,
      date: false,
      llm: false,
      sessions: true,
      runs: false,
      dialogue: true,
      tools: true,
    },

    // Phase E — saved runs
    recentRuns: [],
    loadedRunId: null,
    clearingRecentRuns: false,
    sections: [
      { id: "llm", title: "LLM", icon: "L" },
      { id: "dialogue", title: "Dialogue", icon: "M" },
      { id: "tools", title: "Tools", icon: "⚙" },
    ],

    /* ---------------------------------------------------------- run state */
    isRunning: false,
    // Transient guard for the /api/sessions POST round-trip so a
    // double-click doesn't queue the same run twice.  Distinct from
    // ``isRunning`` (which tracks the *attached* session): a new run
    // can always be started even while another session is running.
    starting: false,
    status: "idle",
    currentNode: "",
    finalDecision: "",
    decisionPayload: null,
    activeTab: "prediction",
    // When reached via /run/{ticker} (from a Prediction tile), this holds the
    // most recent batch prediction stored in the DB so the Workflow tab can
    // show it immediately, before/without a fresh live run.
    storedPrediction: null,
    storedPredictionTicker: "",
    // The "Futures" tab is rendered only for commodity dashboards; the
    // x-show guards in index.html honour ``assetClass``.  We keep it in
    // the array so the order is stable; the template just hides it for
    // stock dashboards.
    // ``scope`` splits high-level tabs (shown on the /stocks | /commodity page)
    // from instrument-specific tabs (shown on the focused /run page).
    // ``asset`` (optional) restricts a tab to one asset class — Prediction &
    // Names/Portfolio are stock-only; Contracts is commodity-only; Training &
    // Episodes are cross-asset.  ``desk`` splits the live universe
    // (/stocks: Prediction · Training · Episodes) from a dated prediction
    // run (/stocks/history/{date}: Names · Portfolio).  ``commodityOnly``
    // gates an instrument tab.
    tabs: [
      { id: "prediction", label: "Prediction", scope: "global", asset: "stock", desk: "live" },
      { id: "contracts", label: "Contracts", scope: "global", asset: "commodity" },
      { id: "training", label: "Training", scope: "global", desk: "live" },
      { id: "episodes", label: "Episodes", scope: "global", desk: "live" },
      { id: "run-names", label: "Names", scope: "global", asset: "stock", desk: "review" },
      { id: "portfolio", label: "Portfolio", scope: "global", asset: "stock", desk: "review" },
      // Reports + Logs are no longer standalone tabs — both are folded into
      // the Workflow tab so clicking a tile shows the report AND the activity
      // log during the run and after it completes (both are persisted in the
      // run record and replayed on reopen).
      { id: "workflow", label: "Workflow", scope: "stock" },
      { id: "charts", label: "Charts", scope: "stock" },
      { id: "futures", label: "Futures", scope: "stock", commodityOnly: true },
      { id: "backtest", label: "Backtest", scope: "stock" },
    ],
    // Whether a specific stock is "open" (via a Prediction tile or /run/{t}).
    // Gates the stock-specific tabs + drives the header title.
    stockActive: false,
    // Hierarchical page mode: true ⇒ the focused per-instrument workbench
    // (URL /run/{ticker} for stocks, /commodity/run/{contract} for commodity)
    // — no left pane, instrument tabs only, a back button to the high-level
    // page.  false ⇒ the high-level page (/stocks or /commodity) with the left
    // pane + global tabs.  Applies to BOTH asset classes.
    focusPage: false,
    // Dedicated /settings page: left pane = run config, right pane = strategy + prompts.
    settingsPage: (() => {
      const p = (typeof window !== "undefined" && window.location && window.location.pathname) || "";
      return p === "/settings" || p.startsWith("/settings/");
    })(),
    // Left sidebar mode: "history" (run dates) | "settings" (full run config).
    sidebarMode: (() => {
      const p = (typeof window !== "undefined" && window.location && window.location.pathname) || "";
      if (p === "/settings" || p.startsWith("/settings/")) return "settings";
      return "history";
    })(),
    settingsSaving: false,
    settingsError: "",
    settingsPromptBlock: "",
    settingsSource: "",
    settingsLivePath: "",
    settingsPackagedPath: "",
    bulkCfgError: "",

    // Signal fusion result — populated by the optional in-run "fusion" event
    // (folded in from the old Signals tab) and rendered on the Workflow tab.
    fusionResult: null,

    // Structured report trend signals (Firefly-style card) — populated by the
    // "report_signals" event live and rehydrated by loadRun, rendered on the
    // Workflow tab beside the analyst reports.
    reportSignals: null,

    // Deterministic price-structure signals (vol-state / regime-age / oversold
    // DIP_BUY / Hurst) computed from OHLCV.  Populated by the "price_signals"
    // event and rehydrated by loadRun; rendered on the Workflow tab.
    priceSignals: null,

    // Per-desk bias board — each analyst desk's directional lean handed to the
    // PM ([{desk, bias}, ...]).  Populated by the "desk_bias_board" event and
    // rehydrated by loadRun; rendered in the Decision-node panel.
    deskBiasBoard: null,

    // Full evidence blackboard (UI shows full reports; debate tools see compact).
    evidenceBlackboard: null,

    // Shared bull/bear scratch pad (analyst TLDRs + debate turns + notes).
    debateScratchpad: null,

    // Display role strings for desks running ReAct-style planning (from
    // /api/options).  Drives the ReAct tile badge in the workflow diagram.
    reactAgents: [],

    /* --------------------------------------------- commodity-only state */
    curveData: null,
    cotData: null,
    seasonalityData: null,
    commodityLoading: { curve: false, cot: false, seasonality: false },
    curveChart: null,
    cotChart: null,
    seasonalityChart: null,

    diagram: null,
    diagramScale: 1,        // effective scale = fitScale * userZoom; powers the toolbar %
    selectedNode: null,
    nodeActivities: {},
    reports: {},          // role -> output text (latest round only — back-compat)
    // role -> [{round, total_rounds, output, degraded}] in chronological order.
    // Single-round roles (analysts, manager, trader, compliance, PM) hold
    // exactly one entry.  Multi-round roles (Bullish / Bearish Researcher,
    // the three Risk personas) hold one entry per round.  This is the new
    // source of truth for the Reports tab so the user can see how round 1
    // differs from round 2 instead of only the last round.
    reportsByRound: {},
    // Roles whose final answer was the degraded placeholder from
    // _patches.py — used by the Reports tab and the workflow diagram
    // to flag the affected analyst with a "degraded" badge instead of
    // rendering the placeholder text as if it were a real report.
    // Kept as an object (not Set) so Alpine's reactivity picks up
    // shallow assignments via spread.
    degradedRoles: {},

    // M1-M5 — populated by post-LLM pipeline events
    cascadeStatus: null,
    actionProposal: null,
    actionProposalMd: "",
    executionResult: null,
    episodeMeta: null,
    reflectionRecords: null,   // M4 — { records: [...], final_action, final_size_pct, revised }

    // M3 — Memory tab
    retrievedEpisodes: [],
    memoryRegime: "",             // optional regime filter
    memoryMeta: null,              // { embedder, regime }
    pruneLoading: false,
    pruneResult: null,

    // L2 agentic training — outcome resolution + reflections
    resolveResult: null,
    resolveLoading: false,
    strategyBoard: null,
    strategyDraft: null,
    strategySaving: false,
    strategyError: "",
    promptKey: "trader",
    promptDraft: { goal: "", backstory: "" },
    promptSaving: false,
    promptError: "",
    experimentKnob: "",
    hermesBusy: false,
    hermesError: "",
    hermesEditor: "strategy",

    // L2 analytics — conviction calibration panel
    calibration: null,
    calibrationBasis: "return",   // "return" | "alpha"
    calibrationCohort: "current", // "current" | "all"
    calibrationTickerOnly: false,

    // M6 — Backtest tab
    backtestConfig: { train_size: 3, embargo_size: 1, test_size: 1 },
    backtestResult: null,
    backtestLoading: false,
    backtestChart: null,
    backtestProposalCount: null,   // null = unknown / not checked yet
    backtestNetworkError: false,   // distinguishes browser-level fetch failure from server's "not enough data"
    backtestLookbackYears: 5,      // OHLCV history fed to the engine (defaults to 5y so the long-term thesis is covered)

    // L3 agentic training — grid search "Auto-tune"
    gridSearchSize: "coarse",
    gridSearchRankBy: "deflated_sharpe",
    applyTune: false,   // persist the winning config → live inference + walk-forward
    // Phase 2C — extra grid axes / robustness toggles.
    gridSweepMaxLeverage: false,
    gridSweepDrawdownKill: false,
    gridSweepCostModel: false,
    backtestValidation: "walk_forward",   // walk_forward | cpcv
    backtestCostModel: "standard",
    // CPCV geometry (used when backtestValidation === "cpcv")
    cpcvGroups: 6,
    cpcvK: 2,
    seedN: 20,
    seedLoading: false,
    seedResult: null,
    gridSearchResult: null,
    gridSearchLoading: false,
    // Swarm intelligence — optimizer choice + PSO + consensus controls
    backtestOptimizer: "grid",             // grid | pso
    swarmSize: 12,
    swarmIterations: 8,
    swarmSeed: 0,
    gridConsensus: false,                  // fuse top-K into a consensus ensemble
    consensusTopK: 5,
    // Multi-asset portfolio backtest (M7 allocator)
    portfolioTickers: "",
    portfolioAllocMethod: "HRP",           // HRP | MEAN_VARIANCE | EQUAL_RISK
    portfolioGrossBudget: 0.8,
    portfolioResult: null,
    portfolioLoading: false,
    portfolioChart: null,
    // Crowd-positioning chip (swarm crowd signal for the current ticker)
    crowdSignal: null,
    crowdLoading: false,

    // L4 agentic training — PPO on the M2 simulator
    rlConfig: {
      total_steps: 10000,
      steps_per_rollout: 512,
      learning_rate: 3e-4,
      entropy_coef: 0.01,
      train_window_days: 750,
      eval_window_days: 60,
      drawdown_kill_pct: 0.4,
      cost_model_name: "standard",
      // Phase 2F — short-vs-long-term mode auto-fills train/eval windows.
      // ``custom`` leaves them at whatever the user has typed manually.
      horizon_mode: "balanced",         // short_term | balanced | long_term | custom
      // Phase 2D — risk + shaping knobs surfaced to the UI.
      algorithm: "ppo",                 // ppo | cvar_ppo | cql | c51 | decision_transformer
      risk_aversion: 0.0,
      cvar_alpha: 0.20,
      turnover_penalty_bps: 0.0,
      drawdown_penalty_coef: 0.5,
      volatility_curriculum: false,
      policy_universe: "",              // comma-separated tickers; "" = single-ticker
    },
    rlStatus: null,            // active run snapshot (poll result)
    rlMetrics: [],             // streamed rollout metrics
    rlRuns: [],                // leaderboard rows
    rlPromoted: [],            // promoted-policy list
    rlSelectedRun: null,       // expanded detail in the leaderboard
    rlRecommendation: null,    // "what would the policy do today" preview
    rlLoading: { start: false, stop: false, runs: false, promote: false, recommend: false },
    rlError: "",
    rlChart: null,
    rlActionChart: null,
    rlPollTimer: null,

    // Forward collection (scale) — live crew over the 200-stock universe
    scaleStatus: null,
    scaleForm: { universe: "all200", workers: 6, useCustom: false, tickers: "" },
    scaleLoading: false,
    scaleError: "",
    scalePollTimer: null,
    // Per-session web-search backend counters for the active batch (trace_id
    // == batch job_id).  Populated by loadSearchStats() alongside scale status.
    searchStats: null,
    get scaleActive() {
      return ["running", "pausing", "stopping"].includes(this.scaleStatus?.status);
    },

    /* ---------------- Prediction tab (universe grid + batch) ------------- */
    predUniverse: { markets: { IN: [], US: [] } },
    predProgress: { total: 0, done: 0, failed: 0 },
    predRunDate: "",
    predExpanded: {},
    predMarket: "all",       // grid filter + batch scope: all | IN | US
    predMode: "bulk",        // Prediction tab: bulk universe | single name
    predForceRerun: false,   // ignore today's done cache on the next single/bulk
    predWorkers: 5,
    // Bulk/scale prediction runs a SAVED snapshot of the Settings form
    // (dialogue rounds + tool select/unselect + LLM preset), so a batch is
    // reproducible even if Settings change mid-run.  Saved with the "Save
    // bulk settings" button; defaults to the current Settings on first load.
    bulkCfg: {},
    bulkCfgSaved: false,
    bulkToolsOpen: false,
    predError: "",
    predSearch: { q: "", market: "IN", results: [], open: false, loading: false, adding: "" },
    predHistoryDays: [],
    predHistDate: "",        // "" = live/today; else viewing a historical day
    reviewReturnTab: "run-names",  // Names | Portfolio to restore after /run/{ticker}

    /* ---------- Virtual portfolio (₹10L growth-weighted paper book) ---------- */
    portfolio: null,          // { run_date, capital, holdings[], totals{}, fx }
    portfolioDate: "",        // run_date currently shown (kept in sync w/ predHistDate)
    portfolioMarket: "all",   // universe filter: all | IN | US (full ₹10L per view)
    portfolioLoading: false,
    portfolioError: "",
    // Click-to-sort state for the holdings table (SQL-viewer style).
    portfolioSort: { key: "current_value", dir: "desc" },
    predHist: { sectors: [], action_counts: {}, total: 0, done: 0, failed: 0 },
    // Action-distribution filter on the history view: null = show All, else a
    // five-tier action key (e.g. "STRONG_BUY") that hides every other tile.
    predActionFilter: null,
    _predEnrichRunning: false,

    logs: [],
    pendingLogCount: 0,
    autoScroll: true,
    userScrolledUp: false,

    chartData: null,
    chartLoading: false,
    chartError: "",
    enabledIndicators: [
      "close_10_ema", "close_50_sma", "close_200_sma",
      "boll_ub", "boll_lb",
    ],

    ws: null,
    _logSeq: 0,

    // -------------------------------------------------------------------
    // Session manager (multi-run background execution)
    // -------------------------------------------------------------------
    // The backend now hosts every run inside a process-wide session
    // manager keyed by ``run_id``.  The frontend keeps a shallow copy
    // of those sessions here so:
    //   * the user can queue multiple stocks and switch between them
    //     without cancelling anything mid-flight (was: closing the WS
    //     killed the kickoff),
    //   * a browser reload resumes tailing whichever session was
    //     active — the buffered events replay from the server so we
    //     rebuild the diagram + reports from scratch,
    //   * completed runs stay visible in the sidebar for review even
    //     after a refresh.
    // ``activeRunId`` names the session we're currently viewing.  It
    // may or may not equal one whose kickoff is still in-flight; a
    // completed run is still "attached" as long as we're rendering it.
    sessions: [],                  // [{ run_id, ticker, status, started_at, completed_at, final_action, error, event_count, subscribers, cfg }]
    activeRunId: null,
    sessionsPollTimer: null,
    runStartedAt: null,
    runCompletedAt: null,
    clockNow: Date.now(),
    elapsedTickTimer: null,

    /* ---------------------------------------------------------- display helpers */
    // Human-friendly rendering of the 5-tier PortfolioDecision.action field.
    // We keep legacy 3-tier records readable by collapsing OVERWEIGHT/
    // UNDERWEIGHT/NEUTRAL to the equivalent BUY/SELL/HOLD label; every
    // colour class in index.html already gates on both the new and the
    // legacy string so no cross-referenced code needs to inspect the raw
    // value directly.
    displayAction(a) {
      if (a === null || a === undefined || a === "") return "";
      return this._canonAction(a).replace(/_/g, " ");
    },
    localCalendarDate() {
      const n = new Date();
      return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, "0")}-${String(n.getDate()).padStart(2, "0")}`;
    },
    // Day the live Prediction index is actually on (server calendar, from
    // /api/predict/universe). Browser midnight can differ from the server.
    get liveRunDate() {
      return this.predRunDate || this.localCalendarDate();
    },
    get predHistoryPastDays() {
      const live = this.liveRunDate;
      return (this.predHistoryDays || []).filter((d) => d && d.run_date !== live);
    },
    get onRunReviewPage() {
      return !!(this.predHistDate && !this.focusPage && !this.settingsPage);
    },
    deskTabs() {
      const desk = this.predHistDate ? "review" : "live";
      return (this.tabs || []).filter((t) => (
        t.scope === "global"
        && !this.focusPage
        && (!t.asset || t.asset === this.assetClass)
        && (!t.desk || t.desk === desk)
      ));
    },

    /* ---------- Horizon helpers ---------- */
    // Bucket a horizon_days number into SHORT / MEDIUM / LONG.  Mirrors
    // the Python ``horizon_days_to_horizon`` in trading_crew/horizons.py
    // so the UI can render the tier badge for legacy runs that don't
    // carry the explicit ``horizon`` field.
    horizonFromDays(days) {
      const d = Number(days) || 0;
      if (d <= 0) return "MEDIUM";
      if (d <= 60) return "SHORT";
      if (d <= 252) return "MEDIUM";
      return "LONG";
    },
    horizonHumanLabel(tier) {
      if (tier === "SHORT") return "Short term (up to 3 months)";
      if (tier === "LONG") return "Long term (1-10 years)";
      return "Medium term (3-12 months)";
    },

    /* ---------------------------------------------------------- boot */
    async init() {
      await this.loadOptions();
      // Bulk/scale runs use a saved snapshot of Settings; seed it (from
      // localStorage, or the current Settings on first load) now that the
      // selected LLM preset is known.
      this._initBulkCfg();
      // Restore the previously-selected concurrency (survives server restart).
      this._initScaleWorkers();

      // Routing off the URL path (hierarchical multi-page, both asset classes):
      //   /run/{ticker} | /commodity/run/{c}  -> focused workbench (no left pane)
      //   /settings[?book=]                   -> run config + strategy/prompts
      //   /stocks/history/{date}              -> run review (Names · Portfolio)
      //   /stocks | /commodity | /            -> live high-level dashboard
      const path = window.location.pathname || "";
      const runMatch = path.match(/^\/(?:commodity\/)?run\/(.+)$/);
      const histMatch = path.match(/^\/(?:stocks|predict)\/history\/(.+)$/);
      const onSettings = path === "/settings" || path.startsWith("/settings/");
      const dateParam = new URLSearchParams(window.location.search).get("date") || "";
      let deepLinkTicker = "";
      if (onSettings) {
        this.settingsPage = true;
        this.focusPage = false;
        this.sidebarMode = "settings";
        const book = new URLSearchParams(window.location.search).get("book") || "";
        if (book === "commodity") this.assetClass = "commodity";
        else if (book === "equity" || book === "stock") this.assetClass = "stock";
      } else if (runMatch) {
        const t = decodeURIComponent(runMatch[1]).trim().toUpperCase();
        if (t) {
          deepLinkTicker = t;
          this.form.ticker = t;
          this.stockActive = true;
          this.focusPage = true;
          this.activeTab = "workflow";
          this.sidebarMode = "settings";
          if (dateParam) {
            this.predHistDate = dateParam;
            this.portfolioDate = dateParam;
          }
          // Hydrate AFTER refreshSessions so a live single-run session
          // wins over the cached Prediction-index HOLD (otherwise
          // loadLatestRun paints "completed" on top of a still-running
          // kickoff).
        }
      } else if (histMatch) {
        this.predHistDate = decodeURIComponent(histMatch[1]);
        this.portfolioDate = this.predHistDate;
        this.activeTab = (new URLSearchParams(window.location.search).get("tab") === "portfolio")
          ? "portfolio"
          : "run-names";
        this.reviewReturnTab = this.activeTab;
      } else if (this.assetClass === "commodity") {
        // Commodity high-level default view is the Contracts picker (its
        // Prediction / Names / Portfolio analogs don't apply).
        this.activeTab = "contracts";
      }

      // Browser back/forward: reload only when crossing the page boundary
      // (focused ⇄ high-level) so each URL behaves like a real page while
      // same-page history entries (e.g. tab/date changes) stay instant.
      window.addEventListener("popstate", () => {
        const p = window.location.pathname || "";
        const onFocusPage = /^\/(?:commodity\/)?run\//.test(p);
        const onSettingsPage = p === "/settings" || p.startsWith("/settings/");
        if (onFocusPage !== this.focusPage || onSettingsPage !== this.settingsPage) {
          window.location.reload();
          return;
        }
        if (this.focusPage || this.settingsPage || this.assetClass !== "stock") return;
        const hist = p.match(/^\/(?:stocks|predict)\/history\/(.+)$/);
        const date = hist ? decodeURIComponent(hist[1]) : "";
        if (date) this.enterRunReview(date, { push: false });
        else if (this.predHistDate) this.backToLive({ push: false });
      });

      // Bootstrap the high-level front page (universe grid, history list, batch
      // status polling).  Skipped on the focused per-instrument page, and the
      // stock-universe loads are skipped for the commodity dashboard (its
      // Contracts grid comes from loadOptions()).
      if (!this.focusPage && !this.settingsPage && this.assetClass === "stock") {
        this.loadPredHistory();
        this.loadPredUniverse();
        if (this.predHistDate) {
          this.loadPredHistDay();
          if (this.activeTab === "portfolio") this.loadPortfolio();
        } else {
          this.predEnrichLoop();
        }
      }
      if (this.settingsPage) {
        this.loadSettingsPage();
      } else {
        this._hydrateBulkFromServer();
      }
      // Batch inflight map is a second source of live run_ids (scale
      // workers). Single-run sessions are handled after refreshSessions.
      this.initScale().then(() => {
        if (!deepLinkTicker) return;
        if (this._sessionIsLive(this.activeRunId)) return;
        const runId = this.predRunIdFor(deepLinkTicker);
        if (runId && this.activeRunId !== runId) {
          this.attachToSession(runId);
        }
      });

      this.mountWorkflowDiagram();

      this.$watch("form.ticker", () => this.loadChart());
      this.$watch("form.trade_date", () => this.loadChart());
      this.$watch("enabledIndicators", () => this.rebuildCharts());
      this.$watch("activeTab", (v) => {
        if (v === "charts") setTimeout(() => this.rebuildCharts(), 60);
        // Episodes tab (replaces the old Memory tab): hydrate both the
        // top-k retrieval inspector and the full-store table on open.
        if (v === "episodes") {
          this.loadMemoryRetrieval();
          this.loadCalibration();
        }
        if (v === "training") {
          this.loadStrategyBoard();
        }
        // Re-fit the workflow diagram whenever the user returns to its
        // tab. The container's clientWidth is 0 while x-show hides it, so
        // any fit() call (window resize, ResizeObserver, etc.) during a
        // hidden state would otherwise leave the inner SVG scale-collapsed.
        // setTimeout pushes the call past Alpine's DOM update so display
        // is back to block before we read clientWidth.
        if (v === "workflow" && this.diagram) {
          setTimeout(() => this.diagram.fit(), 0);
        }
      });
      this.$watch("form.tools_enabled", () => {
        if (this.diagram) this.diagram.setActiveTools(this.form.tools_enabled);
      }, { deep: true });
      this.$watch("form.run_mode", () => {
        if (this.assetClass !== "stock") return;
        this.mountWorkflowDiagram();
      });

      // Glossary highlighter — re-annotate the Reports tab whenever the
      // analyst outputs or the final PM rationale change.  Wrapped in
      // $nextTick so we run after Alpine has flushed the new x-html.
      const annotate = () => {
        if (!window.TcAutoGlossary) return;
        this.$nextTick(() => {
          document.querySelectorAll(".report-body, .pm-rationale").forEach((el) => {
            window.TcAutoGlossary.annotate(el);
          });
        });
      };
      this.$watch("reports", annotate, { deep: true });
      this.$watch("reportsByRound", annotate, { deep: true });
      this.$watch("decisionPayload", annotate);
      this.$watch("activeTab", (v) => {
        if (v === "workflow") annotate();
        if (v === "reports") this.hydrateAllReports();
      });
      this.$watch("selectedNode", (id) => {
        if (id && id !== "decision") this.hydrateRoleReports(id);
      });

      // Charts are a stock/commodity view; the sidebar "Recent runs" panel only
      // exists on the high-level/commodity layouts.  Skip both on the focused
      // per-stock page's first paint (the chart still lazy-loads when its tab
      // opens via the form.ticker watcher) so /run/{ticker} boots fast.
      if (this.focusPage || this.assetClass === "commodity") this.loadChart();
      if (!this.focusPage) this.loadRecentRuns();

      // Session manager bootstrap — grab the list of live/recent
      // background runs and re-attach to whichever one the tab was
      // watching before the reload (persisted in localStorage).
      await this.refreshSessions();
      const savedActive = (typeof localStorage !== "undefined"
        ? localStorage.getItem("tc.activeRunId")
        : null);
      if (deepLinkTicker) {
        const live = this.liveSessionForTicker(deepLinkTicker);
        const saved = savedActive
          ? this.sessions.find((s) => s.run_id === savedActive)
          : null;
        const savedMatches = saved
          && String(saved.ticker || "").toUpperCase() === deepLinkTicker;
        const attachId = (savedMatches && this._sessionIsLive(savedActive))
          ? savedActive
          : (live && live.run_id);
        if (attachId) {
          this.attachToSession(attachId);
        } else {
          this._hydrateStoredRun(deepLinkTicker, this.predHistDate || dateParam || "");
        }
      } else if (savedActive && this.sessions.some((s) => s.run_id === savedActive)) {
        this.attachToSession(savedActive);
      }
      // Poll the sessions list so the sidebar reflects promotion
      // (queued -> running) and completion without a manual refresh.
      // Cheap: single JSON round-trip returning at most ~20 rows.
      this.sessionsPollTimer = setInterval(() => this.refreshSessions(), 3000);
      this.clockNow = Date.now();
      if (this.elapsedTickTimer) clearInterval(this.elapsedTickTimer);
      this.elapsedTickTimer = setInterval(() => { this.clockNow = Date.now(); }, 1000);

      // First-run guided tour (Phase 1E).  Fires on a fresh browser
      // (`localStorage.tcTourCompleted` unset) after the diagram has
      // finished its first paint.  We also expose the tab getter/setter
      // on `window.TcDashboard` so the Help-drawer "Restart tour" button
      // (which lives outside this Alpine scope) can call TcTour.start()
      // with the right callbacks.
      window.TcDashboard = {
        getTab: () => this.activeTab,
        setTab: (tab) => { this.activeTab = tab; },
      };
      this.$nextTick(() => {
        if (window.TcTour && typeof window.TcTour.maybeAutoStart === "function") {
          window.TcTour.maybeAutoStart(window.TcDashboard.getTab, window.TcDashboard.setTab);
        }
      });
    },

    /* --------------------------------------------- stored prediction */
    // Loads the DB-stored batch prediction for a ticker (from /predict tiles).
    async loadStoredPrediction(ticker, runDate) {
      try {
        const qs = runDate ? `?run_date=${encodeURIComponent(runDate)}` : "";
        const r = await fetch(`/api/predict/prediction/${encodeURIComponent(ticker)}${qs}`);
        if (!r.ok) return null;
        const d = await r.json();
        this.storedPredictionTicker = ticker;
        this.storedPrediction = d.prediction || null;
        return this.storedPrediction;
      } catch (_) {
        /* no stored prediction is fine — the user can run a fresh one */
        return null;
      }
    },
    // Replay the stored prediction and, when we have a run_id for that
    // day, that exact workflow — not latest, which can be a newer date.
    async _hydrateStoredRun(ticker, runDate) {
      const pred = await this.loadStoredPrediction(ticker, runDate || "");
      const runId = pred && pred.run_id;
      if (runId) await this.loadRun(ticker, runId);
      else await this.loadLatestRun(ticker);
    },

    /* ---------------------------------------------------------- options */
    // Fetches the bootstrap payload (tickers, agents, tools, LLM info).
    // Exposed as a method so the user can retry from the UI when the
    // initial load failed (network blip, blocked extension, etc.).
    async loadOptions() {
      this.optionsRetrying = true;
      this.optionsError = "";
      this.optionsLoaded = false;
      try {
        const url = `/api/options?asset_class=${this.assetClass}`;
        const resp = await fetch(url, { cache: "no-store" });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
        }
        const res = await resp.json();

        this.tickers = res.tickers || [];
        this.toolsInfo = res.tools || {};
        this.indicatorLabels = res.chart_indicators || {};
        this.llm = res.llm || {};
        this.llmPresets = res.llm_presets || [];
        const persisted = (typeof localStorage !== "undefined"
          ? localStorage.getItem("tc.llm_preset")
          : null);
        const isValid = persisted && this.llmPresets.some((p) => p.id === persisted);
        this.selectedLlmPreset = isValid
          ? persisted
          : (res.default_llm_preset
            || (this.llmPresets[0] && this.llmPresets[0].id)
            || "");

        this.tavily = !!res.tavily_configured;
        this.reactAgents = res.react_agents || [];

        this.form.ticker = res.default_ticker || this.form.ticker;
        this.form.trade_date = res.default_trade_date || this.form.trade_date;
        this.form.max_debate_rounds = res.default_debate_rounds || this.form.max_debate_rounds;
        this.form.max_risk_rounds = res.default_risk_rounds || this.form.max_risk_rounds;

        // Tool owners: agents whose .tools[] is non-empty
        this.toolOwners = (res.agents || []).filter((a) => (a.tools || []).length > 0);

        // Default tool selection mirrors the crew defaults.
        const enabled = {};
        this.toolOwners.forEach((a) => { enabled[a.key] = [...(a.tools || [])]; });
        this.form.tools_enabled = enabled;

        if (!this.toolOwners.length) {
          this.optionsError = `No agents returned for asset_class=${this.assetClass}. Check the backend logs.`;
        } else {
          this.optionsLoaded = true;
        }

        if (this.diagram) {
          this.diagram.setActiveTools(this.form.tools_enabled);
          this.diagram.setReactAgents(this.reactAgents);
        }
      } catch (err) {
        console.error("Failed to load /api/options", err);
        this.optionsError = String(err?.message || err);
      } finally {
        this.optionsRetrying = false;
      }
    },

    /* ---------------------------------------------------------- collapsibles */
    toggle(id) { this.openSections[id] = !this.openSections[id]; },
    openSection(id) {
      this.openSections[id] = true;
    },

    /* ---------------------------------------------------------- workflow diagram */
    diagramLayoutId() {
      if (this.assetClass === "commodity") return "commodity";
      if (this.form.run_mode === "deterministic") return "stock_deterministic";
      if (this.form.run_mode === "harness") return "stock_harness";
      return "stock";
    },
    mountWorkflowDiagram() {
      const el = document.getElementById("workflow-diagram");
      if (!el) return;
      const layoutId = this.diagramLayoutId();
      if (!this.diagram) {
        this.diagram = new WorkflowDiagram(el, {
          assetClass: this.assetClass,
          layoutId,
        });
        this.diagram.onSelect((id) => (this.selectedNode = id));
        this.diagram.onScale((s) => { this.diagramScale = s; });
      } else {
        this.diagram.setLayout(layoutId);
      }
      this.diagram.setActiveTools(this.form.tools_enabled);
      this.diagram.setReactAgents(this.reactAgents);
      this.diagramScale = this.diagram.scale;
      if (this.selectedNode && !this.diagram.nodesById[this.selectedNode]) {
        this.selectedNode = null;
      }
      if (typeof this.diagram.setGraphMeta === "function") {
        this.diagram.setGraphMeta({
          runMode: this.form.run_mode,
          orchestrator: this.assetClass === "commodity" || this.form.run_mode === "deterministic"
            ? "crewai"
            : (this.form.run_mode === "harness" ? "harness" : "langgraph"),
          qualityReview: this.form.quality_review,
          compliance: this.form.compliance,
        });
      }
    },

    /* ---------------------------------------------------------- workflow zoom */
    // Toolbar handlers: + / − step by 20% multiplicatively so the perceived
    // zoom rate stays roughly constant whether you're at 50% or 200%.
    diagramZoomIn()  { this.diagram && this.diagram.zoomBy(1.2); },
    diagramZoomOut() { this.diagram && this.diagram.zoomBy(1 / 1.2); },
    // "Fit": reflow to container width, then reset user zoom to 1.0.
    diagramZoomFit() {
      if (!this.diagram) return;
      this.diagram.userZoom = 1;
      this.diagram.fit();
    },
    // "100%": force a pixel-perfect rendering, useful for sharing screenshots.
    diagramZoomReset() { this.diagram && this.diagram.resetZoom(); },

    get activeLlmPreset() {
      return (this.llmPresets || []).find((p) => p.id === this.selectedLlmPreset) || null;
    },
    get llmSummary() {
      const p = this.activeLlmPreset;
      if (p) return p.model;
      if (!this.llm?.model) return "(unset)";
      return this.llm.model.split("/").pop();
    },
    onLlmPresetChange() {
      try {
        if (typeof localStorage !== "undefined") {
          localStorage.setItem("tc.llm_preset", this.selectedLlmPreset || "");
        }
      } catch (_) { /* private mode etc. */ }
    },
    get operatorPicksDialogue() {
      return this.assetClass === "commodity"
        || (this.form.run_mode || "autonomous") === "deterministic";
    },
    dialogueMeta() {
      if (!this.operatorPicksDialogue) {
        return (this.form.run_mode || "autonomous") === "harness"
          ? "conductor-driven"
          : "graph-driven";
      }
      return `debate ${this.form.max_debate_rounds} · risk ${this.form.max_risk_rounds}`;
    },
    _dialogueSnapshot() {
      if (!this.operatorPicksDialogue) return {};
      return {
        debate_rounds: this.form.max_debate_rounds,
        risk_rounds: this.form.max_risk_rounds,
      };
    },
    get indicatorKeys() { return Object.keys(this.indicatorLabels || {}); },
    get selectedAnalystCount() {
      return this.toolOwners.filter((a) =>
        (this.form.tools_enabled[a.key] || []).length > 0
      ).length;
    },

    /* ---------------------------------------------------------- tools */
    isToolEnabled(agentKey, tool) {
      return (this.form.tools_enabled[agentKey] || []).includes(tool);
    },
    toggleTool(agentKey, tool, on) {
      const cur = new Set(this.form.tools_enabled[agentKey] || []);
      if (on) cur.add(tool); else cur.delete(tool);
      this.form.tools_enabled[agentKey] = Array.from(cur);
    },
    totalEnabledTools() {
      return Object.values(this.form.tools_enabled).reduce((n, arr) => n + (arr?.length || 0), 0);
    },
    totalAvailableTools() {
      return this.toolOwners.reduce((n, a) => n + (a.tools?.length || 0), 0);
    },

    /* ---------------------------------------------------------- charts */
    async loadChart() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker || !this.form.trade_date) return;
      this.chartLoading = true;
      this.chartError = "";
      try {
        const url = `/api/chart?ticker=${encodeURIComponent(ticker)}&trade_date=${encodeURIComponent(this.form.trade_date)}`;
        const res = await fetch(url).then((r) => r.json());
        // Guard against a stale response: on page-reload we fire a default
        // loadChart() and then another for the reattached session's ticker.
        // If the ticker changed while this request was in flight, drop it so
        // the chart doesn't flip back to the wrong (e.g. default) symbol.
        if ((this.form.ticker || "").trim().toUpperCase() !== ticker) return;
        this.chartData = res;
        if (res.error) this.chartError = res.error;
      } catch (err) {
        this.chartError = String(err);
      } finally {
        this.chartLoading = false;
        this.$nextTick(() => this.rebuildCharts());
      }
    },
    reloadChart() { this.loadChart(); },
    rebuildCharts() {
      if (!this.chartData) return;
      window.TcCharts.buildPriceChart(
        document.getElementById("price-chart"),
        this.chartData,
        this.enabledIndicators,
      );
      window.TcCharts.buildRsiChart(document.getElementById("rsi-chart"), this.chartData);
      window.TcCharts.buildMacdChart(document.getElementById("macd-chart"), this.chartData);
    },

    /* ---------------------------------------------------------- run */
    // ``startRun()`` kicks off a *background* session on the server
    // and then attaches this tab to its event stream.  A single tab
    // renders one session at a time (the "active" one), but the
    // backend keeps every session alive independently — so refreshing
    // the page just re-attaches, and pushing "Run" again while an
    // earlier session is still going queues a second one that will
    // start as soon as the concurrency cap allows.
    async startRun() {
      // Allow queuing a new session even while another is running
      // (true multi-session), but guard the POST round-trip against a
      // double-click.
      if (this.starting) return;
      this.starting = true;
      this.form.trade_date = this.localCalendarDate();
      const f = this.form;
      const payload = {
        asset_class: this.assetClass,
        ticker: (f.ticker || "").trim().toUpperCase(),
        ...this._dialogueSnapshot(),
        run_mode: f.run_mode || "autonomous",
        quality_review: f.quality_review,
        compliance: f.compliance,
        critic_enabled: f.critic_enabled,
        fusion_enabled: f.fusion_enabled,
        fusion_include_crowd: f.fusion_include_crowd,
        tools_enabled: f.tools_enabled,
        book: this.form.book || "paper",
        llm_preset: this.selectedLlmPreset || "",
        force_rerun: this.assetClass !== "commodity" && !!this.predForceRerun,
      };
      if (!payload.ticker) {
        this.appendLog("error", "client", "Ticker is required.");
        this.starting = false;
        return;
      }

      // A run configured in Settings opens the focused per-instrument workbench
      // (Workflow tab + instrument tabs) so the user watches it stream — the
      // per-stock page (/run/{ticker}) or per-contract page
      // (/commodity/run/{contract}) depending on asset class.
      this.stockActive = true;
      this.settingsPage = false;
      if (this.activeTab === "prediction" || this.activeTab === "contracts" || this.sidebarMode === "settings") {
        this.activeTab = "workflow";
      }
      if (!this.focusPage) {
        this.focusPage = true;
        const base = this.assetClass === "commodity" ? "/commodity/run/" : "/run/";
        try { window.history.pushState({ ticker: payload.ticker }, "", base + encodeURIComponent(payload.ticker)); } catch (_) {}
      }

      try {
        const res = await fetch("/api/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const errText = await res.text();
          this.appendLog("error", "client", `Start failed: ${errText}`);
          return;
        }
        const body = await res.json();
        if (body.cached) {
          this.status = "completed";
          this.isRunning = false;
          this.appendLog("info", "client",
            `Using today's cached prediction for ${payload.ticker}`
            + (body.run_id ? ` (run_id=${String(body.run_id).slice(0, 16)})` : ""));
          await this.loadStoredPrediction(payload.ticker, body.run_date || "");
          if (body.run_id) {
            await this.loadRun(payload.ticker, body.run_id);
          }
          this.loadPredUniverse();
          this.loadPredHistory();
          this.loadRecentRuns();
          return;
        }
        this.appendLog("info", "client",
          `Queued ${payload.ticker} (run_id=${String(body.run_id || "").slice(0, 16)})`
          + (payload.run_mode === "harness"
            ? " harness"
            : ` debate=${payload.debate_rounds} risk=${payload.risk_rounds}`));
        await this.refreshSessions();
        this.attachToSession(body.run_id);
      } catch (err) {
        this.appendLog("error", "client", `Start failed: ${String(err)}`);
      } finally {
        this.starting = false;
      }
    },

    // Number of sessions currently queued or running — surfaced on the
    // Run button so the user can see the background fan-out at a glance.
    get activeSessionCount() {
      return (this.sessions || []).filter(
        (s) => s.status === "running" || s.status === "queued"
      ).length;
    },

    // Reset every per-run render field so switching between sessions
    // doesn't leak state from the previous view.  Called by
    // attachToSession before we replay the target session's buffer.
    resetRunState() {
      if (this.diagram) {
        this.diagram.reset();
        this.diagram.setActiveTools(this.form.tools_enabled);
      }
      this.logs = [];
      this.pendingLogCount = 0;
      this.reports = {};
      this.reportsByRound = {};
      this.degradedRoles = {};
      this.finalDecision = "";
      this.cascadeStatus = null;
      this.actionProposal = null;
      this.actionProposalMd = "";
      this.executionResult = null;
      this.episodeMeta = null;
      this.reflectionRecords = null;
      this.decisionPayload = null;
      this.fusionResult = null;
      this.reportSignals = null;
      this.priceSignals = null;
      this.deskBiasBoard = null;
      this.evidenceBlackboard = null;
      this.debateScratchpad = null;
      this.nodeActivities = {};
      this.selectedNode = null;
      this.currentNode = "";
      this.runStartedAt = null;
      this.runCompletedAt = null;
    },

    // Attach the current tab to ``runId``'s live event stream.  Closes
    // any existing viewer socket first (attach is exclusive per tab)
    // so we don't double-render events into the wrong panels.  The
    // server sends a full buffer replay before it starts tailing, so
    // the diagram + reports come back to their exact state as of the
    // last event — regardless of when we joined.
    attachToSession(runId) {
      if (!runId) return;
      if (this.ws) {
        try { this.ws.close(); } catch (_) {}
        this.ws = null;
      }
      this.resetRunState();
      this.activeRunId = runId;
      try {
        if (typeof localStorage !== "undefined") {
          localStorage.setItem("tc.activeRunId", runId);
        }
      } catch (_) {}
      // A session that's already terminated is still valid to attach
      // to — we just replay history and immediately hit EOF.  Keep
      // isRunning false so the sticky Run button doesn't spin.
      const s = this.sessions.find((x) => x.run_id === runId);
      this.isRunning = s && (s.status === "queued" || s.status === "running");
      this.status = s ? s.status : "running";
      this.applyRunClock(s && s.started_at, s && s.completed_at);

      // Sync the visible ticker / date / asset-class to the session this
      // tab is now attached to.  Without this a page reload (which resets
      // ``form`` to defaults, e.g. NTNX) would leave the header + Charts
      // tab showing the default ticker while the run was actually for a
      // different symbol.  We reload the chart so Charts reflects THIS run.
      if (s) {
        if (s.ticker) this.form.ticker = s.ticker;
        const cfg = s.cfg || {};
        if (cfg.trade_date || cfg.decision_ts) {
          this.form.trade_date = String(cfg.trade_date || cfg.decision_ts).slice(0, 10);
        }
        // NB: we deliberately do NOT flip ``assetClass`` here.  Sessions
        // are process-global across the /stock and /commodity dashboards;
        // switching asset mode mid-page (without reloading /api/options)
        // would leave stale tickers/tools.  Ticker + date sync is enough
        // to keep the header + Charts correct on either dashboard.
        this.loadChart();
      }

      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/session/${encodeURIComponent(runId)}`);
      this.ws = ws;
      ws.addEventListener("message", (e) => {
        try { this.handleEvent(JSON.parse(e.data)); }
        catch (err) { this.appendLog("error", "parse", String(err)); }
      });
      ws.addEventListener("close", () => {
        // Ignore a close for a socket we already replaced (reattach)
        // or for a kickoff that is still running — keep the header
        // and Run spinner aligned with /api/sessions.
        if (this.ws !== ws) return;
        if (this._sessionIsLive(this.activeRunId)) return;
        this.isRunning = false;
        this.currentNode = "";
      });
      ws.addEventListener("error", () => {
        this.appendLog("error", "ws", `WebSocket error on run ${runId}`);
      });
    },

    // History session row → focused workbench (/run/{ticker} or
    // /commodity/run/{c}). Hard-navigate so we never stay on /settings
    // with a pushed /run URL (settingsPage would hide Workflow).
    openSession(runId) {
      if (!runId) return;
      const s = (this.sessions || []).find((x) => x.run_id === runId);
      const ticker = ((s && s.ticker) || this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      const sessionAsset = String((s && s.cfg && s.cfg.asset_class) || this.assetClass || "stock").toLowerCase();
      try { localStorage.setItem("tc.activeRunId", runId); } catch (_) {}
      const base = sessionAsset === "commodity" ? "/commodity/run/" : "/run/";
      const dest = base + encodeURIComponent(ticker);
      const here = window.location.pathname || "";
      if (here === dest && this.focusPage && !this.settingsPage) {
        this.form.ticker = ticker;
        this.stockActive = true;
        this.activeTab = "workflow";
        this.attachToSession(runId);
        return;
      }
      window.location.assign(dest);
    },

    deskSessions() {
      const want = this.assetClass === "commodity" ? "commodity" : "stock";
      return (this.sessions || []).filter((s) => {
        const ac = String((s.cfg && s.cfg.asset_class) || "").toLowerCase();
        return ac === want;
      });
    },

    liveDeskSessions() {
      return this.deskSessions().filter(
        (s) => s.status === "running" || s.status === "queued",
      );
    },

    liveSessionForTicker(ticker) {
      const t = String(ticker || "").toUpperCase();
      if (!t) return null;
      const live = (this.sessions || []).filter((s) => (
        String(s.ticker || "").toUpperCase() === t
        && (s.status === "running" || s.status === "queued")
      ));
      live.sort((a, b) => String(b.started_at || "").localeCompare(String(a.started_at || "")));
      return live[0] || null;
    },

    _sessionIsLive(runId) {
      if (!runId) return false;
      const s = (this.sessions || []).find((x) => x.run_id === runId);
      return !!(s && (s.status === "running" || s.status === "queued"));
    },

    _hydrateBlockedByLive(ticker) {
      const t = String(ticker || "").toUpperCase();
      if (this.activeRunId && this._sessionIsLive(this.activeRunId)) {
        const s = (this.sessions || []).find((x) => x.run_id === this.activeRunId);
        if (s && String(s.ticker || "").toUpperCase() === t) return true;
      }
      return !!this.liveSessionForTicker(t);
    },

    async cancelRun() {
      if (!this.activeRunId) {
        // No active session — just release the local spinner.
        this.isRunning = false;
        this.status = "idle";
        return;
      }
      await this.cancelSpecificRun(this.activeRunId);
    },

    // Cancel *a specific* background run — used by the per-row Cancel
    // button in the Sessions panel, and by cancelRun() for the tab's
    // currently-attached session.  Kept separate so a user can cancel
    // a *different* queued run without first having to attach to it.
    async cancelSpecificRun(runId) {
      if (!runId) return;
      try {
        const res = await fetch(
          `/api/sessions/${encodeURIComponent(runId)}/cancel`,
          { method: "POST" },
        );
        if (!res.ok) {
          this.appendLog("warn", "client", `Cancel returned ${res.status}`);
        } else {
          this.appendLog("info", "client", `Cancel requested for ${runId.slice(0,16)}`);
        }
      } catch (err) {
        this.appendLog("error", "client", `Cancel failed: ${String(err)}`);
      }
      await this.refreshSessions();
    },

    async refreshSessions() {
      try {
        const res = await fetch("/api/sessions").then((r) => r.json());
        this.sessions = (res.runs || []);
        // If the active run finished, drop the sticky "running" state.
        if (this.activeRunId) {
          const s = this.sessions.find((x) => x.run_id === this.activeRunId);
          if (s) this.applyRunClock(s.started_at, s.completed_at);
          if (s && s.status !== "queued" && s.status !== "running") {
            this.isRunning = false;
            this.status = s.status;
          }
        }
      } catch (err) {
        // Non-fatal — the poller will retry.  Don't flood the log.
        console.warn("refreshSessions failed", err);
      }
    },

    async removeSession(runId) {
      if (!runId) return;
      // Only ask for confirmation on active runs; terminal ones are cheap.
      const s = this.sessions.find((x) => x.run_id === runId);
      if (s && (s.status === "running" || s.status === "queued")) {
        if (!window.confirm(`Session ${runId.slice(0,16)} is still active. Cancel it first?`)) return;
        await fetch(`/api/sessions/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
        // Give the server a moment to transition to CANCELLED before deleting.
        await new Promise((r) => setTimeout(r, 300));
      }
      try {
        await fetch(`/api/sessions/${encodeURIComponent(runId)}`, { method: "DELETE" });
      } catch (err) {
        console.warn("removeSession failed", err);
      }
      if (this.activeRunId === runId) {
        this.activeRunId = null;
        try { if (this.ws) this.ws.close(); } catch (_) {}
        try { localStorage.removeItem("tc.activeRunId"); } catch (_) {}
      }
      await this.refreshSessions();
    },

    // -------------------------------------------------------------------
    // Session-status helpers used by the sidebar template
    // -------------------------------------------------------------------
    parseRunTs(value) {
      if (!value) return null;
      const t = Date.parse(value);
      return Number.isFinite(t) ? t : null;
    },
    applyRunClock(startedAt, completedAt) {
      if (startedAt) this.runStartedAt = startedAt;
      if (completedAt) this.runCompletedAt = completedAt;
    },
    formatElapsed(ms) {
      if (ms == null || ms < 0 || !Number.isFinite(ms)) return "—";
      const s = Math.max(0, Math.round(ms / 1000));
      if (s < 60) return `${s}s`;
      const m = Math.floor(s / 60);
      const rem = s % 60;
      if (m < 60) return rem ? `${m}m ${rem}s` : `${m}m`;
      const h = Math.floor(m / 60);
      return rem ? `${h}h ${m % 60}m ${rem}s` : `${h}h ${m % 60}m`;
    },
    get workflowElapsedLabel() {
      const t0 = this.parseRunTs(this.runStartedAt);
      if (t0 == null) return "—";
      const doneAt = this.parseRunTs(this.runCompletedAt);
      if (doneAt != null) return this.formatElapsed(doneAt - t0);
      const live = this.isRunning || this.status === "running" || this.status === "queued";
      if (!live) return "—";
      return this.formatElapsed(this.clockNow - t0);
    },

    sessionStatusClass(s) {
      switch ((s || "").toLowerCase()) {
        case "running":   return "bg-blue-500/20 text-blue-200";
        case "queued":    return "bg-amber-500/20 text-amber-200";
        case "completed": return "bg-emerald-500/20 text-emerald-200";
        case "error":     return "bg-rose-500/20 text-rose-200";
        case "cancelled": return "bg-slate-500/20 text-slate-300";
        default:          return "bg-slate-500/20 text-slate-300";
      }
    },

    /* ---------------------------------------------------------- events */
    handleEvent(event) {
      switch (event.type) {
        case "run_started":
          // Keep the header + Charts locked to the run we're viewing,
          // even after a reload where ``form`` was reset to defaults.
          if (event.ticker && event.ticker !== this.form.ticker) {
            this.form.ticker = event.ticker;
            this.loadChart();
          }
          if (event.trade_date) this.form.trade_date = String(event.trade_date).slice(0, 10);
          this.appendLog("info", "run",
            `${event.ticker} · ${event.agent_count} agents · ${event.task_count} tasks`
            + (event.orchestrator ? ` · ${event.orchestrator}` : "")
            + (event.run_mode ? ` · ${event.run_mode}` : "")
            + (event.playbook_action ? ` · ${event.playbook_action}` : ""));
          if (event.run_mode) this.form.run_mode = event.run_mode;
          if (!this.runStartedAt) this.runStartedAt = new Date().toISOString();
          this.runCompletedAt = null;
          this.mountWorkflowDiagram();
          if (this.diagram && typeof this.diagram.setGraphMeta === "function") {
            this.diagram.setGraphMeta({
              playbookAction: event.playbook_action,
              skipReports: event.skip_reports,
              runMode: event.run_mode,
              orchestrator: event.orchestrator,
              qualityReview: event.quality_review,
              compliance: event.compliance,
            });
          }
          return;

        case "graph_edge":
          if (this.diagram && typeof this.diagram.activateEdge === "function") {
            this.diagram.activateEdge(event.src, event.dst, event.kind || "activated");
          }
          return;

        case "graph_trace":
          this.appendLog("info", "graph",
            `${(event.edges || []).length} live edges`);
          return;

        case "node_started":
          this.currentNode = event.node;
          this.diagram.setState(event.node, "running");
          this.appendLog("node", event.node, "▶ started");
          return;

        case "node_completed": {
          this.diagram.setState(event.node, event.degraded ? "degraded" : "done");
          const output = event.output || "";
          this.reports[event.node] = output;
          // Append to the per-round history so multi-turn debates render
          // every round in the Reports tab.  Reactivity: re-create the
          // role's array AND the top-level object so Alpine's shallow
          // watchers fire on Alpine.js.
          const round = Number.isFinite(event.round) ? event.round : 1;
          const totalRounds = Number.isFinite(event.total_rounds) ? event.total_rounds : 1;
          const prior = this.reportsByRound[event.node] || [];
          const nextRoundList = prior.concat([{
            round, total_rounds: totalRounds, output, degraded: !!event.degraded,
          }]);
          this.reportsByRound = { ...this.reportsByRound, [event.node]: nextRoundList };
          if (event.degraded) {
            this.degradedRoles = { ...this.degradedRoles, [event.node]: true };
            this.appendLog("warn", event.node, "⚠ degraded: LLM emitted tool-call list as final answer");
          }
          const roundLabel = totalRounds > 1 ? ` · round ${round}/${totalRounds}` : "";
          this.pushActivity(event.node, {
            label: event.degraded ? `report (degraded)${roundLabel}` : `report${roundLabel}`,
            color: event.degraded ? "warn" : "llm",
            subtitle: `${output.length} chars`,
            body: output, markdown: true,
          });
          this.appendLog("node", event.node,
            event.degraded ? `⚠ completed (degraded)${roundLabel}` : `✓ completed${roundLabel}`);
          return;
        }

        case "agent_step": {
          const text = (event.content || "").trim();
          if (!text) return;
          this.appendLog("llm", event.node, text.slice(0, 240));
          this.pushActivity(event.node, {
            label: "step", color: "llm",
            subtitle: `${text.length} chars`,
            body: text, markdown: true,
          });
          return;
        }

        case "tool_call": {
          this.diagram.setToolState(event.node, event.tool, event.call_id, "running",
            typeof event.args === "string" ? event.args : JSON.stringify(event.args || {}, null, 2));
          this.appendLog("tool", event.tool, `[${event.node}] call`);
          this.pushActivity(event.node, {
            label: "tool call", color: "tool",
            subtitle: event.tool,
            body: typeof event.args === "string" ? event.args : JSON.stringify(event.args || {}, null, 2),
          });
          return;
        }

        case "tool_result": {
          const state = event.error ? "error" : "done";
          this.diagram.setToolState(event.node, event.tool, event.call_id, state, (event.output || "").slice(0, 600));
          this.appendLog(event.error ? "error" : "result", event.tool,
            `[${event.node}] result (${event.elapsed_ms || 0}ms)`);
          const boardTool = event.tool === "lookup_blackboard"
            || event.tool === "request_evidence"
            || event.tool === "read_scratchpad"
            || event.tool === "write_scratchpad";
          this.pushActivity(event.node, {
            label: event.error ? "tool error" : "tool result",
            color: event.error ? "error" : "result",
            subtitle: `${event.tool} · ${event.elapsed_ms || 0}ms · ${(event.output || "").length} chars`,
            body: event.output || "",
            markdown: boardTool && !event.error,
          });
          return;
        }

        case "final_decision":
          this.decisionPayload = event.decision;
          this.finalDecision = event.decision?.action || "";
          this.diagram.setDecision(this.finalDecision || "DECISION");
          this.appendLog("final", "decision", this.finalDecision || "(no parsed decision)");
          // Reports are folded into the Workflow tab now — stay put so the
          // decision + report render in place without a tab jump.
          if (this.stockActive) this.activeTab = "workflow";
          return;

        case "report_signals":
          this.reportSignals = {
            signals: event.signals || [],
            net_bias: event.net_bias,
            net_confidence: event.net_confidence,
            bias_score: event.bias_score,
          };
          this.appendLog("info", "signals",
            `guidance bias=${event.net_bias ?? "n/a"} score=${event.bias_score ?? "n/a"}`);
          return;

        case "price_signals":
          this.priceSignals = {
            signals: event.signals || [],
            vol_pct: event.vol_pct,
            ann_vol: event.ann_vol,
            atr_pct: event.atr_pct,
            pos_in_range: event.pos_in_range,
            drawdown_from_high: event.drawdown_from_high,
            regime_age_days: event.regime_age_days,
            hurst: event.hurst,
            dip_buy: event.dip_buy,
            size_risk_mult: event.size_risk_mult,
            size_note: event.size_note,
          };
          this.appendLog("info", "signals",
            `price structure: vol_pct=${event.vol_pct != null ? Number(event.vol_pct).toFixed(2) : "n/a"} ` +
            `regime_age=${event.regime_age_days != null ? Math.round(event.regime_age_days) + "d" : "n/a"} ` +
            `dip_buy=${event.dip_buy ? "YES" : "no"} size×${event.size_risk_mult ?? "1"}`);
          return;

        case "desk_bias_board":
          this.deskBiasBoard = event.desks || [];
          this.appendLog("info", "signals",
            `desk bias board: ${(event.desks || []).map(d => `${d.desk}=${d.bias}`).join(", ")}`);
          return;

        case "evidence_blackboard":
          this.evidenceBlackboard = {
            ticker: event.ticker,
            markdown: event.markdown || "",
            desks: event.desks || [],
            requests: event.requests || [],
          };
          if (this.diagram) this.diagram.setState("Shared State", "running");
          this.appendLog("info", "signals",
            `evidence blackboard: ${(event.desks || []).length} desks, `
            + `${(event.markdown || "").length} chars`);
          return;

        case "debate_scratchpad":
          this.debateScratchpad = {
            ticker: event.ticker,
            markdown: event.markdown || "",
            entries: event.entries || [],
          };
          this.appendLog("info", "signals",
            `debate scratch pad: ${(event.entries || []).length} entries, `
            + `${(event.markdown || "").length} chars`);
          return;

        case "cascade_status":
          this.cascadeStatus = {
            regime: event.regime, route: event.route, reason: event.reason,
          };
          this.appendLog("info", "cascade", `regime=${event.regime} route=${event.route}`);
          return;

        case "action_proposal":
          this.actionProposal = event.proposal;
          this.actionProposalMd = event.markdown || "";
          this.appendLog("info", "action_proposal",
            `${event.proposal?.side} ${(event.proposal?.target_weight * 100).toFixed(2)}%`);
          return;

        case "execution_result":
          this.executionResult = event;
          if (event.fill) {
            this.appendLog("info", "execution",
              `${event.fill.status} qty=${event.fill.qty_filled} @ $${event.fill.avg_price?.toFixed(2)}`);
          } else if (event.note) {
            this.appendLog("info", "execution", event.note);
          }
          return;

        case "reflection_records":
          this.reflectionRecords = event;
          this.appendLog("info", "critic",
            `${event.records?.length} sample(s) · final=${event.final_action} ${event.revised ? '(REVISED)' : ''}`);
          return;

        case "episode_recorded":
          this.episodeMeta = {
            episode_id: event.episode_id, regime: event.regime,
            decision_ts: event.decision_ts, outcome_ts: event.outcome_ts,
          };
          this.appendLog("info", "memory", `episode ${event.episode_id} (${event.regime})`);
          return;

        case "fusion":
          // Optional in-run signal fusion — surfaced in the Workflow tab
          // next to the final decision.  Advisory only.
          this.fusionResult = event.available
            ? { available: true, sources: event.sources, fused: event.fused }
            : { available: false, note: event.note, sources: event.sources };
          if (event.available && event.fused) {
            this.appendLog("info", "fusion",
              `fused=${event.fused.side} conf=${(event.fused.confidence * 100).toFixed(0)}%`);
          }
          return;

        case "run_completed":
          this.status = "completed";
          this.appendLog("info", "run", "completed");
          this.isRunning = false;
          this.currentNode = "";
          if (!this.runCompletedAt) this.runCompletedAt = new Date().toISOString();
          // Refresh both the on-disk audit list AND the in-memory
          // session sidebar so the completion badge flips right away
          // without waiting for the 3 s poll tick.
          this.loadRecentRuns();
          this.refreshSessions();
          if (this.assetClass === "stock") {
            this.loadPredUniverse();
            this.loadPredHistory();
            if (this.form.ticker) {
              this.loadStoredPrediction(this.form.ticker, this.localCalendarDate());
            }
          }
          return;

        case "session_started":
          // Legacy /ws/analyze clients only — the new path knows the
          // run_id up front from POST /api/sessions.  We still handle
          // this so old bookmarks keep working.
          if (event.run_id && !this.activeRunId) {
            this.activeRunId = event.run_id;
            try { localStorage.setItem("tc.activeRunId", event.run_id); } catch (_) {}
          }
          return;

        case "error":
          this.status = "error";
          this.appendLog("error", "error", event.message);
          this.isRunning = false;
          this.currentNode = "";
          if (!this.runCompletedAt) this.runCompletedAt = new Date().toISOString();
          this.refreshSessions();
          return;
      }
    },

    /* ---------------------------------------------------------- logs/activity */
    // The standalone activity log was removed — the diagram edge colours show
    // live progress and per-agent output is read by clicking a tile.  Kept as
    // a no-op so the many existing call sites stay valid without edits.
    appendLog() { /* logs removed */ },
    onLogsScroll() {
      const c = document.getElementById("logs-container");
      if (!c) return;
      this.userScrolledUp = c.scrollTop + c.clientHeight < c.scrollHeight - 40;
      if (!this.userScrolledUp) this.pendingLogCount = 0;
    },
    pushActivity(node, entry) {
      if (!node) return;
      const arr = this.nodeActivities[node] || [];
      arr.push({ ...entry, key: arr.length + ":" + Date.now() });
      this.nodeActivities[node] = arr;
    },
    nodeActivity(node) { return this.nodeActivities[node] || []; },
    nodeKindLabel(node) {
      if (!node) return "";
      if (node.includes("Researcher")) return "researcher";
      if (node.includes("Risk")) return "risk analyst";
      if (node.includes("Analyst")) return "analyst";
      if (node === "Trader" || node === "Senior Futures Trader") return "trader";
      if (node === "Portfolio Manager") return "portfolio manager";
      if (node === "Research Manager" || node === "Futures Research Manager") return "research manager";
      if (node === "Quality Reviewer") return "quality reviewer";
      if (node === "Compliance Officer") return "compliance officer";
      if (node === "Playbook Router") return "router";
      if (node === "Shared State") return "shared state";
      return "node";
    },
    entryBadgeClass(entry) {
      switch (entry.color) {
        case "tool":   return "bg-amber-500/10 text-amber-300 border border-amber-500/30";
        case "result": return "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30";
        case "error":  return "bg-red-500/10 text-red-300 border border-red-500/30";
        case "llm":    return "bg-purple-500/10 text-purple-300 border border-purple-500/30";
        case "warn":   return "bg-amber-500/10 text-amber-300 border border-amber-500/40";
        default:       return "bg-slate-800 text-slate-400 border border-slate-700";
      }
    },

    /* ---------------------------------------------------------- reports */
    get reportList() {
      const colors = {
        analyst:  "#06b6d4",
        bull:     "#22c55e",
        bear:     "#ef4444",
        manager:  "#3b82f6",
        reviewer: "#facc15",
        trader:   "#a855f7",
        risk_a:   "#ef4444",
        risk_n:   "#f59e0b",
        risk_c:   "#3b82f6",
      };
      const KIND_BY_ROLE = {
        "Market Analyst": "analyst",
        "News Analyst": "analyst",
        "Fundamentals Analyst": "analyst",
        "Reports Analyst": "analyst",
        "Macro Analyst": "analyst",
        "Bullish Researcher": "bull",
        "Bearish Researcher": "bear",
        "Research Manager": "manager",
        "Quality Reviewer": "reviewer",
        "Trader": "trader",
        "Aggressive Risk Analyst": "risk_a",
        "Neutral Risk Analyst": "risk_n",
        "Conservative Risk Analyst": "risk_c",
        "Compliance Officer": "reviewer",
        "Portfolio Manager": "manager",
      };
      const DEGRADED_MARKER = "[[DEGRADED_TOOL_CALL_OUTPUT]]";
      // The original (pre-patch) failure mode dumped the Python repr of
      // an OpenAI-style tool-call list directly into the task output.
      // Detect that shape too so saved runs produced before this build
      // still get the amber badge instead of showing raw repr text.
      const LEGACY_REPR_PREFIX = "[ChatCompletionMessageFunctionToolCall";
      const renderOne = (role, round, totalRounds, body, perRoundDegraded) => {
        const flagged = !!this.degradedRoles[role] || !!perRoundDegraded;
        const trimmed = typeof body === "string" ? body.trimStart() : "";
        const markerHit = trimmed.startsWith(DEGRADED_MARKER);
        const legacyHit = trimmed.startsWith(LEGACY_REPR_PREFIX);
        const degraded = flagged || markerHit || legacyHit;
        let cleanBody = body;
        if (markerHit) {
          cleanBody = body.replace(DEGRADED_MARKER, "").trim();
        } else if (legacyHit) {
          // Wrap the raw repr in a fenced code block so marked.js doesn't
          // try to parse the embedded JSON as italics / lists.  The
          // explainer above sets context for the operator.
          cleanBody =
            "**Analyst report missing — degraded LLM output.**\n\n" +
            "The model emitted a tool-call list as its final answer " +
            "instead of a synthesised report. Raw tool-call payload " +
            "captured at the time of the run:\n\n" +
            "```\n" + body.trim() + "\n```";
        }
        const title = totalRounds > 1
          ? `${role} — round ${round}/${totalRounds}`
          : role;
        return {
          role,
          round,
          totalRounds,
          title,
          color: colors[KIND_BY_ROLE[role]] || "#94a3b8",
          html: this.renderMarkdown(cleanBody),
          degraded,
        };
      };

      const byRound = this.reportsByRound || {};
      if (Object.keys(byRound).length) {
        // Preserve the order roles first appeared (matches the workflow
        // pipeline order: analysts → debate rounds → manager → trader …).
        // Within a role, rounds are already in chronological order from
        // the runner.
        const out = [];
        for (const [role, entries] of Object.entries(byRound)) {
          for (const entry of entries) {
            out.push(renderOne(
              role,
              entry.round || 1,
              entry.total_rounds || 1,
              entry.output || "",
              !!entry.degraded,
            ));
          }
        }
        return out;
      }

      // Legacy fall-back: pre-rounds runs and the in-flight prefix
      // before the first node_completed event arrives.
      return Object.entries(this.reports).map(([role, body]) =>
        renderOne(role, 1, 1, body, false)
      );
    },

    renderMarkdown(raw) {
      const cleaned = this.cleanModelOutput(raw);
      if (!window.marked) return `<pre>${this.escape(cleaned)}</pre>`;
      try { window.marked.setOptions({ breaks: true, gfm: true }); } catch (_) {}
      return this.enhanceHtml(window.marked.parse(cleaned));
    },
    // Format a PLAIN-TEXT model field (rationale, bullets, sources) into safe
    // HTML: escape, turn bare URLs into shortened coloured links, and badge the
    // inline `[source: …]` provenance tags so they read distinctly from the
    // reasoning prose.
    formatText(raw) {
      if (raw == null || raw === "") return "";
      const cleaned = this.cleanModelOutput(String(raw));
      const urlRe = /(https?:\/\/[^\s<>()\[\]]+)/g;
      let out = "", last = 0, m;
      while ((m = urlRe.exec(cleaned)) !== null) {
        out += this.escape(cleaned.slice(last, m.index));
        out += this._anchor(m[1]);
        last = m.index + m[1].length;
      }
      out += this.escape(cleaned.slice(last));
      out = this._badgeSourceTags(out);
      return out.replace(/\n/g, "<br>");
    },
    // Post-process rendered markdown: normalise every anchor to a shortened,
    // coloured, new-tab link and badge any `[source: …]` tags marked.js left
    // as text.  Shared by renderMarkdown so tile reports match the decision card.
    enhanceHtml(html) {
      html = html.replace(/<a\b[^>]*?href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, (_m, href, inner) => {
        const plain = inner.replace(/<[^>]+>/g, "").trim();
        const label = (plain === href || /^https?:\/\/\S+$/.test(plain)) ? this.shortenUrl(href) : plain;
        return this._anchor(href, label);
      });
      return this._badgeSourceTags(html);
    },
    _anchor(href, label) {
      const text = this.escape(label != null ? label : this.shortenUrl(href));
      const h = this.escapeAttr(href);
      return `<a href="${h}" target="_blank" rel="noopener" class="src-link" title="${h}">${text}</a>`;
    },
    _badgeSourceTags(html) {
      // Wrap "[source: …]" (may already contain an <a> from linkifying) in a
      // muted badge.  Negative-lookahead on "]" keeps the match on one tag.
      return html.replace(/\[source:((?:(?!\]).)*)\]/gi, (m) => `<span class="src-tag">${m}</span>`);
    },
    // Compact display label for a link — host (minus www) + a "pdf" hint, so a
    // 120-char BSE filing URL reads as "bseindia.com · pdf ↗" while the full URL
    // stays in href + title.  (A local, dependency-free "shortener" — no external
    // service, no API key, fully reversible on hover/click.)
    shortenUrl(url) {
      try {
        const u = new URL(url);
        const host = u.hostname.replace(/^www\./, "");
        const isPdf = /\.pdf(?:$|[?#])/i.test(u.pathname + u.search);
        return host + (isPdf ? " · pdf" : "") + " ↗";
      } catch (_) { return url; }
    },
    // Source-list entry → display label (shorten if it leads with a URL).
    sourceLabel(s) {
      const first = (s || "").split(" ")[0];
      return /^https?:\/\//.test(first) ? this.shortenUrl(first) : s;
    },
    escapeAttr(s) {
      return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    },
    cleanModelOutput(s) {
      if (!s) return "";
      return s
        .replace(/<\|?channel\|?>[\s\S]*?<\|?channel\|?>/g, "")
        .replace(/<\|?channel\|?>/g, "")
        .replace(/<\|?(start|end|return|message|im_start|im_end|assistant|system|user)\|?>/g, "")
        .replace(/^\s*<\|.*?\|?>\s*$/gm, "")
        .trim();
    },
    escape(s) {
      return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    },

    /* ---------------------------------------------------------- M3 memory */
    async loadMemoryRetrieval() {
      try {
        const ticker = (this.form.ticker || "").trim().toUpperCase();
        if (!ticker) return;
        const asOf = this.form.trade_date || new Date().toISOString().slice(0, 10);
        const q = new URLSearchParams({ ticker, as_of: asOf, k: "5" });
        if (this.memoryRegime) q.set("regime", this.memoryRegime);
        const res = await fetch(`/api/memory/retrieve?${q}`).then(r => r.json());
        this.retrievedEpisodes = res.results || [];
        this.memoryMeta = { embedder: res.embedder, regime: res.regime };
      } catch (err) {
        console.error("loadMemoryRetrieval failed", err);
        this.retrievedEpisodes = [];
        this.memoryMeta = null;
      }
    },

    async loadStrategyBoard() {
      try {
        const book = this.assetClass === "commodity" ? "commodity" : "equity";
        const res = await fetch(`/api/strategy?book=${book}`).then(r => r.json());
        this.strategyBoard = res;
        this.strategyDraft = res.strategy_edit
          ? JSON.parse(JSON.stringify(res.strategy_edit))
          : null;
        const prompts = res.prompts || [];
        if (prompts.length && !prompts.some(p => p.key === this.promptKey)) {
          this.promptKey = prompts[0].key;
        }
        this.syncPromptDraft();
        const catalog = (res.knob_catalog || []).filter(k => k.enabled);
        if (catalog.length && !catalog.some(k => k.id === this.experimentKnob)) {
          this.experimentKnob = catalog[0].id;
        }
        this.strategyError = "";
        this.promptError = "";
        this.hermesError = "";
      } catch (err) {
        console.error("loadStrategyBoard failed", err);
        this.strategyBoard = null;
        this.strategyDraft = null;
      }
    },

    syncPromptDraft() {
      const row = (this.strategyBoard?.prompts || []).find(p => p.key === this.promptKey);
      this.promptDraft = {
        goal: row?.goal || "",
        backstory: row?.backstory || "",
      };
    },

    _hermesBook() {
      return this.assetClass === "commodity" ? "commodity" : "equity";
    },

    _settingsUrl() {
      return this._hermesBook() === "commodity" ? "/settings?book=commodity" : "/settings";
    },

    _dashboardUrl() {
      return this.assetClass === "commodity" ? "/commodity" : "/stocks";
    },

    _apiDetail(body) {
      if (!body) return "";
      if (typeof body.detail === "string") return body.detail;
      if (body.detail != null) return JSON.stringify(body.detail);
      return "";
    },

    async setAssetClass(next) {
      if (this.assetClass === next) return;
      this.assetClass = next;
      await this.loadOptions();
      this._initBulkCfg();
      if (this.settingsPage) {
        try { window.history.pushState({}, "", this._settingsUrl()); } catch (_) {}
        await this.loadSettingsPage();
      } else {
        this._hydrateBulkFromServer();
      }
    },

    openSettingsPage() {
      this.sidebarMode = "settings";
      if (this.settingsPage) return;
      this.settingsPage = true;
      this.focusPage = false;
      try { window.history.pushState({}, "", this._settingsUrl()); } catch (_) {}
      this.loadSettingsPage();
    },

    leaveSettingsPage() {
      window.location.assign(this._dashboardUrl());
    },

    goToHomepage() {
      window.location.assign("/");
    },

    commodityDeskRuns() {
      const allowed = new Set(
        (this.tickers || []).map((t) => String(t.symbol || "").toUpperCase()).filter(Boolean),
      );
      return (this.recentRuns || []).filter((r) => allowed.has(String(r.ticker || "").toUpperCase()));
    },

    _applyBulkToForm(c) {
      if (!c || typeof c !== "object") return;
      if ("debate_rounds" in c) this.form.max_debate_rounds = c.debate_rounds;
      if ("risk_rounds" in c) this.form.max_risk_rounds = c.risk_rounds;
      if ("run_mode" in c) this.form.run_mode = c.run_mode;
      if ("quality_review" in c) this.form.quality_review = c.quality_review;
      if ("compliance" in c) this.form.compliance = c.compliance;
      if ("critic_enabled" in c) this.form.critic_enabled = c.critic_enabled;
      if ("fusion_enabled" in c) this.form.fusion_enabled = c.fusion_enabled;
      if ("fusion_include_crowd" in c) this.form.fusion_include_crowd = c.fusion_include_crowd;
      if ("tools_enabled" in c) this.form.tools_enabled = c.tools_enabled;
      if ("book" in c) this.form.book = c.book;
      if ("llm_preset" in c && c.llm_preset) this.selectedLlmPreset = c.llm_preset;
    },

    _applySettingsPayload(body) {
      this.strategyDraft = body.strategy ? JSON.parse(JSON.stringify(body.strategy)) : null;
      this.settingsPromptBlock = body.prompt_block || "";
      this.settingsSource = body.strategy_source || "";
      this.settingsLivePath = body.live_path || "";
      this.settingsPackagedPath = body.packaged_path || "";
      this.strategyBoard = {
        ...(this.strategyBoard || {}),
        prompts: body.prompts || [],
        prompt_block: body.prompt_block,
        strategy_source: body.strategy_source,
        knob_catalog: body.knob_catalog || [],
      };
      if ((body.prompts || []).length && !body.prompts.some(p => p.key === this.promptKey)) {
        this.promptKey = body.prompts[0].key;
      }
      this.syncPromptDraft();
      if (body.bulk && typeof body.bulk === "object") {
        this.bulkCfg = body.bulk;
        this.bulkCfgSaved = true;
        this._applyBulkToForm(body.bulk);
        try { localStorage.setItem(this._bulkCfgStorageKey(), JSON.stringify(body.bulk)); } catch (_) {}
      }
    },

    async loadSettingsPage() {
      this.settingsError = "";
      try {
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`);
        const body = await res.json();
        if (!res.ok) throw new Error(this._apiDetail(body) || res.statusText);
        this._applySettingsPayload(body);
      } catch (err) {
        this.settingsError = String(err.message || err);
        this.strategyDraft = null;
      }
    },

    async _hydrateBulkFromServer() {
      try {
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`);
        const body = await res.json();
        if (!res.ok) return;
        if (body.bulk && typeof body.bulk === "object") {
          this.bulkCfg = body.bulk;
          this.bulkCfgSaved = true;
          this._applyBulkToForm(body.bulk);
          try { localStorage.setItem(this._bulkCfgStorageKey(), JSON.stringify(body.bulk)); } catch (_) {}
        }
      } catch (_) { /* keep local snapshot */ }
    },

    async saveAllSettings() {
      if (!this.strategyDraft) return;
      this.settingsSaving = true;
      this.settingsError = "";
      try {
        const payload = {
          strategy: this.strategyDraft,
          bulk: this._bulkCfgFromForm(),
        };
        if ((this.promptDraft.goal || "").trim() && (this.promptDraft.backstory || "").trim()) {
          payload.prompts = [{
            key: this.promptKey,
            goal: this.promptDraft.goal,
            backstory: this.promptDraft.backstory,
          }];
        }
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(this._apiDetail(body) || res.statusText);
        this._applySettingsPayload(body);
        this.bulkCfg = body.bulk || payload.bulk;
        this.bulkCfgSaved = true;
        try { localStorage.setItem(this._bulkCfgStorageKey(), JSON.stringify(this.bulkCfg)); } catch (_) {}
      } catch (err) {
        this.settingsError = String(err.message || err);
      } finally {
        this.settingsSaving = false;
      }
    },

    async revertStrategyFromSettings() {
      if (!window.confirm("Revert to the packaged desk_strategy.yaml? Live edits will be deleted.")) return;
      this.settingsSaving = true;
      this.settingsError = "";
      try {
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ revert_strategy: true }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(this._apiDetail(body) || res.statusText);
        this._applySettingsPayload(body);
      } catch (err) {
        this.settingsError = String(err.message || err);
      } finally {
        this.settingsSaving = false;
      }
    },

    async unsetPromptFromSettings() {
      this.settingsSaving = true;
      this.settingsError = "";
      try {
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ unset_prompts: [this.promptKey] }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(this._apiDetail(body) || res.statusText);
        this._applySettingsPayload(body);
      } catch (err) {
        this.settingsError = String(err.message || err);
      } finally {
        this.settingsSaving = false;
      }
    },

    async saveStrategyDraft() {
      if (!this.strategyDraft) return;
      this.strategySaving = true;
      this.strategyError = "";
      try {
        const res = await fetch(`/api/strategy?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(this.strategyDraft),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        this.strategyBoard = body;
        this.strategyDraft = body.strategy_edit
          ? JSON.parse(JSON.stringify(body.strategy_edit))
          : this.strategyDraft;
      } catch (err) {
        this.strategyError = String(err.message || err);
      } finally {
        this.strategySaving = false;
      }
    },

    async revertStrategyDraft() {
      if (!window.confirm("Revert to the packaged desk_strategy.yaml? Live edits will be deleted.")) return;
      this.strategySaving = true;
      this.strategyError = "";
      try {
        const res = await fetch(`/api/strategy/revert?book=${this._hermesBook()}`, { method: "POST" });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        this.strategyBoard = body;
        this.strategyDraft = body.strategy_edit
          ? JSON.parse(JSON.stringify(body.strategy_edit))
          : null;
      } catch (err) {
        this.strategyError = String(err.message || err);
      } finally {
        this.strategySaving = false;
      }
    },

    async savePromptDraft() {
      this.promptSaving = true;
      this.promptError = "";
      try {
        const res = await fetch(`/api/strategy/prompts?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            key: this.promptKey,
            goal: this.promptDraft.goal,
            backstory: this.promptDraft.backstory,
          }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        await this.loadStrategyBoard();
      } catch (err) {
        this.promptError = String(err.message || err);
      } finally {
        this.promptSaving = false;
      }
    },

    async revertPromptDraft() {
      this.promptSaving = true;
      this.promptError = "";
      try {
        const res = await fetch(
          `/api/strategy/prompts/${encodeURIComponent(this.promptKey)}?book=${this._hermesBook()}`,
          { method: "DELETE" },
        );
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        await this.loadStrategyBoard();
      } catch (err) {
        this.promptError = String(err.message || err);
      } finally {
        this.promptSaving = false;
      }
    },

    async hermesExperiment(action) {
      this.hermesBusy = true;
      this.hermesError = "";
      try {
        const payload = { action };
        if (action === "open") payload.knob = this.experimentKnob;
        const res = await fetch(`/api/strategy/experiment?book=${this._hermesBook()}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || res.statusText);
        this.strategyBoard = body;
        if (body.strategy_edit) {
          this.strategyDraft = JSON.parse(JSON.stringify(body.strategy_edit));
        }
      } catch (err) {
        this.hermesError = String(err.message || err);
      } finally {
        this.hermesBusy = false;
      }
    },

    addStrategyRule(section) {
      if (!this.strategyDraft?.rules?.[section]) return;
      this.strategyDraft.rules[section].push("");
    },

    removeStrategyRule(section, idx) {
      if (!this.strategyDraft?.rules?.[section]) return;
      this.strategyDraft.rules[section].splice(idx, 1);
    },

    /* ------------------------------------------- L2: conviction calibration */
    async loadCalibration() {
      try {
        const q = new URLSearchParams({
          basis: this.calibrationBasis || "return",
          cohort: this.calibrationCohort || "current",
        });
        if (this.calibrationTickerOnly) {
          const ticker = (this.form.ticker || "").trim().toUpperCase();
          if (ticker) q.set("ticker", ticker);
        }
        const res = await fetch(`/api/calibration?${q}`).then(r => r.json());
        this.calibration = res;
      } catch (err) {
        console.error("loadCalibration failed", err);
        this.calibration = null;
      }
    },

    /* ----------------------------------------- Phase 2A.3 prune memory */
    async pruneMemory() {
      // Destructive — confirm() prompts the user before hitting the
      // server.  Defaults match the API defaults (10k cap, 365 day
      // window, zero retrieval-count threshold).
      if (!window.confirm(
        "Prune episodes older than 365 days (kept if ever retrieved) and cap to 10,000 records?"
      )) return;
      this.pruneLoading = true;
      try {
        const res = await fetch("/api/memory/evict", { method: "POST" }).then(r => r.json());
        this.pruneResult = res;
        await this.loadMemoryRetrieval();
      } catch (err) {
        console.error("pruneMemory failed", err);
        this.pruneResult = { removed: 0, remaining: null, error: String(err) };
      } finally {
        this.pruneLoading = false;
      }
    },

    /* ------------------------------------------------- L2: outcome resolver */
    async resolveOutcomes(skipLlm = false) {
      // Triggers the server's reflection sweep. We scope to the active ticker
      // by default so the user sees only the relevant episodes — passing no
      // ticker would resolve the entire book, which is a different operation.
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      this.resolveLoading = true;
      this.resolveResult = null;
      try {
        const q = new URLSearchParams();
        if (ticker) q.set("ticker", ticker);
        if (skipLlm) q.set("skip_llm", "true");
        const res = await fetch(`/api/memory/resolve?${q}`, { method: "POST" }).then(r => r.json());
        this.resolveResult = res;
        // Refresh retrieval so newly-resolved episodes immediately appear
        // in the top-k list with their fresh reflections.
        await this.loadMemoryRetrieval();
        // Resolving is exactly what feeds calibration, so refresh it too.
        await this.loadCalibration();
        await this.loadStrategyBoard();
      } catch (err) {
        console.error("resolveOutcomes failed", err);
        this.resolveResult = { error: String(err), count: 0, by_status: {}, records: [] };
      } finally {
        this.resolveLoading = false;
      }
    },

    /* ---------------------------------------------------------- Phase E — saved runs */
    async loadRecentRuns() {
      try {
        const res = await fetch("/api/runs/recent?limit=30").then(r => r.json());
        this.recentRuns = res.runs || [];
      } catch (err) {
        console.error("loadRecentRuns failed", err);
        this.recentRuns = [];
      }
    },
    async clearRecentRuns() {
      // Confirm before nuking — this endpoint moves the runs/ tree into
      // a timestamped trash folder under $TRADINGCREW_CACHE_DIR so the
      // user *can* recover, but the UI button feels destructive so an
      // explicit confirmation is friendlier.  The empty-state copy
      // ("No saved runs yet.") doubles as the success indicator once
      // the panel re-renders with recentRuns=[].
      const count = this.recentRuns?.length || 0;
      if (count === 0) return;
      const ok = window.confirm(
        `Clear all ${count} saved run${count === 1 ? "" : "s"}?\n\n` +
        "Records are moved into a timestamped backup folder on disk " +
        "(runs.trash-<ts>/) so they can be recovered manually if needed."
      );
      if (!ok) return;
      this.clearingRecentRuns = true;
      try {
        const res = await fetch("/api/runs/recent", { method: "DELETE" }).then(r => r.json());
        const removed = (res && res.removed) || 0;
        this.recentRuns = [];
        this.loadedRunId = null;
        console.info(`Cleared ${removed} saved runs (trash: ${res?.trash_path || "n/a"})`);
      } catch (err) {
        console.error("clearRecentRuns failed", err);
        window.alert("Failed to clear recent runs — see browser console for details.");
      } finally {
        this.clearingRecentRuns = false;
      }
    },
    // Repaint the Workflow diagram + reports from the most recent stored
    // run for ``ticker`` (used when a prediction tile is opened for a
    // name that isn't currently live-streaming).  Silently no-ops when
    // the ticker has no persisted run yet, leaving a clean empty diagram.
    async loadLatestRun(ticker) {
      try {
        if (this._hydrateBlockedByLive(ticker)) return;
        const res = await fetch(`/api/runs/${encodeURIComponent(ticker)}/latest`);
        if (!res.ok) return;
        const record = await res.json();
        if (!record || record.error || !record.run_id) return;
        // Ignore if the user has since switched to a different name.
        if (this.form.ticker && this.form.ticker.toUpperCase() !== ticker.toUpperCase()) return;
        if (this._hydrateBlockedByLive(ticker)) return;
        await this.loadRun(ticker, record.run_id);
      } catch (_) {
        /* no stored run is fine */
      }
    },
    async loadRun(ticker, runId) {
      try {
        if (this._hydrateBlockedByLive(ticker)) return;
        const res = await fetch(`/api/runs/${encodeURIComponent(ticker)}/${encodeURIComponent(runId)}`).then(r => r.json());
        if (res.error) {
          console.warn("loadRun error:", res.error);
          return;
        }
        if (this._hydrateBlockedByLive(ticker)) return;
        this.loadedRunId = runId;
        // Lock the header + Charts (and every other ticker-scoped tab:
        // Memory, Backtest, Portfolio, Futures) to the run being viewed.
        // loadRun repaints the report panels from disk but historically
        // left ``form.ticker`` at its default (NTNX), so the title and the
        // Charts tab showed the wrong symbol for a historical run.  The
        // form.ticker / form.trade_date $watchers reload the chart; the
        // other tabs read form.ticker at click time.
        if (ticker) this.form.ticker = String(ticker).toUpperCase();
        const runDate = res.config?.trade_date || res.config?.decision_ts;
        if (runDate) {
          this.form.trade_date = String(runDate).slice(0, 10);
        } else if (res.started_at) {
          this.form.trade_date = String(res.started_at).slice(0, 10);
        }
        // Repaint all the post-LLM panels from the saved record.
        this.cascadeStatus = res.cascade_status || null;
        this.actionProposal = res.action_proposal || null;
        this.actionProposalMd = res.action_proposal_markdown || "";
        this.executionResult = res.execution_result || null;
        this.episodeMeta = res.episode_meta || null;
        this.reflectionRecords = res.reflection_records || null;
        this.decisionPayload = res.final_decision || null;
        this.finalDecision = res.final_decision?.action || "";
        this.fusionResult = null;
        this.reportSignals = res.report_signals || null;
        this.priceSignals = res.price_signals || null;
        this.deskBiasBoard = res.desk_bias_board || null;
        this.evidenceBlackboard = res.evidence_blackboard || null;
        this.debateScratchpad = res.debate_scratchpad || null;
        // Reflect the persisted terminal status so the header/badges match
        // reality post-completion (not the default "running").
        this.status = res.status || "completed";
        this.isRunning = false;
        this.currentNode = "";
        this.applyRunClock(res.started_at, res.completed_at);
        // Repaint the per-agent reports (rendered inside the Workflow tab).
        this.reports = { ...(res.reports || {}) };
        // ``reports_by_round`` is the new per-round source of truth for
        // the Reports tab.  Old run records pre-date the field, so fall
        // back to wrapping the legacy single-round ``reports`` dict into
        // one synthetic round so the UI keeps rendering them.
        const rawByRound = res.reports_by_round || null;
        if (rawByRound && Object.keys(rawByRound).length) {
          this.reportsByRound = JSON.parse(JSON.stringify(rawByRound));
        } else {
          this.reportsByRound = Object.fromEntries(
            Object.entries(res.reports || {}).map(([role, body]) => [
              role,
              [{ round: 1, total_rounds: 1, output: body, degraded: false }],
            ])
          );
        }
        // Rebuild the per-node activity map from the persisted per-round
        // reports so clicking an agent tile shows THAT agent's output after
        // completion — the same tile-specific view the live run shows (tool
        // calls / intermediate steps aren't persisted, so post-run the tile
        // shows the agent's final report(s), which is the output users want).
        const rebuilt = {};
        for (const [role, rounds] of Object.entries(this.reportsByRound)) {
          rebuilt[role] = (rounds || []).map((r, i) => {
            const rl = (r.total_rounds > 1) ? ` · round ${r.round}/${r.total_rounds}` : "";
            const chars = (r.output || "").length;
            return {
              key: `${role}-report-${i}`,
              label: r.degraded ? `report (degraded)${rl}` : `report${rl}`,
              color: r.degraded ? "warn" : "llm",
              subtitle: chars ? `${chars} chars` : (r.has_body ? "loading…" : "0 chars"),
              body: r.output || "",
              has_body: !!r.has_body,
              markdown: true,
            };
          });
        }
        this.nodeActivities = rebuilt;
        this.selectedNode = null;
        // Rehydrate the degraded-roles set so the badges and the diagram
        // colouring match the persisted state of the run (the saved record
        // carries one entry per analyst whose final answer was substituted
        // with the OSS-LLM placeholder).
        this.degradedRoles = Object.fromEntries(
          (res.degraded_roles || []).map((role) => [role, true])
        );
        if (res.config?.run_mode) this.form.run_mode = res.config.run_mode;
        this.mountWorkflowDiagram();
        if (this.diagram && typeof this.diagram.setGraphMeta === "function") {
          this.diagram.setGraphMeta({
            playbookAction: res.config?.playbook_action,
            skipReports: res.config?.skip_reports,
            runMode: res.config?.run_mode,
            orchestrator: res.config?.orchestrator,
            qualityReview: res.config?.quality_review,
            compliance: res.config?.compliance,
          });
        }
        if (this.diagram && res.expected_role_order) {
          res.expected_role_order.forEach(role => {
            try {
              this.diagram.setState(
                role, this.degradedRoles[role] ? "degraded" : "done",
              );
            } catch (_) {}
          });
        }
        this.activeTab = "workflow";
        // The diagram may have collapsed (scale=0) earlier if a fit() ran
        // while the workflow tab was hidden. Force a re-fit after Alpine
        // restores display:block on the workflow section so the saved run
        // actually renders. Without this, "Recent runs" picks the right
        // state under the hood but the user sees an empty diagram area.
        if (this.diagram) {
          this.$nextTick(() => this.diagram.fit());
        }
      } catch (err) {
        console.error("loadRun failed", err);
      }
    },
    _roleNeedsHydrate(role) {
      const rounds = (this.reportsByRound || {})[role] || [];
      return rounds.some((r) => r && r.has_body && !r.output);
    },
    _applyRoleReports(role, rounds) {
      if (!role || !Array.isArray(rounds) || !rounds.length) return;
      this.reportsByRound = { ...this.reportsByRound, [role]: rounds };
      const last = rounds[rounds.length - 1] || {};
      this.reports = { ...this.reports, [role]: last.output || "" };
      this.nodeActivities = {
        ...this.nodeActivities,
        [role]: rounds.map((r, i) => {
          const rl = (r.total_rounds > 1) ? ` · round ${r.round}/${r.total_rounds}` : "";
          return {
            key: `${role}-report-${i}`,
            label: r.degraded ? `report (degraded)${rl}` : `report${rl}`,
            color: r.degraded ? "warn" : "llm",
            subtitle: `${(r.output || "").length} chars`,
            body: r.output || "",
            markdown: true,
          };
        }),
      };
    },
    async hydrateRoleReports(role) {
      const ticker = this.form.ticker;
      const runId = this.loadedRunId;
      if (!ticker || !runId || !role || role === "decision") return;
      if (!this._roleNeedsHydrate(role)) return;
      this._hydratingRoles = this._hydratingRoles || {};
      if (this._hydratingRoles[role]) return;
      this._hydratingRoles[role] = true;
      try {
        const url = `/api/runs/${encodeURIComponent(ticker)}/${encodeURIComponent(runId)}/reports?role=${encodeURIComponent(role)}`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        const rounds = (data.reports_by_round && data.reports_by_round[role]) || [];
        this._applyRoleReports(role, rounds);
      } catch (err) {
        console.error("hydrateRoleReports failed", err);
      } finally {
        delete this._hydratingRoles[role];
      }
    },
    async hydrateAllReports() {
      const ticker = this.form.ticker;
      const runId = this.loadedRunId;
      if (!ticker || !runId) return;
      const needed = Object.keys(this.reportsByRound || {}).filter((role) => this._roleNeedsHydrate(role));
      if (!needed.length) return;
      this._hydratingRoles = this._hydratingRoles || {};
      if (this._hydratingRoles["*"]) return;
      this._hydratingRoles["*"] = true;
      try {
        const url = `/api/runs/${encodeURIComponent(ticker)}/${encodeURIComponent(runId)}/reports`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        for (const [role, rounds] of Object.entries(data.reports_by_round || {})) {
          this._applyRoleReports(role, rounds || []);
        }
      } catch (err) {
        console.error("hydrateAllReports failed", err);
      } finally {
        delete this._hydratingRoles["*"];
      }
    },

    /* ---------------------------------------------------------- M6 backtest */
    get backtestRequiredProposals() {
      const c = this.backtestConfig || {};
      return (c.train_size || 0) + (c.embargo_size || 0) + (c.test_size || 0);
    },
    get backtestEnoughData() {
      if (this.backtestProposalCount === null) return null;  // unknown
      return this.backtestProposalCount >= this.backtestRequiredProposals;
    },
    async checkBacktestEligibility() {
      // Phase E reuse — counts how many runs (and therefore proposals) exist
      // for the active ticker, so we can tell the user up front whether
      // a walk-forward backtest is even feasible.
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) { this.backtestProposalCount = 0; return; }
      try {
        const res = await fetch(`/api/runs/recent?limit=200&ticker=${encodeURIComponent(ticker)}`).then(r => r.json());
        this.backtestProposalCount = (res.runs || []).length;
      } catch (_) {
        this.backtestProposalCount = null;
      }
    },
    /* ===================== COMMODITY FUTURES TAB ============================ */
    async loadCommodityFutures() {
      // Fetch all three panels in parallel since they're independent
      // and the user always wants all three visible together.
      await Promise.all([
        this.loadCommodityCurve(),
        this.loadCommodityCOT(),
        this.loadCommoditySeasonality(),
      ]);
    },
    async loadCommodityCurve() {
      const ticker = (this.form.ticker || "").trim();
      if (!ticker) return;
      this.commodityLoading.curve = true;
      try {
        const res = await fetch(`/api/commodity/curve?ticker=${encodeURIComponent(ticker)}&n_months=12`).then(r => r.json());
        this.curveData = res;
        this.$nextTick(() => this.renderCurveChart());
      } catch (err) {
        console.error("loadCommodityCurve failed", err);
        this.curveData = { error: String(err) };
      } finally {
        this.commodityLoading.curve = false;
      }
    },
    renderCurveChart() {
      if (!window.Chart || !this.curveData?.contracts?.length) return;
      const ctx = document.getElementById("futures-curve-chart");
      if (!ctx) return;
      if (this.curveChart) this.curveChart.destroy();
      const labels = this.curveData.contracts.map(c => c.symbol.replace(/\.[A-Z]+$/, ""));
      const data = this.curveData.contracts.map(c => c.close);
      this.curveChart = new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [{
            label: this.curveData.name || "Curve",
            data,
            borderColor: "#fbbf24",  // amber
            backgroundColor: "rgba(251, 191, 36, 0.1)",
            tension: 0.2,
            pointRadius: 4,
            pointHoverRadius: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10, family: "monospace" } } },
            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          },
        },
      });
    },
    async loadCommodityCOT() {
      const ticker = (this.form.ticker || "").trim();
      if (!ticker) return;
      this.commodityLoading.cot = true;
      try {
        const res = await fetch(`/api/commodity/cot?ticker=${encodeURIComponent(ticker)}&weeks=24`).then(r => r.json());
        this.cotData = res;
        this.$nextTick(() => this.renderCOTChart());
      } catch (err) {
        console.error("loadCommodityCOT failed", err);
        this.cotData = { error: String(err) };
      } finally {
        this.commodityLoading.cot = false;
      }
    },
    renderCOTChart() {
      if (!window.Chart || !this.cotData?.rows?.length) return;
      const ctx = document.getElementById("cot-chart");
      if (!ctx) return;
      if (this.cotChart) this.cotChart.destroy();
      const labels = this.cotData.rows.map(r => r.date);
      this.cotChart = new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Managed Money net",
              data: this.cotData.rows.map(r => r.managed_money_net),
              borderColor: "#fbbf24",
              backgroundColor: "rgba(251, 191, 36, 0.08)",
              tension: 0.15,
              pointRadius: 2,
              fill: true,
            },
            {
              label: "Commercials net",
              data: this.cotData.rows.map(r => r.commercials_net),
              borderColor: "#60a5fa",
              backgroundColor: "rgba(96, 165, 250, 0.08)",
              tension: 0.15,
              pointRadius: 2,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "index" },
          plugins: { legend: { labels: { color: "#94a3b8", font: { size: 11 } } } },
          scales: {
            x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10, family: "monospace" } } },
            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10, family: "monospace" }, callback: (v) => v.toLocaleString() } },
          },
        },
      });
    },
    async loadCommoditySeasonality() {
      const ticker = (this.form.ticker || "").trim();
      if (!ticker) return;
      this.commodityLoading.seasonality = true;
      try {
        const res = await fetch(`/api/commodity/seasonality?ticker=${encodeURIComponent(ticker)}&years=5`).then(r => r.json());
        this.seasonalityData = res;
        this.$nextTick(() => this.renderSeasonalityChart());
      } catch (err) {
        console.error("loadCommoditySeasonality failed", err);
        this.seasonalityData = { error: String(err) };
      } finally {
        this.commodityLoading.seasonality = false;
      }
    },
    renderSeasonalityChart() {
      if (!window.Chart || !this.seasonalityData?.months?.length) return;
      const ctx = document.getElementById("seasonality-chart");
      if (!ctx) return;
      if (this.seasonalityChart) this.seasonalityChart.destroy();
      const labels = this.seasonalityData.months.map(m => m.label);
      const data = this.seasonalityData.months.map(m => m.mean_return_pct);
      this.seasonalityChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Avg monthly return %",
            data,
            backgroundColor: data.map(v => v >= 0 ? "rgba(16, 185, 129, 0.6)" : "rgba(244, 63, 94, 0.6)"),
            borderColor: data.map(v => v >= 0 ? "rgb(16, 185, 129)" : "rgb(244, 63, 94)"),
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
            y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", font: { size: 10 }, callback: (v) => v + "%" } },
          },
        },
      });
    },

    /* ----------------------------------- L3 agentic training: grid search */
    async runGridSearch() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      this.gridSearchLoading = true;
      this.gridSearchResult = null;
      try {
        const q = new URLSearchParams({
          ticker,
          grid_size: this.gridSearchSize,
          rank_by: this.gridSearchRankBy,
          cost_model: this.backtestCostModel || "standard",
          sweep_max_leverage: String(this.gridSweepMaxLeverage),
          sweep_drawdown_kill: String(this.gridSweepDrawdownKill),
          sweep_cost_model: String(this.gridSweepCostModel),
          validation: this.backtestValidation || "walk_forward",
          cpcv_groups: String(this.cpcvGroups || 6),
          cpcv_k: String(this.cpcvK || 2),
          optimizer: this.backtestOptimizer || "grid",
          swarm_size: String(this.swarmSize || 12),
          swarm_iterations: String(this.swarmIterations || 8),
          swarm_seed: String(this.swarmSeed || 0),
          consensus: String(this.gridConsensus),
          consensus_top_k: String(this.consensusTopK || 5),
          train_size: String(this.backtestConfig.train_size),
          embargo_size: String(this.backtestConfig.embargo_size),
          test_size: String(this.backtestConfig.test_size),
          lookback_years: String(this.backtestLookbackYears || 5),
          apply: String(this.applyTune),
        });
        const res = await fetch(`/api/grid_search?${q}`, { method: "POST" }).then(r => r.json());
        this.gridSearchResult = res;
        this.$nextTick(() => this.renderSwarmHistoryChart());
      } catch (err) {
        console.error("runGridSearch failed", err);
        this.gridSearchResult = { error: String(err) };
      } finally {
        this.gridSearchLoading = false;
      }
    },

    /* ----------------------- Phase 2C — synthetic proposal seeder */
    async seedSyntheticProposals() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      this.seedLoading = true;
      this.seedResult = null;
      try {
        const q = new URLSearchParams({
          ticker,
          n: String(this.seedN || 20),
        });
        const res = await fetch(`/api/backtest/seed?${q}`, { method: "POST" }).then(r => r.json());
        this.seedResult = res;
      } catch (err) {
        console.error("seedSyntheticProposals failed", err);
        this.seedResult = { error: String(err) };
      } finally {
        this.seedLoading = false;
      }
    },

    /* ------------------------- L4 agentic training: PPO on M2 simulator */

    // Phase 2F — flip the horizon-mode radio onto the train/eval windows
    // so short_term / balanced / long_term users don't have to know that
    // 252 ≈ 1y of trading days.  ``custom`` is a no-op so the user can
    // type the windows by hand and still see "Custom" selected.
    applyRlHorizonMode() {
      const presets = {
        short_term: { train: 252,  eval: 30  },
        balanced:   { train: 750,  eval: 60  },
        long_term:  { train: 1260, eval: 252 },
      };
      const p = presets[this.rlConfig.horizon_mode];
      if (!p) return;
      this.rlConfig.train_window_days = p.train;
      this.rlConfig.eval_window_days = p.eval;
    },

    // Kicks off a background training run.  The backend returns the
    // record immediately and we start polling /status until it
    // transitions to completed / failed / stopped.
    async startRlTraining() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) { this.rlError = "Pick a ticker first."; return; }
      this.rlError = "";
      this.rlLoading.start = true;
      this.rlMetrics = [];
      this.rlStatus = null;
      try {
        const universe = (this.rlConfig.policy_universe || "")
          .split(",")
          .map(s => s.trim().toUpperCase())
          .filter(Boolean);
        const payload = {
          ticker,
          asset_class: this.assetClass,
          algorithm: this.rlConfig.algorithm || "ppo",
          train_window_days: this.rlConfig.train_window_days,
          eval_window_days: this.rlConfig.eval_window_days,
          horizon_mode: this.rlConfig.horizon_mode || "balanced",
          policy_universe: universe,
          ppo_config: {
            total_steps: this.rlConfig.total_steps,
            steps_per_rollout: this.rlConfig.steps_per_rollout,
            learning_rate: this.rlConfig.learning_rate,
            entropy_coef: this.rlConfig.entropy_coef,
            risk_aversion: this.rlConfig.risk_aversion,
            cvar_alpha: this.rlConfig.cvar_alpha,
          },
          env_config: {
            cost_model_name: this.rlConfig.cost_model_name,
            drawdown_kill_pct: this.rlConfig.drawdown_kill_pct,
            turnover_penalty_bps: this.rlConfig.turnover_penalty_bps,
            drawdown_penalty_coef: this.rlConfig.drawdown_penalty_coef,
            volatility_curriculum: !!this.rlConfig.volatility_curriculum,
          },
        };
        const res = await fetch("/api/training/rl/start", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(detail.detail || res.statusText);
        }
        this.rlStatus = await res.json();
        this._startRlPolling();
        await this.loadRlRuns();
      } catch (err) {
        console.error("startRlTraining failed", err);
        this.rlError = String(err.message || err);
      } finally {
        this.rlLoading.start = false;
      }
    },

    async stopRlTraining() {
      if (!this.rlStatus || !this.rlStatus.run_id) return;
      this.rlLoading.stop = true;
      try {
        await fetch("/api/training/rl/stop", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ run_id: this.rlStatus.run_id }),
        });
        await this.pollRlStatus();
      } catch (err) {
        console.error("stopRlTraining failed", err);
        this.rlError = String(err.message || err);
      } finally {
        this.rlLoading.stop = false;
      }
    },

    _startRlPolling() {
      if (this.rlPollTimer) clearInterval(this.rlPollTimer);
      this.rlPollTimer = setInterval(async () => {
        await this.pollRlStatus();
        const stillRunning = this.rlStatus && this.rlStatus.active;
        if (!stillRunning) {
          clearInterval(this.rlPollTimer);
          this.rlPollTimer = null;
          // Refresh the leaderboard so the just-completed run lands at the top.
          await this.loadRlRuns();
        }
      }, 2000);
    },

    /* ------------------------- Forward collection (scale) job controls */
    async initScale() {
      await this.refreshScaleStatus();
      this._startScalePolling();
    },
    async refreshScaleStatus() {
      try {
        const prevActive = this.scaleActive;
        this.scaleStatus = await fetch("/api/scale/status").then(r => r.json());
        // Default the universe select to a shipped one if unset.
        if (!this.scaleForm.universe && this.scaleStatus?.universes?.length) {
          this.scaleForm.universe = this.scaleStatus.universes[0].name;
        }
        // While a batch runs, keep the Prediction tiles' status fresh so they
        // flip to green as decisions land; refresh once more when it finishes.
        if (!this.predHistDate && (this.scaleActive || prevActive)) {
          this.loadPredUniverse();
          if (prevActive && !this.scaleActive) this.loadPredHistory();
        }
        // Refresh the per-backend search counters for this batch's session.
        this.loadSearchStats(this.scaleStatus?.job_id);
      } catch (err) {
        console.error("refreshScaleStatus failed", err);
      }
    },
    // Fetch per-backend web-search counters for a session (batch job_id or a
    // run_id).  Group-by-session view so backend health is visible in the UI
    // without grepping logs.
    async loadSearchStats(traceId) {
      if (!traceId) { this.searchStats = null; return; }
      try {
        const r = await fetch(`/api/search_stats/${encodeURIComponent(traceId)}`);
        if (!r.ok) return;
        this.searchStats = await r.json();
      } catch (err) {
        console.error("loadSearchStats failed", err);
      }
    },
    get searchBackendRows() {
      // Nest engine breakdown rows (e.g. "searxng/bing") under their parent
      // backend ("searxng") so the panel *expands* SearXNG into its engines.
      const b = (this.searchStats && this.searchStats.backends) || {};
      let all = Object.entries(b).map(([name, v]) => ({ name, ...v }));
      // Circular-ring mode records each engine as its own canonical top-level
      // node ("searxng:bing"). The legacy fan-out breakdown ("searxng/bing"
      // under a "searxng" parent) is then a redundant duplicate, so drop the
      // slash rows AND the empty "searxng" parent whenever ring nodes exist.
      if (all.some((e) => e.name.startsWith("searxng:"))) {
        all = all.filter((e) => !e.name.includes("/") && e.name !== "searxng");
      }
      const tops = all.filter((e) => !e.name.includes("/"))
                      .sort((a, c) => (c.total || 0) - (a.total || 0));
      const subs = {};
      all.filter((e) => e.name.includes("/")).forEach((e) => {
        const parent = e.name.split("/")[0];
        (subs[parent] = subs[parent] || []).push({
          ...e, label: e.name.split("/").slice(1).join("/"), isSub: true,
        });
      });
      Object.values(subs).forEach((l) => l.sort((a, c) => (c.total || 0) - (a.total || 0)));
      const out = [];
      const seen = new Set();
      tops.forEach((t) => {
        out.push({ ...t, label: t.name, isSub: false });
        seen.add(t.name);
        (subs[t.name] || []).forEach((s) => out.push(s));
      });
      // Orphan breakdowns whose parent never logged a top-level row.
      Object.keys(subs).forEach((p) => {
        if (seen.has(p)) return;
        out.push({ name: p, label: p, isSub: false, success: 0, empty: 0, throttled: 0, error: 0, total: 0 });
        subs[p].forEach((s) => out.push(s));
      });
      return out;
    },

    /* ==================================================================== */
    /* Prediction tab: universe grid, batch controls, search-add, history   */
    /* ==================================================================== */
    async loadPredUniverse() {
      try {
        // While a batch is active, its predictions persist under the batch's
        // own `day` (scaleStatus.day), which may differ from "today" if the
        // batch straddles midnight.  Query that day so the completed count
        // tracks the current run instead of today's (possibly empty) row.
        const day = this.scaleActive ? (this.scaleStatus && this.scaleStatus.day) : "";
        const qs = day ? `?run_date=${encodeURIComponent(day)}` : "";
        const r = await fetch(`/api/predict/universe${qs}`);
        if (!r.ok) return;
        const d = await r.json();
        this.predUniverse = { markets: { IN: [], US: [], ...(d.markets || {}) } };
        this.predProgress = d.progress || { total: 0, done: 0, failed: 0 };
        this.predRunDate = d.run_date || "";
      } catch (err) {
        console.error("loadPredUniverse failed", err);
      }
    },
    predToggle(id) { this.predExpanded[id] = !this.predExpanded[id]; },
    get predMarkets() { return this.predMarket === "all" ? ["IN", "US"] : [this.predMarket]; },
    get predProgressPct() {
      const t = this.predProgress.total || 0;
      return t ? Math.round((100 * (this.predProgress.done || 0)) / t) : 0;
    },
    // Tickers currently mid-run in the active batch → their live session run_id.
    // Sourced from the orchestrator's authoritative active_runs map.
    get predInflight() {
      return (this.scaleStatus && this.scaleStatus.active_runs) || {};
    },
    // Resolve a ticker to its live batch session run_id (if any).
    predRunIdFor(ticker) {
      const m = (this.scaleStatus && this.scaleStatus.active_runs) || {};
      return m[ticker] || m[(ticker || "").toUpperCase()] || null;
    },
    predTileClass(co) {
      if (co.status === "done") return "tile-done";       // green — completed
      if (this.predInflight[co.ticker]) return "tile-running"; // orange — ongoing
      return "";                                            // none — not run
    },
    /* ---------- Virtual portfolio ---------- */
    // ₹10L book for the prediction-run date currently on /stocks/history/{date}.
    async loadPortfolio() {
      this.portfolioLoading = true;
      this.portfolioError = "";
      try {
        if (!this.predHistoryDays.length) await this.loadPredHistory();
        const days = this.predHistoryDays;
        this.portfolioDate = this.predHistDate || this.portfolioDate
          || (days[0] && days[0].run_date) || "";
        if (!this.portfolioDate) { this.portfolio = null; return; }
        const q = new URLSearchParams({ market: this.portfolioMarket || "all" });
        const r = await fetch(`/api/portfolio/virtual/${encodeURIComponent(this.portfolioDate)}?${q}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.portfolio = await r.json();
      } catch (err) {
        this.portfolioError = "Failed to load portfolio: " + (err.message || err);
        this.portfolio = null;
      } finally {
        this.portfolioLoading = false;
      }
    },
    // Top-right run-date dropdown → sync the left History pane + reload.
    onPortfolioDateChange() {
      if (!this.portfolioDate) return;
      this.activeTab = "portfolio";
      this.enterRunReview(this.portfolioDate);
    },
    // SQL-viewer style: click a header to sort; click again to flip direction.
    sortPortfolio(key) {
      const s = this.portfolioSort;
      if (s.key === key) {
        s.dir = s.dir === "asc" ? "desc" : "asc";
      } else {
        s.key = key;
        // Text columns default A→Z; numeric columns default high→low.
        s.dir = (key === "name" || key === "ticker" || key === "action") ? "asc" : "desc";
      }
    },
    sortedHoldings() {
      if (!this.portfolio || !this.portfolio.holdings) return [];
      const { key, dir } = this.portfolioSort;
      const mul = dir === "asc" ? 1 : -1;
      return [...this.portfolio.holdings].sort((a, b) => {
        const av = a[key], bv = b[key];
        if (typeof av === "string" || typeof bv === "string") {
          return mul * String(av ?? "").localeCompare(String(bv ?? ""));
        }
        return mul * ((av ?? 0) - (bv ?? 0));
      });
    },
    sortIndicator(key) {
      if (this.portfolioSort.key !== key) return "";
      return this.portfolioSort.dir === "asc" ? " ▲" : " ▼";
    },
    // Compact Indian-format currency (₹, lakh/crore aware for big totals).
    fmtINR(v, decimals = 0) {
      if (v == null || isNaN(v)) return "—";
      return "₹" + Number(v).toLocaleString("en-IN", {
        minimumFractionDigits: decimals, maximumFractionDigits: decimals,
      });
    },
    pctClass(p) {
      if (p == null) return "text-slate-400";
      return p > 0 ? "text-emerald-300" : (p < 0 ? "text-red-300" : "text-slate-400");
    },
    fmtPct(p) {
      if (p == null || isNaN(p)) return "—";
      return (p * 100 >= 0 ? "+" : "") + (p * 100).toFixed(2) + "%";
    },

    predActionClass(a) {
      const x = (a || "").toUpperCase();
      if (x === "STRONG_BUY") return "bg-emerald-500/40 text-emerald-100 ring-1 ring-emerald-400/50";
      if (x === "BUY") return "bg-emerald-500/20 text-emerald-300";
      if (x === "SELL") return "bg-red-500/20 text-red-300";
      if (x === "STRONG_SELL") return "bg-red-500/40 text-red-100 ring-1 ring-red-400/50";
      if (x === "HOLD") return "bg-slate-500/20 text-slate-300";
      return "bg-blue-500/20 text-blue-300";
    },
    // Canonical (underscore) form so filter comparison is legacy-tolerant
    // (OVERWEIGHT/UNDERWEIGHT/NEUTRAL -> BUY/SELL/HOLD).
    _canonAction(a) {
      let s = String(a == null ? "" : a).toUpperCase();
      if (s.includes(".")) s = s.split(".").pop();
      const legacy = { OVERWEIGHT: "BUY", UNDERWEIGHT: "SELL", NEUTRAL: "HOLD" };
      return legacy[s] || s;
    },
    // Total predictions on the day (drives the "All" filter tile count).
    predActionTotal() {
      return Object.values(this.predHist.action_counts || {}).reduce((a, b) => a + (b || 0), 0);
    },
    // Toggle the action-distribution filter. Passing null (the "All" tile) or
    // re-clicking the active tile clears it.
    setPredActionFilter(act) {
      this.predActionFilter = act && this.predActionFilter !== act ? act : null;
    },
    // Sectors for the history grid with the action filter applied: keep only
    // matching companies and drop sectors that end up empty.
    predSectorsFiltered() {
      const sectors = this.predHist.sectors || [];
      if (!this.predActionFilter) return sectors;
      const want = this._canonAction(this.predActionFilter);
      return sectors
        .map((s) => ({ ...s, companies: (s.companies || []).filter((co) => this._canonAction(co.action) === want) }))
        .filter((s) => s.companies.length);
    },
    predRunNames() {
      const rows = [];
      for (const s of this.predSectorsFiltered()) {
        for (const co of s.companies || []) {
          rows.push({ ...co, sector: s.sector, market: s.market });
        }
      }
      return rows.sort((a, b) => {
        const aa = this._canonAction(a.action);
        const ba = this._canonAction(b.action);
        return aa.localeCompare(ba) || String(a.ticker || "").localeCompare(String(b.ticker || ""));
      });
    },
    // Group the loaded contract catalog (this.tickers, from /api/options?
    // asset_class=commodity) by category for the commodity Contracts grid —
    // the futures analog of the stock Prediction universe.
    contractGroups() {
      const order = ["energy", "metals", "grains", "softs", "livestock", "financials", "other"];
      const groups = {};
      (this.tickers || []).forEach((t) => {
        const cat = t.category || "other";
        (groups[cat] = groups[cat] || []).push({ symbol: t.symbol, name: t.name || t.symbol });
      });
      return Object.keys(groups)
        .sort((a, b) => {
          const ia = order.indexOf(a), ib = order.indexOf(b);
          return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
        })
        .map((k) => ({ category: k, items: groups[k] }));
    },
    openTile(ticker, runId) {
      this.form.ticker = ticker;
      this.stockActive = true;
      this.activeTab = "workflow";
      this.storedPredictionTicker = ticker;
      // Navigate into the focused per-instrument page (drops the left pane +
      // global tabs).  The URL is pushed at the tail of this method.
      this.focusPage = true;
      const isCommodity = this.assetClass === "commodity";
      const date = isCommodity ? "" : (this.predHistDate || "");
      // Commodity has no batch forward-collection, so there is never a live
      // batch session to attach to — predRunIdFor is stock-only.
      const live = this.liveSessionForTicker(ticker);
      const liveId = isCommodity ? null : ((live && live.run_id) || this.predRunIdFor(ticker));
      if (liveId && !date) {
        // Live batch run for this name → attach to its session so the
        // Workflow diagram streams node-by-node (same path as a manual run).
        this.attachToSession(liveId);
        this.loadStoredPrediction(ticker, "");
      } else {
        // Not currently running.  Detach from any live session we were
        // streaming so its events stop repainting THIS instrument's diagram,
        // wipe the previous view, then repaint the selected name's stored
        // run (that day's run_id when reviewing a prediction run).
        if (this.ws) { try { this.ws.close(); } catch (_) {} this.ws = null; }
        this.activeRunId = null;
        this.isRunning = false;
        this.resetRunState();
        if (!isCommodity) {
          if (runId) {
            this.loadStoredPrediction(ticker, date);
            this.loadRun(ticker, runId);
          } else {
            this._hydrateStoredRun(ticker, date);
          }
        } else {
          this.loadLatestRun(ticker);
        }
        if (typeof this.loadChart === "function") this.loadChart();
      }
      const base = isCommodity ? "/commodity/run/" : "/run/";
      const q = date ? ("?date=" + encodeURIComponent(date)) : "";
      try { window.history.pushState({}, "", base + encodeURIComponent(ticker) + q); } catch (_) {}
    },
    backToUniverse() {
      this.stockActive = false;
      this.activeTab = "prediction";
      try { window.history.pushState({}, "", "/stocks"); } catch (_) {}
    },
    // Back on the focused workbench.  A dated /run/{ticker}?date= return
    // lands on that prediction run (Names); otherwise /stocks or /commodity.
    backToHome() {
      this.focusPage = false;
      this.stockActive = false;
      if (this.assetClass === "commodity") {
        this.activeTab = "contracts";
        try { window.history.pushState({}, "", "/commodity"); } catch (_) {}
        return;
      }
      if (this.predHistDate) {
        this.enterRunReview(this.predHistDate);
        return;
      }
      this.activeTab = "prediction";
      try { window.history.pushState({}, "", "/stocks"); } catch (_) {}
      if (!this.predHistoryDays.length) this.loadPredHistory();
      const uni = this.predUniverse && this.predUniverse.markets;
      if (!uni || (!(uni.IN || []).length && !(uni.US || []).length)) {
        this.loadPredUniverse();
        this.predEnrichLoop();
      }
    },

    /* ----- background metadata enrichment (names + hover links) ----- */
    async predEnrichLoop() {
      if (this._predEnrichRunning) return;
      this._predEnrichRunning = true;
      try {
        for (let i = 0; i < 25; i++) {
          const r = await fetch("/api/predict/enrich?limit=20", { method: "POST" });
          if (!r.ok) break;
          const d = await r.json();
          if (d.enriched > 0) await this.loadPredUniverse();
          if (!d.remaining) break;
        }
      } catch (_) { /* best-effort */ } finally {
        this._predEnrichRunning = false;
      }
    },

    /* ----- search / add / remove ----- */
    async predDoSearch() {
      const q = (this.predSearch.q || "").trim();
      if (q.length < 2) { this.predSearch.results = []; return; }
      this.predSearch.loading = true;
      try {
        const url = `/api/predict/search?q=${encodeURIComponent(q)}&market=${this.predSearch.market}`;
        const d = await fetch(url).then((r) => r.json());
        this.predSearch.results = d.results || [];
        this.predSearch.open = true;
      } catch (err) {
        this.predError = "Search failed: " + (err.message || err);
      } finally {
        this.predSearch.loading = false;
      }
    },
    async predAddCompany(r) {
      this.predSearch.adding = r.symbol;
      try {
        const resp = await fetch("/api/predict/company", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ticker: r.symbol, name: r.name, market: r.market }),
        });
        if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || resp.status); }
        this.predSearch.q = ""; this.predSearch.results = []; this.predSearch.open = false;
        await this.loadPredUniverse();
      } catch (err) {
        this.predError = "Could not add " + r.symbol + ": " + (err.message || err);
      } finally {
        this.predSearch.adding = "";
      }
    },
    async predRemove(co) {
      if (!window.confirm(`Remove ${co.name} (${co.ticker}) from the universe?`)) return;
      try {
        const r = await fetch("/api/predict/company/" + encodeURIComponent(co.ticker), { method: "DELETE" });
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.status); }
        await this.loadPredUniverse();
      } catch (err) {
        this.predError = "Could not remove " + co.ticker + ": " + (err.message || err);
      }
    },

    /* ----- bulk/scale config snapshot (reuse Settings, save before run) ----- */
    _bulkCfgFromForm() {
      // Mirrors the interactive Run Analysis payload (minus ticker / date).
      // Autonomous stock omits debate/risk — graph + Hermes own those loops.
      const f = this.form;
      return {
        ...this._dialogueSnapshot(),
        run_mode: f.run_mode || "autonomous",
        quality_review: f.quality_review,
        compliance: f.compliance,
        critic_enabled: f.critic_enabled,
        fusion_enabled: f.fusion_enabled,
        fusion_include_crowd: f.fusion_include_crowd,
        tools_enabled: f.tools_enabled || {},
        book: f.book || "paper",
        llm_preset: this.selectedLlmPreset || "",
      };
    },
    _bulkCfgStorageKey() {
      return "tc.bulk_cfg." + (this.assetClass || "stock");
    },
    _initScaleWorkers() {
      // Keep the forward-collection worker count stable across server restarts
      // so a fresh run doesn't silently fall back to the hardcoded default and
      // overwhelm the local LLM server (e.g. previously 2 workers -> restart ->
      // back to 5/6).  Restored into both the Prediction-tile batch (predWorkers)
      // and the scale/bulk form (scaleForm.workers).
      const clamp = (n) => Math.max(1, Math.min(24, parseInt(n, 10) || -1));
      try {
        const pw = localStorage.getItem("tc.predWorkers");
        if (pw !== null) {
          const n = clamp(pw);
          if (n >= 1) this.predWorkers = n;
        }
        const sw = localStorage.getItem("tc.scaleWorkers");
        if (sw !== null) {
          const n = clamp(sw);
          if (n >= 1) this.scaleForm.workers = n;
        }
      } catch (_) { /* private mode etc. */ }
    },
    _persistScaleWorkers() {
      try {
        localStorage.setItem("tc.predWorkers", String(this.predWorkers));
        localStorage.setItem("tc.scaleWorkers", String(this.scaleForm.workers));
      } catch (_) { /* private mode etc. */ }
    },
    _initBulkCfg() {
      let saved = null;
      try {
        saved = JSON.parse(localStorage.getItem(this._bulkCfgStorageKey()));
      } catch (_) { /* private mode / corrupt JSON */ }
      if (saved && typeof saved === "object" && Object.keys(saved).length) {
        // Backfill keys added to the snapshot after this save (notably
        // tools_enabled) so an old snapshot can't silently run the batch
        // with default tools / missing settings.
        const fromForm = this._bulkCfgFromForm();
        let changed = false;
        for (const k of Object.keys(fromForm)) {
          if (!(k in saved)) { saved[k] = fromForm[k]; changed = true; }
        }
        // A legacy snapshot with an EMPTY tools_enabled object is
        // indistinguishable from "all defaults" on the crew side; the form
        // always carries every agent key, so rewrite it from the form.
        if (!saved.tools_enabled || !Object.keys(saved.tools_enabled).length) {
          saved.tools_enabled = fromForm.tools_enabled;
          changed = true;
        }
        if (changed) {
          try { localStorage.setItem(this._bulkCfgStorageKey(), JSON.stringify(saved)); } catch (_) {}
        }
        this.bulkCfg = saved;
        this.bulkCfgSaved = true;
      } else {
        // No saved snapshot yet — default to the current Settings so bulk
        // mirrors Run Analysis out of the box.
        this.bulkCfg = this._bulkCfgFromForm();
        this.bulkCfgSaved = false;
      }
    },
    _batchRunCfg() {
      // Single and bulk use the same Settings form (hydrated from
      // GET /api/settings). No second snapshot.
      return this._bulkCfgFromForm();
    },
    async saveBulkCfg() {
      // Pin the CURRENT Settings as the snapshot used by the next bulk run.
      this.bulkCfgError = "";
      this.bulkCfg = this._bulkCfgFromForm();
      try {
        const res = await fetch(`/api/settings?book=${this._hermesBook()}`, {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ bulk: this.bulkCfg }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(this._apiDetail(body) || res.statusText);
        this.bulkCfg = body.bulk || this.bulkCfg;
        this.bulkCfgSaved = true;
        try { localStorage.setItem(this._bulkCfgStorageKey(), JSON.stringify(this.bulkCfg)); } catch (_) {}
      } catch (err) {
        this.bulkCfgSaved = false;
        this.bulkCfgError = String(err.message || err);
      }
    },
    bulkCfgSummary() {
      const c = this._bulkCfgFromForm();
      const toolsCount = Object.values(c.tools_enabled || {})
        .reduce((n, a) => n + (a?.length || 0), 0);
      let model = c.llm_preset || "default";
      const preset = (this.llmPresets || []).find((p) => p.id === c.llm_preset);
      if (preset) model = preset.resolved_model || preset.model || preset.label;
      const mode = c.run_mode === "deterministic"
        ? "deterministic"
        : (c.run_mode === "harness" ? "harness" : "autonomous");
      if (mode === "autonomous" && this.assetClass === "stock") {
        return `${mode} · graph-driven debate · ${toolsCount} tools · ${model}`;
      }
      if (mode === "harness" && this.assetClass === "stock") {
        return `${mode} · conductor loop · ${toolsCount} tools · ${model}`;
      }
      return `${mode} · ${c.debate_rounds} debate · ${c.risk_rounds} risk · ${toolsCount} tools · ${model}`;
    },
    bulkCfgToolRows() {
      const tools = this.form.tools_enabled || {};
      return (this.toolOwners || []).map((a) => ({
        role: a.role,
        key: a.key,
        tools: tools[a.key] || [],
        inSnapshot: a.key in tools,
      }));
    },

    /* ----- batch controls (reuse /api/scale/*) ----- */
    _predBatchTickers() {
      const out = [];
      this.predMarkets.forEach((mk) => {
        (this.predUniverse.markets[mk] || []).forEach((cat) => {
          (cat.companies || []).forEach((co) => out.push(co.ticker));
        });
      });
      return out;
    },
    async predStartBatch() {
      const tickers = this._predBatchTickers();
      if (!tickers.length) { this.predError = "No companies to run."; return; }
      this.scaleLoading = true; this.predError = "";
      this._persistScaleWorkers();
      // Poll immediately so the UI goes live even while the start POST is
      // still in flight (the server may briefly gate on the LLM endpoint
      // being ready — the status endpoint reflects that via llm_wait).
      this._startScalePolling();
      try {
        const res = await fetch("/api/scale/start", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            tickers,
            workers: this.predWorkers,
            // Use the saved bulk-config snapshot (full Settings form) so the
            // batch honors the configured dialogue rounds + tool selection.
            run_cfg: this._batchRunCfg(),
            force_rerun: !!this.predForceRerun,
          }),
        });
        if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || res.status); }
        this.scaleStatus = await res.json();
        this._startScalePolling();
      } catch (err) {
        this.predError = "Could not start batch: " + (err.message || err);
      } finally {
        this.scaleLoading = false;
      }
    },

    /* ----- history (left sidebar dates -> that day's snapshot) ----- */
    async loadPredHistory() {
      try {
        const d = await fetch("/api/predict/history").then((r) => r.json());
        this.predHistoryDays = d.days || [];
      } catch (_) { /* non-fatal */ }
    },
    async loadPredHistDay() {
      if (!this.predHistDate) return;
      try {
        const r = await fetch("/api/predict/history/" + encodeURIComponent(this.predHistDate));
        if (r.ok) {
          this.predHist = await r.json();
          this.predActionFilter = null;   // reset filter for the freshly loaded day
        }
      } catch (err) {
        this.predError = "Failed to load " + this.predHistDate + ": " + (err.message || err);
      }
    },
    openHistDate(date) {
      this.enterRunReview(date);
    },
    enterRunReview(date, { push = true } = {}) {
      if (!date) return;
      this.predHistDate = date;
      this.portfolioDate = date;
      this.stockActive = false;
      this.focusPage = false;
      if (this.activeTab !== "portfolio" && this.activeTab !== "run-names") {
        this.activeTab = this.reviewReturnTab || "run-names";
      }
      if (!this.predHistoryDays.length) this.loadPredHistory();
      this.loadPredHistDay();
      if (this.activeTab === "portfolio") this.loadPortfolio();
      if (push) {
        try { window.history.pushState({}, "", "/stocks/history/" + encodeURIComponent(date)); } catch (_) {}
      }
    },
    backToLive({ push = true } = {}) {
      this.predHistDate = "";
      this.stockActive = false;
      this.focusPage = false;
      this.activeTab = "prediction";
      this.loadPredUniverse();
      this.predEnrichLoop();
      if (push) {
        try { window.history.pushState({}, "", "/stocks"); } catch (_) {}
      }
    },
    _startScalePolling() {
      if (this.scalePollTimer) clearInterval(this.scalePollTimer);
      this.scalePollTimer = setInterval(() => this.refreshScaleStatus(), 3000);
    },
    async startScale() {
      this.scaleLoading = true;
      this.scaleError = "";
      this._persistScaleWorkers();
      // Poll immediately so the LLM-readiness wait (llm_wait) is visible even
      // while the start POST is still in flight.
      this._startScalePolling();
      try {
        const body = {
          workers: this.scaleForm.workers,
          // Use the saved bulk-config snapshot (full Settings form).
          run_cfg: this._batchRunCfg(),
          force_rerun: !!this.predForceRerun,
        };
        if (this.scaleForm.useCustom) {
          body.tickers = this.scaleForm.tickers;
        } else {
          body.universe = this.scaleForm.universe;
        }
        const res = await fetch("/api/scale/start", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        this.scaleStatus = await res.json();
        this._startScalePolling();
      } catch (err) {
        console.error("startScale failed", err);
        this.scaleError = String(err.message || err);
      } finally {
        this.scaleLoading = false;
      }
    },
    async _scaleAction(path) {
      this.scaleError = "";
      try {
        const res = await fetch(path, { method: "POST" }).then(r => r.json());
        if (res && res.ok === false) this.scaleError = res.reason || "action failed";
        await this.refreshScaleStatus();
      } catch (err) {
        console.error(`${path} failed`, err);
        this.scaleError = String(err.message || err);
      }
    },
    pauseScale() { return this._scaleAction("/api/scale/pause"); },
    resumeScale() { return this._scaleAction("/api/scale/resume"); },
    killScale() {
      if (!window.confirm("Kill the forward-collection run? In-flight names finish first; progress is kept and can be re-started later.")) return;
      return this._scaleAction("/api/scale/stop");
    },

    async pollRlStatus() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      try {
        const res = await fetch(`/api/training/rl/status?ticker=${encodeURIComponent(ticker)}`).then(r => r.json());
        this.rlStatus = res;
        this.rlMetrics = Array.isArray(res.metrics) ? res.metrics : [];
        this.$nextTick(() => this.renderRlChart());
      } catch (err) {
        console.warn("pollRlStatus failed", err);
      }
    },

    async loadRlRuns() {
      this.rlLoading.runs = true;
      try {
        const ticker = (this.form.ticker || "").trim().toUpperCase();
        const url = ticker
          ? `/api/training/rl/runs?ticker=${encodeURIComponent(ticker)}`
          : `/api/training/rl/runs`;
        const res = await fetch(url).then(r => r.json());
        this.rlRuns = Array.isArray(res.runs) ? res.runs : [];
        const promoted = await fetch(`/api/training/rl/promoted`).then(r => r.json());
        this.rlPromoted = Array.isArray(promoted.promoted) ? promoted.promoted : [];
      } catch (err) {
        console.error("loadRlRuns failed", err);
      } finally {
        this.rlLoading.runs = false;
      }
    },

    isRunPromoted(run) {
      return this.rlPromoted.some(p => p.ticker === run.ticker && p.run_id === run.run_id);
    },

    async promoteRun(run) {
      if (!run || run.status !== "completed") {
        this.rlError = "Only completed runs can be promoted.";
        return;
      }
      this.rlLoading.promote = true;
      this.rlError = "";
      try {
        const res = await fetch("/api/training/rl/promote", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ticker: run.ticker, run_id: run.run_id }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(detail.detail || res.statusText);
        }
        await this.loadRlRuns();
      } catch (err) {
        console.error("promoteRun failed", err);
        this.rlError = String(err.message || err);
      } finally {
        this.rlLoading.promote = false;
      }
    },

    async previewRlRecommendation() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      this.rlLoading.recommend = true;
      this.rlRecommendation = null;
      try {
        const res = await fetch("/api/training/rl/recommend", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ticker }),
        });
        this.rlRecommendation = await res.json();
      } catch (err) {
        console.error("previewRlRecommendation failed", err);
        this.rlRecommendation = { error: String(err) };
      } finally {
        this.rlLoading.recommend = false;
      }
    },

    renderRlChart() {
      const canvas = document.getElementById("rl-training-chart");
      if (!canvas || typeof Chart === "undefined") return;
      if (!this.rlMetrics || !this.rlMetrics.length) {
        if (this.rlChart) { this.rlChart.destroy(); this.rlChart = null; }
        return;
      }
      const labels = this.rlMetrics.map(m => `r${m.rollout_index}`);
      const data = {
        labels,
        datasets: [
          {
            label: "Mean episode PnL %",
            data: this.rlMetrics.map(m => (m.mean_episode_pnl_pct || 0) * 100),
            borderColor: "#60a5fa", backgroundColor: "#60a5fa22",
            yAxisID: "yPnl", tension: 0.25,
          },
          {
            label: "Entropy",
            data: this.rlMetrics.map(m => m.entropy || 0),
            borderColor: "#fbbf24", backgroundColor: "#fbbf2422",
            yAxisID: "yEntropy", tension: 0.25,
          },
          {
            label: "Sharpe (per-step)",
            data: this.rlMetrics.map(m => m.sharpe_per_step || 0),
            borderColor: "#34d399", backgroundColor: "#34d39922",
            yAxisID: "ySharpe", tension: 0.25,
          },
        ],
      };
      const opts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
          yPnl: {
            position: "left",
            ticks: { color: "#60a5fa" },
            grid: { color: "#1e293b" },
            title: { display: true, text: "PnL %", color: "#60a5fa" },
          },
          yEntropy: {
            position: "right",
            ticks: { color: "#fbbf24" },
            grid: { display: false },
            title: { display: true, text: "Entropy", color: "#fbbf24" },
          },
          ySharpe: {
            position: "right",
            offset: true,
            ticks: { color: "#34d399" },
            grid: { display: false },
            title: { display: true, text: "Sharpe", color: "#34d399" },
            display: false,   // hidden axis; toggle via legend click
          },
        },
      };
      if (this.rlChart) {
        this.rlChart.data = data;
        this.rlChart.update("none");
      } else {
        this.rlChart = new Chart(canvas.getContext("2d"), { type: "line", data, options: opts });
      }
    },

    rlActionLabel(weight) {
      const pct = Math.round(weight * 100);
      if (pct > 0) return `LONG ${pct}%`;
      if (pct < 0) return `SHORT ${Math.abs(pct)}%`;
      return "FLAT";
    },

    async loadBacktest() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      this.backtestNetworkError = false;
      // Guard: if we know we don't have enough proposals, don't burn the
      // round-trip — the UI will show a friendly "need N more" message.
      await this.checkBacktestEligibility();
      if (this.backtestEnoughData === false) {
        this.backtestResult = null;
        return;
      }
      this.backtestLoading = true;
      try {
        const q = new URLSearchParams({
          ticker,
          train_size: String(this.backtestConfig.train_size),
          embargo_size: String(this.backtestConfig.embargo_size),
          test_size: String(this.backtestConfig.test_size),
          validation: this.backtestValidation || "walk_forward",
          cpcv_groups: String(this.cpcvGroups || 6),
          cpcv_k: String(this.cpcvK || 2),
          lookback_years: String(this.backtestLookbackYears || 5),
        });
        const res = await fetch(`/api/backtest?${q}`).then(r => r.json());
        this.backtestResult = res;
        this.$nextTick(() => this.renderBacktestChart());
      } catch (err) {
        // Browser-level fetch failure (network drop, server restart, CORS).
        // Distinct from the server returning {error: "..."} which is handled
        // in the same UI block but means "data wasn't enough", not "server
        // unreachable".
        console.error("loadBacktest network failure", err);
        this.backtestNetworkError = true;
        this.backtestResult = null;
      } finally {
        this.backtestLoading = false;
      }
    },
    renderBacktestChart() {
      const ctx = document.getElementById("backtest-equity-chart");
      if (!ctx || !this.backtestResult?.combined_equity?.length || !window.Chart) return;
      if (this.backtestChart) {
        try { this.backtestChart.destroy(); } catch (_) {}
        this.backtestChart = null;
      }
      const equity = this.backtestResult.combined_equity;
      const labels = this.backtestResult.combined_timestamps || equity.map((_, i) => `t${i}`);
      const datasets = [{
        label: "Strategy",
        data: equity,
        borderColor: "#22d3ee",
        backgroundColor: "rgba(34, 211, 238, 0.1)",
        fill: true, tension: 0.1, pointRadius: 0,
      }];
      // Overlay the buy-and-hold benchmark so alpha is visible, not just stated.
      const bench = this.backtestResult.combined_benchmark;
      if (bench && bench.length === equity.length) {
        datasets.push({
          label: "Buy & hold",
          data: bench,
          borderColor: "#a78bfa",
          borderDash: [5, 4],
          backgroundColor: "transparent",
          fill: false, tension: 0.1, pointRadius: 0,
        });
      }
      this.backtestChart = new window.Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
            y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
          },
          plugins: { legend: { display: bench && bench.length === equity.length, labels: { color: "#94a3b8" } } },
        },
      });
    },

    /* ----------------------- Swarm PSO convergence sparkline */
    renderSwarmHistoryChart() {
      const ctx = document.getElementById("swarm-history-chart");
      const hist = this.gridSearchResult?.history;
      if (!ctx || !hist || !hist.length || !window.Chart) return;
      if (this._swarmChart) { try { this._swarmChart.destroy(); } catch (_) {} this._swarmChart = null; }
      this._swarmChart = new window.Chart(ctx, {
        type: "line",
        data: {
          labels: hist.map((_, i) => `it${i}`),
          datasets: [{
            label: "Global best score",
            data: hist,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.12)",
            fill: true, tension: 0.2, pointRadius: 2,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
            y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
          },
          plugins: { legend: { display: false } },
        },
      });
    },

    /* ----------------------- Multi-asset portfolio backtest (M7) */
    async runPortfolioBacktest() {
      const tickers = (this.portfolioTickers || "").trim();
      if (!tickers) return;
      this.portfolioLoading = true;
      this.portfolioResult = null;
      try {
        const q = new URLSearchParams({
          tickers,
          cost_model: this.backtestCostModel || "standard",
          allocation_method: this.portfolioAllocMethod || "HRP",
          gross_budget: String(this.portfolioGrossBudget || 0.8),
          lookback_years: String(this.backtestLookbackYears || 5),
        });
        const res = await fetch(`/api/portfolio_backtest?${q}`).then(r => r.json());
        this.portfolioResult = res;
        this.$nextTick(() => this.renderPortfolioChart());
      } catch (err) {
        console.error("runPortfolioBacktest failed", err);
        this.portfolioResult = { error: String(err) };
      } finally {
        this.portfolioLoading = false;
      }
    },
    renderPortfolioChart() {
      const ctx = document.getElementById("portfolio-equity-chart");
      const eq = this.portfolioResult?.equity_curve;
      if (!ctx || !eq || !eq.length || !window.Chart) return;
      if (this.portfolioChart) { try { this.portfolioChart.destroy(); } catch (_) {} this.portfolioChart = null; }
      const labels = this.portfolioResult.timestamps || eq.map((_, i) => `t${i}`);
      const datasets = [{
        label: "Portfolio",
        data: eq,
        borderColor: "#34d399",
        backgroundColor: "rgba(52, 211, 153, 0.1)",
        fill: true, tension: 0.1, pointRadius: 0,
      }];
      const bench = this.portfolioResult.benchmark_curve;
      if (bench && bench.length === eq.length) {
        datasets.push({
          label: "Equal-weight B&H",
          data: bench,
          borderColor: "#a78bfa",
          borderDash: [5, 4],
          backgroundColor: "transparent",
          fill: false, tension: 0.1, pointRadius: 0,
        });
      }
      this.portfolioChart = new window.Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
            y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
          },
          plugins: { legend: { display: true, labels: { color: "#94a3b8" } } },
        },
      });
    },

    /* ----------------------- Crowd-positioning chip */
    async loadCrowdSignal() {
      const ticker = (this.form.ticker || "").trim().toUpperCase();
      if (!ticker) return;
      this.crowdLoading = true;
      this.crowdSignal = null;
      try {
        const res = await fetch(`/api/crowd_positioning?ticker=${encodeURIComponent(ticker)}`).then(r => r.json());
        this.crowdSignal = res;
      } catch (err) {
        console.error("loadCrowdSignal failed", err);
        this.crowdSignal = { error: String(err) };
      } finally {
        this.crowdLoading = false;
      }
    },

    /* ----------------------- Formatting helpers for the new metric cards */
    fmtPct(v, dp = 2) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return (v * 100).toFixed(dp) + "%";
    },
    fmtNum(v, dp = 2) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      if (v === Infinity || v === -Infinity) return v > 0 ? "∞" : "-∞";
      return Number(v).toFixed(dp);
    },
  };
}

window.tradingCrewApp = tradingCrewApp;
