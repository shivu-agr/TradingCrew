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
    embeddingPresets: [],   // mirror of llmPresets for the embedding endpoint
    selectedEmbeddingPreset: "",
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
      trade_date: new Date().toISOString().slice(0, 10),
      max_debate_rounds: 2,
      max_risk_rounds: 1,
      memory: true,
      tools_enabled: {},   // agent_key -> [tool_name, ...]
      // Phase 2B.3 — partition the on-disk portfolio between a
      // sandboxed "paper" book and a live "prod" book.  Defaults to
      // paper so casual experimentation can't corrupt prod state.
      book: "paper",
    },

    /* ---------------------------------------------------------- collapsible state */
    sidebarCollapsed: false,
    advancedOpen: true,
    openSections: {
      ticker: true,
      date: false,
      llm: false,
      runs: false,
      dialogue: true,
      tools: true,
    },

    // Phase E — saved runs
    recentRuns: [],
    loadedRunId: null,
    clearingRecentRuns: false,
    sections: [
      { id: "ticker", title: "Ticker", icon: "T" },
      { id: "date", title: "Date", icon: "D" },
      { id: "llm", title: "LLM", icon: "L" },
      { id: "dialogue", title: "Dialogue", icon: "M" },
      { id: "tools", title: "Tools", icon: "⚙" },
    ],

    /* ---------------------------------------------------------- run state */
    isRunning: false,
    status: "idle",
    currentNode: "",
    finalDecision: "",
    decisionPayload: null,
    activeTab: "workflow",
    // The "Futures" tab is rendered only for commodity dashboards; the
    // x-show guards in index.html honour ``assetClass``.  We keep it in
    // the array so the order is stable; the template just hides it for
    // stock dashboards.
    tabs: [
      { id: "workflow", label: "Workflow" },
      { id: "charts", label: "Charts" },
      { id: "futures", label: "Futures", commodityOnly: true },
      { id: "reports", label: "Reports" },
      { id: "memory", label: "Memory" },
      { id: "portfolio", label: "Portfolio" },
      { id: "backtest", label: "Backtest" },
      { id: "training", label: "RL Training" },
      { id: "logs", label: "Logs" },
    ],

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
    reports: {},          // role -> output text
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

    // M7 — Portfolio tab
    portfolio: null,
    allocatorMethod: "HRP",       // HRP / MV / EQR (Phase 2B.1)

    // L2 agentic training — outcome resolution + reflections
    resolveResult: null,
    resolveLoading: false,

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
    // Phase 2C — extra grid axes / robustness toggles.
    gridSweepMaxLeverage: false,
    gridSweepDrawdownKill: false,
    gridSweepCostModel: false,
    backtestValidation: "walk_forward",   // walk_forward | cpcv
    backtestCostModel: "standard",
    seedN: 20,
    seedLoading: false,
    seedResult: null,
    gridSearchResult: null,
    gridSearchLoading: false,

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

    /* ---------------------------------------------------------- boot */
    async init() {
      await this.loadOptions();

      this.diagram = new WorkflowDiagram(
        document.getElementById("workflow-diagram"),
        { assetClass: this.assetClass },
      );
      this.diagram.setActiveTools(this.form.tools_enabled);
      this.diagram.onSelect((id) => (this.selectedNode = id));
      // Keep the toolbar's "%" label in sync with the diagram's actual
      // effective scale (fit * userZoom) so the number is always the
      // truth, not a stale UI guess.
      this.diagramScale = this.diagram.scale;
      this.diagram.onScale((s) => { this.diagramScale = s; });

      this.$watch("form.ticker", () => this.loadChart());
      this.$watch("form.trade_date", () => this.loadChart());
      this.$watch("enabledIndicators", () => this.rebuildCharts());
      this.$watch("activeTab", (v) => {
        if (v === "charts") setTimeout(() => this.rebuildCharts(), 60);
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
        this.diagram.setActiveTools(this.form.tools_enabled);
      }, { deep: true });

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
      this.$watch("decisionPayload", annotate);
      this.$watch("activeTab", (v) => { if (v === "reports") annotate(); });

      this.loadChart();
      this.loadRecentRuns();

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

        this.embeddingPresets = res.embedding_presets || [];
        const persistedEmb = (typeof localStorage !== "undefined"
          ? localStorage.getItem("tc.embedding_preset")
          : null);
        const isValidEmb = persistedEmb
          && this.embeddingPresets.some((p) => p.id === persistedEmb);
        this.selectedEmbeddingPreset = isValidEmb
          ? persistedEmb
          : (res.default_embedding_preset
            || (this.embeddingPresets[0] && this.embeddingPresets[0].id)
            || "");

        this.tavily = !!res.tavily_configured;

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

        if (this.diagram) this.diagram.setActiveTools(this.form.tools_enabled);
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
      if (["dialogue", "tools"].includes(id)) this.advancedOpen = true;
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

    get activeEmbeddingPreset() {
      return (this.embeddingPresets || [])
        .find((p) => p.id === this.selectedEmbeddingPreset) || null;
    },
    get embeddingSummary() {
      const p = this.activeEmbeddingPreset;
      if (!p) return "(unset)";
      return p.resolved_model || p.model || p.label;
    },
    onEmbeddingPresetChange() {
      try {
        if (typeof localStorage !== "undefined") {
          localStorage.setItem(
            "tc.embedding_preset",
            this.selectedEmbeddingPreset || "",
          );
        }
      } catch (_) { /* private mode etc. */ }
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

    /* -------------------------------------- memory-LLM call estimation
     *
     * When ``form.memory`` is ON, CrewAI's unified Memory runs an LLM
     * analysis pass for every save and recall — concretely the four
     * functions in ``crewai/memory/analyze.py``:
     *
     *   • analyze_for_save              (per extracted memory)
     *   • extract_memories_from_content (per task save)
     *   • analyze_for_consolidation     (when similar memories exist)
     *   • analyze_query                 (per search_memory tool call)
     *
     * The total scales with task count, which itself depends on the
     * debate / risk rounds chosen in the sidebar.  We compute the
     * estimate client-side so the user sees the cost before they
     * launch a run — and so it stays reactive to slider changes.
     *
     * Calibration: this matches the ``62 × MemoryLLM disabled`` count
     * observed when memory was wired to a fail-fast stub on a default
     * (debate=2, risk=1) NTNX run.  We treat that as the lower bound
     * for the "real" LLM case and add headroom for consolidation /
     * recall fan-out on heavier runs.
     */
    estimateMemoryLlmCalls() {
      const isCommodity = this.assetClass === "commodity";
      // Analyst grid: 8 (stock) / 7 (commodity) with ~2 tasks each.
      const numAnalysts = isCommodity ? 7 : 8;
      const analystTasks = numAnalysts * 2;
      // Debate: bull + bear per round.
      const debateTasks = 2 * (this.form.max_debate_rounds || 1);
      // Sequential mid stages: research mgr, quality reviewer, trader.
      const midTasks = 3;
      // Risk team: 3 personas per round.
      const riskTasks = 3 * (this.form.max_risk_rounds || 1);
      // Final stages: compliance + portfolio manager.
      const finalTasks = 2;
      const numTasks = analystTasks + debateTasks + midTasks + riskTasks + finalTasks;

      // Per-task analyze calls: 1 extract + ~2 save (avg 2 extracted
      // memories) + occasional consolidation = ~3-4 calls.  Plus a
      // small per-run recall budget (search_memory tool calls from
      // analysts).
      const callsPerTask = 3.5;
      const recallBudget = 12;
      const count = Math.round(numTasks * callsPerTask + recallBudget);

      // Wall-clock estimate.  Local LLM caps at ~60 req/min — with 8
      // parallel save workers + the inner 10-worker pool inside
      // EncodingFlow we can saturate that, so effective rate ≈ 60/min.
      const minutes = Math.max(0.1, count / 60);
      return { count, minutes: minutes.toFixed(1), tasks: numTasks };
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
    startRun() {
      if (this.isRunning) return;
      this.diagram.reset();
      this.diagram.setActiveTools(this.form.tools_enabled);
      this.logs = [];
      this.pendingLogCount = 0;
      this.reports = {};
      this.degradedRoles = {};
      this.finalDecision = "";
      this.cascadeStatus = null;
      this.actionProposal = null;
      this.actionProposalMd = "";
      this.executionResult = null;
      this.episodeMeta = null;
      this.reflectionRecords = null;
      this.decisionPayload = null;
      this.nodeActivities = {};
      this.selectedNode = null;
      this.status = "running";
      this.isRunning = true;
      this.currentNode = "";

      const f = this.form;
      const payload = {
        asset_class: this.assetClass,
        ticker: (f.ticker || "").trim().toUpperCase(),
        debate_rounds: f.max_debate_rounds,
        risk_rounds: f.max_risk_rounds,
        memory: f.memory,
        tools_enabled: f.tools_enabled,
        // Phase 2B.3 — paper/prod book partition.  The runner uses this
        // as the PortfolioState portfolio_id so paper experiments don't
        // pollute the prod book's audit trail.
        book: this.form.book || "paper",
        // Phase 2F — UI-selectable LLM. The runner applies this as a
        // thread-local override that ``get_llm()`` consults BEFORE the
        // env chain, so each WS session can use a different model
        // (e.g. one closed-source PM, the rest open-source).
        llm_preset: this.selectedLlmPreset || "",
        embedding_preset: this.selectedEmbeddingPreset || "",
      };

      this.appendLog("info", "client", `Starting ${payload.ticker} debate=${payload.debate_rounds} risk=${payload.risk_rounds}`);

      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/analyze`);
      this.ws = ws;
      ws.addEventListener("open", () => ws.send(JSON.stringify(payload)));
      ws.addEventListener("message", (e) => {
        try { this.handleEvent(JSON.parse(e.data)); }
        catch (err) { this.appendLog("error", "parse", String(err)); }
      });
      ws.addEventListener("close", () => this.finishRun());
      ws.addEventListener("error", () => {
        this.appendLog("error", "ws", "WebSocket error");
        this.status = "error";
      });
    },
    cancelRun() {
      if (this.ws && this.ws.readyState === 1) {
        try { this.ws.close(); } catch (_) {}
      }
      this.finishRun();
      this.status = "idle";
    },
    finishRun() {
      this.isRunning = false;
      this.currentNode = "";
    },

    /* ---------------------------------------------------------- events */
    handleEvent(event) {
      switch (event.type) {
        case "run_started":
          this.appendLog("info", "run",
            `${event.ticker} · ${event.agent_count} agents · ${event.task_count} tasks`);
          return;

        case "node_started":
          this.currentNode = event.node;
          this.diagram.setState(event.node, "running");
          this.appendLog("node", event.node, "▶ started");
          return;

        case "node_completed":
          this.diagram.setState(event.node, event.degraded ? "degraded" : "done");
          this.reports[event.node] = event.output || "";
          if (event.degraded) {
            this.degradedRoles = { ...this.degradedRoles, [event.node]: true };
            this.appendLog("warn", event.node, "⚠ degraded: LLM emitted tool-call list as final answer");
          }
          this.pushActivity(event.node, {
            label: event.degraded ? "report (degraded)" : "report",
            color: event.degraded ? "warn" : "llm",
            subtitle: `${(event.output || "").length} chars`,
            body: event.output || "", markdown: true,
          });
          this.appendLog("node", event.node, event.degraded ? "⚠ completed (degraded)" : "✓ completed");
          return;

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
          this.pushActivity(event.node, {
            label: event.error ? "tool error" : "tool result",
            color: event.error ? "error" : "result",
            subtitle: `${event.tool} · ${event.elapsed_ms || 0}ms`,
            body: (event.output || "").slice(0, 4000),
          });
          return;
        }

        case "final_decision":
          this.decisionPayload = event.decision;
          this.finalDecision = event.decision?.action || "";
          this.diagram.setDecision(this.finalDecision || "DECISION");
          this.appendLog("final", "decision", this.finalDecision || "(no parsed decision)");
          // Auto-flip to reports tab
          this.activeTab = "reports";
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

        case "run_completed":
          this.status = "completed";
          this.appendLog("info", "run", "completed");
          this.finishRun();
          // Phase E — the new record is on disk; refresh the sidebar list.
          this.loadRecentRuns();
          return;

        case "error":
          this.status = "error";
          this.appendLog("error", "error", event.message);
          this.finishRun();
          return;
      }
    },

    /* ---------------------------------------------------------- logs/activity */
    appendLog(kind, tag, message) {
      const id = ++this._logSeq;
      const time = new Date().toTimeString().slice(0, 8);
      const colorMap = {
        info: "log-tag-info", node: "log-tag-node", tool: "log-tag-tool",
        result: "log-tag-result", error: "log-tag-error", llm: "log-tag-llm",
        report: "log-tag-report", final: "log-tag-final", warn: "log-tag-warn",
      };
      this.logs.push({ id, time, tag, color: colorMap[kind] || "log-tag-info", message: String(message) });
      if (this.activeTab !== "logs") this.pendingLogCount++;
      this.$nextTick(() => {
        if (!this.autoScroll || this.userScrolledUp) return;
        const c = document.getElementById("logs-container");
        if (c) c.scrollTop = c.scrollHeight;
      });
    },
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
      if (node === "Trader") return "trader";
      if (node === "Portfolio Manager") return "portfolio manager";
      if (node === "Research Manager") return "research manager";
      if (node === "Quality Reviewer") return "quality reviewer";
      if (node === "Compliance Officer") return "compliance officer";
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
        "Social Analyst": "analyst",
        "News Analyst": "analyst",
        "Fundamentals Analyst": "analyst",
        "Macro Analyst": "analyst",
        "Geopolitical Analyst": "analyst",
        "Sector / Peer Analyst": "analyst",
        "Quant / Options Analyst": "analyst",
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
      return Object.entries(this.reports).map(([role, body]) => {
        const flagged = !!this.degradedRoles[role];
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
        return {
          role,
          title: role,
          color: colors[KIND_BY_ROLE[role]] || "#94a3b8",
          html: this.renderMarkdown(cleanBody),
          degraded,
        };
      });
    },

    renderMarkdown(raw) {
      const cleaned = this.cleanModelOutput(raw);
      if (!window.marked) return `<pre>${this.escape(cleaned)}</pre>`;
      try { window.marked.setOptions({ breaks: true, gfm: true }); } catch (_) {}
      return window.marked.parse(cleaned);
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
      } catch (err) {
        console.error("resolveOutcomes failed", err);
        this.resolveResult = { error: String(err), count: 0, by_status: {}, records: [] };
      } finally {
        this.resolveLoading = false;
      }
    },

    /* ---------------------------------------------------------- M7 portfolio */
    async loadPortfolio() {
      try {
        const q = new URLSearchParams({
          method: this.allocatorMethod || "HRP",
          book: this.form.book || "paper",
        });
        const res = await fetch(`/api/portfolio?${q}`).then(r => r.json());
        this.portfolio = res;
      } catch (err) {
        console.error("loadPortfolio failed", err);
        this.portfolio = null;
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
    async loadRun(ticker, runId) {
      try {
        const res = await fetch(`/api/runs/${encodeURIComponent(ticker)}/${encodeURIComponent(runId)}`).then(r => r.json());
        if (res.error) {
          console.warn("loadRun error:", res.error);
          return;
        }
        this.loadedRunId = runId;
        // Repaint all the post-LLM panels from the saved record.
        this.cascadeStatus = res.cascade_status || null;
        this.actionProposal = res.action_proposal || null;
        this.actionProposalMd = res.action_proposal_markdown || "";
        this.executionResult = res.execution_result || null;
        this.episodeMeta = res.episode_meta || null;
        this.reflectionRecords = res.reflection_records || null;
        this.decisionPayload = res.final_decision || null;
        this.finalDecision = res.final_decision?.action || "";
        // Repaint the per-agent reports (Workflow tab + Reports tab).
        this.reports = { ...(res.reports || {}) };
        // Rehydrate the degraded-roles set so the Reports tab badges
        // and the diagram colouring match the persisted state of the
        // run (the saved record carries one entry per analyst whose
        // final answer was substituted with the OSS-LLM placeholder).
        this.degradedRoles = Object.fromEntries(
          (res.degraded_roles || []).map((role) => [role, true])
        );
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
          train_size: String(this.backtestConfig.train_size),
          embargo_size: String(this.backtestConfig.embargo_size),
          test_size: String(this.backtestConfig.test_size),
          lookback_years: String(this.backtestLookbackYears || 5),
        });
        const res = await fetch(`/api/grid_search?${q}`, { method: "POST" }).then(r => r.json());
        this.gridSearchResult = res;
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
      this.backtestChart = new window.Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [{
            label: "Equity",
            data: equity,
            borderColor: "#22d3ee",
            backgroundColor: "rgba(34, 211, 238, 0.1)",
            fill: true, tension: 0.1, pointRadius: 0,
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
  };
}

window.tradingCrewApp = tradingCrewApp;
