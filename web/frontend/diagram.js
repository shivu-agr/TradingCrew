/* TradingCrew — workflow diagram (SVG nodes + edges).
 *
 * LAYOUT
 * ------
 * Single-row LEFT-TO-RIGHT pipeline laid out as 5 ranks, with a 2-column
 * analyst grid so the 8 parallel analysts don't stretch the whole canvas
 * vertically:
 *
 *   ┌─────────────────┐  ┌────────┐  ┌─────────┐  ┌────────┐  ┌─────┐  ┌────────┐  ┌─────┐
 *   │ ANALYSTS (4×2)  │→ │ BULL   │→ │ R.MGR   │→ │ TRADER │→ │ AGG │→ │ COMPL  │→ │ DEC │
 *   │                 │  │ BEAR   │  │ Q.REV   │  │        │  │ NTR │  │ PM     │  │     │
 *   └─────────────────┘  └────────┘  └─────────┘  └────────┘  │ CON │  └────────┘  └─────┘
 *                                                              └─────┘
 *
 * Width ~1240, height ~340 (4:1 aspect — sized for wide screens).  The
 * old layout was 1480×980 (cramped), the failed serpentine was 980×1380
 * (wasted vertical space, awkward pivot).  This is what the user gets
 * by default; Ctrl+wheel / +/− buttons zoom in to read details.
 *
 * Two layouts ship with the diagram:
 *   - stock     (default) : 18-agent debate pipeline
 *   - commodity           : 17-agent futures debate pipeline
 * Both use the same shape; only the personas differ.  The right layout
 * is picked from the ``assetClass`` option passed to the
 * ``WorkflowDiagram`` constructor (read from window.location.pathname
 * in app.js). Node IDs MUST match the role strings the runner emits.
 *
 * Edges + state semantics (industry standard, mirrors React-Flow / n8n /
 * Argo / GitLab CI):
 *
 *   idle      : thin gray (the default static look)
 *   running   : cyan + animated dashed flow (stroke-dashoffset keyframe
 *               in style.css — "flow" effect that makes direction obvious)
 *   complete  : solid green
 *
 * Repeat-fire agents (Bull/Bear in debate rounds, Risk team in risk rounds)
 * flip back to "running" each time their next task starts — the runner
 * emits a fresh node_started in those cases, which retriggers the
 * animation on their outgoing edges.
 *
 * Zoom + pan: the diagram is wrapped in an ``overflow:auto`` viewport.
 * The toolbar in index.html calls ``zoomBy()`` / ``fit()`` / ``resetZoom()``;
 * Ctrl+wheel inside the viewport also zooms anchored at the cursor.
 */

(function () {
  // Maps the diagram's node.id (which == the runner's role string) to the
  // help_content.js anchor.  Add new entries here when new agents are
  // wired into the crew.  Entries that aren't in HELP_CONTENT simply
  // fall through to "no tooltip" — they don't crash.
  const AGENT_HELP_ANCHORS = {
    "Market Analyst": "agent.market_analyst",
    "Social Analyst": "agent.social_analyst",
    "News Analyst": "agent.news_analyst",
    "Fundamentals Analyst": "agent.fundamentals_analyst",
    "Macro Analyst": "agent.macro_analyst",
    "Geopolitical Analyst": "agent.geopolitical_analyst",
    "Sector / Peer Analyst": "agent.sector_analyst",
    "Quant / Options Analyst": "agent.quant_analyst",
    "Bullish Researcher": "agent.bull_researcher",
    "Bearish Researcher": "agent.bear_researcher",
    "Research Manager": "agent.research_manager",
    "Quality Reviewer": "agent.quality_reviewer",
    "Trader": "agent.trader",
    "Aggressive Risk Analyst": "agent.risk_aggressive",
    "Neutral Risk Analyst": "agent.risk_neutral",
    "Conservative Risk Analyst": "agent.risk_conservative",
    "Compliance Officer": "agent.compliance_officer",
    "Portfolio Manager": "agent.portfolio_manager",
  };

  function agentHelpAnchor(nodeId) {
    return AGENT_HELP_ANCHORS[nodeId] || null;
  }

  const NODE_COLORS = {
    analyst: "#06b6d4",
    bull: "#22c55e",
    bear: "#ef4444",
    manager: "#3b82f6",
    reviewer: "#facc15",
    trader: "#a855f7",
    risk_a: "#ef4444",
    risk_n: "#f59e0b",
    risk_c: "#3b82f6",
    decision: "#facc15",
  };

  // ====================== Stock layout (18-agent debate, strict LTR) ============
  //
  // 10 ranks, all flowing LEFT → RIGHT, with one column per pipeline
  // stage.  Parallel agents at the SAME stage (Bull/Bear in the debate,
  // 3-analyst risk team) stack vertically inside their own column;
  // every SEQUENTIAL hand-off (Research Mgr → Quality Reviewer → Trader,
  // Compliance → PM → Decision) gets its own column to the right.
  // No more two-row "RM on top, QR below" stacks — every step reads as
  // a clean horizontal left-to-right pipeline.
  //
  //   col 1  (x=20)    : Analysts column A (Market, Social, News, Fundamentals)
  //   col 2  (x=170)   : Analysts column B (Macro, Geopolitical, Sector, Quant)
  //   col 3  (x=325)   : Bullish / Bearish researchers (parallel)
  //   col 4  (x=475)   : Research Manager  (sequential)
  //   col 5  (x=625)   : Quality Reviewer  (sequential after RM)
  //   col 6  (x=775)   : Trader            (sequential)
  //   col 7  (x=925)   : Risk team (Aggressive / Neutral / Conservative — parallel)
  //   col 8  (x=1075)  : Compliance        (sequential)
  //   col 9  (x=1225)  : Portfolio Manager (sequential after Compliance)
  //   col 10 (x=1375)  : Decision          (sequential, final emission)
  //
  // Total: 1540×340 — wider than before so the extra columns fit, but
  // still 4.5:1 aspect so it reads as a single horizontal strip.  Nodes
  // remain compact pills (140×38) plus a short tool strip where
  // applicable.  fit() auto-scales the diagram into the viewport width.

  const STOCK_WIDTH = 1540;
  const STOCK_HEIGHT = 340;

  // Two-column analyst grid: 4 rows × 2 columns = 8 analysts.
  // ROW_GAP (~68) leaves room for the 30px tool-chip strip below each node.
  const STOCK_ANALYST_ROW_GAP = 68;
  const STOCK_ANALYST_Y0 = 25;
  const STOCK_ANALYST_X_A = 20;
  const STOCK_ANALYST_X_B = 170;

  // Inter-column spacing past the analyst grid (each downstream column
  // gets a steady ~150 px gap so the diagram has a regular horizontal
  // rhythm).  Sequential stages each get their own column; parallel
  // stages (Bull/Bear, Risk team) stack inside one.
  const STOCK_COL_BULL_BEAR = 325;
  const STOCK_COL_RES_MGR   = 475;
  const STOCK_COL_Q_REV     = 625;
  const STOCK_COL_TRADER    = 775;
  const STOCK_COL_RISK      = 925;
  const STOCK_COL_COMP      = 1075;
  const STOCK_COL_PM        = 1225;
  const STOCK_COL_DECISION  = 1375;

  // All single-node columns center on the same vertical mid-line so the
  // sequential pipeline reads as one straight horizontal track.
  const STOCK_MID_Y = 150;

  const STOCK_NODES = [
    // Analyst team — 2 columns × 4 rows (parallel)
    { id: "Market Analyst",          label: "Market",          kind: "analyst",  x: STOCK_ANALYST_X_A, y: STOCK_ANALYST_Y0 + 0 * STOCK_ANALYST_ROW_GAP, hasTools: "market_analyst" },
    { id: "Social Analyst",          label: "Social",          kind: "analyst",  x: STOCK_ANALYST_X_A, y: STOCK_ANALYST_Y0 + 1 * STOCK_ANALYST_ROW_GAP, hasTools: "social_analyst" },
    { id: "News Analyst",            label: "News",            kind: "analyst",  x: STOCK_ANALYST_X_A, y: STOCK_ANALYST_Y0 + 2 * STOCK_ANALYST_ROW_GAP, hasTools: "news_analyst" },
    { id: "Fundamentals Analyst",    label: "Fundamentals",    kind: "analyst",  x: STOCK_ANALYST_X_A, y: STOCK_ANALYST_Y0 + 3 * STOCK_ANALYST_ROW_GAP, hasTools: "fundamentals_analyst" },
    { id: "Macro Analyst",           label: "Macro",           kind: "analyst",  x: STOCK_ANALYST_X_B, y: STOCK_ANALYST_Y0 + 0 * STOCK_ANALYST_ROW_GAP, hasTools: "macro_analyst" },
    { id: "Geopolitical Analyst",    label: "Geopolitical",    kind: "analyst",  x: STOCK_ANALYST_X_B, y: STOCK_ANALYST_Y0 + 1 * STOCK_ANALYST_ROW_GAP, hasTools: "geopolitical_analyst" },
    { id: "Sector / Peer Analyst",   label: "Sector / Peer",   kind: "analyst",  x: STOCK_ANALYST_X_B, y: STOCK_ANALYST_Y0 + 2 * STOCK_ANALYST_ROW_GAP, hasTools: "sector_analyst" },
    { id: "Quant / Options Analyst", label: "Quant / Options", kind: "analyst",  x: STOCK_ANALYST_X_B, y: STOCK_ANALYST_Y0 + 3 * STOCK_ANALYST_ROW_GAP, hasTools: "quant_analyst" },

    // Researcher team — Bull / Bear stack (genuine parallel debate)
    { id: "Bullish Researcher",      label: "Bullish",         kind: "bull",     x: STOCK_COL_BULL_BEAR, y: 80 },
    { id: "Bearish Researcher",      label: "Bearish",         kind: "bear",     x: STOCK_COL_BULL_BEAR, y: 220 },

    // Sequential hand-offs — each gets its own column, centered vertically
    { id: "Research Manager",        label: "Research Mgr",    kind: "manager",  x: STOCK_COL_RES_MGR,   y: STOCK_MID_Y },
    { id: "Quality Reviewer",        label: "Quality Review",  kind: "reviewer", x: STOCK_COL_Q_REV,     y: STOCK_MID_Y },
    { id: "Trader",                  label: "Trader",          kind: "trader",   x: STOCK_COL_TRADER,    y: STOCK_MID_Y, hasTools: "trader" },

    // Risk team — 3 stacked vertically (parallel risk perspectives)
    { id: "Aggressive Risk Analyst",   label: "Aggressive",    kind: "risk_a",   x: STOCK_COL_RISK,      y: 60 },
    { id: "Neutral Risk Analyst",      label: "Neutral",       kind: "risk_n",   x: STOCK_COL_RISK,      y: 150 },
    { id: "Conservative Risk Analyst", label: "Conservative",  kind: "risk_c",   x: STOCK_COL_RISK,      y: 240 },

    // Compliance → Portfolio Manager — sequential, each in its own column
    { id: "Compliance Officer",      label: "Compliance",      kind: "reviewer", x: STOCK_COL_COMP,      y: STOCK_MID_Y },
    { id: "Portfolio Manager",       label: "Portfolio Mgr",   kind: "manager",  x: STOCK_COL_PM,        y: STOCK_MID_Y },

    // Decision (the final emission)
    { id: "decision",                label: "Decision",        kind: "decision", x: STOCK_COL_DECISION,  y: STOCK_MID_Y },
  ];

  const STOCK_EDGES = [
    // analysts -> bull
    ["Market Analyst", "Bullish Researcher"],
    ["Social Analyst", "Bullish Researcher"],
    ["News Analyst", "Bullish Researcher"],
    ["Fundamentals Analyst", "Bullish Researcher"],
    ["Macro Analyst", "Bullish Researcher"],
    ["Geopolitical Analyst", "Bullish Researcher"],
    ["Sector / Peer Analyst", "Bullish Researcher"],
    ["Quant / Options Analyst", "Bullish Researcher"],
    // analysts -> bear
    ["Market Analyst", "Bearish Researcher"],
    ["Social Analyst", "Bearish Researcher"],
    ["News Analyst", "Bearish Researcher"],
    ["Fundamentals Analyst", "Bearish Researcher"],
    ["Macro Analyst", "Bearish Researcher"],
    ["Geopolitical Analyst", "Bearish Researcher"],
    ["Sector / Peer Analyst", "Bearish Researcher"],
    ["Quant / Options Analyst", "Bearish Researcher"],
    // bull/bear -> Research Manager + Quality Reviewer
    ["Bullish Researcher", "Research Manager"],
    ["Bearish Researcher", "Research Manager"],
    ["Research Manager", "Quality Reviewer"],
    // research -> trader
    ["Research Manager", "Trader"],
    ["Quality Reviewer", "Trader"],
    // trader -> risk (the SERPENTINE PIVOT — trader on top row, risk
    // team directly below it on row 2; we route this with a vertical
    // bezier instead of the horizontal LR one)
    ["Trader", "Aggressive Risk Analyst"],
    ["Trader", "Neutral Risk Analyst"],
    ["Trader", "Conservative Risk Analyst"],
    // risk -> compliance + pm (RTL within row 2)
    ["Aggressive Risk Analyst", "Compliance Officer"],
    ["Neutral Risk Analyst", "Compliance Officer"],
    ["Conservative Risk Analyst", "Compliance Officer"],
    ["Aggressive Risk Analyst", "Portfolio Manager"],
    ["Neutral Risk Analyst", "Portfolio Manager"],
    ["Conservative Risk Analyst", "Portfolio Manager"],
    ["Compliance Officer", "Portfolio Manager"],
    // PM -> decision (final RTL leg)
    ["Portfolio Manager", "decision"],
  ];

  // Group rectangles hug their contents tightly so they read as labels,
  // not as giant empty boxes.  Each group box wraps the cells of its
  // column(s) with a small padding (8 px each side).
  const STOCK_GROUPS = [
    // Analyst grid (2 cols × 4 rows) — spans x=12..308 / y=6..276
    { label: "Analyst team",     x: 12,   y: 6,   w: 296, h: 270 },
    // Researcher team wraps Bull/Bear (col 3) + Research Mgr (col 4) +
    // Quality Reviewer (col 5).  Left edge = col_bull_bear - 8, right
    // edge = col_q_rev + 140 + 8.
    { label: "Researcher team",  x: 317,  y: 70,  w: 456, h: 200 },
    // Trader — single node centered at mid-y
    { label: "Trader",           x: 767,  y: 140, w: 156, h: 80 },
    // Risk team — 3 stacked vertically
    { label: "Risk management",  x: 917,  y: 50,  w: 156, h: 240 },
    // Compliance + PM — now two side-by-side columns
    { label: "Compliance + PM",  x: 1067, y: 140, w: 306, h: 80 },
    // Decision (the terminal)
    { label: "Decision",         x: 1367, y: 140, w: 156, h: 80 },
  ];

  // ====================== Commodity layout (17-agent debate) ==================
  // Topology mirrors the stock crew (debate -> trader plan -> risk -> PM)
  // but with futures-aware personas. ID strings MUST match the YAML
  // ``role:`` fields in commodity_crew/config/agents.yaml — that's what
  // the runner emits over the WebSocket.
  //
  // Personas reused verbatim from the stock crew (Quality Reviewer, risk
  // team, Compliance Officer, Portfolio Manager) share their role
  // strings so NODE_KIND can map them uniformly — only the bull / bear /
  // research-manager / trader carry distinct futures-only IDs.

  // Commodity uses the same strict-LTR shape as stock; only personas change.
  // 7 commodity analysts vs 8 stock analysts — laid out as 2 cols × 4 rows
  // (last cell empty) so the column structure stays consistent.  All
  // sequential stages (Research Mgr → Quality Reviewer → Trader,
  // Compliance → PM → Decision) each get their own column.
  const COMM_WIDTH = 1540;
  const COMM_HEIGHT = 340;
  const COMM_ANALYST_ROW_GAP = 68;
  const COMM_ANALYST_Y0 = 25;
  const COMM_ANALYST_X_A = 20;
  const COMM_ANALYST_X_B = 170;

  const COMM_COL_BULL_BEAR = 325;
  const COMM_COL_RES_MGR   = 475;
  const COMM_COL_Q_REV     = 625;
  const COMM_COL_TRADER    = 775;
  const COMM_COL_RISK      = 925;
  const COMM_COL_COMP      = 1075;
  const COMM_COL_PM        = 1225;
  const COMM_COL_DECISION  = 1375;

  const COMM_MID_Y = 150;

  const COMM_NODES = [
    // Analyst team — 7 analysts split across 2 cols (col A = 4, col B = 3)
    { id: "Commodity Market Analyst",           label: "Market",         kind: "analyst",  x: COMM_ANALYST_X_A, y: COMM_ANALYST_Y0 + 0 * COMM_ANALYST_ROW_GAP, hasTools: "market_analyst" },
    { id: "Term Structure Analyst",             label: "Curve",          kind: "analyst",  x: COMM_ANALYST_X_A, y: COMM_ANALYST_Y0 + 1 * COMM_ANALYST_ROW_GAP, hasTools: "curve_analyst" },
    { id: "Inventories & Stocks Analyst",       label: "Inventories",    kind: "analyst",  x: COMM_ANALYST_X_A, y: COMM_ANALYST_Y0 + 2 * COMM_ANALYST_ROW_GAP, hasTools: "inventories_analyst" },
    { id: "Supply & Demand Analyst",            label: "Supply / Demand",kind: "analyst",  x: COMM_ANALYST_X_A, y: COMM_ANALYST_Y0 + 3 * COMM_ANALYST_ROW_GAP, hasTools: "supply_demand_analyst" },
    { id: "Macro Analyst",                      label: "Macro",          kind: "analyst",  x: COMM_ANALYST_X_B, y: COMM_ANALYST_Y0 + 0 * COMM_ANALYST_ROW_GAP, hasTools: "macro_analyst" },
    { id: "Geopolitical & Supply-Risk Analyst", label: "Geopolitical",   kind: "analyst",  x: COMM_ANALYST_X_B, y: COMM_ANALYST_Y0 + 1 * COMM_ANALYST_ROW_GAP, hasTools: "geopolitical_analyst" },
    { id: "Positioning & Seasonality Quant",    label: "Positioning",    kind: "analyst",  x: COMM_ANALYST_X_B, y: COMM_ANALYST_Y0 + 2 * COMM_ANALYST_ROW_GAP, hasTools: "quant_analyst" },

    // Researcher team — Bull/Bear parallel stack
    { id: "Bullish Futures Researcher",         label: "Bullish",        kind: "bull",     x: COMM_COL_BULL_BEAR, y: 80 },
    { id: "Bearish Futures Researcher",         label: "Bearish",        kind: "bear",     x: COMM_COL_BULL_BEAR, y: 220 },

    // Sequential hand-offs — each in its own column
    { id: "Futures Research Manager",           label: "Research Mgr",   kind: "manager",  x: COMM_COL_RES_MGR,   y: COMM_MID_Y },
    { id: "Quality Reviewer",                   label: "Quality Review", kind: "reviewer", x: COMM_COL_Q_REV,     y: COMM_MID_Y },
    { id: "Senior Futures Trader",              label: "Futures Trader", kind: "trader",   x: COMM_COL_TRADER,    y: COMM_MID_Y, hasTools: "trader" },

    // Risk team — 3 stacked vertically (parallel risk perspectives)
    { id: "Aggressive Risk Analyst",            label: "Aggressive",     kind: "risk_a",   x: COMM_COL_RISK,      y: 60 },
    { id: "Neutral Risk Analyst",               label: "Neutral",        kind: "risk_n",   x: COMM_COL_RISK,      y: 150 },
    { id: "Conservative Risk Analyst",          label: "Conservative",   kind: "risk_c",   x: COMM_COL_RISK,      y: 240 },

    // Compliance → PM — sequential, each in its own column
    { id: "Compliance Officer",                 label: "Compliance",     kind: "reviewer", x: COMM_COL_COMP,      y: COMM_MID_Y, hasTools: "compliance_officer" },
    { id: "Portfolio Manager",                  label: "Portfolio Mgr",  kind: "manager",  x: COMM_COL_PM,        y: COMM_MID_Y },

    // Decision
    { id: "decision",                           label: "Decision",       kind: "decision", x: COMM_COL_DECISION,  y: COMM_MID_Y },
  ];

  const COMM_EDGES = [
    // analysts -> bull
    ["Commodity Market Analyst",           "Bullish Futures Researcher"],
    ["Term Structure Analyst",             "Bullish Futures Researcher"],
    ["Inventories & Stocks Analyst",       "Bullish Futures Researcher"],
    ["Supply & Demand Analyst",            "Bullish Futures Researcher"],
    ["Macro Analyst",                      "Bullish Futures Researcher"],
    ["Geopolitical & Supply-Risk Analyst", "Bullish Futures Researcher"],
    ["Positioning & Seasonality Quant",    "Bullish Futures Researcher"],
    // analysts -> bear
    ["Commodity Market Analyst",           "Bearish Futures Researcher"],
    ["Term Structure Analyst",             "Bearish Futures Researcher"],
    ["Inventories & Stocks Analyst",       "Bearish Futures Researcher"],
    ["Supply & Demand Analyst",            "Bearish Futures Researcher"],
    ["Macro Analyst",                      "Bearish Futures Researcher"],
    ["Geopolitical & Supply-Risk Analyst", "Bearish Futures Researcher"],
    ["Positioning & Seasonality Quant",    "Bearish Futures Researcher"],
    // bull/bear -> Research Manager + Quality Reviewer
    ["Bullish Futures Researcher",         "Futures Research Manager"],
    ["Bearish Futures Researcher",         "Futures Research Manager"],
    ["Futures Research Manager",           "Quality Reviewer"],
    // research -> trader
    ["Futures Research Manager",           "Senior Futures Trader"],
    ["Quality Reviewer",                   "Senior Futures Trader"],
    // trader -> risk
    ["Senior Futures Trader",              "Aggressive Risk Analyst"],
    ["Senior Futures Trader",              "Neutral Risk Analyst"],
    ["Senior Futures Trader",              "Conservative Risk Analyst"],
    // risk -> compliance + pm
    ["Aggressive Risk Analyst",            "Compliance Officer"],
    ["Neutral Risk Analyst",               "Compliance Officer"],
    ["Conservative Risk Analyst",          "Compliance Officer"],
    ["Aggressive Risk Analyst",            "Portfolio Manager"],
    ["Neutral Risk Analyst",               "Portfolio Manager"],
    ["Conservative Risk Analyst",          "Portfolio Manager"],
    ["Compliance Officer",                 "Portfolio Manager"],
    // PM -> decision
    ["Portfolio Manager",                  "decision"],
  ];

  const COMM_GROUPS = [
    { label: "Analyst team",     x: 12,   y: 6,   w: 296, h: 270 },
    { label: "Researcher team",  x: 317,  y: 70,  w: 456, h: 200 },
    { label: "Futures trader",   x: 767,  y: 140, w: 156, h: 80 },
    { label: "Risk management",  x: 917,  y: 50,  w: 156, h: 240 },
    { label: "Compliance + PM",  x: 1067, y: 140, w: 306, h: 80 },
    { label: "Decision",         x: 1367, y: 140, w: 156, h: 80 },
  ];

  const LAYOUTS = {
    stock:     { width: STOCK_WIDTH, height: STOCK_HEIGHT, NODES: STOCK_NODES, EDGES: STOCK_EDGES, GROUPS: STOCK_GROUPS },
    commodity: { width: COMM_WIDTH,  height: COMM_HEIGHT,  NODES: COMM_NODES,  EDGES: COMM_EDGES,  GROUPS: COMM_GROUPS },
  };

  function el(tag, attrs, ...children) {
    const isSvg = ["svg", "path", "rect", "circle", "line", "polygon", "defs", "marker", "g"].includes(tag);
    const node = isSvg
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "style") Object.assign(node.style, v);
      else if (k === "html") node.innerHTML = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) {
      if (c == null) continue;
      node.append(c.nodeType ? c : document.createTextNode(c));
    }
    return node;
  }

  // Horizontal LTR bezier router. All edges flow left → right in this
  // layout (no vertical pivot or RTL leg anymore), so the control points
  // are always X-offset; the magnitude of the offset scales with the
  // horizontal distance so short edges curve gently and long edges have
  // enough slack to curve smoothly without overshooting.
  function curvePath(x1, y1, x2, y2) {
    const dx = Math.max(40, Math.abs(x2 - x1) * 0.55);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  class WorkflowDiagram {
    constructor(container, options) {
      this.container = container;
      const opts = options || {};
      const assetClass = (opts.assetClass === "commodity") ? "commodity" : "stock";
      const layout = LAYOUTS[assetClass];
      this.assetClass = assetClass;
      this.LAYOUT_WIDTH = layout.width;
      this.LAYOUT_HEIGHT = layout.height;
      this.NODES = layout.NODES;
      this.EDGES = layout.EDGES;
      this.GROUPS = layout.GROUPS;
      this.nodesById = {};
      this.toolStrips = {};
      this.edges = {};
      this.selected = null;
      this.selectListeners = [];
      this.scaleListeners = [];    // notified on zoom changes (UI percentage)
      this.nodeRunCount = {};      // role -> # task_completed events received
      // Zoom state.  ``fitScale`` is what fits the diagram into the
      // viewport width; ``userZoom`` is the multiplier the user applies
      // on top via the +/- toolbar or Ctrl+wheel.  Effective scale is
      // ``fitScale * userZoom``.  Default user zoom = 1.0.
      this.fitScale = 1;
      this.userZoom = 1;
      this.MIN_USER_ZOOM = 0.4;
      this.MAX_USER_ZOOM = 3.0;
      this.render();
    }

    render() {
      this.container.innerHTML = "";
      this.container.style.position = "relative";
      this.container.style.width = "100%";
      this.container.style.height = this.LAYOUT_HEIGHT + "px";

      const inner = document.createElement("div");
      inner.style.position = "relative";
      inner.style.width = this.LAYOUT_WIDTH + "px";
      inner.style.height = this.LAYOUT_HEIGHT + "px";
      inner.style.transformOrigin = "top left";
      this.inner = inner;
      this.container.appendChild(inner);

      // Group rectangles
      this.GROUPS.forEach((g) => {
        const grp = document.createElement("div");
        grp.className = "flow-group";
        grp.style.left = g.x + "px";
        grp.style.top = g.y + "px";
        grp.style.width = g.w + "px";
        grp.style.height = g.h + "px";
        const lbl = document.createElement("div");
        lbl.className = "group-label";
        lbl.textContent = g.label;
        grp.appendChild(lbl);
        inner.appendChild(grp);
      });

      // SVG with edges
      const svg = el("svg", {
        class: "flow-svg",
        viewBox: `0 0 ${this.LAYOUT_WIDTH} ${this.LAYOUT_HEIGHT}`,
        preserveAspectRatio: "none",
        style: { position: "absolute", left: 0, top: 0, width: "100%", height: "100%" },
      });
      const defs = el("defs");
      defs.appendChild(
        el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
                       markerWidth: 6, markerHeight: 6, orient: "auto" },
          el("path", { d: "M0,0 L10,5 L0,10 Z", fill: "#475569" }))
      );
      defs.appendChild(
        el("marker", { id: "arrow-active", viewBox: "0 0 10 10", refX: 9, refY: 5,
                       markerWidth: 6, markerHeight: 6, orient: "auto" },
          el("path", { d: "M0,0 L10,5 L0,10 Z", fill: "#60a5fa" }))
      );
      defs.appendChild(
        el("marker", { id: "arrow-done", viewBox: "0 0 10 10", refX: 9, refY: 5,
                       markerWidth: 6, markerHeight: 6, orient: "auto" },
          el("path", { d: "M0,0 L10,5 L0,10 Z", fill: "#22c55e" }))
      );
      svg.appendChild(defs);
      this.edgesGroup = el("g");
      svg.appendChild(this.edgesGroup);
      inner.appendChild(svg);
      this.svg = svg;

      // Nodes
      this.NODES.forEach((cfg) => {
        const node = document.createElement("div");
        node.className = "flow-node";
        node.dataset.nodeId = cfg.id;
        node.dataset.kind = cfg.kind;
        node.dataset.state = "idle";
        node.style.left = cfg.x + "px";
        node.style.top = cfg.y + "px";
        node.style.color = NODE_COLORS[cfg.kind] || "#94a3b8";

        // help.js wiring: hover shows a tooltip via .tc-tooltip, click on
        // the (i) icon opens the Help drawer with the matching anchor.
        // The mapping below covers every agent in NODE_KIND.  Decision and
        // any role we haven't authored an entry for simply skip the tag.
        const helpAnchor = agentHelpAnchor(cfg.id);
        if (helpAnchor) {
          node.dataset.helpTip = helpAnchor;
        }

        const title = document.createElement("div");
        title.className = "title";
        title.innerHTML = `<span class="ndot" style="background:${NODE_COLORS[cfg.kind]}"></span><span>${cfg.label}</span>`;
        node.appendChild(title);

        const subtitle = document.createElement("div");
        subtitle.className = "subtitle";
        subtitle.textContent = cfg.kind.replace("_", " ");
        node.appendChild(subtitle);

        if (helpAnchor) {
          const help = document.createElement("span");
          help.className = "node-help info-i";
          help.dataset.help = helpAnchor;
          help.setAttribute("role", "button");
          help.setAttribute("tabindex", "0");
          help.setAttribute("aria-label", `About ${cfg.label}`);
          help.textContent = "i";
          // Stop the click from selecting the parent node (that opens the
          // activity-inspector panel instead).  We *do* want the click to
          // open the help drawer, so we call the store directly rather
          // than relying on bubbling — stopPropagation would otherwise
          // also kill help.js's document-level click delegation.
          help.addEventListener("click", (ev) => {
            ev.stopPropagation();
            const store = window.Alpine && window.Alpine.store("help");
            if (store) store.openHelp(helpAnchor);
          });
          node.appendChild(help);
        }

        node.addEventListener("click", () => this.select(cfg.id));
        inner.appendChild(node);
        this.nodesById[cfg.id] = node;

        if (cfg.hasTools) {
          const strip = document.createElement("div");
          strip.className = "tools-strip";
          strip.style.left = cfg.x + "px";
          // 40 = 38 px node body + 2 px gap.  Strip itself is height-capped
          // by CSS so a tool-heavy analyst can't crowd the next row.
          strip.style.top = (cfg.y + 40) + "px";
          inner.appendChild(strip);
          this.toolStrips[cfg.id] = { el: strip, agentKey: cfg.hasTools, chips: {}, counters: {} };
        }
      });

      // Edges (after nodes so we know their actual heights)
      this.renderEdges();

      // Auto-fit on resize.
      this.fit();
      window.addEventListener("resize", () => this.refit());
      // ResizeObserver picks up the container going from display:none -> visible
      // when the user switches tabs (Alpine x-show toggles display). Without
      // this, fit() last ran with clientWidth=0 (because the tab was hidden
      // mid-run) and the diagram stays collapsed to height 0 even after the
      // user returns. The observer refires fit() the moment the container
      // gets a real width again. Guarded by typeof check so the diagram still
      // boots in environments without ResizeObserver (jsdom etc.).
      if (typeof ResizeObserver !== "undefined") {
        try {
          this._resizeObserver = new ResizeObserver(() => this.refit());
          this._resizeObserver.observe(this.container);
        } catch (_) {
          // ignore — refit() will still recover when the user resizes.
        }
      }

      // Ctrl + wheel = zoom (anchored at the cursor position).  We
      // listen on the SVG layer so panning by scrolling normally
      // (without Ctrl) still works through the parent overflow:auto.
      this.container.addEventListener("wheel", (ev) => {
        if (!(ev.ctrlKey || ev.metaKey)) return;
        ev.preventDefault();
        const factor = ev.deltaY > 0 ? 0.9 : 1.1;
        this.zoomBy(factor, ev.clientX, ev.clientY);
      }, { passive: false });
    }

    /**
     * Pick the (entry, exit) anchors for an edge.  The layout is pure
     * LTR — every edge exits the right side of A and enters the left
     * side of B at their respective vertical mid-points.
     */
    edgeAnchors(a, b) {
      const ax = parseFloat(a.style.left);
      const ay = parseFloat(a.style.top);
      const aw = a.offsetWidth;
      const ah = a.offsetHeight;
      const bx = parseFloat(b.style.left);
      const by = parseFloat(b.style.top);
      const bh = b.offsetHeight;
      return { x1: ax + aw, y1: ay + ah / 2, x2: bx, y2: by + bh / 2 };
    }

    renderEdges() {
      this.edges = {};
      this.edgesGroup.innerHTML = "";
      this.EDGES.forEach(([from, to]) => {
        const a = this.nodesById[from];
        const b = this.nodesById[to];
        if (!a || !b) return;
        const { x1, y1, x2, y2 } = this.edgeAnchors(a, b);
        const path = el("path", {
          d: curvePath(x1, y1, x2, y2),
          class: "flow-edge",
          "marker-end": "url(#arrow)",
        });
        this.edgesGroup.appendChild(path);
        this.edges[`${from}->${to}`] = path;
      });
    }

    /**
     * Recompute fitScale to fit the diagram into the current container
     * width *at userZoom = 1*.  Then apply ``fitScale * userZoom``.
     *
     * Called from the resize observer and from the toolbar "Fit" button
     * (which also resets userZoom to 1).
     */
    fit() {
      const containerWidth = this.container.clientWidth;
      if (containerWidth <= 0) return;
      this.fitScale = Math.min(1, containerWidth / this.LAYOUT_WIDTH);
      this.applyScale();
    }

    // Recompute fitScale without touching userZoom (resize-driven path).
    refit() {
      const containerWidth = this.container.clientWidth;
      if (containerWidth <= 0) return;
      this.fitScale = Math.min(1, containerWidth / this.LAYOUT_WIDTH);
      this.applyScale();
    }

    /** Total effective scale = fit-to-width × user zoom. */
    get scale() { return this.fitScale * this.userZoom; }

    applyScale() {
      const s = this.scale;
      this.inner.style.transform = `scale(${s})`;
      // The container's height grows with the scaled diagram so the
      // surrounding viewport scroll bar lines up correctly.
      this.container.style.height = (this.LAYOUT_HEIGHT * s) + "px";
      this.container.style.width = (this.LAYOUT_WIDTH * s) + "px";
      requestAnimationFrame(() => this.renderEdges());
      this.scaleListeners.forEach((cb) => cb(s));
    }

    /** Multiply the user zoom by ``factor`` and re-apply.
     *
     * If anchor (clientX, clientY) is supplied, we also adjust the
     * containing viewport's scroll so the point under the cursor stays
     * roughly put after the zoom — this is what makes Ctrl+wheel feel
     * natural at the cursor instead of always zooming about (0,0).
     */
    zoomBy(factor, clientX, clientY) {
      const next = Math.max(this.MIN_USER_ZOOM, Math.min(this.MAX_USER_ZOOM, this.userZoom * factor));
      if (Math.abs(next - this.userZoom) < 1e-3) return;
      const viewport = this.container.parentElement;
      let preX = null, preY = null;
      if (viewport && clientX !== undefined && clientY !== undefined) {
        const rect = viewport.getBoundingClientRect();
        // Position in unscaled (diagram-local) coords BEFORE the zoom.
        preX = (clientX - rect.left + viewport.scrollLeft) / this.scale;
        preY = (clientY - rect.top + viewport.scrollTop) / this.scale;
      }
      this.userZoom = next;
      this.applyScale();
      if (viewport && preX !== null) {
        const rect = viewport.getBoundingClientRect();
        viewport.scrollLeft = preX * this.scale - (clientX - rect.left);
        viewport.scrollTop  = preY * this.scale - (clientY - rect.top);
      }
    }

    setUserZoom(z) {
      this.userZoom = Math.max(this.MIN_USER_ZOOM, Math.min(this.MAX_USER_ZOOM, z));
      this.applyScale();
    }

    /** Reset to 100% (1:1 with the layout coords, ignoring fitScale). */
    resetZoom() {
      this.fitScale = 1;
      this.userZoom = 1;
      this.applyScale();
    }

    /** Listen for scale changes — the toolbar UI subscribes to update %. */
    onScale(cb) { this.scaleListeners.push(cb); }

    setState(nodeId, state) {
      const node = this.nodesById[nodeId];
      if (!node) return;
      node.dataset.state = state;
      // Outgoing edges of a "running" node light up; outgoing edges of a
      // "done" node turn green so the user can trace finished branches.
      if (state === "running") {
        this.highlightOutgoingEdges(nodeId, true, false);
        // pulse the node's incoming edges too — hints at "currently active"
        this.highlightIncomingEdges(nodeId, true, false);
      } else if (state === "done" || state === "degraded") {
        this.highlightIncomingEdges(nodeId, false, true);
        this.nodeRunCount[nodeId] = (this.nodeRunCount[nodeId] || 0) + 1;
        const counter = this.nodeRunCount[nodeId];
        const subtitle = node.querySelector(".subtitle");
        if (subtitle) {
          const base = node.dataset.kind.replace("_", " ");
          if (state === "degraded") {
            // Annotate the subtitle so the diagram itself shows which
            // analyst came back empty without the user having to flip
            // to the Reports tab.  Re-runs on the same role (debate /
            // risk rounds) keep the count visible.
            subtitle.textContent = counter > 1
              ? `${base} · ${counter}× · degraded`
              : `${base} · degraded`;
          } else if (counter > 1) {
            subtitle.textContent = `${base} · ${counter}×`;
          }
        }
      }
    }

    highlightIncomingEdges(nodeId, active, doneClass) {
      Object.entries(this.edges).forEach(([key, path]) => {
        if (key.endsWith(`->${nodeId}`)) {
          path.classList.toggle("active", !!active);
          if (doneClass) path.classList.add("done");
          path.setAttribute("marker-end",
            doneClass ? "url(#arrow-done)" : (active ? "url(#arrow-active)" : "url(#arrow)"));
        }
      });
    }

    highlightOutgoingEdges(nodeId, active, doneClass) {
      Object.entries(this.edges).forEach(([key, path]) => {
        if (key.startsWith(`${nodeId}->`)) {
          path.classList.toggle("active", !!active);
          if (doneClass) path.classList.add("done");
        }
      });
    }

    /**
     * Update the chip for a tool. Indexed by tool name so repeated calls
     * collapse to a single chip with a count badge.
     */
    setToolState(analystNodeId, toolName, callId, state, tooltip) {
      const strip = this.toolStrips[analystNodeId];
      if (!strip) return;

      const counters = strip.counters || (strip.counters = {});
      const counter = counters[toolName] || (counters[toolName] = { running: 0, done: 0, error: 0, total: 0 });

      if (state === "running") {
        counter.running += 1;
        counter.total += 1;
      } else if (state === "done") {
        counter.running = Math.max(0, counter.running - 1);
        counter.done += 1;
      } else if (state === "error") {
        counter.running = Math.max(0, counter.running - 1);
        counter.error += 1;
      }

      let chip = strip.chips[toolName];
      if (!chip) {
        chip = document.createElement("span");
        chip.className = "tool-chip";
        strip.el.appendChild(chip);
        strip.chips[toolName] = chip;
      }
      delete chip.dataset.kind;
      chip.style.opacity = "1";

      let effective = "idle";
      if (counter.error > 0 && counter.running === 0) effective = "error";
      else if (counter.running > 0) effective = "running";
      else if (counter.done > 0) effective = "done";
      chip.dataset.state = effective;

      const calls = counter.done + counter.error + counter.running;
      chip.textContent = calls > 1 ? `${toolName} ×${calls}` : toolName;

      if (tooltip) chip.title = tooltip;
    }

    /**
     * Seed each agent's tool strip with idle pre-configured chips so the
     * user can see the tool inventory before the run.
     */
    setActiveTools(agentToTools) {
      Object.entries(agentToTools || {}).forEach(([agentKey, tools]) => {
        // Find the node id whose tools-strip belongs to this agent
        const nodeId = Object.entries(this.toolStrips)
          .find(([_, s]) => s.agentKey === agentKey)?.[0];
        if (!nodeId) return;
        const strip = this.toolStrips[nodeId];
        strip.el.innerHTML = "";
        strip.chips = {};
        strip.counters = {};
        (tools || []).forEach((toolName) => {
          const chip = document.createElement("span");
          chip.className = "tool-chip";
          chip.textContent = toolName;
          chip.dataset.state = "idle";
          chip.dataset.kind = "preconfig";
          chip.style.opacity = "0.55";
          strip.el.appendChild(chip);
          strip.chips[toolName] = chip;
        });
      });
    }

    select(nodeId) {
      if (this.selected === nodeId) nodeId = null;
      Object.values(this.nodesById).forEach((n) => n.classList.remove("selected"));
      this.selected = nodeId;
      if (nodeId && this.nodesById[nodeId]) {
        this.nodesById[nodeId].classList.add("selected");
      }
      this.selectListeners.forEach((cb) => cb(nodeId));
    }

    onSelect(cb) { this.selectListeners.push(cb); }

    setDecision(text) {
      const dec = this.nodesById["decision"];
      if (!dec) return;
      const titleSpan = dec.querySelector(".title span:last-child");
      if (titleSpan) titleSpan.textContent = text || "Decision";
      dec.dataset.state = text ? "done" : "idle";
    }

    reset() {
      Object.values(this.nodesById).forEach((n) => (n.dataset.state = "idle"));
      Object.values(this.toolStrips).forEach((s) => {
        s.el.innerHTML = "";
        s.chips = {};
        s.counters = {};
      });
      Object.values(this.edges).forEach((p) => {
        p.classList.remove("active");
        p.classList.remove("done");
        p.setAttribute("marker-end", "url(#arrow)");
      });
      this.nodeRunCount = {};
      // restore subtitles
      this.NODES.forEach((cfg) => {
        const node = this.nodesById[cfg.id];
        if (!node) return;
        const sub = node.querySelector(".subtitle");
        if (sub) sub.textContent = cfg.kind.replace("_", " ");
      });
      this.setDecision("");
    }
  }

  window.WorkflowDiagram = WorkflowDiagram;
})();
