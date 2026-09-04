/* TradingCrew — workflow diagram (SVG nodes + edges).
 *
 * LAYOUT
 * ------
 * v3 stock is a LangGraph, not a left-to-right CrewAI pipe.  The stock
 * canvas is a two-band flowchart sized to Fit at ~100% in the workbench
 * (~1020×560), not a 1560px strip that collapses to ~30% zoom.
 * Research: Playbook → Blackboard, analysts into the board, Bull/Bear
 * under it. Decision: RM → Trader → risk → QR/Compliance (parallel)
 * → PM → Decision. Cycles: debate, evidence, Neutral→Trader reopen.
 * Reviewers sit on the same Decision row as risk (dimmed when off).
 * Commodity uses the same two-band canvas (7 futures analysts, QR and
 * Compliance stay on the decision spine).
 *
 *
 * Three layouts ship with the diagram:
 *   - stock                 : LangGraph two-band (fan-out, blackboard, cycles)
 *   - stock_deterministic   : exact v2 LTR crew (Analyst team → Researcher
 *                             team → Trader → Risk → Compliance + PM → Decision)
 *   - commodity             : 17-agent futures two-band crew
 * The right layout is picked from ``assetClass`` + ``layoutId`` (app.js
 * remounts when Settings ``run_mode`` changes). Node IDs MUST match the
 * role strings the runner emits.
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
    "Conductor": "sidebar.run_mode",
    "Investigator": "agent.news_analyst",
    "News Analyst": "agent.news_analyst",
    "Fundamentals Analyst": "agent.fundamentals_analyst",
    "Reports Analyst": "agent.reports_analyst",
    "Macro Analyst": "agent.macro_analyst",
    "Commodity Market Analyst": "agent.market_analyst",
    "Bullish Futures Researcher": "agent.bull_researcher",
    "Bearish Futures Researcher": "agent.bear_researcher",
    "Futures Research Manager": "agent.research_manager",
    "Senior Futures Trader": "agent.trader",
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
    router: "#e879f9",
    board: "#38bdf8",
    trader: "#a855f7",
    risk_a: "#ef4444",
    risk_n: "#f59e0b",
    risk_c: "#3b82f6",
    decision: "#facc15",
  };

  // ====================== Stock layout (LangGraph, v3) ========================
  // Two bands so Fit stays readable (~1020px, not a 1560px LTR strip).
  // Idle edges are the live spine + three labeled cycles.  Fan-out /
  // extra-debate / advisory edges are drawn when the runner emits them.

  const STOCK_WIDTH = 1020;
  const STOCK_HEIGHT = 500;
  const STOCK_AX = [188, 348, 508, 668, 828];
  const STOCK_AY = 40;
  const STOCK_BOARD_X = 508;
  const STOCK_BOARD_Y = 124;
  const STOCK_PLAY_Y = 124;
  const STOCK_DEBATE_Y = 212;
  const STOCK_DEC_Y = 368;
  const STOCK_RISK_X = 336;
  const STOCK_REV_X = 508;

  const STOCK_NODES = [
    { id: "Playbook Router",           label: "Playbook",        kind: "router",   x: 16,              y: STOCK_PLAY_Y },
    { id: "Market Analyst",            label: "Market + Quant",  kind: "analyst",  x: STOCK_AX[0],     y: STOCK_AY, hasTools: "market_analyst" },
    { id: "News Analyst",              label: "News + Social",   kind: "analyst",  x: STOCK_AX[1],     y: STOCK_AY, hasTools: "news_analyst" },
    { id: "Fundamentals Analyst",      label: "Fund. + Sector",  kind: "analyst",  x: STOCK_AX[2],     y: STOCK_AY, hasTools: "fundamentals_analyst" },
    { id: "Reports Analyst",           label: "Reports/Filings", kind: "analyst",  x: STOCK_AX[3],     y: STOCK_AY, hasTools: "reports_analyst" },
    { id: "Macro Analyst",             label: "Macro + Geo",     kind: "analyst",  x: STOCK_AX[4],     y: STOCK_AY, hasTools: "macro_analyst" },
    { id: "Shared State",              label: "Blackboard",      kind: "board",    x: STOCK_BOARD_X,   y: STOCK_BOARD_Y },
    { id: "Bullish Researcher",        label: "Bullish",         kind: "bull",     x: 428,             y: STOCK_DEBATE_Y, hasTools: "bull_researcher" },
    { id: "Bearish Researcher",        label: "Bearish",         kind: "bear",     x: 588,             y: STOCK_DEBATE_Y, hasTools: "bear_researcher" },
    { id: "Research Manager",          label: "Research Mgr",    kind: "manager",  x: 16,              y: STOCK_DEC_Y, hasTools: "research_manager" },
    { id: "Trader",                    label: "Trader",          kind: "trader",   x: 172,             y: STOCK_DEC_Y, hasTools: "trader" },
    { id: "Aggressive Risk Analyst",   label: "Aggressive",      kind: "risk_a",   x: STOCK_RISK_X,    y: 312 },
    { id: "Neutral Risk Analyst",      label: "Neutral",         kind: "risk_n",   x: STOCK_RISK_X,    y: STOCK_DEC_Y },
    { id: "Conservative Risk Analyst", label: "Conservative",    kind: "risk_c",   x: STOCK_RISK_X,    y: 424 },
    { id: "Quality Reviewer",          label: "Quality Review",  kind: "reviewer", x: STOCK_REV_X,     y: 312, dimmed: true },
    { id: "Compliance Officer",        label: "Compliance",      kind: "reviewer", x: STOCK_REV_X,     y: 424, dimmed: true },
    { id: "Portfolio Manager",         label: "Portfolio Mgr",   kind: "manager",  x: 688,             y: STOCK_DEC_Y },
    { id: "decision",                  label: "Decision",        kind: "decision", x: 856,             y: STOCK_DEC_Y },
  ];

  const STOCK_EDGES = [
    { from: "Playbook Router", to: "Shared State", kind: "forward", route: "right" },
    { from: "Market Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "News Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Fundamentals Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Reports Analyst", to: "Shared State", kind: "optional", route: "down" },
    { from: "Macro Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Shared State", to: "Bullish Researcher", kind: "forward", route: "down" },
    { from: "Shared State", to: "Bearish Researcher", kind: "forward", route: "down" },
    { from: "Bullish Researcher", to: "Bearish Researcher", kind: "cycle", route: "right", label: "debate" },
    { from: "Bearish Researcher", to: "Shared State", kind: "cycle", route: "up", label: "evidence" },
    { from: "Bearish Researcher", to: "Research Manager", kind: "forward", route: "down-rail" },
    { from: "Research Manager", to: "Trader", kind: "forward", route: "right" },
    { from: "Trader", to: "Aggressive Risk Analyst", kind: "forward" },
    { from: "Trader", to: "Neutral Risk Analyst", kind: "forward", route: "right" },
    { from: "Trader", to: "Conservative Risk Analyst", kind: "forward" },
    { from: "Aggressive Risk Analyst", to: "Neutral Risk Analyst", kind: "forward", route: "down" },
    { from: "Conservative Risk Analyst", to: "Neutral Risk Analyst", kind: "forward", route: "up" },
    { from: "Neutral Risk Analyst", to: "Quality Reviewer", kind: "optional", route: "right" },
    { from: "Neutral Risk Analyst", to: "Compliance Officer", kind: "optional", route: "right" },
    { from: "Quality Reviewer", to: "Portfolio Manager", kind: "optional", route: "right" },
    { from: "Compliance Officer", to: "Portfolio Manager", kind: "optional", route: "right" },
    { from: "Neutral Risk Analyst", to: "Portfolio Manager", kind: "forward", route: "right" },
    { from: "Neutral Risk Analyst", to: "Trader", kind: "cycle", route: "left-low", label: "reopen" },
    { from: "Portfolio Manager", to: "decision", kind: "forward", route: "right" },
  ];

  const STOCK_GROUPS = [
    { label: "1 · Research · LangGraph fan-out + blackboard", x: 16, y: 10, w: 988, h: 18, caption: true },
    { label: "2 · Decision · evidence / reopen cycles", x: 16, y: 292, w: 988, h: 18, caption: true },
  ];

  // Deterministic = exact v2 LTR crew (port 8001). One column per stage;
  // parallel desks (analysts, Bull/Bear, risk trio) stack in their column.
  const DET_WIDTH = 1540;
  const DET_HEIGHT = 400;
  const DET_ANALYST_ROW_GAP = 68;
  const DET_ANALYST_Y0 = 25;
  const DET_ANALYST_X = 20;
  const DET_COL_BULL_BEAR = 325;
  const DET_COL_RES_MGR   = 475;
  const DET_COL_Q_REV     = 625;
  const DET_COL_TRADER    = 775;
  const DET_COL_RISK      = 925;
  const DET_COL_COMP      = 1075;
  const DET_COL_PM        = 1225;
  const DET_COL_DECISION  = 1375;
  const DET_MID_Y = 150;

  const DET_NODES = [
    { id: "Market Analyst",            label: "Market + Quant",  kind: "analyst",  x: DET_ANALYST_X, y: DET_ANALYST_Y0 + 0 * DET_ANALYST_ROW_GAP, hasTools: "market_analyst" },
    { id: "News Analyst",              label: "News + Social",   kind: "analyst",  x: DET_ANALYST_X, y: DET_ANALYST_Y0 + 1 * DET_ANALYST_ROW_GAP, hasTools: "news_analyst" },
    { id: "Fundamentals Analyst",      label: "Fund. + Sector",  kind: "analyst",  x: DET_ANALYST_X, y: DET_ANALYST_Y0 + 2 * DET_ANALYST_ROW_GAP, hasTools: "fundamentals_analyst" },
    { id: "Reports Analyst",           label: "Reports/Filings", kind: "analyst",  x: DET_ANALYST_X, y: DET_ANALYST_Y0 + 3 * DET_ANALYST_ROW_GAP, hasTools: "reports_analyst" },
    { id: "Macro Analyst",             label: "Macro + Geo",     kind: "analyst",  x: DET_ANALYST_X, y: DET_ANALYST_Y0 + 4 * DET_ANALYST_ROW_GAP, hasTools: "macro_analyst" },
    { id: "Bullish Researcher",        label: "Bullish",         kind: "bull",     x: DET_COL_BULL_BEAR, y: 80, hasTools: "bull_researcher" },
    { id: "Bearish Researcher",        label: "Bearish",         kind: "bear",     x: DET_COL_BULL_BEAR, y: 220, hasTools: "bear_researcher" },
    { id: "Research Manager",          label: "Research Mgr",    kind: "manager",  x: DET_COL_RES_MGR, y: DET_MID_Y, hasTools: "research_manager" },
    { id: "Quality Reviewer",          label: "Quality Review",  kind: "reviewer", x: DET_COL_Q_REV, y: DET_MID_Y, dimmed: true },
    { id: "Trader",                    label: "Trader",          kind: "trader",   x: DET_COL_TRADER, y: DET_MID_Y, hasTools: "trader" },
    { id: "Aggressive Risk Analyst",   label: "Aggressive",      kind: "risk_a",   x: DET_COL_RISK, y: 60 },
    { id: "Neutral Risk Analyst",      label: "Neutral",         kind: "risk_n",   x: DET_COL_RISK, y: 150 },
    { id: "Conservative Risk Analyst", label: "Conservative",    kind: "risk_c",   x: DET_COL_RISK, y: 240 },
    { id: "Compliance Officer",        label: "Compliance",      kind: "reviewer", x: DET_COL_COMP, y: DET_MID_Y, dimmed: true },
    { id: "Portfolio Manager",         label: "Portfolio Mgr",   kind: "manager",  x: DET_COL_PM, y: DET_MID_Y },
    { id: "decision",                  label: "Decision",        kind: "decision", x: DET_COL_DECISION, y: DET_MID_Y },
  ];

  const DET_EDGES = [
    ["Market Analyst", "Bullish Researcher"],
    ["News Analyst", "Bullish Researcher"],
    ["Fundamentals Analyst", "Bullish Researcher"],
    ["Reports Analyst", "Bullish Researcher"],
    ["Macro Analyst", "Bullish Researcher"],
    ["Market Analyst", "Bearish Researcher"],
    ["News Analyst", "Bearish Researcher"],
    ["Fundamentals Analyst", "Bearish Researcher"],
    ["Reports Analyst", "Bearish Researcher"],
    ["Macro Analyst", "Bearish Researcher"],
    ["Bullish Researcher", "Research Manager"],
    ["Bearish Researcher", "Research Manager"],
    ["Research Manager", "Trader"],
    ["Research Manager", "Quality Reviewer"],
    ["Quality Reviewer", "Portfolio Manager"],
    ["Trader", "Aggressive Risk Analyst"],
    ["Trader", "Neutral Risk Analyst"],
    ["Trader", "Conservative Risk Analyst"],
    ["Aggressive Risk Analyst", "Compliance Officer"],
    ["Neutral Risk Analyst", "Compliance Officer"],
    ["Conservative Risk Analyst", "Compliance Officer"],
    ["Aggressive Risk Analyst", "Portfolio Manager"],
    ["Neutral Risk Analyst", "Portfolio Manager"],
    ["Conservative Risk Analyst", "Portfolio Manager"],
    ["Compliance Officer", "Portfolio Manager"],
    ["Portfolio Manager", "decision"],
  ];

  const DET_GROUPS = [
    { label: "Analyst team",     x: 12,   y: 6,   w: 156, h: 360 },
    { label: "Researcher team",  x: 317,  y: 70,  w: 456, h: 200 },
    { label: "Trader",           x: 767,  y: 140, w: 156, h: 80 },
    { label: "Risk management",  x: 917,  y: 50,  w: 156, h: 240 },
    { label: "Compliance + PM",  x: 1067, y: 140, w: 306, h: 80 },
    { label: "Decision",         x: 1367, y: 140, w: 156, h: 80 },
  ];

  // ====================== Commodity layout (17-agent debate) ==================
  // Same two-band canvas as stock so Fit stays readable.  ID strings MUST
  // match the YAML ``role:`` fields in commodity_crew/config/agents.yaml.
  // QR and Compliance stay on the decision spine (they are in this crew).
  // Personas reused from the stock crew keep the same role strings.

  const COMM_WIDTH = 1032;
  const COMM_HEIGHT = 560;
  const COMM_AX = [12, 156, 300, 444, 588, 732, 876];
  const COMM_AY = 40;
  const COMM_DEBATE_Y = 124;
  const COMM_DEC_Y = 368;
  const COMM_RISK_X = 448;
  const COMM_DX = [16, 160, 304, 448, 592, 736, 880];

  const COMM_NODES = [
    { id: "Commodity Market Analyst",           label: "Market",         kind: "analyst",  x: COMM_AX[0], y: COMM_AY, hasTools: "market_analyst" },
    { id: "Term Structure Analyst",             label: "Curve",          kind: "analyst",  x: COMM_AX[1], y: COMM_AY, hasTools: "curve_analyst" },
    { id: "Inventories & Stocks Analyst",       label: "Inventories",    kind: "analyst",  x: COMM_AX[2], y: COMM_AY, hasTools: "inventories_analyst" },
    { id: "Supply & Demand Analyst",            label: "Supply / Demand",kind: "analyst",  x: COMM_AX[3], y: COMM_AY, hasTools: "supply_demand_analyst" },
    { id: "Macro Analyst",                      label: "Macro",          kind: "analyst",  x: COMM_AX[4], y: COMM_AY, hasTools: "macro_analyst" },
    { id: "Geopolitical & Supply-Risk Analyst", label: "Geopolitical",   kind: "analyst",  x: COMM_AX[5], y: COMM_AY, hasTools: "geopolitical_analyst" },
    { id: "Positioning & Seasonality Quant",    label: "Positioning",    kind: "analyst",  x: COMM_AX[6], y: COMM_AY, hasTools: "quant_analyst" },
    { id: "Bullish Futures Researcher",         label: "Bullish",        kind: "bull",     x: 364, y: COMM_DEBATE_Y },
    { id: "Bearish Futures Researcher",         label: "Bearish",        kind: "bear",     x: 520, y: COMM_DEBATE_Y },
    { id: "Futures Research Manager",           label: "Research Mgr",   kind: "manager",  x: COMM_DX[0], y: COMM_DEC_Y },
    { id: "Quality Reviewer",                   label: "Quality Review", kind: "reviewer", x: COMM_DX[1], y: COMM_DEC_Y },
    { id: "Senior Futures Trader",              label: "Futures Trader", kind: "trader",   x: COMM_DX[2], y: COMM_DEC_Y, hasTools: "trader" },
    { id: "Aggressive Risk Analyst",            label: "Aggressive",     kind: "risk_a",   x: COMM_RISK_X, y: 312 },
    { id: "Neutral Risk Analyst",               label: "Neutral",        kind: "risk_n",   x: COMM_RISK_X, y: COMM_DEC_Y },
    { id: "Conservative Risk Analyst",          label: "Conservative",   kind: "risk_c",   x: COMM_RISK_X, y: 424 },
    { id: "Compliance Officer",                 label: "Compliance",     kind: "reviewer", x: COMM_DX[4], y: COMM_DEC_Y, hasTools: "compliance_officer" },
    { id: "Portfolio Manager",                  label: "Portfolio Mgr",  kind: "manager",  x: COMM_DX[5], y: COMM_DEC_Y },
    { id: "decision",                           label: "Decision",       kind: "decision", x: COMM_DX[6], y: COMM_DEC_Y },
  ];

  const COMM_ANALYSTS = [
    "Commodity Market Analyst",
    "Term Structure Analyst",
    "Inventories & Stocks Analyst",
    "Supply & Demand Analyst",
    "Macro Analyst",
    "Geopolitical & Supply-Risk Analyst",
    "Positioning & Seasonality Quant",
  ];

  const COMM_EDGES = [
    ...COMM_ANALYSTS.map((id) => ({ from: id, to: "Bullish Futures Researcher", kind: "forward", route: "down" })),
    ...COMM_ANALYSTS.map((id) => ({ from: id, to: "Bearish Futures Researcher", kind: "forward", route: "down" })),
    { from: "Bullish Futures Researcher", to: "Bearish Futures Researcher", kind: "cycle", route: "right", label: "debate" },
    { from: "Bullish Futures Researcher", to: "Futures Research Manager", kind: "forward", route: "down-rail" },
    { from: "Bearish Futures Researcher", to: "Futures Research Manager", kind: "forward", route: "down-rail" },
    { from: "Futures Research Manager", to: "Quality Reviewer", kind: "forward", route: "right" },
    { from: "Quality Reviewer", to: "Senior Futures Trader", kind: "forward", route: "right" },
    { from: "Senior Futures Trader", to: "Aggressive Risk Analyst", kind: "forward" },
    { from: "Senior Futures Trader", to: "Neutral Risk Analyst", kind: "forward", route: "right" },
    { from: "Senior Futures Trader", to: "Conservative Risk Analyst", kind: "forward" },
    { from: "Aggressive Risk Analyst", to: "Neutral Risk Analyst", kind: "forward", route: "down" },
    { from: "Conservative Risk Analyst", to: "Neutral Risk Analyst", kind: "forward", route: "up" },
    { from: "Neutral Risk Analyst", to: "Compliance Officer", kind: "forward", route: "right" },
    { from: "Compliance Officer", to: "Portfolio Manager", kind: "forward", route: "right" },
    { from: "Portfolio Manager", to: "decision", kind: "forward", route: "right" },
  ];

  const COMM_GROUPS = [
    { label: "1 · Research", x: 16, y: 10, w: 988, h: 18, caption: true },
    { label: "2 · Decision", x: 16, y: 292, w: 988, h: 18, caption: true },
  ];

  const HARNESS_NODES = [
    { id: "Conductor",                 label: "Conductor",       kind: "router",   x: 16,              y: STOCK_PLAY_Y },
    { id: "Market Analyst",            label: "Market + Quant",  kind: "analyst",  x: STOCK_AX[0],     y: STOCK_AY, hasTools: "market_analyst" },
    { id: "News Analyst",              label: "News + Social",   kind: "analyst",  x: STOCK_AX[1],     y: STOCK_AY, hasTools: "news_analyst" },
    { id: "Fundamentals Analyst",      label: "Fund. + Sector",  kind: "analyst",  x: STOCK_AX[2],     y: STOCK_AY, hasTools: "fundamentals_analyst" },
    { id: "Reports Analyst",           label: "Reports/Filings", kind: "analyst",  x: STOCK_AX[3],     y: STOCK_AY, hasTools: "reports_analyst" },
    { id: "Macro Analyst",             label: "Macro + Geo",     kind: "analyst",  x: STOCK_AX[4],     y: STOCK_AY, hasTools: "macro_analyst" },
    { id: "Shared State",              label: "Transcript",      kind: "board",    x: STOCK_BOARD_X,   y: STOCK_BOARD_Y },
    { id: "Investigator",              label: "Investigator",    kind: "analyst",  x: 268,             y: STOCK_DEBATE_Y },
    { id: "Bullish Researcher",        label: "Bullish",         kind: "bull",     x: 428,             y: STOCK_DEBATE_Y, hasTools: "bull_researcher" },
    { id: "Bearish Researcher",        label: "Bearish",         kind: "bear",     x: 588,             y: STOCK_DEBATE_Y, hasTools: "bear_researcher" },
    { id: "Research Manager",          label: "Research Mgr",    kind: "manager",  x: 16,              y: STOCK_DEC_Y, hasTools: "research_manager" },
    { id: "Trader",                    label: "Trader",          kind: "trader",   x: 172,             y: STOCK_DEC_Y, hasTools: "trader" },
    { id: "Aggressive Risk Analyst",   label: "Aggressive",      kind: "risk_a",   x: STOCK_RISK_X,    y: 312 },
    { id: "Neutral Risk Analyst",      label: "Neutral",         kind: "risk_n",   x: STOCK_RISK_X,    y: STOCK_DEC_Y },
    { id: "Conservative Risk Analyst", label: "Conservative",    kind: "risk_c",   x: STOCK_RISK_X,    y: 424 },
    { id: "Portfolio Manager",         label: "Portfolio Mgr",   kind: "manager",  x: 688,             y: STOCK_DEC_Y },
    { id: "decision",                  label: "Decision",        kind: "decision", x: 856,             y: STOCK_DEC_Y },
  ];
  const HARNESS_EDGES = [
    { from: "Conductor", to: "Shared State", kind: "forward", route: "right" },
    { from: "Conductor", to: "Investigator", kind: "optional", route: "right" },
    { from: "Market Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "News Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Fundamentals Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Reports Analyst", to: "Shared State", kind: "optional", route: "down" },
    { from: "Macro Analyst", to: "Shared State", kind: "forward", route: "down" },
    { from: "Investigator", to: "Shared State", kind: "forward", route: "right" },
    { from: "Shared State", to: "Bullish Researcher", kind: "optional", route: "down" },
    { from: "Shared State", to: "Bearish Researcher", kind: "optional", route: "down" },
    { from: "Bullish Researcher", to: "Bearish Researcher", kind: "cycle", route: "right", label: "debate" },
    { from: "Research Manager", to: "Trader", kind: "optional", route: "right" },
    { from: "Trader", to: "Neutral Risk Analyst", kind: "optional", route: "right" },
    { from: "Aggressive Risk Analyst", to: "Conservative Risk Analyst", kind: "cycle", route: "right", label: "risk" },
    { from: "Conservative Risk Analyst", to: "Neutral Risk Analyst", kind: "cycle", route: "right" },
    { from: "Portfolio Manager", to: "decision", kind: "forward", route: "right" },
  ];
  const HARNESS_GROUPS = [
    { label: "1 · Harness · conductor picks tools and sub-agents", x: 16, y: 10, w: 988, h: 18, caption: true },
    { label: "2 · Decision · PM writes the primary ticket", x: 16, y: 292, w: 988, h: 18, caption: true },
  ];

  const LAYOUTS = {
    stock:                 { width: STOCK_WIDTH, height: STOCK_HEIGHT, NODES: STOCK_NODES, EDGES: STOCK_EDGES, GROUPS: STOCK_GROUPS },
    stock_deterministic:   { width: DET_WIDTH,   height: DET_HEIGHT,   NODES: DET_NODES,   EDGES: DET_EDGES,   GROUPS: DET_GROUPS },
    stock_harness:         { width: STOCK_WIDTH, height: STOCK_HEIGHT, NODES: HARNESS_NODES, EDGES: HARNESS_EDGES, GROUPS: HARNESS_GROUPS },
    commodity:             { width: COMM_WIDTH,  height: COMM_HEIGHT,  NODES: COMM_NODES,  EDGES: COMM_EDGES,  GROUPS: COMM_GROUPS },
  };

  function el(tag, attrs, ...children) {
    const isSvg = ["svg", "path", "rect", "circle", "line", "polygon", "defs", "marker", "g", "text"].includes(tag);
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
      const layoutId = opts.layoutId
        || (assetClass === "commodity" ? "commodity" : "stock");
      this.assetClass = assetClass;
      this._applyLayout(layoutId);
      this.nodesById = {};
      this.toolStrips = {};
      this.edges = {};
      this.dynamicEdges = [];
      this.edgeLive = {};
      this.edgeSpecs = {};
      this.selected = null;
      this.selectListeners = [];
      this.scaleListeners = [];    // notified on zoom changes (UI percentage)
      this.nodeRunCount = {};      // role -> # task_completed events received
      this.qualityReview = false;
      this.compliance = false;
      this.runMode = "autonomous";
      this.orchestrator = assetClass === "commodity" ? "crewai" : "langgraph";
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

    _applyLayout(layoutId) {
      const layout = LAYOUTS[layoutId] || LAYOUTS.stock;
      this.layoutId = LAYOUTS[layoutId] ? layoutId : "stock";
      this.LAYOUT_WIDTH = layout.width;
      this.LAYOUT_HEIGHT = layout.height;
      this.NODES = layout.NODES;
      this.EDGES = layout.EDGES;
      this.GROUPS = layout.GROUPS;
    }

    setLayout(layoutId) {
      if (!layoutId || layoutId === this.layoutId) return false;
      this._applyLayout(layoutId);
      this.nodesById = {};
      this.toolStrips = {};
      this.edges = {};
      this.dynamicEdges = [];
      this.edgeLive = {};
      this.edgeSpecs = {};
      this.nodeRunCount = {};
      this.render();
      return true;
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
      if (this.layoutId === "stock" || this.layoutId === "commodity") {
        inner.classList.add("flow-layout-banded");
      }
      if (this.assetClass === "commodity") inner.classList.add("flow-layout-commodity");
      if (this.layoutId === "stock_deterministic") inner.classList.add("flow-layout-v2");
      this.container.appendChild(inner);

      // Group rectangles (stock uses caption-only stage labels, no boxes)
      this.groupEls = [];
      this.GROUPS.forEach((g) => {
        const grp = document.createElement("div");
        grp.className = g.caption ? "flow-group caption-only" : "flow-group";
        grp.style.left = g.x + "px";
        grp.style.top = g.y + "px";
        grp.style.width = g.w + "px";
        grp.style.height = g.h + "px";
        const lbl = document.createElement("div");
        lbl.className = "group-label";
        lbl.textContent = g.label;
        grp.appendChild(lbl);
        inner.appendChild(grp);
        this.groupEls.push({ cfg: g, el: grp, label: lbl });
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
        if (cfg.dimmed) node.classList.add("flow-node-dim");
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
        if (cfg.dimmed) subtitle.textContent = "off";
        else if (cfg.kind === "router") subtitle.textContent = "router";
        else if (cfg.kind === "board") subtitle.textContent = "shared state";
        else subtitle.textContent = cfg.kind.replace("_", " ");
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
          if (this.layoutId === "stock_deterministic") {
            strip.style.left = cfg.x + "px";
            strip.style.top = (cfg.y + 40) + "px";
            inner.appendChild(strip);
          } else {
            node.appendChild(strip);
          }
          this.toolStrips[cfg.id] = { el: strip, agentKey: cfg.hasTools, chips: {}, counters: {} };
        }
      });

      // Edges (after nodes so we know their actual heights)
      this.renderEdges();

      this.fit();
      this._bindViewportOnce();
    }

    _bindViewportOnce() {
      if (this._viewportBound) return;
      this._viewportBound = true;
      window.addEventListener("resize", () => this.refit());
      if (typeof ResizeObserver !== "undefined") {
        try {
          if (this._resizeObserver) this._resizeObserver.disconnect();
          this._resizeObserver = new ResizeObserver(() => this.refit());
          const vp = this.container.parentElement || this.container;
          this._resizeObserver.observe(vp);
        } catch (_) { /* refit() still recovers on window resize */ }
      }
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
    _normEdge(e) {
      if (Array.isArray(e)) return { from: e[0], to: e[1], kind: e[2] || "forward" };
      return {
        from: e.from,
        to: e.to,
        kind: e.kind || "forward",
        route: e.route,
        label: e.label,
      };
    }

    _lookupEdgeSpec(from, to, kind) {
      const found = (this.EDGES || []).map((raw) => this._normEdge(raw))
        .find((e) => e.from === from && e.to === to);
      return found || { from, to, kind: kind || "forward" };
    }

    edgeAnchors(a, b, route) {
      const ax = parseFloat(a.style.left);
      const ay = parseFloat(a.style.top);
      const aw = a.offsetWidth;
      const ah = a.offsetHeight;
      const bx = parseFloat(b.style.left);
      const by = parseFloat(b.style.top);
      const bw = b.offsetWidth;
      const bh = b.offsetHeight;
      const aCx = ax + aw / 2;
      const aCy = ay + ah / 2;
      const bCx = bx + bw / 2;
      const bCy = by + bh / 2;

      if (this.layoutId === "stock_deterministic") {
        return { x1: ax + aw, y1: aCy, x2: bx, y2: bCy };
      }
      if (this.assetClass === "stock" || this.assetClass === "commodity") {
        const forced = route || "";
        if (forced === "down") {
          return { x1: aCx, y1: ay + ah, x2: bCx, y2: by, elbow: true, vertFirst: true };
        }
        if (forced === "down-rail") {
          return { x1: aCx, y1: ay + ah, x2: bCx, y2: by, elbow: true, railY: ay + ah + 16 };
        }
        if (forced === "up") {
          return { x1: aCx, y1: ay, x2: bCx, y2: by + bh, elbow: true, vertFirst: true };
        }
        if (forced === "right") {
          return { x1: ax + aw, y1: aCy, x2: bx, y2: bCy, elbow: true };
        }
        if (forced === "left") {
          return { x1: ax, y1: aCy, x2: bx + bw, y2: bCy, elbow: true };
        }
        if (forced === "left-low") {
          return { x1: ax, y1: aCy + 10, x2: bx + bw, y2: bCy + 10, elbow: true };
        }
        const dx = bCx - aCx;
        const dy = bCy - aCy;
        if (Math.abs(dx) >= Math.abs(dy)) {
          if (dx >= 0) return { x1: ax + aw, y1: aCy, x2: bx, y2: bCy, elbow: true };
          return { x1: ax, y1: aCy, x2: bx + bw, y2: bCy, elbow: true };
        }
        if (dy >= 0) return { x1: aCx, y1: ay + ah, x2: bCx, y2: by, elbow: true, vertFirst: true };
        return { x1: aCx, y1: ay, x2: bCx, y2: by + bh, elbow: true, vertFirst: true };
      }

      if (Math.abs(bx - ax) < 24) {
        return { x1: ax + aw, y1: ay + ah / 2, x2: bx + bw, y2: by + bh / 2, loop: "right" };
      }
      if (bx + 8 < ax) {
        return { x1: ax, y1: ay + ah / 2, x2: bx + bw, y2: by + bh / 2, reverse: true };
      }
      return { x1: ax + aw, y1: ay + ah / 2, x2: bx, y2: by + bh / 2 };
    }

    _pathForAnchors(anchors) {
      const { x1, y1, x2, y2, reverse, loop, elbow, vertFirst } = anchors;
      if (elbow) {
        if (anchors.railY != null) {
          return `M ${x1} ${y1} L ${x1} ${anchors.railY} L ${x2} ${anchors.railY} L ${x2} ${y2}`;
        }
        if (anchors.drop) {
          const y = Math.max(y1, y2) + anchors.drop;
          return `M ${x1} ${y1} L ${x1} ${y} L ${x2} ${y} L ${x2} ${y2}`;
        }
        if (Math.abs(x1 - x2) < 2 || Math.abs(y1 - y2) < 2) {
          return `M ${x1} ${y1} L ${x2} ${y2}`;
        }
        if (vertFirst) {
          const my = (y1 + y2) / 2;
          return `M ${x1} ${y1} L ${x1} ${my} L ${x2} ${my} L ${x2} ${y2}`;
        }
        const mx = (x1 + x2) / 2;
        return `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`;
      }
      if (loop === "right") {
        const bulge = 42;
        return `M ${x1} ${y1} C ${x1 + bulge} ${y1}, ${x2 + bulge} ${y2}, ${x2} ${y2}`;
      }
      if (reverse) {
        const lift = Math.max(36, Math.abs(y2 - y1) * 0.35 + 28);
        return `M ${x1} ${y1} C ${x1 - 48} ${y1 - lift}, ${x2 + 48} ${y2 - lift}, ${x2} ${y2}`;
      }
      return curvePath(x1, y1, x2, y2);
    }

    _styleIdleEdge(path, kind) {
      path.classList.remove("active", "done", "skipped", "cycled", "cycle-idle", "optional-idle");
      if (kind === "cycle") path.classList.add("cycle-idle");
      if (kind === "optional") path.classList.add("optional-idle");
      path.setAttribute("marker-end", "url(#arrow)");
    }

    _drawEdge(spec) {
      const from = spec.from;
      const to = spec.to;
      const kind = spec.kind || "forward";
      const a = this.nodesById[from];
      const b = this.nodesById[to];
      if (!a || !b) return null;
      const anchors = this.edgeAnchors(a, b, spec.route);
      const path = el("path", {
        d: this._pathForAnchors(anchors),
        class: "flow-edge",
        "marker-end": "url(#arrow)",
      });
      this._styleIdleEdge(path, kind);
      this.edgesGroup.appendChild(path);
      if (spec.label) {
        const lx = (anchors.x1 + anchors.x2) / 2;
        const ly = (anchors.y1 + anchors.y2) / 2 - 7;
        const text = el("text", {
          class: "flow-edge-label",
          x: lx,
          y: ly,
          "text-anchor": "middle",
        });
        text.textContent = spec.label;
        this.edgesGroup.appendChild(text);
      }
      return path;
    }

    renderEdges() {
      this.edges = {};
      this.dynamicEdges = [];
      this.edgeSpecs = this.edgeSpecs || {};
      this.edgesGroup.innerHTML = "";
      this.EDGES.forEach((raw) => {
        const e = this._normEdge(raw);
        const path = this._drawEdge(e);
        if (path) {
          this.edges[`${e.from}->${e.to}`] = path;
          this.edgeSpecs[`${e.from}->${e.to}`] = e;
        }
      });
      Object.entries(this.edgeLive || {}).forEach(([key, live]) => {
        const [from, to] = key.split("->");
        if (!this.edges[key]) {
          const spec = this._lookupEdgeSpec(from, to, live === "cycled" ? "cycle" : "forward");
          const path = this._drawEdge(spec);
          if (path) {
            this.edges[key] = path;
            this.edgeSpecs[key] = spec;
            this.dynamicEdges.push(key);
          }
        }
        if (this.edges[key]) this._applyLive(this.edges[key], live, from);
      });
    }

    _applyLive(path, kind, from) {
      path.classList.remove("active", "done", "skipped", "cycled");
      if (kind === "skipped") {
        path.classList.add("skipped");
        const node = from && this.nodesById[from];
        if (node) node.dataset.state = "skipped";
      } else if (kind === "cycled") {
        path.classList.add("cycled", "active");
        path.setAttribute("marker-end", "url(#arrow-active)");
      } else {
        path.classList.add("active");
        path.setAttribute("marker-end", "url(#arrow-active)");
      }
    }

    activateEdge(from, to, kind) {
      kind = kind || "activated";
      const key = `${from}->${to}`;
      this.edgeLive[key] = kind;
      let path = this.edges[key];
      if (!path) {
        const spec = this._lookupEdgeSpec(from, to, kind === "cycled" ? "cycle" : "forward");
        path = this._drawEdge(spec);
        if (!path) return;
        this.edges[key] = path;
        this.edgeSpecs[key] = spec;
        this.dynamicEdges.push(key);
      }
      this._applyLive(path, kind, from);
    }

    setGraphMeta(meta) {
      meta = meta || {};
      if (meta.qualityReview !== undefined) this.qualityReview = !!meta.qualityReview;
      if (meta.compliance !== undefined) this.compliance = !!meta.compliance;
      if (meta.runMode) this.runMode = meta.runMode;
      if (meta.orchestrator) this.orchestrator = meta.orchestrator;
      this._applyReviewerVisibility();
      this._applyOrchestratorChrome();
      const router = this.nodesById["Playbook Router"];
      if (router && this.orchestrator !== "crewai") {
        router.dataset.state = "done";
        const sub = router.querySelector(".subtitle");
        if (sub) sub.textContent = meta.playbookAction || "default";
      }
      if (meta.skipReports && this.orchestrator !== "crewai") {
        this.activateEdge("Playbook Router", "Reports Analyst", "skipped");
        this.activateEdge("Reports Analyst", "Shared State", "skipped");
        const reports = this.nodesById["Reports Analyst"];
        if (reports) reports.dataset.state = "skipped";
      }
    }

    _advisoryOn() {
      return !!(this.qualityReview || this.compliance);
    }

    _applyReviewerVisibility() {
      if (this.assetClass !== "stock") return;
      const flags = {
        "Quality Reviewer": !!this.qualityReview,
        "Compliance Officer": !!this.compliance,
      };
      Object.keys(flags).forEach((id) => {
        const node = this.nodesById[id];
        if (!node) return;
        const on = flags[id];
        node.classList.toggle("flow-node-dim", !on);
        const sub = node.querySelector(".subtitle");
        if (sub) sub.textContent = on ? "advisory" : "off";
      });
      if (this.layoutId === "stock_deterministic") {
        if (this.qualityReview) {
          this.activateEdge("Research Manager", "Quality Reviewer", "activated");
          this.activateEdge("Quality Reviewer", "Portfolio Manager", "activated");
        }
        if (this.compliance) {
          this.activateEdge("Compliance Officer", "Portfolio Manager", "activated");
        }
        return;
      }
      if (this.qualityReview) {
        this.activateEdge("Neutral Risk Analyst", "Quality Reviewer", "activated");
        this.activateEdge("Quality Reviewer", "Portfolio Manager", "activated");
      }
      if (this.compliance) {
        this.activateEdge("Neutral Risk Analyst", "Compliance Officer", "activated");
        this.activateEdge("Compliance Officer", "Portfolio Manager", "activated");
      }
      if (this._advisoryOn()) {
        this.activateEdge("Neutral Risk Analyst", "Portfolio Manager", "skipped");
      }
    }

    _applyOrchestratorChrome() {
      if (this.assetClass !== "stock") return;
      const router = this.nodesById["Playbook Router"];
      if (!router) return;
      const sequential = this.orchestrator === "crewai" || this.runMode === "deterministic";
      router.classList.toggle("flow-node-dim", sequential);
      const sub = router.querySelector(".subtitle");
      if (sub && sequential) sub.textContent = "sequential";
    }

    /**
     * Recompute fitScale to fit the diagram into the current container
     * width *at userZoom = 1*.  Then apply ``fitScale * userZoom``.
     *
     * Called from the resize observer and from the toolbar "Fit" button
     * (which also resets userZoom to 1).
     */
    _viewportBox() {
      const vp = this.container.parentElement;
      const w = (vp && vp.clientWidth) || this.container.clientWidth;
      return { w, h: window.innerHeight * 0.8 };
    }

    _computeFitScale() {
      const { w, h } = this._viewportBox();
      if (w <= 0) return this.fitScale;
      const byW = (w - 8) / this.LAYOUT_WIDTH;
      const byH = (h - 8) / this.LAYOUT_HEIGHT;
      return Math.min(1, byW, byH);
    }

    fit() {
      const next = this._computeFitScale();
      if (!(next > 0)) return;
      this.fitScale = next;
      this.applyScale();
    }

    // Recompute fitScale without touching userZoom (resize-driven path).
    refit() {
      this.fit();
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
    /* Tag the tiles whose desks run ReAct-style planning (plan→act→observe).
     * ``roles`` is the list of node ids from /api/options#react_agents, so the
     * badge reflects the LIVE backend config (empty when planning is off) —
     * never a hardcoded assumption.  Idempotent: safe to call on reload. */
    setReactAgents(roles) {
      const want = new Set(roles || []);
      Object.entries(this.nodesById).forEach(([nodeId, node]) => {
        const existing = node.querySelector(".react-badge");
        if (want.has(nodeId)) {
          if (!existing) {
            const badge = document.createElement("span");
            badge.className = "react-badge";
            badge.textContent = "ReAct";
            badge.title = "Runs a plan → act → observe/replan loop before deciding";
            node.appendChild(badge);
          }
        } else if (existing) {
          existing.remove();
        }
      });
    }

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
      this.qualityReview = false;
      this.compliance = false;
      Object.values(this.nodesById).forEach((n) => (n.dataset.state = "idle"));
      Object.values(this.toolStrips).forEach((s) => {
        s.el.innerHTML = "";
        s.chips = {};
        s.counters = {};
      });
      this.edgeLive = {};
      (this.dynamicEdges || []).forEach((key) => {
        const p = this.edges[key];
        if (p && p.parentNode) p.parentNode.removeChild(p);
        delete this.edges[key];
      });
      this.dynamicEdges = [];
      this.EDGES.forEach((raw) => {
        const e = this._normEdge(raw);
        const p = this.edges[`${e.from}->${e.to}`];
        if (p) this._styleIdleEdge(p, e.kind);
      });
      this.nodeRunCount = {};
      this.NODES.forEach((cfg) => {
        const node = this.nodesById[cfg.id];
        if (!node) return;
        const sub = node.querySelector(".subtitle");
        if (!sub) return;
        if (cfg.dimmed) sub.textContent = "off";
        else if (cfg.kind === "router") sub.textContent = "router";
        else if (cfg.kind === "board") sub.textContent = "shared state";
        else sub.textContent = cfg.kind.replace("_", " ");
      });
      this.setDecision("");
      this._applyReviewerVisibility();
      this._applyOrchestratorChrome();
    }
  }

  window.WorkflowDiagram = WorkflowDiagram;
})();
