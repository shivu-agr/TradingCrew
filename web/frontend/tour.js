/* TradingCrew first-run guided tour.
 *
 * Walks new users through Workflow -> Reports -> Memory -> Backtest ->
 * RL Training -> Help drawer on their first visit.  Gated by
 * `localStorage.tcTourCompleted` so it only fires once.  Re-launchable
 * from the Help drawer footer (TcTour.start()).
 *
 * Implementation: pure DOM, no extra deps.  Each step takes a selector
 * (the element to spotlight), an anchor (the help_content.js entry to
 * mirror), a placement hint, and optional `activeTab` so we can switch
 * to the right tab before showing that step.
 *
 * The overlay div has a "spotlight" cutout achieved via a giant
 * `box-shadow` (the same trick Intro.js uses), so we don't need SVG
 * masks.  See style.css (.tour-spotlight / .tour-card / .tour-overlay).
 */

(function () {
  "use strict";

  const STORAGE_KEY = "tcTourCompleted";

  // Each step lists the *primary* selector (CSS standard) and an
  // optional fallback in case the primary isn't present yet (e.g. the
  // user collapsed the advanced section).  We use real CSS attribute
  // selectors only — no jQuery/Playwright extensions.
  const STEPS = [
    {
      selector: "aside.sidebar .info-i[data-help='sidebar.ticker']",
      fallbackSelector: "aside.sidebar .section",
      title: "Pick a ticker",
      body: "Type a stock symbol or click a preset chip. Non-US tickers like MAZDOCK, HSBA or 0700 auto-resolve to the right exchange (.NS, .L, .HK, etc.).",
      tab: null,
      placement: "right",
    },
    {
      selector: "aside.sidebar .info-i[data-help='sidebar.tools']",
      fallbackSelector: "aside.sidebar .advanced-group",
      title: "Configure tools + dialogue",
      body: "Open Advanced settings to toggle per-agent tools and dial in how many debate / risk rounds the crew runs. More rounds = thorough but more LLM cost.",
      tab: null,
      placement: "right",
    },
    {
      selector: "aside.sidebar button.bg-blue-600, aside.sidebar .sticky.bottom-0 button",
      title: "Launch the crew",
      body: "Hit Run analysis to kick off the 18-agent debate. Cancel any time.",
      tab: null,
      placement: "right",
    },
    {
      selector: "main nav .info-i[data-help='tab.workflow']",
      title: "Workflow tab",
      body: "Live SVG diagram of every agent + the deterministic risk / sizing / execution pipeline that runs after the LLM crew. Edges light up as agents progress.",
      tab: "workflow",
      placement: "bottom",
    },
    {
      selector: "main nav .info-i[data-help='tab.reports']",
      title: "Reports tab",
      body: "Each agent's full markdown report with inline [source: …] tags. Finance jargon (RSI, MACD, Kelly, CVaR, Sharpe) is auto-highlighted — hover the dotted underline for an inline definition.",
      tab: "reports",
      placement: "bottom",
    },
    {
      selector: "main nav .info-i[data-help='tab.memory']",
      title: "Memory tab",
      body: "Audit-grade episodic memory. After a horizon elapses, click Resolve outcomes & reflect to score realised return + alpha and let the LLM write a one-paragraph lesson that future runs retrieve.",
      tab: "memory",
      placement: "bottom",
    },
    {
      selector: "main nav .info-i[data-help='tab.backtest']",
      title: "Backtest tab",
      body: "Walk-forward replay of logged proposals with embargo. The L3 grid search auto-tunes sizing + risk gates and ranks configurations by Deflated Sharpe (selection-bias adjusted).",
      tab: "backtest",
      placement: "bottom",
    },
    {
      selector: "main nav .info-i[data-help='tab.training']",
      title: "RL Training tab",
      body: "L4 — closed-loop reinforcement learning. Agents act, the simulator pays out, gradients flow into the policy. Promote a run to expose it to the LLM crew.",
      tab: "training",
      placement: "bottom",
    },
    {
      selector: ".help-fab",
      title: "Help drawer (Cmd/Ctrl+K)",
      body: "Click the ? button or press Cmd/Ctrl+K any time to search the help. Every (i) icon in the UI deep-links into this drawer.",
      tab: null,
      placement: "left",
    },
  ];

  let _overlayEl = null;
  let _cardEl = null;
  let _spotEl = null;
  let _activeIndex = -1;
  let _getTab = null;
  let _setTab = null;

  function _q(sel) {
    // :contains() is jQuery — fall back to attribute / text matching.
    try { return document.querySelector(sel); } catch (_) {}
    return null;
  }

  function _resolve(step) {
    const node = _q(step.selector) || _q(step.fallbackSelector || "");
    return node;
  }

  function _ensureOverlay() {
    if (_overlayEl) return;
    _overlayEl = document.createElement("div");
    _overlayEl.className = "tour-overlay";
    _spotEl = document.createElement("div");
    _spotEl.className = "tour-spotlight";
    _overlayEl.appendChild(_spotEl);
    _cardEl = document.createElement("div");
    _cardEl.className = "tour-card";
    document.body.appendChild(_overlayEl);
    document.body.appendChild(_cardEl);
  }

  function _renderStep(idx) {
    _activeIndex = idx;
    const step = STEPS[idx];
    if (!step) { _finish(); return; }
    if (step.tab && _setTab) _setTab(step.tab);
    // wait a tick for the new tab to paint.
    setTimeout(() => {
      const target = _resolve(step);
      if (!target) {
        _next(); return;
      }
      const r = target.getBoundingClientRect();
      const pad = 6;
      _spotEl.style.top = (r.top - pad) + "px";
      _spotEl.style.left = (r.left - pad) + "px";
      _spotEl.style.width = (r.width + 2 * pad) + "px";
      _spotEl.style.height = (r.height + 2 * pad) + "px";

      // Card placement.
      const cardW = 320;
      const cardH = 160;
      let cardX, cardY;
      switch (step.placement) {
        case "right":
          cardX = r.right + 16;
          cardY = Math.max(16, r.top);
          break;
        case "left":
          cardX = Math.max(16, r.left - cardW - 16);
          cardY = Math.max(16, r.top);
          break;
        case "bottom":
        default:
          cardX = Math.min(window.innerWidth - cardW - 16, r.left);
          cardY = r.bottom + 12;
          break;
      }
      cardX = Math.max(12, Math.min(cardX, window.innerWidth - cardW - 12));
      cardY = Math.max(12, Math.min(cardY, window.innerHeight - cardH - 12));
      _cardEl.style.left = cardX + "px";
      _cardEl.style.top = cardY + "px";
      _cardEl.innerHTML = `
        <h3>${_esc(step.title)}</h3>
        <div class="tour-card-body">${_esc(step.body)}</div>
        <div class="tour-card-foot">
          <button type="button" class="tour-btn" data-tour="skip">Skip tour</button>
          <button type="button" class="tour-btn" data-tour="prev" ${idx === 0 ? "disabled" : ""}>Back</button>
          <button type="button" class="tour-btn is-primary" data-tour="next">
            ${idx === STEPS.length - 1 ? "Done" : "Next"}
          </button>
        </div>
        <div style="margin-top:6px; font-size:10.5px; color:#64748b; text-align:right;">
          ${idx + 1} / ${STEPS.length}
        </div>
      `;
      _cardEl.querySelectorAll("button[data-tour]").forEach((b) => {
        b.addEventListener("click", () => {
          const a = b.dataset.tour;
          if (a === "skip") _finish();
          else if (a === "prev") _renderStep(Math.max(0, idx - 1));
          else _next();
        });
      });
    }, step.tab ? 80 : 0);
  }

  function _next() {
    if (_activeIndex >= STEPS.length - 1) { _finish(); return; }
    _renderStep(_activeIndex + 1);
  }

  function _finish() {
    if (_overlayEl) _overlayEl.remove();
    if (_cardEl) _cardEl.remove();
    _overlayEl = _cardEl = _spotEl = null;
    _activeIndex = -1;
    try { localStorage.setItem(STORAGE_KEY, "1"); } catch (_) {}
  }

  function _esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function start(opts) {
    if (opts && opts.getTab) _getTab = opts.getTab;
    if (opts && opts.setTab) _setTab = opts.setTab;
    _ensureOverlay();
    _renderStep(0);
  }

  function maybeAutoStart(getTab, setTab) {
    try {
      if (localStorage.getItem(STORAGE_KEY) === "1") return;
    } catch (_) { /* private-browsing mode — just run the tour */ }
    _getTab = getTab;
    _setTab = setTab;
    // Defer so the dashboard has had a chance to lay out.
    setTimeout(() => start({ getTab, setTab }), 600);
  }

  function reset() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
  }

  window.TcTour = { start, maybeAutoStart, reset };
})();
