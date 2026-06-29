/* TradingCrew help system — tooltip + drawer + glossary + tour.
 *
 * Loads BEFORE app.js (see <script> order in index.html) so Alpine's
 * x-init handlers in the dashboard see `$store.help` ready.
 *
 * Wiring contract
 * ---------------
 * - Markup: any element with class `info-i` (an "(i)" info-icon) and a
 *   `data-help="anchor.id"` attribute opens the drawer on click and
 *   shows a tooltip on hover.  Plain elements with
 *   `data-help-tip="anchor.id"` get the tooltip only (no click handler).
 * - JS: `Alpine.store('help')` exposes `.openHelp(id)`, `.close()`,
 *   `.search()`, `.tooltip{visible,title,body,x,y}`, `.groups`,
 *   `.results`, `.entry`, `.query`, `.isOpen`.
 * - Source of truth: `window.HELP_CONTENT` from `help_content.js`.
 *
 * The drawer + tooltip + FAB markup lives in index.html (top-level so
 * they don't get trapped inside the Alpine `tradingCrewApp()` scope).
 */

(function () {
  "use strict";

  const DEFAULT_INTRO = "_intro";

  // Section ordering for the drawer TOC.
  const GROUPS = [
    { name: "tabs",     label: "Tabs" },
    { name: "panels",   label: "Workflow panels" },
    { name: "agents",   label: "Agents" },
    { name: "sidebar",  label: "Sidebar controls" },
    { name: "glossary", label: "Finance glossary" },
  ];

  // Tiny fallback markdown renderer for the brief moment before marked.js
  // finishes loading (it has `defer` so it isn't ready when the drawer
  // first paints).  Once marked is available we use it.
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function tinyMarkdown(src) {
    let s = escapeHtml(src);
    s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c}</code></pre>`);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/^(?:[-*] .*(?:\n|$))+?/gm, (block) => {
      const items = block.trim().split(/\n/).map(
        (line) => `<li>${line.replace(/^[-*] /, "")}</li>`
      ).join("");
      return `<ul>${items}</ul>`;
    });
    s = s.split(/\n{2,}/).map((para) => {
      if (/^<(?:ul|ol|pre|h\d|table)/.test(para)) return para;
      return `<p>${para.replace(/\n/g, "<br>")}</p>`;
    }).join("\n");
    return s;
  }

  document.addEventListener("alpine:init", () => {
    if (typeof window.Alpine === "undefined") return;

    window.Alpine.store("help", {
      isOpen: false,
      query: "",
      results: [],
      entry: null,
      groups: [],
      tooltip: { visible: false, title: "", body: "", x: 0, y: 0 },

      _content: window.HELP_CONTENT || {},
      _tooltipTimer: null,
      _searchTimer: null,
      _ready: false,

      // Lifecycle — called once from initListeners() below.  Alpine
      // stores don't auto-run init(), and stores aren't reactive
      // contexts so we wire DOM listeners imperatively.
      _boot() {
        if (this._ready) return;
        this._ready = true;
        this.groups = this._buildGroups();

        const params = new URLSearchParams(window.location.search);
        const deep = params.get("help");
        if (deep) this.openHelp(deep);

        window.addEventListener("keydown", (ev) => {
          if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
            ev.preventDefault();
            this.openHelp(this.entry ? this.entry.id : DEFAULT_INTRO);
            setTimeout(() => {
              const el = document.querySelector('input[placeholder^="Search help"]');
              if (el) { el.focus(); el.select(); }
            }, 30);
          }
        });

        document.addEventListener("click", (ev) => this._onClick(ev));
        document.addEventListener("mouseover", (ev) => this._onHover(ev));
        document.addEventListener("mouseout", (ev) => this._onHoverEnd(ev));
        document.addEventListener("focusin", (ev) => this._onHover(ev));
        document.addEventListener("focusout", (ev) => this._onHoverEnd(ev));
      },

      openHelp(anchorId) {
        const id = anchorId || DEFAULT_INTRO;
        const entry = this._content[id];
        this.entry = entry ? { id, ...entry } : null;
        this.query = "";
        this.results = [];
        this.isOpen = true;
      },

      close() {
        this.isOpen = false;
        this.tooltip.visible = false;
      },

      renderLong(src) {
        if (!src) return "";
        if (typeof window.marked !== "undefined") {
          try { return window.marked.parse(src); } catch (_) { /* fallthrough */ }
        }
        return tinyMarkdown(src);
      },

      search() {
        clearTimeout(this._searchTimer);
        this._searchTimer = setTimeout(() => this._runSearch(), 120);
      },

      _runSearch() {
        const q = this.query.trim().toLowerCase();
        if (!q) { this.results = []; return; }
        const rows = [];
        for (const [id, e] of Object.entries(this._content)) {
          if (id === DEFAULT_INTRO) continue;
          const hay = [
            e.title || "",
            e.short || "",
            e.long || "",
            ...(e.keywords || []),
          ].join(" ").toLowerCase();
          if (hay.includes(q)) {
            rows.push({ id, ...e });
          }
        }
        rows.sort((a, b) => {
          const aTitle = (a.title || "").toLowerCase().includes(q);
          const bTitle = (b.title || "").toLowerCase().includes(q);
          if (aTitle && !bTitle) return -1;
          if (bTitle && !aTitle) return 1;
          return (a.title || "").localeCompare(b.title || "");
        });
        this.results = rows.slice(0, 30);
      },

      _buildGroups() {
        const buckets = {};
        for (const g of GROUPS) buckets[g.name] = [];
        for (const [id, e] of Object.entries(this._content)) {
          if (id === DEFAULT_INTRO) continue;
          const sec = e.section || "panels";
          if (!buckets[sec]) buckets[sec] = [];
          buckets[sec].push({ id, ...e });
        }
        return GROUPS.map((g) => ({
          name: g.name,
          label: g.label,
          items: (buckets[g.name] || []).sort((a, b) => (a.title || "").localeCompare(b.title || "")),
        })).filter((g) => g.items.length > 0);
      },

      // -----------------------------------------------------------------
      // Tooltip + click delegation
      // -----------------------------------------------------------------
      _entryFor(el) {
        const id = el.dataset.help || el.dataset.helpTip;
        if (!id) return null;
        const e = this._content[id];
        return e ? { id, ...e } : null;
      },

      _onClick(ev) {
        // Treat a click as "open help" only when the user clicked an
        // explicit (.info-i) icon, a [data-help-action=open] trigger,
        // or an auto-glossary span (.glossary-term).  Plain
        // [data-help-tip] elements (e.g. tab labels) get tooltips
        // only — clicking them should keep their original action.
        const trigger = ev.target.closest(".info-i, .glossary-term, [data-help-action='open']");
        if (!trigger) return;
        const e = this._entryFor(trigger);
        if (!e) return;
        ev.preventDefault();
        ev.stopPropagation();
        this.openHelp(e.id);
      },

      _onHover(ev) {
        const trigger = ev.target.closest("[data-help], [data-help-tip]");
        if (!trigger) return;
        const e = this._entryFor(trigger);
        if (!e) return;
        clearTimeout(this._tooltipTimer);
        this._tooltipTimer = setTimeout(() => {
          this._positionTooltip(trigger, e);
        }, 180);
      },

      _onHoverEnd(ev) {
        const trigger = ev.target.closest("[data-help], [data-help-tip]");
        if (!trigger) return;
        clearTimeout(this._tooltipTimer);
        this.tooltip.visible = false;
      },

      _positionTooltip(trigger, e) {
        const rect = trigger.getBoundingClientRect();
        const TOOLTIP_W = 280;
        const margin = 8;
        const x = Math.min(
          Math.max(margin, rect.left + rect.width / 2 - TOOLTIP_W / 2),
          window.innerWidth - TOOLTIP_W - margin,
        );
        const y = rect.bottom + margin;
        this.tooltip = {
          visible: true,
          title: e.title || "",
          body: e.short || "",
          x, y,
        };
      },
    });

    // Boot once Alpine has finished initialising all components.  We
    // queue this on alpine:initialized so any deferred scripts that
    // load HELP_CONTENT after alpine:init are picked up.
    document.addEventListener("alpine:initialized", () => {
      const store = window.Alpine.store("help");
      // Refresh content + groups in case help_content.js loaded after
      // alpine:init (e.g. when served with the slow CDN order).
      store._content = window.HELP_CONTENT || {};
      store.groups = store._buildGroups();
      store._boot();
    });
  });
})();
