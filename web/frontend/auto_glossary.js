/* TradingCrew auto-glossary — annotates rendered report bodies with
 * hover-tooltips for common finance terms.
 *
 * Usage from Alpine:
 *   $store.help.autoGlossary(rootElement)   // re-scan & wrap
 *
 * The scanner walks text nodes (no innerHTML re-parse, which would lose
 * Alpine bindings on the host element) and replaces matched terms with
 * <span class="glossary-term" data-help-tip="glossary.<id>">...</span>.
 * help.js's existing document-level hover delegation surfaces the
 * tooltip; clicking opens the Help drawer.
 *
 * Skipped: <code>, <pre>, <a> children (so we don't recursively wrap
 * an already-wrapped term, and we don't disturb hyperlinks).
 */

(function () {
  "use strict";

  // Term → glossary anchor.  Keep regex source single-word or short
  // phrases — \b boundaries make sure we don't match inside identifiers.
  // Where a term has multiple printable forms (RSI / RSI(14), MACD vs
  // MACD), the regex captures both and the same anchor handles them.
  const TERMS = [
    { rx: /\bOHLCV\b/g,                                anchor: "glossary.ohlcv" },
    { rx: /\bSMA\s?(?:20|50|200)?\b/g,                  anchor: "glossary.sma" },
    { rx: /\bEMA\s?(?:10|20|50)?\b/g,                   anchor: "glossary.sma" },
    { rx: /\bRSI(?:\s?\(?\s?14\s?\)?)?\b/g,             anchor: "glossary.rsi" },
    { rx: /\bMACD\b/g,                                  anchor: "glossary.macd" },
    { rx: /\bBollinger(?:\s+bands?)?\b/gi,              anchor: "glossary.bollinger" },
    { rx: /\bATR(?:\s?\(\s?14\s?\))?\b/g,               anchor: "glossary.atr" },
    { rx: /\bP\s?\/\s?E\b/g,                            anchor: "glossary.pe" },
    { rx: /\bP\s?\/\s?B\b/g,                            anchor: "glossary.pb" },
    { rx: /\bFCF\b/g,                                   anchor: "glossary.fcf" },
    { rx: /\bfree\s+cash\s+flow\b/gi,                   anchor: "glossary.fcf" },
    { rx: /\bKelly(?:\s+criterion)?\b/g,                anchor: "glossary.kelly" },
    { rx: /\bCVaR\b/g,                                  anchor: "glossary.cvar" },
    { rx: /\bVaR\b/g,                                   anchor: "glossary.cvar" },
    { rx: /\bSharpe(?:\s+ratio)?\b/g,                   anchor: "glossary.sharpe" },
    { rx: /\bSortino(?:\s+ratio)?\b/g,                  anchor: "glossary.sortino" },
    { rx: /\bCalmar(?:\s+ratio)?\b/g,                   anchor: "glossary.calmar" },
    { rx: /\bDeflated\s+Sharpe(?:\s+ratio)?\b/gi,       anchor: "glossary.deflated_sharpe" },
    { rx: /\bcontango\b/gi,                             anchor: "glossary.contango" },
    { rx: /\bbackwardation\b/gi,                        anchor: "glossary.backwardation" },
    { rx: /\bopen\s+interest\b/gi,                      anchor: "glossary.open_interest" },
    { rx: /\bput[\s/]?call\s+ratio\b/gi,                anchor: "glossary.put_call_ratio" },
    { rx: /\bimplied\s+volatility\b/gi,                 anchor: "glossary.iv" },
    { rx: /\bIV(?=[\s.,;:)])/g,                         anchor: "glossary.iv" },
    { rx: /\bADV\b/g,                                   anchor: "glossary.adv" },
    { rx: /\bmax(?:imum)?\s+drawdown\b/gi,              anchor: "glossary.drawdown" },
    { rx: /\bdrawdown\b/gi,                             anchor: "glossary.drawdown" },
    { rx: /\bz[-\s]?score\b/gi,                         anchor: "glossary.z_score" },
    { rx: /\bbid[-\s]?ask\s+spread\b/gi,                anchor: "glossary.spread" },
    // alpha / beta are short — match only when used as standalone words.
    { rx: /\balpha\b/gi,                                anchor: "glossary.alpha" },
    { rx: /\bbeta\b/gi,                                 anchor: "glossary.beta" },
  ];

  // Combine into a single regex with alternation so we only walk the
  // node once.  Each branch carries its anchor via a parallel array
  // indexed by the matched group.  Build with the help of (?:) groups
  // so we can use named-group emulation.
  function compileCombined() {
    const parts = TERMS.map((t, i) => `(${t.rx.source})`);
    const combined = new RegExp(parts.join("|"), "gi");
    return { combined, anchors: TERMS.map((t) => t.anchor) };
  }

  const COMBINED = compileCombined();

  // Skip these tags entirely.
  const SKIP_TAGS = new Set([
    "CODE", "PRE", "A", "SCRIPT", "STYLE",
    "SPAN" /* skip already-wrapped */,
  ]);

  function shouldSkipParent(node) {
    let p = node.parentElement;
    while (p) {
      if (SKIP_TAGS.has(p.tagName)) return true;
      if (p.classList && p.classList.contains("glossary-term")) return true;
      p = p.parentElement;
    }
    return false;
  }

  function wrapTextNode(textNode) {
    const text = textNode.nodeValue;
    if (!text || !text.trim()) return;
    COMBINED.combined.lastIndex = 0;
    let match;
    let lastIdx = 0;
    const frag = document.createDocumentFragment();
    let any = false;
    while ((match = COMBINED.combined.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      // Find which alternation branch matched.
      let anchor = null;
      for (let g = 1; g < match.length; g++) {
        if (match[g] !== undefined) {
          anchor = COMBINED.anchors[g - 1];
          break;
        }
      }
      if (!anchor) continue;
      if (start > lastIdx) {
        frag.appendChild(document.createTextNode(text.slice(lastIdx, start)));
      }
      const span = document.createElement("span");
      span.className = "glossary-term";
      span.dataset.helpTip = anchor;
      span.dataset.help = anchor;
      span.textContent = match[0];
      frag.appendChild(span);
      lastIdx = end;
      any = true;
    }
    if (!any) return;
    if (lastIdx < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIdx)));
    }
    textNode.parentNode.replaceChild(frag, textNode);
  }

  function annotate(root) {
    if (!root) return 0;
    let count = 0;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (shouldSkipParent(node)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);
    for (const t of targets) {
      const before = t.parentNode.childNodes.length;
      wrapTextNode(t);
      const after = t.parentNode ? t.parentNode.childNodes.length : before;
      if (after > before) count += (after - before);
    }
    return count;
  }

  window.TcAutoGlossary = { annotate };
})();
