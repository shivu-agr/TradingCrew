"""Report trend signals — structured lift of the Firefly-style signal card.

Two complementary sources feed the same machine-readable shape
(:class:`ReportSignals`):

* **LLM-read filing signals** — the ``## REPORT TREND SIGNALS`` card the
  reports desk emits from a company's OWN primary filings (see
  ``tools.get_report_insights`` / ``_REPORT_INSIGHT_PROMPT``): OUTLOOK_TONE,
  REVENUE_GUIDANCE, MARGIN_GUIDANCE, DELIVERY_TRACK_RECORD, DEMAND_MOMENTUM,
  CREDIT_TRAJECTORY and a synthesised NET_GUIDANCE_BIAS.  These are qualitative
  directions the model reads out of management commentary.

* **Deterministic financial-trend signals** — REVENUE_TREND / EARNINGS_TREND /
  MARGIN_TREND computed here from HARD quarterly numbers (screener.in's
  ``#quarters`` table for Indian names, which the reports tool already has in
  hand; yfinance's quarterly income statement otherwise).  These are the
  "revenue increased/decreased vs last year, and is that growth accelerating"
  read the user asked for — no LLM, no fabrication: if the numbers aren't there
  the signal is INSUFFICIENT_DATA.

Both are emitted as one canonical line so a single parser lifts the whole
digest into structure::

    SIGNAL | <NAME> | <DIRECTION> | conf=<0.00-1.00> | <rationale> [source: <url>]

``parse_signal_card`` is deliberately tolerant (it scans any text for lines
matching that grammar) so it survives an analyst reformatting the digest.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured shape
# ---------------------------------------------------------------------------

class ReportSignal(BaseModel):
    """One directional, confidence-scored trend signal."""

    name: str
    direction: str
    confidence: float = 0.0
    rationale: str = ""
    source: Optional[str] = None
    # True → deterministic (computed from hard financials); False → LLM-read
    # from the primary filings.  Lets the UI / fusion weight hard numbers
    # differently from a qualitative management-tone read.
    computed: bool = False


class ReportSignals(BaseModel):
    """The full signal set lifted from one report digest."""

    signals: List[ReportSignal] = Field(default_factory=list)

    def get(self, name: str) -> Optional[ReportSignal]:
        name = name.upper()
        for s in self.signals:
            if s.name.upper() == name:
                return s
        return None

    @property
    def net_bias(self) -> Optional[str]:
        s = self.get("NET_GUIDANCE_BIAS")
        return s.direction if s else None

    @property
    def net_confidence(self) -> float:
        s = self.get("NET_GUIDANCE_BIAS")
        return s.confidence if s else 0.0

    def is_empty(self) -> bool:
        return not any(
            s.direction and s.direction.upper() != "INSUFFICIENT_DATA"
            for s in self.signals
        )


# Direction → signed guidance score in [-1, 1].  Used to fold the qualitative
# card into ONE numeric bias (``guidance_bias_score``) that a downstream
# consumer (dashboard badge, optional fusion input) can reason about without
# re-deriving the mapping.  Only directions that carry a clear positive /
# negative lean are scored; neutral / structural directions map to 0.0 and
# INSUFFICIENT_DATA is skipped entirely by the aggregator.
_DIRECTION_SCORE = {
    # tone
    "BULLISH": 1.0, "CAUTIOUS": -0.3, "DEFENSIVE": -1.0,
    # guidance
    "RAISED": 1.0, "MAINTAINED": 0.0, "CUT": -1.0,
    # margin
    "EXPANDING": 1.0, "STABLE": 0.0, "CONTRACTING": -1.0,
    # delivery
    "BEAT": 1.0, "IN_LINE": 0.0, "MISS": -1.0,
    # momentum / computed trend
    "ACCELERATING": 1.0, "DECELERATING": -1.0,
    "GROWING": 0.6, "DECLINING": -1.0, "FLAT": 0.0,
    # credit
    "IMPROVING": 1.0, "DETERIORATING": -1.0,
    # net bias
    "POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0,
}


def guidance_bias_score(signals: ReportSignals) -> Optional[float]:
    """Confidence-weighted signed bias in [-1, 1] across all directional
    signals, or ``None`` when nothing is decisive.

    NET_GUIDANCE_BIAS is the analyst's own synthesis, so when present with
    non-trivial confidence it anchors the score; otherwise we average the
    component directions weighted by their confidence.  INSUFFICIENT_DATA and
    unknown directions are ignored (never fabricate a lean from missing data).
    """
    net = signals.get("NET_GUIDANCE_BIAS")
    if net is not None and net.confidence >= 0.3 and net.direction.upper() in _DIRECTION_SCORE:
        base = _DIRECTION_SCORE[net.direction.upper()]
        # scale the anchor by the confidence the analyst stood behind
        return round(base * net.confidence, 4)

    num = 0.0
    den = 0.0
    for s in signals.signals:
        if s.name.upper() == "NET_GUIDANCE_BIAS":
            continue
        d = s.direction.upper()
        if d == "INSUFFICIENT_DATA" or d not in _DIRECTION_SCORE:
            continue
        w = max(0.0, min(1.0, s.confidence))
        if w <= 0.0:
            continue
        num += _DIRECTION_SCORE[d] * w
        den += w
    if den <= 0.0:
        return None
    return round(num / den, 4)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SIGNAL_RE = re.compile(
    r"SIGNAL\s*\|\s*([A-Z0-9_]+)\s*\|\s*([A-Za-z0-9_\- ]+?)\s*\|\s*"
    r"conf\s*=\s*([0-9]*\.?[0-9]+)\s*\|\s*(.*)",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"\[source:\s*([^\]]+)\]\s*$", re.IGNORECASE)


def parse_signal_card(text: str) -> ReportSignals:
    """Lift every ``SIGNAL | ...`` line out of ``text`` into structure.

    Tolerant by design: scans line-by-line, ignores anything that isn't a
    signal line, de-dupes by name (keeping the FIRST occurrence, which is the
    canonical card — a later analyst paraphrase won't clobber it).
    """
    out = ReportSignals()
    if not text:
        return out
    seen: set = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        m = _SIGNAL_RE.search(line)
        if not m:
            continue
        name = m.group(1).strip().upper()
        if name in seen:
            continue
        direction = m.group(2).strip().upper().replace(" ", "_").replace("-", "_")
        try:
            conf = max(0.0, min(1.0, float(m.group(3))))
        except ValueError:
            conf = 0.0
        tail = m.group(4).strip()
        source = None
        sm = _SOURCE_RE.search(tail)
        if sm:
            source = sm.group(1).strip()
            tail = tail[: sm.start()].strip()
        computed = bool(source and "computed" in source.lower())
        seen.add(name)
        out.signals.append(ReportSignal(
            name=name, direction=direction, confidence=conf,
            rationale=tail, source=source, computed=computed,
        ))
    return out


def render_signal_line(sig: ReportSignal) -> str:
    src = f" [source: {sig.source}]" if sig.source else ""
    return (
        f"SIGNAL | {sig.name} | {sig.direction} | conf={sig.confidence:.2f} | "
        f"{sig.rationale}{src}"
    )


# ---------------------------------------------------------------------------
# Deterministic financial-trend signals (hard numbers, no LLM)
# ---------------------------------------------------------------------------

_COMPUTED_HEADER = "## COMPUTED FINANCIAL TREND SIGNALS"


def _to_float(s) -> Optional[float]:
    """Parse a screener / yfinance numeric cell to float, or None.

    Handles Indian grouping ("2,35,000"), a trailing "%", parentheses for
    negatives, and blanks.  Never raises.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            v = float(s)
            return None if v != v else v  # drop NaN
        except (TypeError, ValueError):
            return None
    txt = str(s).strip()
    if not txt or txt in {"-", "—", "NA", "N/A"}:
        return None
    neg = txt.startswith("(") and txt.endswith(")")
    txt = txt.strip("()").replace(",", "").replace("%", "").replace("₹", "").strip()
    try:
        v = float(txt)
    except ValueError:
        return None
    return -v if neg else v


def _yoy(series: List[float], back: int = 4) -> Optional[float]:
    """Year-over-year growth of the last point vs ``back`` points earlier."""
    if len(series) <= back:
        return None
    prev = series[-1 - back]
    if prev == 0:
        return None
    return series[-1] / prev - 1.0


def _trend_signal(name: str, series: List[float], label: str, source: str) -> ReportSignal:
    """Build REVENUE_TREND / EARNINGS_TREND from an oldest→newest series."""
    series = [v for v in series if v is not None]
    yoy = _yoy(series, 4)
    if yoy is None:
        return ReportSignal(
            name=name, direction="INSUFFICIENT_DATA", confidence=0.0,
            rationale=f"insufficient quarterly {label} history", computed=True,
        )
    # prior quarter's YoY, to measure acceleration of the growth rate
    yoy_prev = _yoy(series[:-1], 4)
    if yoy_prev is not None:
        accel = yoy - yoy_prev
        if accel > 0.02:
            direction = "ACCELERATING"
        elif accel < -0.02:
            direction = "DECELERATING"
        else:
            direction = "STABLE"
        conf = 0.8
        rationale = (
            f"{label} YoY {yoy:+.0%} vs prior-quarter YoY {yoy_prev:+.0%} "
            f"(growth {'accelerating' if accel > 0 else 'slowing'})"
        )
    else:
        direction = "GROWING" if yoy > 0 else "DECLINING"
        conf = 0.5
        rationale = f"{label} YoY {yoy:+.0%} (no prior-quarter compare available)"
    return ReportSignal(
        name=name, direction=direction, confidence=conf,
        rationale=rationale, source=source, computed=True,
    )


def _margin_signal(revenue: List[float], profit: List[float], source: str) -> ReportSignal:
    """MARGIN_TREND from net-margin latest quarter vs the year-ago quarter."""
    n = min(len(revenue), len(profit))
    if n <= 4:
        return ReportSignal(
            name="MARGIN_TREND", direction="INSUFFICIENT_DATA", confidence=0.0,
            rationale="insufficient history to compute margin trend", computed=True,
        )
    rev = revenue[-n:]
    prof = profit[-n:]
    if rev[-1] in (0, None) or rev[-5] in (0, None):
        return ReportSignal(
            name="MARGIN_TREND", direction="INSUFFICIENT_DATA", confidence=0.0,
            rationale="revenue base unavailable for margin trend", computed=True,
        )
    m_latest = prof[-1] / rev[-1]
    m_prior = prof[-5] / rev[-5]
    delta_pp = (m_latest - m_prior) * 100.0
    if delta_pp > 0.5:
        direction = "EXPANDING"
    elif delta_pp < -0.5:
        direction = "CONTRACTING"
    else:
        direction = "STABLE"
    return ReportSignal(
        name="MARGIN_TREND", direction=direction, confidence=0.7,
        rationale=(
            f"net margin {m_latest:.1%} vs {m_prior:.1%} a year ago "
            f"({delta_pp:+.1f}pp)"
        ),
        source=source, computed=True,
    )


def _series_from_screener_quarters(quarters, *metric_aliases: str) -> List[float]:
    """Pull one metric row (oldest→newest) from a screener quarters table."""
    if quarters is None or not getattr(quarters, "rows", None):
        return []
    aliases = [a.lower() for a in metric_aliases]
    for metric, values in quarters.rows:
        m = str(metric).lower().strip()
        if any(a in m for a in aliases):
            nums = [_to_float(v) for v in values]
            # trailing blanks are unreported future columns — drop them
            while nums and nums[-1] is None:
                nums.pop()
            return [v for v in nums if v is not None]
    return []


def _build_trend_signals(revenue: List[float], profit: List[float], source: str) -> List[ReportSignal]:
    """Assemble REVENUE/EARNINGS/MARGIN signals from a series pair.

    Returns ``[]`` unless at least one signal is decisive, so a caller can tell
    "this source produced nothing usable" apart from "this source is bearish".
    """
    signals: List[ReportSignal] = []
    if revenue:
        signals.append(_trend_signal("REVENUE_TREND", revenue, "revenue", source))
    if profit:
        signals.append(_trend_signal("EARNINGS_TREND", profit, "net profit", source))
    if revenue and profit:
        signals.append(_margin_signal(revenue, profit, source))
    if any(s.direction != "INSUFFICIENT_DATA" for s in signals):
        return signals
    return []


def compute_financial_trend_signals(ticker: str, screener_quarters=None) -> List[ReportSignal]:
    """Deterministic revenue / earnings / margin trend signals for ``ticker``.

    Prefers the screener ``#quarters`` table already fetched by the reports tool
    for Indian names.  If that table is absent OR can't be parsed into a usable
    series (e.g. an unexpected screener layout — the COLPAL case), it falls back
    to yfinance's quarterly income statement so an Indian name isn't left with
    no computed signals just because its screener quarters didn't parse.
    Best-effort: returns ``[]`` (never fabricates) when no hard numbers resolve.
    """
    if screener_quarters is not None:
        revenue = _series_from_screener_quarters(screener_quarters, "sales", "revenue", "income")
        profit = _series_from_screener_quarters(screener_quarters, "net profit", "profit after tax")
        signals = _build_trend_signals(revenue, profit, "screener.in #quarters (computed)")
        if signals:
            return signals
        logger.info(
            "compute_financial_trend_signals: screener quarters unusable for %s "
            "— falling back to yfinance", ticker,
        )

    revenue, profit = _yfinance_quarterly(ticker)
    return _build_trend_signals(revenue, profit, "yfinance quarterly financials (computed)")


def _yfinance_quarterly(ticker: str):
    """(revenue, net_income) oldest→newest from yfinance, or ([], [])."""
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        df = tk.quarterly_income_stmt
        if df is None or df.empty:
            return [], []
        # columns are period Timestamps, newest first → reverse to oldest→newest
        cols = list(df.columns)[::-1]

        def _row(*names: str) -> List[float]:
            for nm in names:
                if nm in df.index:
                    vals = [_to_float(df.at[nm, c]) for c in cols]
                    return [v for v in vals if v is not None]
            return []

        revenue = _row("Total Revenue", "TotalRevenue", "OperatingRevenue")
        profit = _row("Net Income", "NetIncome", "NetIncomeCommonStockholders")
        return revenue, profit
    except Exception as exc:  # noqa: BLE001 — deterministic signals are best-effort
        logger.info("compute_financial_trend_signals: yfinance read failed for %s: %s", ticker, exc)
        return [], []


def render_computed_block(signals: List[ReportSignal]) -> str:
    """Format computed signals as an appendable card block (empty str if none)."""
    if not signals:
        return ""
    lines = [_COMPUTED_HEADER] + [render_signal_line(s) for s in signals]
    return "\n".join(lines)


def render_pm_ledger(signals: "ReportSignals") -> str:
    """Authoritative reports-desk block injected into the Portfolio Manager
    prompt so the deterministically-parsed guidance / trend signals ALWAYS
    reach the final decision — even when the reports desk's prose dropped the
    card on the way to the PM (observed: the card survived into only ~23% of
    decisions despite being computed for ~96% of runs).

    This is a *completeness* guarantee, not a directive: it forces the signal
    to be considered and cited; it never prescribes a side, so it introduces
    no hardcoded bias.  Returns an empty string when there is nothing decisive
    to inject (so the plain prompt runs unchanged).
    """
    if signals is None or signals.is_empty():
        return ""
    score = guidance_bias_score(signals)
    lines = [
        "## REPORTS DESK — MACHINE-PARSED GUIDANCE SIGNALS (authoritative; do not ignore)",
        "Parsed directly from the primary-filing digest, independent of any "
        "analyst summary. Weigh these alongside the other desks. Weight "
        "computed REVENUE/EARNINGS/MARGIN_TREND (hard quarterly numbers) at "
        "least as heavily as LLM-read OUTLOOK_TONE — management commentary "
        "is structurally optimistic:",
    ]
    for s in signals.signals:
        if s.direction and s.direction.upper() != "INSUFFICIENT_DATA":
            lines.append(render_signal_line(s))
    nb = signals.net_bias
    if nb:
        tail = f", bias_score {score:+.2f}" if score is not None else ""
        lines.append(
            f"NET_GUIDANCE_BIAS: {nb} (confidence {signals.net_confidence:.2f}{tail})"
        )
    lines.append(
        "You MUST explicitly weigh this parsed guidance read and cite "
        "NET_GUIDANCE_BIAS in your rationale. If your call disagrees with it, "
        "name the specific evidence that overrides it. (Completeness check — it "
        "does not prescribe a side.)"
    )
    return "\n".join(lines)
