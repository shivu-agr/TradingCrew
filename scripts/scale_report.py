"""scale_report.py - post-run analysis for the daily 200-stock forward collect.

Turns a day's run artifacts into an operator report so you can answer, after
each collect run:

  * did it scale?          throughput, per-name latency (avg/median/p95),
                           runs/hour, wall clock.
  * what failed?           per-error-class histogram + the failed-name list.
  * are predictions sane?  action distribution overall / by market / by sector,
                           with directional-bias and sector-skew flags, plus
                           conviction + size distributions.
  * did all data reach it? evidence coverage — which analyst lenses actually
                           survive into each decision's rationale (a signal
                           computed but never cited did NOT affect the call),
                           with a flag when a lens is systematically dropped.
  * is the call stable?    intra-day consistency — same name + same day flip
                           rate (BUY<->SELL on ~identical data = workflow
                           stochasticity, tracked as a standing diagnostic).
  * is scraping healthy?   screener 429 / login / search-backend-failure counts
                           parsed from the crew log.

Inputs (all produced by trading_crew.agentic.scale — the dashboard collect):
  prediction.db collect_state table    per-name status/elapsed/error/episode_id
                                        (keyed by date; replaces the old
                                        <cache>/forward/collect-<day>.json)
  <cache>/memory/episodes.jsonl        the recorded decisions (joined by id)
  logs/scale/<day>/run.log             crew stderr (scraping health)

Outputs:
  stdout                               markdown summary (the collector tees
                                       this to logs/scale/<day>/report.md)
  <log-dir>/report.json                machine-readable version

Stdlib only + fast: safe to run without the venv.

Usage:
  python scripts/scale_report.py --day 2026-07-20
  python scripts/scale_report.py --day 2026-07-20 --universe universe/all200.txt --log-dir logs/scale/2026-07-20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# side -> directional bucket
_BULL = {"BUY", "STRONG_BUY", "OVERWEIGHT"}
_BEAR = {"SELL", "STRONG_SELL", "UNDERWEIGHT"}
_NEUTRAL = {"HOLD", "NEUTRAL", "ABSTAIN"}

# ---------------------------------------------------------------------------
# Evidence-coverage diagnostic — WHICH analyst lenses actually survive into the
# final decision.  We measure this because a signal computed upstream but never
# cited in the PM's rationale is a signal that did NOT affect the decision (the
# reports/guidance lens was reaching the decision only ~23% of the time despite
# being computed for ~96% of runs).  Classifies each ``[source: <id>]`` tag in
# the rationale into a lens by matching the tool-footer identifiers the analyst
# tasks emit.  Purely diagnostic — no thresholds gate a trade.
# ---------------------------------------------------------------------------
_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
# order matters — first match wins (macro basket before generic yfinance/market)
_LENS_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("macro", ("macro basket", "india vix", " vix", "dxy", "treasury", "10y ust", "brent", "wti", "sensex", "nifty")),
    ("reports", ("edgar", "10-k", "10-q", "8-k", "concall", "annual-report", "annual report", "investor", "credit", "report_insights", "guidance", "computed")),
    ("geopolitical", ("geopol", "sanction", "supply_chain", "supply chain", "strait", "opec")),
    ("fundamentals", ("long_term_thesis", "screener", "profit-loss", "#ratios", "balance", "cashflow", "income", "fundamental", ".info", "earnings_estimate", "revenue_estimate", "eps_revision", "analyst_price_targets", "forward")),
    ("market", ("ohlcv", "indicator", "option", "recommendation", "sma", "rsi")),
    ("news", ("insider", "corporate_actions", "http://", "https://", "news", "article")),
]
# lenses that a "yfinance-numbers-only" decision would MISS — their absence is
# the tell-tale of a decision that ignored the qualitative/forward lenses.
_NON_YFINANCE_LENSES = {"news", "macro", "geopolitical", "reports"}


def _lens_of(tag: str) -> str:
    t = tag.lower()
    for lens, needles in _LENS_RULES:
        if any(n in t for n in needles):
            return lens
    return "other"


def _lenses_in_rationale(rationale: str) -> set:
    return {_lens_of(tag) for tag in _SOURCE_TAG_RE.findall(rationale or "")}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _cache_dir(cli: Optional[str]) -> Path:
    return Path(
        cli
        or os.environ.get("TRADINGCREW_CACHE_DIR")
        or os.path.expanduser("~/.trading_crew")
    )


def _parse_universe(path: Path) -> Dict[str, Dict[str, str]]:
    """ticker(upper) -> {market, sector} from a universe file.

    Sector is taken from the nearest preceding ``# --- <Sector> (n) ---``
    comment; market is inferred from the ``.NS`` / ``.BO`` suffix.
    """
    meta: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return meta
    sector = "Unknown"
    sec_re = re.compile(r"^#\s*-+\s*(.+?)\s*(?:\(\d+\))?\s*-+\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw.startswith("#"):
            m = sec_re.match(raw)
            if m and "----" not in m.group(1) and "###" not in raw:
                sector = m.group(1).strip()
            continue
        ticker = raw.split("#", 1)[0].strip()
        if not ticker:
            continue
        up = ticker.upper()
        market = "IN" if up.endswith((".NS", ".BO")) else "US"
        meta[up] = {"market": market, "sector": sector}
    return meta


def _load_state(cache: Path, day: str) -> Dict[str, dict]:
    """Per-name collect status for ``day``, now read from the DB
    (``collect_state`` table) instead of the retired JSON checkpoint."""
    try:
        from web.backend import prediction_db as db

        return db.load_collect_state(day)
    except Exception:
        return {}


def _index_episodes(cache: Path) -> Dict[str, dict]:
    """episode_id -> episode dict (from the SQLite episodic store)."""
    from trading_crew.agentic.memory import EpisodicMemory

    out: Dict[str, dict] = {}
    try:
        mem = EpisodicMemory(cache / "memory" / "episodes.jsonl")
        for ep in mem.all_episodes():
            out[ep.episode_id] = ep.to_dict()
    except Exception:
        return out
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _err_class(error: str) -> str:
    """'ScreenerRateLimitError: ...' -> 'ScreenerRateLimitError'."""
    if not error:
        return "Unknown"
    head = error.split(":", 1)[0].strip()
    return head[:60] or "Unknown"


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _direction(side: str) -> str:
    s = (side or "").upper()
    if s in _BULL:
        return "bull"
    if s in _BEAR:
        return "bear"
    return "neutral"


def _scrape_health(log_path: Path) -> Dict[str, int]:
    patterns = {
        "screener_429": re.compile(r"screener\.in 429|ScreenerRateLimitError", re.I),
        "screener_403": re.compile(r"returned 403", re.I),
        "screener_login_fail": re.compile(r"screener\.in: login .*anonymously|no sessionid", re.I),
        "search_backend_failed": re.compile(r"search backend .* failed", re.I),
        "search_no_results": re.compile(r"no results from any enabled backend", re.I),
        "llm_call_failed": re.compile(r"summarizer LLM call failed|LLM .*failed", re.I),
    }
    counts = {k: 0 for k in patterns}
    if not log_path.exists():
        return counts
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for k, rx in patterns.items():
            if rx.search(line):
                counts[k] += 1
    return counts


def build_report(
    *,
    day: str,
    universe_path: Path,
    cache: Path,
    log_dir: Path,
    bias_threshold: float,
    slow_n: int,
) -> dict:
    meta = _parse_universe(universe_path)
    state = _load_state(cache, day)
    episodes = _index_episodes(cache)

    # ---- throughput ----
    status_counts: Counter = Counter()
    elapsed: List[float] = []
    ts_all: List[str] = []
    failures: List[Tuple[str, str]] = []
    per_name_elapsed: List[Tuple[str, float]] = []

    for name, rec in state.items():
        if not isinstance(rec, dict):
            continue
        st = rec.get("status", "unknown")
        status_counts[st] += 1
        if rec.get("ts"):
            ts_all.append(rec["ts"])
        if st == "done" and isinstance(rec.get("elapsed_sec"), (int, float)):
            elapsed.append(float(rec["elapsed_sec"]))
            per_name_elapsed.append((name, float(rec["elapsed_sec"])))
        if st == "failed":
            failures.append((name, rec.get("error", "")))

    def _p(vals: List[float], q: float) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
        return round(s[idx], 1)

    wall_sec = None
    if len(ts_all) >= 2:
        try:
            dts = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in ts_all]
            wall_sec = (max(dts) - min(dts)).total_seconds()
        except Exception:
            wall_sec = None

    done = status_counts.get("done", 0)
    throughput = {
        "universe": len(meta) or len(state),
        "attempted": len(state),
        "done": done,
        "failed": status_counts.get("failed", 0),
        "other_status": {k: v for k, v in status_counts.items() if k not in ("done", "failed")},
        "elapsed_sec": {
            "avg": round(statistics.mean(elapsed), 1) if elapsed else None,
            "median": round(statistics.median(elapsed), 1) if elapsed else None,
            "p95": _p(elapsed, 0.95),
            "max": round(max(elapsed), 1) if elapsed else None,
        },
        "wall_clock_sec": round(wall_sec, 1) if wall_sec else None,
        "wall_clock_hours": round(wall_sec / 3600.0, 2) if wall_sec else None,
        "runs_per_hour": round(done / (wall_sec / 3600.0), 1) if wall_sec else None,
    }

    # ---- failures ----
    fail_hist = Counter(_err_class(err) for _, err in failures)

    # ---- predictions / bias ----
    # group -> Counter(side); also collect conviction + size
    overall_sides: Counter = Counter()
    by_market: Dict[str, Counter] = defaultdict(Counter)
    by_sector: Dict[str, Counter] = defaultdict(Counter)
    convictions: List[float] = []
    sizes: List[float] = []
    decisions = 0
    # evidence-coverage accumulators
    lens_present: Counter = Counter()
    zero_nonyf = 0

    for name, rec in state.items():
        if not isinstance(rec, dict) or rec.get("status") != "done":
            continue
        ep = episodes.get(rec.get("episode_id", ""))
        if not ep:
            continue
        ap = ep.get("action_proposal") or {}
        side = (ap.get("side") or ap.get("action") or "").upper()
        if not side:
            continue
        decisions += 1
        overall_sides[side] += 1
        info = meta.get(name.upper(), {"market": "IN" if name.upper().endswith((".NS", ".BO")) else "US", "sector": "Unknown"})
        by_market[info["market"]][side] += 1
        by_sector[info["sector"]][side] += 1
        cs = ap.get("conviction_score")
        if isinstance(cs, (int, float)):
            convictions.append(float(cs))
        tw = ap.get("target_weight")
        if isinstance(tw, (int, float)):
            sizes.append(abs(float(tw)))
        # which lenses survived into THIS decision's rationale
        lenses = _lenses_in_rationale(ap.get("rationale") or ep.get("state_summary") or "")
        for L in lenses:
            lens_present[L] += 1
        if not (lenses & _NON_YFINANCE_LENSES):
            zero_nonyf += 1

    # ---- evidence coverage: is all the data reaching the decision? ----
    _cov_lenses = ("market", "fundamentals", "reports", "news", "macro", "geopolitical")
    evidence_coverage = {
        "decisions": decisions,
        "lens_present_pct": {L: _pct(lens_present.get(L, 0), decisions) for L in _cov_lenses},
        "zero_non_yfinance_lens": zero_nonyf,
        "zero_non_yfinance_pct": _pct(zero_nonyf, decisions),
    }
    # Flag lenses that reach fewer than half the decisions — a computed-but-
    # dropped signal (the reports/macro/geopolitical leak we found).
    coverage_flags = [
        f"{L} lens reaches only {evidence_coverage['lens_present_pct'][L]}% of decisions"
        for L in ("reports", "macro", "news", "fundamentals", "market")
        if decisions >= 10 and evidence_coverage["lens_present_pct"][L] < 50.0
    ]
    if decisions >= 10 and evidence_coverage["zero_non_yfinance_pct"] >= 15.0:
        coverage_flags.append(
            f"{evidence_coverage['zero_non_yfinance_pct']}% of decisions cite NO "
            "news/macro/geopolitical/reports lens (yfinance-numbers-only)"
        )
    evidence_coverage["flags"] = coverage_flags

    # ---- intra-day consistency: same name + same day, conflicting call? ----
    # A stochastic workflow will flip BUY<->SELL on ~identical data.  We keep
    # this as a standing diagnostic (a real regime read should be stable), not
    # something to optimise away.  Groups ALL episodes decided on ``day``.
    by_symbol_day: Dict[str, list] = defaultdict(list)
    for ep in episodes.values():
        if (ep.get("decision_ts") or "")[:10] != day:
            continue
        ap = ep.get("action_proposal") or {}
        side = (ap.get("side") or ap.get("action") or "").upper()
        if side:
            by_symbol_day[str(ep.get("symbol") or "").upper()].append(_direction(side))
    rerun_groups = {s: dirs for s, dirs in by_symbol_day.items() if len(dirs) > 1}
    conflicting = {s: dirs for s, dirs in rerun_groups.items()
                   if len({d for d in dirs if d in ("bull", "bear")}) > 1}
    intraday_consistency = {
        "rerun_groups": len(rerun_groups),
        "direction_conflicting": len(conflicting),
        "flip_rate_pct": _pct(len(conflicting), len(rerun_groups)) if rerun_groups else 0.0,
        "conflicting_symbols": sorted(conflicting)[:20],
    }

    def _dir_shares(counter: Counter) -> Dict[str, float]:
        total = sum(counter.values())
        agg = Counter()
        for side, n in counter.items():
            agg[_direction(side)] += n
        return {d: _pct(agg.get(d, 0), total) for d in ("bull", "bear", "neutral")}

    # bias flags
    bias_flags: List[str] = []
    thr = bias_threshold * 100.0
    overall_shares = _dir_shares(overall_sides)
    for d, share in overall_shares.items():
        if decisions >= 10 and share >= thr:
            bias_flags.append(f"overall {d} share {share}% >= {round(thr)}% threshold")
    for mkt, counter in by_market.items():
        sh = _dir_shares(counter)
        tot = sum(counter.values())
        for d, share in sh.items():
            if tot >= 10 and share >= thr:
                bias_flags.append(f"market {mkt} {d} share {share}% (n={tot})")
    for sec, counter in by_sector.items():
        tot = sum(counter.values())
        if tot >= 4 and len({_direction(s) for s in counter.elements()}) == 1:
            only = _direction(next(iter(counter)))
            bias_flags.append(f"sector '{sec}' unanimous {only} (n={tot})")

    predictions = {
        "decisions": decisions,
        "overall_sides": dict(overall_sides),
        "overall_direction_shares": overall_shares,
        "by_market": {m: {"sides": dict(c), "direction_shares": _dir_shares(c)} for m, c in by_market.items()},
        "by_sector": {s: {"sides": dict(c), "direction_shares": _dir_shares(c)} for s, c in by_sector.items()},
        "conviction": {
            "mean": round(statistics.mean(convictions), 3) if convictions else None,
            "median": round(statistics.median(convictions), 3) if convictions else None,
            "n": len(convictions),
        },
        "size_target_weight": {
            "mean": round(statistics.mean(sizes), 4) if sizes else None,
            "median": round(statistics.median(sizes), 4) if sizes else None,
            "n": len(sizes),
        },
        "bias_flags": bias_flags,
        "evidence_coverage": evidence_coverage,
        "intraday_consistency": intraday_consistency,
    }

    slowest = sorted(per_name_elapsed, key=lambda kv: kv[1], reverse=True)[:slow_n]

    return {
        "day": day,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_file": str(universe_path),
        "throughput": throughput,
        "failures": {
            "count": len(failures),
            "by_class": dict(fail_hist),
            "names": [{"ticker": n, "error": (e or "")[:200]} for n, e in failures],
        },
        "predictions": predictions,
        "scrape_health": _scrape_health(log_dir / "run.log"),
        "slowest": [{"ticker": n, "elapsed_sec": s} for n, s in slowest],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(rep: dict) -> str:
    t = rep["throughput"]
    p = rep["predictions"]
    f = rep["failures"]
    h = rep["scrape_health"]
    L: List[str] = []
    L.append(f"# Scale run report - {rep['day']}")
    L.append(f"_generated {rep['generated_at']} - universe `{rep['universe_file']}`_\n")

    L.append("## Throughput")
    L.append(f"- Attempted **{t['attempted']}** / universe {t['universe']} - "
             f"done **{t['done']}**, failed **{t['failed']}**"
             + (f", other {t['other_status']}" if t['other_status'] else ""))
    el = t["elapsed_sec"]
    L.append(f"- Per-name latency: avg **{el['avg']}s**, median {el['median']}s, p95 {el['p95']}s, max {el['max']}s")
    if t["wall_clock_hours"] is not None:
        L.append(f"- Wall clock **{t['wall_clock_hours']}h** - throughput **{t['runs_per_hour']} runs/hour**")
    L.append("")

    L.append("## Failures")
    if f["count"] == 0:
        L.append("- None \U0001F389")
    else:
        L.append(f"- **{f['count']}** failed. By class:")
        for cls, n in sorted(f["by_class"].items(), key=lambda kv: -kv[1]):
            L.append(f"  - `{cls}`: {n}")
        L.append("- Names:")
        for row in f["names"][:25]:
            L.append(f"  - `{row['ticker']}` - {row['error']}")
    L.append("")

    L.append("## Predictions & bias")
    L.append(f"- Decisions analysed: **{p['decisions']}**")
    sh = p["overall_direction_shares"]
    L.append(f"- Overall direction: bull {sh['bull']}% / bear {sh['bear']}% / neutral {sh['neutral']}%  (sides: {p['overall_sides']})")
    L.append("- By market:")
    for m, d in p["by_market"].items():
        ds = d["direction_shares"]
        L.append(f"  - **{m}**: bull {ds['bull']}% / bear {ds['bear']}% / neutral {ds['neutral']}%  {d['sides']}")
    L.append("- By sector:")
    for s, d in sorted(p["by_sector"].items()):
        ds = d["direction_shares"]
        L.append(f"  - {s}: bull {ds['bull']}% / bear {ds['bear']}% / neutral {ds['neutral']}%  {d['sides']}")
    cv = p["conviction"]
    L.append(f"- Conviction: mean {cv['mean']}, median {cv['median']} (n={cv['n']})")
    sz = p["size_target_weight"]
    L.append(f"- Target weight |size|: mean {sz['mean']}, median {sz['median']} (n={sz['n']})")
    if p["bias_flags"]:
        L.append("- \u26A0\uFE0F Bias flags:")
        for flag in p["bias_flags"]:
            L.append(f"  - {flag}")
    else:
        L.append("- No bias flags tripped.")
    L.append("")

    # ---- Evidence coverage — did all the data reach the decision? ----
    ec = p.get("evidence_coverage")
    if ec:
        L.append("## Evidence coverage (which lenses reached the decision)")
        lp = ec["lens_present_pct"]
        L.append("- Lens present in rationale (% of decisions): "
                 + ", ".join(f"{k} {lp[k]}%" for k in ("market", "fundamentals", "news", "reports", "macro", "geopolitical")))
        L.append(f"- Decisions citing NO non-yfinance lens (news/macro/geo/reports): "
                 f"**{ec['zero_non_yfinance_pct']}%** ({ec['zero_non_yfinance_lens']}/{ec['decisions']})")
        if ec.get("flags"):
            L.append("- \u26A0\uFE0F Coverage flags (computed-but-dropped signal):")
            for flag in ec["flags"]:
                L.append(f"  - {flag}")
        else:
            L.append("- No coverage flags: every lens reaches most decisions.")
        L.append("")

    # ---- Intra-day consistency — same name+day, does the call flip? ----
    ic = p.get("intraday_consistency")
    if ic:
        L.append("## Intra-day consistency (stochasticity diagnostic)")
        L.append(f"- Re-run groups (same name, same day): **{ic['rerun_groups']}** - "
                 f"direction-conflicting: **{ic['direction_conflicting']}** "
                 f"(**{ic['flip_rate_pct']}%** flip rate)")
        L.append("  _A high flip rate means ~identical data produced opposite calls "
                 "(workflow stochasticity), not a genuine regime change. Track it; "
                 "don't optimise it away._")
        if ic["conflicting_symbols"]:
            L.append(f"- Conflicting names: {', '.join('`'+s+'`' for s in ic['conflicting_symbols'])}")
        L.append("")

    L.append("## Scraping / search health (from run.log)")
    if any(h.values()):
        for k, n in h.items():
            if n:
                L.append(f"- `{k}`: {n}")
    else:
        L.append("- Clean (no rate-limit / login / backend-failure lines).")
    L.append("")

    if rep["slowest"]:
        L.append("## Slowest names")
        for row in rep["slowest"]:
            L.append(f"- `{row['ticker']}` - {row['elapsed_sec']}s")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-run analysis for the daily scale collect.")
    ap.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="UTC day (YYYY-MM-DD)")
    ap.add_argument("--universe", default="universe/all200.txt", help="universe file (for sector/market map)")
    ap.add_argument("--log-dir", default=None, help="logs/scale/<day> (defaults from --day)")
    ap.add_argument("--cache-dir", default=None, help="override TRADINGCREW_CACHE_DIR")
    ap.add_argument("--bias-threshold", type=float, default=0.70, help="directional share that trips a bias flag")
    ap.add_argument("--slow-n", type=int, default=10, help="how many slowest names to list")
    args = ap.parse_args()

    cache = _cache_dir(args.cache_dir)
    universe_path = Path(args.universe)
    if not universe_path.is_absolute():
        universe_path = _PROJECT_ROOT / universe_path
    log_dir = Path(args.log_dir) if args.log_dir else (_PROJECT_ROOT / "logs" / "scale" / args.day)

    rep = build_report(
        day=args.day,
        universe_path=universe_path,
        cache=cache,
        log_dir=log_dir,
        bias_threshold=args.bias_threshold,
        slow_n=args.slow_n,
    )

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[scale_report] WARN: could not write report.json: {exc}")

    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
