"""Sector-bias diagnostic — run the TradingCrew on a 10-ticker basket
spanning US + India and multiple sectors, then summarise the resulting
PortfolioDecision distribution so the user can spot sector / country
bias empirically.

This is the **live** counterpart to ``tests/test_sector_bias.py``.  It
expects the LLM endpoints in ``.env`` to be reachable and will call them
once per ticker (≈ 10 × full crew kickoff).

Usage::

    # Activate the workspace venv first (./run_web.sh uses the same one):
    ../.venv/bin/python scripts/sector_bias_basket.py

    # Pick a smaller / different basket:
    ../.venv/bin/python scripts/sector_bias_basket.py --tickers NVDA,AAPL,RELIANCE

    # Override debate / risk budgets (default = 2 / 1 from main.py):
    ../.venv/bin/python scripts/sector_bias_basket.py --debate-rounds 1 --risk-rounds 1

The script writes:
    * ``reports/sector_bias_<timestamp>.json`` — full per-ticker decision
    * ``reports/sector_bias_<timestamp>.md``   — markdown summary the
                                                  user can paste into a
                                                  ticket / chat

Bias signals to look for in the markdown summary:
    * All US tickers OVERWEIGHT / all India tickers NEUTRAL (or vice versa)
    * Confidence band materially different by country / sector
    * Compliance status BLOCKED concentrated in one country
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Make ``trading_crew`` importable when running the script directly from
# the project root (``python scripts/sector_bias_basket.py``).  Without
# this, Python's CWD-based import only sees ``scripts/`` itself.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load ``.env`` so LLM / embedder env vars are available even when the
# user did not pre-export them in this shell.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------
# Default basket — covers US tech, US financials, US consumer, Indian
# defence, Indian auto, Indian conglomerates, Indian FMCG.  Override via
# ``--tickers``.
# ---------------------------------------------------------------------
DEFAULT_BASKET: list[tuple[str, str, str]] = [
    ("NVDA",       "US",    "Technology / AI & Semiconductors"),
    ("MSFT",       "US",    "Technology / Software & Cloud"),
    ("AAPL",       "US",    "Consumer Electronics"),
    ("JPM",        "US",    "Financials / Banking"),
    ("MAZDOCK",    "India", "Defence & Capital Goods"),
    ("BEL",        "India", "Defence Electronics"),
    ("TATAMOTORS", "India", "Automotive / EV"),
    ("RELIANCE",   "India", "Conglomerate (Energy, Retail, Telecom)"),
    ("LT",         "India", "Engineering & Infrastructure"),
    ("HINDUNILVR", "India", "FMCG / Consumer Goods"),
]


@dataclass
class BasketRow:
    ticker: str
    resolved: str
    country: str
    sector: str
    action: Optional[str]
    confidence: Optional[float]
    size_pct: Optional[float]
    compliance: Optional[str]
    horizon_days: Optional[int]
    expected_return_pct: Optional[float]
    error: Optional[str]
    runtime_s: float


# Lazy imports so ``--help`` works without crewai installed.
def _import_crew():
    from trading_crew import TradingCrew  # noqa: WPS433
    from trading_crew.market_context import resolve_ticker  # noqa: WPS433
    return TradingCrew, resolve_ticker


def run_single(ticker: str, debate_rounds: int, risk_rounds: int, use_memory: bool) -> BasketRow:
    TradingCrew, resolve_ticker = _import_crew()
    t0 = time.time()
    resolved = ticker
    try:
        resolved = resolve_ticker(ticker)
        tc = TradingCrew(
            ticker=resolved,
            debate_rounds=debate_rounds,
            risk_rounds=risk_rounds,
            memory=use_memory,
        )
        crew = tc.crew()
        result = crew.kickoff(inputs={"ticker": resolved})
        pyd = result.pydantic
        if pyd is None:
            return BasketRow(
                ticker=ticker, resolved=resolved,
                country="", sector="",
                action=None, confidence=None, size_pct=None,
                compliance=None, horizon_days=None, expected_return_pct=None,
                error="No PortfolioDecision parsed (raw output)",
                runtime_s=round(time.time() - t0, 1),
            )
        return BasketRow(
            ticker=ticker, resolved=resolved,
            country="", sector="",
            action=pyd.action,
            confidence=float(pyd.confidence),
            size_pct=float(pyd.size_pct_of_book),
            compliance=pyd.compliance_status,
            horizon_days=int(pyd.horizon_days),
            expected_return_pct=float(pyd.expected_return_pct),
            error=None,
            runtime_s=round(time.time() - t0, 1),
        )
    except Exception as exc:
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return BasketRow(
            ticker=ticker, resolved=resolved,
            country="", sector="",
            action=None, confidence=None, size_pct=None,
            compliance=None, horizon_days=None, expected_return_pct=None,
            error=tb,
            runtime_s=round(time.time() - t0, 1),
        )


def _summarise(rows: list[BasketRow]) -> dict:
    actions = Counter(r.action for r in rows if r.action)
    by_country: dict[str, Counter] = {}
    for r in rows:
        by_country.setdefault(r.country or "?", Counter())[r.action or "ERROR"] += 1
    confidences = [r.confidence for r in rows if r.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    sizes = [r.size_pct for r in rows if r.size_pct is not None]
    avg_size = sum(sizes) / len(sizes) if sizes else None
    return {
        "actions_total": dict(actions),
        "actions_by_country": {c: dict(v) for c, v in by_country.items()},
        "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
        "avg_size_pct": round(avg_size, 3) if avg_size is not None else None,
        "n_errors": sum(1 for r in rows if r.error),
        "n_total": len(rows),
    }


def _write_report(rows: list[BasketRow], summary: dict, out_dir: Path) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"sector_bias_{ts}.json"
    md_path = out_dir / f"sector_bias_{ts}.md"
    json_path.write_text(
        json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2, default=str),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# Sector-bias basket run — {ts}")
    lines.append("")
    lines.append(f"- Total tickers: **{summary['n_total']}** (errors: {summary['n_errors']})")
    lines.append(f"- Decision mix: `{summary['actions_total']}`")
    lines.append(f"- Avg confidence: **{summary['avg_confidence']}**")
    lines.append(f"- Avg size_pct_of_book: **{summary['avg_size_pct']}**")
    lines.append("")
    lines.append("## Decision mix by country")
    lines.append("")
    lines.append("| Country | OVERWEIGHT | NEUTRAL | UNDERWEIGHT | ERROR |")
    lines.append("|---|---|---|---|---|")
    for country, c in summary["actions_by_country"].items():
        lines.append(
            f"| {country} | {c.get('OVERWEIGHT', 0)} | {c.get('NEUTRAL', 0)} | "
            f"{c.get('UNDERWEIGHT', 0)} | {c.get('ERROR', 0)} |"
        )
    lines.append("")
    lines.append("## Per-ticker decisions")
    lines.append("")
    lines.append("| Ticker | Resolved | Country | Sector | Action | Conf | Size% | Compliance | Horizon | E[ret%] | Runtime |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        action = r.action or f"ERR: {(r.error or '')[:30]}"
        lines.append(
            f"| {r.ticker} | {r.resolved} | {r.country} | {r.sector} | "
            f"{action} | "
            f"{r.confidence if r.confidence is not None else '—'} | "
            f"{r.size_pct if r.size_pct is not None else '—'} | "
            f"{r.compliance or '—'} | "
            f"{r.horizon_days if r.horizon_days is not None else '—'} | "
            f"{r.expected_return_pct if r.expected_return_pct is not None else '—'} | "
            f"{r.runtime_s}s |"
        )

    lines.append("")
    lines.append("## Bias diagnostics")
    lines.append("")
    by_country = summary["actions_by_country"]
    if "US" in by_country and "India" in by_country:
        us_overweight_rate = by_country["US"].get("OVERWEIGHT", 0) / max(1, sum(by_country["US"].values()))
        in_overweight_rate = by_country["India"].get("OVERWEIGHT", 0) / max(1, sum(by_country["India"].values()))
        gap = abs(us_overweight_rate - in_overweight_rate)
        if gap >= 0.5:
            lines.append(
                f"- ⚠️ Large country gap: US overweight rate {us_overweight_rate:.0%} "
                f"vs India {in_overweight_rate:.0%} (Δ {gap:.0%}). Investigate the "
                "agent backstories, the geopolitical task, and the news-tool's "
                "coverage parity."
            )
        else:
            lines.append(
                f"- ✅ Country gap within tolerance: US overweight rate "
                f"{us_overweight_rate:.0%} vs India {in_overweight_rate:.0%}."
            )
    if summary["actions_total"].get("NEUTRAL", 0) / max(1, summary["n_total"]) >= 0.7:
        lines.append(
            "- ⚠️ More than 70% of decisions are NEUTRAL — the system is biased "
            "toward inaction. Inspect the PM hard rules / quality reviewer / "
            "backtest hit-rate caps before drawing other conclusions."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sector-bias diagnostic basket.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated list of tickers (default: 10-ticker US+India basket).",
    )
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--risk-rounds", type=int, default=1)
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable episodic memory (avoids cross-run interference).")
    parser.add_argument("--out-dir", default="reports",
                        help="Directory for the JSON + markdown report (default: ./reports).")
    args = parser.parse_args()

    if args.tickers.strip():
        basket = []
        for raw in (t.strip() for t in args.tickers.split(",") if t.strip()):
            # If the user passed a known ticker, use its metadata; otherwise mark unknown.
            row = next((b for b in DEFAULT_BASKET if b[0].upper() == raw.upper()), None)
            if row:
                basket.append(row)
            else:
                basket.append((raw.upper(), "?", "?"))
    else:
        basket = DEFAULT_BASKET

    print(f"Sector-bias basket: {len(basket)} tickers")
    for tkr, country, sector in basket:
        print(f"  - {tkr:<12} [{country}] {sector}")
    print()

    rows: list[BasketRow] = []
    for tkr, country, sector in basket:
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] Running {tkr} …", flush=True)
        row = run_single(tkr, args.debate_rounds, args.risk_rounds, not args.no_memory)
        row.country = country
        row.sector = sector
        rows.append(row)
        print(
            f"  -> action={row.action} conf={row.confidence} size%={row.size_pct} "
            f"compliance={row.compliance} runtime={row.runtime_s}s "
            f"{'(error: ' + row.error + ')' if row.error else ''}",
            flush=True,
        )

    summary = _summarise(rows)
    out_dir = Path(args.out_dir).resolve()
    json_path, md_path = _write_report(rows, summary, out_dir)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
