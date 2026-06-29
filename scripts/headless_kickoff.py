"""Headless WebSocket client — trigger an analysis run and block until it
completes, then save the saved-run path so it can be fed straight to
``scripts/audit_run.py``.

Why
---
The web UI is the canonical entry point for a kickoff, but a CLI client
is useful for: (a) running automation, smoke tests, or audits without
opening a browser; (b) reproducing a UI run from logs (the saved
RunRecord is the same).  The UI and this script speak the same
``/ws/analyze`` protocol — the first frame is the config JSON, the
rest is a stream of events ending at ``run_completed`` or
``run_error``.

Usage
-----
::

    ../.venv/bin/python scripts/headless_kickoff.py \\
        --ticker LT.NS \\
        --port 8365 \\
        --memory \\
        --debate-rounds 2 \\
        --risk-rounds 1

The script prints a compact one-line summary for each node_started /
node_completed / tool_call / tool_result / final_decision /
run_completed event so you can follow progress in the terminal while
the workflow runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from typing import Any, Dict

try:
    import websockets
except ImportError:
    sys.exit(
        "Missing dependency 'websockets'. Install with: "
        "../.venv/bin/python -m pip install websockets"
    )


def _short(text: str, limit: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_event(ev: Dict[str, Any]) -> str:
    t = ev.get("type", "?")
    role = ev.get("role") or ev.get("node") or ""
    if t == "node_started":
        return f"  ▶ {role}"
    if t == "node_completed":
        chars = len(ev.get("output") or "")
        flag = " (DEGRADED)" if ev.get("degraded") else ""
        return f"  ✓ {role}{flag} — {chars} chars"
    if t == "tool_call":
        return f"     · {role} → {ev.get('tool')}({_short(json.dumps(ev.get('args') or {}), 80)})"
    if t == "tool_result":
        body = ev.get("output") or ev.get("result") or ""
        err = ev.get("error")
        if err:
            return f"     · {role} ← {ev.get('tool')} ERROR: {_short(err, 80)}"
        return f"     · {role} ← {ev.get('tool')} ({len(body)} chars)"
    if t == "agent_step":
        return f"     · {role}: {_short(ev.get('content') or '', 120)}"
    if t == "cascade_status":
        return (
            f"  CASCADE  regime={ev.get('regime')} "
            f"route={ev.get('route')} reason={_short(ev.get('reason') or '', 80)}"
        )
    if t == "final_decision":
        d = ev.get("decision") or {}
        return (
            f"  FINAL  action={d.get('action')} "
            f"size={d.get('size_pct_of_book')} conf={d.get('confidence')}"
        )
    if t == "run_completed":
        return f"  ── RUN COMPLETED ──"
    if t == "error":
        return f"  ── RUN ERROR: {ev.get('message') or ev.get('error')} ──"
    return f"  [{t}]"


async def _run(args: argparse.Namespace) -> int:
    uri = f"ws://127.0.0.1:{args.port}/ws/analyze"
    cfg = {
        "ticker": args.ticker,
        "trade_date": args.trade_date or date.today().isoformat(),
        "asset_class": args.asset_class,
        "book": args.book,
        "memory": args.memory,
        "max_debate_rounds": args.debate_rounds,
        "max_risk_rounds": args.risk_rounds,
        "tools_enabled": None,  # server defaults
        "llm_preset": args.llm_preset,
        "embedding_preset": args.embedding_preset,
    }
    print(f"Connecting to {uri} …")
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps(cfg))
        print(f"Sent config: ticker={cfg['ticker']} "
              f"debate={cfg['max_debate_rounds']} risk={cfg['max_risk_rounds']}")
        run_id = None
        async for raw in ws:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                print("  [non-json frame]", raw[:200])
                continue
            line = _format_event(ev)
            print(line, flush=True)
            if ev.get("type") in ("run_completed", "error"):
                run_id = ev.get("run_id")
                break
        if run_id:
            print(f"\nrun_id={run_id}")
            print(
                f"Audit with: ../.venv/bin/python scripts/audit_run.py "
                f"--ticker {args.ticker} --run-id {run_id}"
            )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", required=True)
    p.add_argument("--trade-date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--port", type=int, default=8365)
    p.add_argument("--asset-class", default="stock", choices=["stock", "commodity"])
    p.add_argument("--book", default="paper", choices=["paper", "prod"])
    p.add_argument("--memory", action="store_true", default=True)
    p.add_argument("--no-memory", dest="memory", action="store_false")
    p.add_argument("--debate-rounds", type=int, default=2)
    p.add_argument("--risk-rounds", type=int, default=1)
    p.add_argument("--llm-preset", default=None, help="LLM preset id (default: server default)")
    p.add_argument("--embedding-preset", default=None, help="Embedding preset id (default: server default)")
    args = p.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
