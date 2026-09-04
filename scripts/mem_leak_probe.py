"""Deterministic per-run memory-leak probe.

Runs the EXACT batch per-name path (``scale._collect_one`` → ``TradingCrew``
→ ``crew().kickoff``) sequentially N times in ONE process and reports, after
each run (and after gc + malloc_trim):

    * process RSS (VmRSS from /proc/self/status)
    * live thread count
    * tracemalloc top allocation sites (cumulative), so a site whose retained
      size grows run-over-run is the leak.

Usage:
    ./.venv/bin/python scripts/mem_leak_probe.py TICKER1 TICKER2 TICKER3
"""

from __future__ import annotations

import gc
import sys
import threading
import tracemalloc
from datetime import datetime, timezone


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _report(tag: str, snap_prev):
    from trading_crew.memutil import release_memory

    release_memory()  # gc.collect + malloc_trim (mirror the production hook)
    snap = tracemalloc.take_snapshot()
    print(f"\n===== {tag} | RSS={_rss_mb():.0f}MB | threads={threading.active_count()} =====")
    top = snap.statistics("lineno")[:12]
    for st in top:
        print(f"   {st.size/1e6:8.1f} MB  {st.count:>8} objs  {st.traceback[0]}")
    if snap_prev is not None:
        print("   --- growth vs previous run ---")
        for st in snap.compare_to(snap_prev, "lineno")[:12]:
            if st.size_diff > 0:
                print(f"   +{st.size_diff/1e6:7.1f} MB  {st.traceback[0]}")
    return snap


def main() -> None:
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "KO"]
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic import scale

    tracemalloc.start(15)
    memory = EpisodicMemory(scale._memory_path())
    lock = threading.Lock()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"baseline RSS={_rss_mb():.0f}MB threads={threading.active_count()}")
    prev = None
    for i, tk in enumerate(tickers, 1):
        t0 = datetime.now()
        try:
            scale._collect_one(
                tk, debate_rounds=1, risk_rounds=1, use_memory=True,
                day=day, memory=memory, lock=lock,
            )
            ok = "ok"
        except Exception as exc:  # noqa: BLE001
            ok = f"FAILED {type(exc).__name__}: {exc}"
        dt = (datetime.now() - t0).total_seconds()
        prev = _report(f"after run {i}/{len(tickers)} [{tk}] {ok} ({dt:.0f}s)", prev)


if __name__ == "__main__":
    main()
