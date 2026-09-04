#!/usr/bin/env python3
"""google_search_experiment.py — find the best config for the google-headless backend.

Drives ``scripts/google-headless/search.js`` exactly the way production does
(same env: DISPLAY / GH_MODE / NODE_EXTRA_CA_CERTS, fresh subprocess per query,
same exit-code contract) but lets us sweep two levers in isolation:

  * ``--gap`` / ``--jitter`` — the break *between* successive google queries
    (the CAPTCHA / rate lever).
  * ``--concurrency`` — how many google-headless subprocesses run at once
    (the profile-contention lever: they all share one Chromium user-data-dir).

For each query it records the exit code, link count, latency and an outcome
class, then prints a per-config summary and appends every attempt to a JSONL so
several runs can be compared.  It touches nothing in the app — safe to run while
the server is up.

Exit-code contract (from search.js / _search_google_headless):
  0  + links   -> success
  4            -> captcha        (throttled; real rate-limit signal)
  5            -> empty          (page parsed, zero organic links)
  3            -> nav_error      ("execution context destroyed" — nav race)
  1            -> launch_error   (profile lock / spawn failure)
  timeout      -> timeout
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_SCRIPT = _HERE.parents[1] / "google-headless" / "search.js"

# Realistic batch-style queries: a mix of Indian + US names and a few long,
# multi-clause ones (the batch builds long queries) so we exercise the same
# consent / redirect paths the real workflow hits.
_QUERIES = [
    "Reliance Industries quarterly results revenue guidance FY27",
    "TCS management commentary margin outlook latest quarter",
    "Infosys deal wins large contract pipeline guidance",
    "HDFC Bank net interest margin trend analyst view",
    "ICICI Bank asset quality slippage guidance outlook",
    "Larsen Toubro order book execution guidance FY27",
    "Apple services revenue growth guidance analyst estimate",
    "Nvidia data center demand outlook next quarter guidance",
    "Microsoft Azure cloud growth deceleration guidance",
    "Tesla vehicle delivery guidance margin pressure outlook",
    "JPMorgan net interest income guidance rate cut impact",
    "Exxon Mobil production guidance capex outlook crude",
    "Tata Motors JLR margin guidance demand outlook",
    "Sun Pharma US generics pricing guidance specialty",
    "Amazon AWS reacceleration operating margin guidance",
    "Colgate Palmolive India volume growth guidance rural demand",
]


def _classify(rc: int, n_links: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if rc == 0 and n_links > 0:
        return "success"
    if rc == 0:
        return "empty"          # exited clean but nothing parsed
    if rc == 4:
        return "captcha"
    if rc == 5:
        return "empty"
    if rc == 3:
        return "nav_error"
    if rc == 1:
        return "launch_error"
    return f"exit_{rc}"


def _run_one(query: str, mode: str, timeout: float) -> dict:
    env = {**os.environ, "WS_NUM": "10", "GH_MODE": mode}
    node = shutil.which("node") or "node"
    t0 = time.monotonic()
    timed_out = False
    rc = -1
    n_links = 0
    err_head = ""
    try:
        proc = subprocess.run(
            [node, str(_SCRIPT), query],
            cwd=str(_SCRIPT.parent),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        rc = proc.returncode
        n_links = sum(1 for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("http"))
        err_head = (proc.stderr or "").strip().replace("\n", " ")[:160]
    except subprocess.TimeoutExpired:
        timed_out = True
    dt = time.monotonic() - t0
    return {
        "query": query,
        "rc": rc,
        "links": n_links,
        "latency_s": round(dt, 2),
        "outcome": _classify(rc, n_links, timed_out),
        "err": err_head,
    }


def run_config(gap: float, jitter: float, concurrency: int, num: int,
               mode: str, timeout: float, out_path: Path, label: str) -> dict:
    queries = [_QUERIES[i % len(_QUERIES)] for i in range(num)]
    results: list[dict] = []
    sem = threading.Semaphore(concurrency)
    launch_lock = threading.Lock()
    next_slot = [time.monotonic()]

    def worker(q: str) -> None:
        # Space *launch times* by gap+jitter (mirrors production's slot scheduler).
        import random
        with launch_lock:
            now = time.monotonic()
            slot = max(now, next_slot[0])
            step = gap + (random.uniform(0.0, jitter) if jitter > 0 else 0.0)
            next_slot[0] = slot + step
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        with sem:
            r = _run_one(q, mode, timeout)
        r["ts"] = datetime.now(timezone.utc).isoformat()
        r["config"] = label
        results.append(r)

    wall0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as ex:
        list(ex.map(worker, queries))
    wall = time.monotonic() - wall0

    with out_path.open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    counts = Counter(r["outcome"] for r in results)
    lats = [r["latency_s"] for r in results]
    ok = counts.get("success", 0)
    summary = {
        "config": label,
        "gap": gap, "jitter": jitter, "concurrency": concurrency,
        "num": num, "mode": mode,
        "success": ok,
        "success_pct": round(100.0 * ok / max(num, 1), 1),
        "captcha": counts.get("captcha", 0),
        "empty": counts.get("empty", 0),
        "nav_error": counts.get("nav_error", 0),
        "launch_error": counts.get("launch_error", 0),
        "timeout": counts.get("timeout", 0),
        "p50_latency_s": round(statistics.median(lats), 2) if lats else 0.0,
        "p95_latency_s": round(sorted(lats)[max(0, int(0.95 * len(lats)) - 1)], 2) if lats else 0.0,
        "wall_s": round(wall, 1),
    }
    return summary


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("config", 22), ("conc", 4), ("gap", 5), ("jit", 4),
        ("succ%", 6), ("succ", 5), ("capt", 5), ("empty", 6),
        ("nav", 4), ("lock", 5), ("tmo", 4), ("p50", 6), ("p95", 6), ("wall", 6),
    ]
    hdr = " ".join(name.rjust(w) if i else name.ljust(w) for i, (name, w) in enumerate(cols))
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        vals = [
            (r["config"][:22]).ljust(22),
            str(r["concurrency"]).rjust(4),
            f"{r['gap']:.0f}".rjust(5),
            f"{r['jitter']:.0f}".rjust(4),
            f"{r['success_pct']:.0f}".rjust(6),
            str(r["success"]).rjust(5),
            str(r["captcha"]).rjust(5),
            str(r["empty"]).rjust(6),
            str(r["nav_error"]).rjust(4),
            str(r["launch_error"]).rjust(5),
            str(r["timeout"]).rjust(4),
            f"{r['p50_latency_s']:.1f}".rjust(6),
            f"{r['p95_latency_s']:.1f}".rjust(6),
            f"{r['wall_s']:.0f}".rjust(6),
        ]
        print(" ".join(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=0.0, help="seconds between launch slots")
    ap.add_argument("--jitter", type=float, default=0.0, help="extra random 0..jitter per slot")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel google subprocesses")
    ap.add_argument("--num", type=int, default=12, help="number of queries")
    ap.add_argument("--mode", default=os.environ.get("GOOGLE_HEADLESS_MODE", "headful"))
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--out", default=str(_HERE.parent / "google_experiment_results.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--sweep", action="store_true",
                    help="run the built-in config sweep instead of a single config")
    args = ap.parse_args()

    if not _SCRIPT.is_file():
        print(f"search.js not found at {_SCRIPT}", file=sys.stderr)
        return 2
    if "DISPLAY" not in os.environ and args.mode == "headful":
        print("WARNING: headful mode but DISPLAY is unset (need Xvfb :99)", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    if args.sweep:
        # Levers: prove serialization (concurrency) then tune the break (gap).
        plan = [
            # label,                 gap, jitter, conc, num
            ("conc4-gap0 (batch-like)", 0.0, 0.0, 4, 12),
            ("serial-gap0",            0.0, 0.0, 1, 12),
            ("serial-gap3",            3.0, 2.0, 1, 12),
            ("serial-gap8",            8.0, 4.0, 1, 12),
            ("serial-gap15",          15.0, 5.0, 1, 12),
        ]
        for label, gap, jit, conc, num in plan:
            print(f"\n>>> running config: {label}", flush=True)
            s = run_config(gap, jit, conc, num, args.mode, args.timeout, out_path, label)
            summaries.append(s)
            print(f"    {s['success_pct']:.0f}% success "
                  f"({s['success']}/{num}), captcha={s['captcha']}, "
                  f"nav={s['nav_error']}, lock={s['launch_error']}, wall={s['wall_s']:.0f}s",
                  flush=True)
            # A short breather between configs so cooldowns / captcha state settle.
            time.sleep(10)
    else:
        label = args.label or f"conc{args.concurrency}-gap{args.gap:.0f}"
        s = run_config(args.gap, args.jitter, args.concurrency, args.num,
                       args.mode, args.timeout, out_path, label)
        summaries.append(s)

    _print_table(summaries)
    print(f"\nraw attempts appended to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
