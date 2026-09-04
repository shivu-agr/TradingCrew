#!/usr/bin/env python3
"""LEGACY Mass promote — install confirmed 1PO/2TO/3PO into promoted.json.

Hermes already applies the open experiment via ``apply_harness`` + the graph
router. This CLI is kept for Mass-alone A/B. It writes ``promoted.json`` only
if the held-out cohort beats the current book on WIN rate AND mean alpha.
A single LOSS never promotes.

Usage:

    .venv/bin/python scripts/promote_graph.py \\
        --current-version <tag> --candidate-version <tag>

    .venv/bin/python scripts/promote_graph.py --held-out-after 2026-06-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_crew.agentic.mass import (
    collect_candidate_prompts,
    evaluate_cohort,
    load_promoted,
    promote,
    stage_2to_propose_action,
    stage_3po_global,
)
from trading_crew.agentic.memory.episodic import EpisodicMemory, OutcomeStatus
from trading_crew.agentic.versioning import workflow_version


def _memory() -> EpisodicMemory:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return EpisodicMemory(Path(cache) / "memory" / "episodes.jsonl")


def _resolved(mem: EpisodicMemory):
    out = []
    for ep in mem.iter_all():
        if ep.outcome_status != OutcomeStatus.RESOLVED:
            continue
        out.append(ep)
    return out


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gated Mass promote for v3 graph/prompts.")
    p.add_argument("--current-version", default="", help="workflow_version of the live cohort")
    p.add_argument("--candidate-version", default="", help="workflow_version of the candidate cohort")
    p.add_argument(
        "--held-out-after",
        default="",
        help="ISO date: episodes at/after this date are the candidate slice",
    )
    p.add_argument("--playbook-key", default="", help="topology key to stamp promoted_action on")
    p.add_argument("--action", default="", help="PLAYBOOK_ACTION to install (default: 2TO proposal)")
    p.add_argument("--dry-run", action="store_true", help="Print the gate result without writing")
    return p.parse_args()


def main() -> int:
    args = _parse()
    mem = _memory()
    eps = _resolved(mem)
    current_ver = args.current_version or (workflow_version() or "")
    cand_ver = args.candidate_version
    cutoff = args.held_out_after

    if cand_ver:
        current = [e for e in eps if (e.workflow_version or "") == current_ver]
        candidate = [e for e in eps if (e.workflow_version or "") == cand_ver]
    elif cutoff:
        current = [e for e in eps if str(e.outcome_ts or "")[:10] < cutoff]
        candidate = [e for e in eps if str(e.outcome_ts or "")[:10] >= cutoff]
    else:
        print("error: pass --candidate-version or --held-out-after", file=sys.stderr)
        return 2

    cur_m = evaluate_cohort(current)
    cand_m = evaluate_cohort(candidate)
    print(f"current n={cur_m['n']:.0f} win_rate={cur_m['win_rate']:.3f} mean_alpha={cur_m['mean_alpha']:+.4f}")
    print(f"candidate n={cand_m['n']:.0f} win_rate={cand_m['win_rate']:.3f} mean_alpha={cand_m['mean_alpha']:+.4f}")

    action = args.action or stage_2to_propose_action() or "default"
    prompts = collect_candidate_prompts()
    global_add = ""
    gpath = Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")) / "v3" / "candidates" / "global_3po.md"
    if gpath.is_file():
        global_add = gpath.read_text(encoding="utf-8")
    elif not args.dry_run:
        global_add = stage_3po_global() or ""

    if args.dry_run:
        already = load_promoted()
        print(f"dry-run action={action} prompts={list(prompts)} existing={bool(already)}")
        beats = (
            cand_m["n"] >= 3
            and cand_m["win_rate"] >= cur_m["win_rate"]
            and cand_m["mean_alpha"] >= cur_m["mean_alpha"]
        )
        print("would_promote" if beats else "would_refuse")
        return 0 if beats else 1

    ok = promote(
        candidate_action=action,
        desk_instructions=prompts,
        global_addendum=global_add,
        candidate_metrics=cand_m,
        current_metrics=cur_m,
        playbook_key_name=args.playbook_key,
    )
    print("promoted" if ok else "refused")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
