"""Scale driver — forward episode collection + daily resolution for RL.

The RL training data pipeline has three stages (all verified working):

    collect   run the crew LIVE over a ticker universe and record a
              versioned ``source="forward"`` PENDING episode per name.
    resolve   sweep the whole episode store and price every episode whose
              horizon has elapsed into a RESOLVED, alpha-labelled row.
    status    report progress toward the RL training goal (how many
              forward episodes collected, how many resolved / RL-ready).

Why a dedicated forward collector (and not the backfill harness)
----------------------------------------------------------------
The backfill harness wraps every crew run in a strict point-in-time
``as_of_scope`` and stamps ``source="backfill"``.  That is correct for
*historical* labels but wrong for *forward* collection:

  * As-of=today PIT strictness can needlessly fail-close live tools that
    lack a date parameter, degrading a decision that has no future to leak.
  * Non-PIT tools still leak (e.g. ``get_event_proximity`` returns the
    current forward earnings calendar) — which is exactly why historical
    backfill labels are not trustworthy for RL and forward collection is.

So this collector runs the crew with no as_of scope, tags the episode
``source="forward"`` (a clean cohort you can train on), and otherwise
mirrors the live web runner's episode-recording (versioning stamps,
outcome_ts = decision_ts + the PM's horizon).

Scale plan
----------
Run ``collect`` on a ~200-name universe once per trading day; the
episodes accumulate as PENDING.  Run ``resolve`` daily (cheap; it only
touches episodes whose horizon has elapsed) so short-horizon calls turn
into RL-ready rows within weeks while medium/long ones mature over the
campaign.  Over ~100 days that is ~20k decisions feeding the Hermes
prediction harness and the L3 sizing trainer.

CLI
---
    python -m trading_crew.agentic.scale collect --tickers RELIANCE.NS,TCS.NS
    python -m trading_crew.agentic.scale collect --universe universe.txt --workers 2
    python -m trading_crew.agentic.scale resolve --reflect
    python -m trading_crew.agentic.scale status

``collect`` is resumable: it keys a per-day state file by date so re-running
the same day skips names already collected (episode ids are date-based, so a
re-run upserts rather than duplicates).  Ctrl-C is safe.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    return Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew"))


def _memory_path() -> Path:
    return _cache_dir() / "memory" / "episodes.jsonl"


def _collect_state_path(day: str) -> Path:
    """Legacy JSON checkpoint path — kept only for the one-time import below."""
    return _cache_dir() / "forward" / f"collect-{day}.json"


# Collection state now lives in the DB (``collect_state`` table, keyed by date)
# instead of ``~/.trading_crew/forward/collect-*.json``.  A legacy JSON file is
# imported into the DB once per process so an in-flight day's progress survives
# the cutover, then renamed aside so it's no longer the active store.
_MIGRATED_COLLECT_DAYS: set = set()


def _migrate_legacy_collect_json(day: str) -> None:
    if day in _MIGRATED_COLLECT_DAYS:
        return
    _MIGRATED_COLLECT_DAYS.add(day)
    path = _collect_state_path(day)
    if not path.exists():
        return
    try:
        legacy = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt checkpoint must not abort a run
        logger.warning("corrupt legacy collect state at %s; ignoring", path)
        return
    if not isinstance(legacy, dict) or not legacy:
        return
    try:
        from web.backend import prediction_db as db

        existing = db.load_collect_state(day)
        merged = {**legacy, **existing}  # DB rows (newer) win over the JSON
        db.save_collect_state(day, merged)
        path.rename(path.with_suffix(".json.migrated"))
        logger.info("imported %d legacy collect cells for %s into the DB", len(legacy), day)
    except Exception as exc:  # noqa: BLE001 — migration is best-effort
        logger.warning("legacy collect-state import failed for %s: %s", day, exc)


def _load_collect_state(day: str) -> dict:
    from web.backend import prediction_db as db

    _migrate_legacy_collect_json(day)
    return db.load_collect_state(day)


def _save_collect_state(day: str, state: dict) -> None:
    from web.backend import prediction_db as db

    db.save_collect_state(day, state)


# ---------------------------------------------------------------------------
# Forward episode recording (live — mirrors runner._record_episode)
# ---------------------------------------------------------------------------


def _record_forward_episode(memory, ticker: str, proposal, decision_ts: str, day: str) -> str:
    """Write a PENDING ``source="forward"`` episode; returns its id.

    The id is date-based (``TICKER-YYYY-MM-DD``) so one decision per name per
    day is recorded and a same-day re-run upserts instead of duplicating.
    """
    from trading_crew.agentic.memory import Episode, OutcomeStatus, Regime
    from trading_crew.agentic.versioning import (
        code_git_sha,
        prompts_hash,
        workflow_version,
    )

    try:
        d = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
    except ValueError:
        d = datetime.now(timezone.utc)
    horizon = max(int(getattr(proposal, "horizon_days", 0) or 0), 1)
    outcome_dt = d + timedelta(days=horizon)
    ep = Episode(
        episode_id=f"{ticker.upper()}-{day}",
        symbol=ticker.upper(),
        decision_ts=decision_ts,
        state_summary=(proposal.rationale or "")[:1000],
        regime=Regime.UNKNOWN,
        action_proposal=proposal.model_dump(),
        outcome_ts=outcome_dt.isoformat(),
        outcome_status=OutcomeStatus.PENDING,
        embargo_days=horizon,
        workflow_version=workflow_version(),
        prompts_hash=prompts_hash(),
        code_git_sha=code_git_sha(),
        source="forward",
    )
    memory.add(ep)
    return ep.episode_id


def _collect_one(ticker: str, *, debate_rounds: int, risk_rounds: int, use_memory: bool, day: str, memory, lock: threading.Lock):
    """Run the crew LIVE for one ticker and record a forward episode.

    Returns ``(episode_id, proposal)`` — the proposal (an ``ActionProposal``) is
    handed back so callers (the UI scale runner) can persist the decision without
    re-reading the episode store.
    """
    from trading_crew import TradingCrew
    from trading_crew.agentic.bridge import portfolio_decision_to_action_proposal

    canonical = ticker
    try:
        from trading_crew.market_context import resolve_ticker

        resolved = resolve_ticker(ticker)
        if resolved:
            canonical = resolved
    except Exception:  # noqa: BLE001
        pass

    decision_ts = datetime.now(timezone.utc).isoformat()
    tc = TradingCrew(
        ticker=canonical,
        debate_rounds=debate_rounds,
        risk_rounds=risk_rounds,
        memory=use_memory,
    )
    crew = tc.crew()
    try:
        result = crew.kickoff(inputs={"ticker": canonical, "trade_date": day})
    finally:
        # Release this run's CrewAI Memory (LanceDB handle + 8-worker save
        # pool) before the next name — otherwise a 200-name collect leaks
        # per-run threads/handles and RSS creeps until the VM OOMs.
        try:
            from trading_crew.memutil import close_crew_memory

            close_crew_memory(crew)
        except Exception:  # noqa: BLE001 — cleanup must never break a run
            pass
    decision = getattr(result, "pydantic", None)
    if decision is None:
        raise RuntimeError("crew produced no structured PortfolioDecision")
    proposal = portfolio_decision_to_action_proposal(
        decision, symbol=canonical, decision_ts=decision_ts
    )
    with lock:  # serialize episode writes across concurrent collector workers
        episode_id = _record_forward_episode(memory, canonical, proposal, decision_ts, day)
    return episode_id, proposal


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def run_collect(
    tickers: List[str],
    *,
    debate_rounds: int = 1,
    risk_rounds: int = 1,
    use_memory: bool = True,
    gap_seconds: float = 3.0,
    workers: int = 1,
    day: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_log: Optional[Callable[[dict], None]] = None,
    memory=None,
) -> dict:
    """Run the crew LIVE over ``tickers`` today; record forward episodes.

    Resumable: names already ``done`` in today's state file are skipped.
    """
    from trading_crew.agentic.memory import EpisodicMemory

    log = on_log or (lambda payload: None)
    should_stop = should_stop or (lambda: False)
    memory = memory or EpisodicMemory(_memory_path())
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lock = threading.Lock()

    universe: List[str] = []
    seen = set()
    for t in tickers:
        u = (t or "").strip()
        if u and u.upper() not in seen:
            seen.add(u.upper())
            universe.append(u)

    state = _load_collect_state(day)
    todo = [t for t in universe if state.get(t.upper(), {}).get("status") != "done"]
    log({"event": "start", "day": day, "universe": len(universe), "todo": len(todo), "workers": workers})

    done = failed = 0

    def _process(ticker: str) -> None:
        nonlocal done, failed
        if should_stop():
            return
        t0 = time.time()
        log({"event": "cell_start", "ticker": ticker})
        try:
            episode_id, proposal = _collect_one(
                ticker,
                debate_rounds=debate_rounds,
                risk_rounds=risk_rounds,
                use_memory=use_memory,
                day=day,
                memory=memory,
                lock=lock,
            )
            with lock:
                state[ticker.upper()] = {
                    "status": "done",
                    "episode_id": episode_id,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                _save_collect_state(day, state)
                done += 1
            try:
                decision = proposal.model_dump(mode="json")
            except Exception:  # noqa: BLE001 — logging must never break the run
                decision = None
            log({
                "event": "cell_done",
                "ticker": ticker,
                "episode_id": episode_id,
                "elapsed_sec": round(time.time() - t0, 1),
                "side": getattr(getattr(proposal, "side", None), "value", None),
                "conviction": getattr(proposal, "conviction_score", None),
                "target_weight": getattr(proposal, "target_weight", None),
                "expected_return_pct": getattr(proposal, "expected_return_pct", None),
                "decision": decision,
            })
        except Exception as exc:  # noqa: BLE001 — one bad name must not kill the run
            logger.exception("collect failed for %s", ticker)
            with lock:
                state[ticker.upper()] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                _save_collect_state(day, state)
                failed += 1
            log({"event": "cell_failed", "ticker": ticker, "error": str(exc)})
        finally:
            # Return this name's freed heap to the OS so RSS doesn't creep over
            # a 200-name batch (see trading_crew.memutil).
            try:
                from trading_crew.memutil import release_memory

                release_memory()
            except Exception:  # noqa: BLE001 — cleanup must never break the run
                pass

    if workers <= 1:
        for idx, ticker in enumerate(todo):
            if should_stop():
                break
            _process(ticker)
            if gap_seconds > 0 and idx < len(todo) - 1 and not should_stop():
                time.sleep(gap_seconds)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_process, todo))

    summary = {"day": day, "universe": len(universe), "done": done, "failed": failed, "skipped": len(universe) - len(todo)}
    log({"event": "finished", **summary})
    return summary


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def run_resolve(
    *,
    reflect: bool = False,
    ticker: Optional[str] = None,
    on_log: Optional[Callable[[dict], None]] = None,
    memory=None,
) -> dict:
    """Sweep the store: 30-day learning reviews and elapsed thesis horizons.

    ``resolve_pending_episodes`` with ``ticker=None`` walks all symbols; it
    scores the 30-day clock from desk_strategy.yaml even while a 180-day
    thesis is still pending, and skips clocks that are still in the future.
    """
    from trading_crew.agentic.memory import EpisodicMemory
    from trading_crew.agentic.reflection import resolve_pending_episodes

    log = on_log or (lambda payload: None)
    memory = memory or EpisodicMemory(_memory_path())

    llm = None
    if reflect:
        try:
            from trading_crew._common import get_llm

            llm = get_llm(temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reflection LLM unavailable (%s); resolving without it", exc)
            llm = None

    records = resolve_pending_episodes(
        ticker=ticker, memory=memory, llm=llm, skip_llm=(llm is None)
    )
    by_status: dict = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    summary = {"considered": len(records), "by_status": by_status}
    log({"event": "resolved", **summary})
    return summary


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def collection_status(*, goal: int = 20_000, memory=None) -> dict:
    """Report progress toward the RL training goal."""
    from trading_crew.agentic.memory import EpisodicMemory, OutcomeStatus

    memory = memory or EpisodicMemory(_memory_path())
    eps = list(memory.all_episodes())

    by_status: dict = {}
    by_source: dict = {}
    forward_resolved = 0
    rl_ready = 0  # resolved with a realised return (usable RL row)
    symbols = set()
    for e in eps:
        st = getattr(e.outcome_status, "value", str(e.outcome_status))
        by_status[st] = by_status.get(st, 0) + 1
        src = e.source or "unknown"
        by_source[src] = by_source.get(src, 0) + 1
        symbols.add(e.symbol)
        resolved = e.outcome_status == OutcomeStatus.RESOLVED and e.realised_return is not None
        if resolved:
            rl_ready += 1
            if src == "forward":
                forward_resolved += 1

    return {
        "total_episodes": len(eps),
        "unique_symbols": len(symbols),
        "by_status": by_status,
        "by_source": by_source,
        "rl_ready_rows": rl_ready,
        "forward_resolved": forward_resolved,
        "goal": goal,
        "goal_pct": round(100.0 * rl_ready / goal, 2) if goal else None,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_universe(path: str) -> List[str]:
    """One ticker per line; ``#`` comments and blank lines ignored."""
    out: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _configure_logging() -> None:
    """Route library ``logging`` to stderr at INFO for the CLI.

    The daily runner tees stderr into ``run.log`` so the post-run report can
    parse search-backend / screener / login / retry lines.  ``force=False`` so
    we don't clobber a handler an embedding host (e.g. the web app) already
    installed; the ``TRADINGCREW_SCALE_LOGLEVEL`` env var overrides the level.
    """
    level_name = os.environ.get("TRADINGCREW_SCALE_LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m trading_crew.agentic.scale",
        description="Forward episode collection + daily resolution for RL training.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("collect", help="run the crew live over a universe; record forward episodes")
    p_c.add_argument("--tickers", help="comma-separated tickers")
    p_c.add_argument("--universe", help="file with one ticker per line")
    p_c.add_argument("--debate-rounds", type=int, default=1)
    p_c.add_argument("--risk-rounds", type=int, default=1)
    p_c.add_argument("--no-memory", action="store_true", help="disable L1 episodic-memory retrieval during the run")
    p_c.add_argument("--gap-seconds", type=float, default=3.0, help="polite gap between names (workers=1 only)")
    p_c.add_argument("--workers", type=int, default=1, help="parallel crew runs (mind LLM/web rate limits)")

    p_r = sub.add_parser("resolve", help="resolve every episode whose horizon elapsed")
    p_r.add_argument("--reflect", action="store_true", help="generate LLM reflections for resolved episodes")
    p_r.add_argument("--ticker", default=None, help="restrict to one symbol")

    p_s = sub.add_parser("status", help="report progress toward the RL training goal")
    p_s.add_argument("--goal", type=int, default=20_000)

    args = parser.parse_args(argv)

    if args.cmd == "collect":
        tickers: List[str] = []
        if args.universe:
            tickers += _read_universe(args.universe)
        if args.tickers:
            tickers += [t for t in args.tickers.split(",")]
        if not tickers:
            print("error: pass --tickers and/or --universe", file=sys.stderr)
            return 2
        stop = threading.Event()
        try:
            summary = run_collect(
                tickers,
                debate_rounds=args.debate_rounds,
                risk_rounds=args.risk_rounds,
                use_memory=not args.no_memory,
                gap_seconds=args.gap_seconds,
                workers=args.workers,
                should_stop=stop.is_set,
                on_log=_emit,
            )
        except KeyboardInterrupt:
            stop.set()
            print("\nInterrupted — progress saved; re-run the same command to resume.", file=sys.stderr)
            return 130
        print("\n" + json.dumps(summary, indent=2))
        return 0

    if args.cmd == "resolve":
        summary = run_resolve(reflect=args.reflect, ticker=args.ticker, on_log=_emit)
        print("\n" + json.dumps(summary, indent=2))
        return 0

    if args.cmd == "status":
        print(json.dumps(collection_status(goal=args.goal), indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
