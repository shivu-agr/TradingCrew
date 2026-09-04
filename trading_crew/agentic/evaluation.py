"""Reusable end-to-end evaluation service (the "compact" glue layer).

Everything needed to turn a store of logged decisions into a validated
prediction lived, until now, *inside* the FastAPI handlers in
``web/backend/app.py`` (and was re-implemented ad-hoc in throwaway
``verify.py`` scripts).  That made "run all the parts together" impossible
to call, test, or script outside the web server.

This module extracts that orchestration into one place so the web app, the
``trading_crew.agentic.validate`` CLI, and tests share a single code path:

    episodes  --(cohort filter)-->  ActionProposals
              --(OHLCV)-->          walk-forward / CPCV backtest   (M6)
              --(trades)-->         PnL attribution                (M2/M5)
              --(L3 PSO)-->         tuned expectancy signal
              --(fuse)-->           guardrailed live signal        (fusion)

It is deliberately dependency-light: the backtest / attribution / fusion
primitives it calls are pure Python, and the only heavy import (yfinance,
for the default OHLCV provider) is lazy so importing this module stays
cheap.  Callers may inject their own ``ohlcv_provider`` (the web app passes
its cached ``charts._fetch_ohlcv``) to keep a single fetch/caching policy.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from trading_crew.agentic.execution.contracts import ActionProposal, ActionSide

OHLCVProvider = Callable[[str, datetime, int], "object"]  # (ticker, end_date, lookback_days) -> DataFrame


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------

def default_cache_dir() -> str:
    """The episode/memory root, matching the web app + backfill convention."""
    return os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")


def load_cohort(
    ticker: str,
    *,
    cache_dir: Optional[str] = None,
    workflow_version: Optional[str] = None,
    source: Optional[str] = None,
) -> List["object"]:
    """Load episodes for ``ticker``, optionally restricted to a cohort.

    ``workflow_version`` / ``source`` apply the Phase-0 provenance filter so
    callers can evaluate *only* trustworthy (stamped) decisions instead of a
    mix of legacy and current-workflow episodes.
    """
    from trading_crew.agentic.memory import EpisodicMemory

    root = Path(cache_dir or default_cache_dir())
    mem = EpisodicMemory(root / "memory" / "episodes.jsonl")
    episodes = [e for e in mem.all_episodes() if e.symbol.upper() == ticker.upper()]
    if workflow_version or source:
        from trading_crew.agentic.versioning import filter_episodes_by_cohort

        episodes = filter_episodes_by_cohort(
            episodes,
            workflow_version=workflow_version or None,
            source=source or None,
        )
    return episodes


def latest_workflow_version(episodes: Sequence["object"]) -> Optional[str]:
    """Return the ``workflow_version`` stamped on the most recent episode.

    ``workflow_version`` is a ``{git8}-{prompts8}`` tag — NOT lexically
    orderable — so "latest" is defined by the newest ``decision_ts`` among
    episodes that actually carry a stamp.  Returns ``None`` when no episode is
    stamped (pre-Phase-0 history), in which case the caller should fall back to
    "all versions".
    """
    stamped = [e for e in episodes if getattr(e, "workflow_version", None)]
    if not stamped:
        return None
    latest = max(stamped, key=lambda e: getattr(e, "decision_ts", "") or "")
    return getattr(latest, "workflow_version", None)


def proposals_from_episodes(episodes: Sequence["object"]) -> List[ActionProposal]:
    """Rebuild the sorted ``ActionProposal`` timeline from episodes.

    Malformed proposals are skipped (never fatal) so one bad row can't sink
    an otherwise-valid cohort.
    """
    proposals: List[ActionProposal] = []
    for ep in episodes:
        try:
            proposals.append(ActionProposal(**ep.action_proposal))
        except Exception:  # noqa: BLE001 — a single bad row must not abort eval
            continue
    proposals.sort(key=lambda p: p.decision_ts)
    return proposals


# ---------------------------------------------------------------------------
# Default OHLCV provider (PIT-guarded yfinance daily fetch)
# ---------------------------------------------------------------------------

def default_ohlcv_provider(ticker: str, end_date: datetime, lookback_days: int):
    """Daily OHLCV via yfinance, clamped to the active PIT cutoff.

    Mirrors ``web.backend.charts._fetch_ohlcv`` (buffered window so long
    warm-up indicators seed) but lives in the library so evaluation has no
    dependency on the web package.
    """
    from datetime import timedelta

    import pandas as pd
    import yfinance as yf

    try:
        from trading_crew.data_sources import pit

        cutoff = pit.cutoff()
        if cutoff is not None:
            cutoff_naive = cutoff.replace(tzinfo=None)
            if end_date > cutoff_naive:
                end_date = cutoff_naive
    except Exception:  # noqa: BLE001 — PIT clamp is best-effort, never fatal
        pass

    buffer_days = max(lookback_days + 360, 540)
    start = end_date - timedelta(days=buffer_days)
    df = yf.Ticker(ticker).history(start=start.date(), end=(end_date + timedelta(days=1)).date())
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    ticker: str
    n_episodes: int = 0
    n_proposals: int = 0
    sides: Dict[str, int] = field(default_factory=dict)
    validation: str = "walk_forward"
    n_folds: int = 0
    metrics: Optional[dict] = None
    trade_status_counts: Dict[str, int] = field(default_factory=dict)
    n_filled: int = 0
    attribution: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_episodes": self.n_episodes,
            "n_proposals": self.n_proposals,
            "sides": self.sides,
            "validation": self.validation,
            "n_folds": self.n_folds,
            "metrics": self.metrics,
            "trade_status_counts": self.trade_status_counts,
            "n_filled": self.n_filled,
            "attribution": self.attribution,
            "error": self.error,
        }


def _finite(v):
    try:
        if v is None or v != v or v in (float("inf"), float("-inf")):
            return None
    except Exception:  # noqa: BLE001
        return None
    return v


def _metrics_dict(m) -> dict:
    return {
        "total_return_pct": m.total_return_pct,
        "cagr": m.cagr,
        "annualised_vol": m.annualised_vol,
        "sharpe": _finite(m.sharpe),
        "sortino": _finite(m.sortino),
        "calmar": _finite(m.calmar),
        "max_drawdown": m.max_drawdown,
        "deflated_sharpe": _finite(m.deflated_sharpe),
        "n_periods": m.n_periods,
        "win_rate": m.win_rate,
        "expectancy": m.expectancy,
        "profit_factor": _finite(m.profit_factor),
        "benchmark_return_pct": m.benchmark_return_pct,
        "alpha": _finite(m.alpha),
        "sharpe_ci_low": _finite(m.sharpe_ci_low),
        "sharpe_ci_high": _finite(m.sharpe_ci_high),
    }


# ---------------------------------------------------------------------------
# The evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_ticker(
    ticker: str,
    *,
    cache_dir: Optional[str] = None,
    workflow_version: Optional[str] = None,
    source: Optional[str] = None,
    lookback_years: int = 5,
    validation: str = "walk_forward",
    train_size: int = 3,
    embargo_size: int = 1,
    test_size: int = 1,
    cpcv_groups: int = 6,
    cpcv_k: int = 2,
    cost_model: str = "standard",
    ohlcv_provider: Optional[OHLCVProvider] = None,
) -> EvalReport:
    """Replay a ticker's logged decisions through the M6 backtest + attribution.

    Returns an :class:`EvalReport` — never raises for "not enough data";
    those become ``error`` strings so a caller iterating a cohort keeps going.
    """
    from trading_crew.agentic.backtest import (
        CPCVConfig,
        WalkForwardConfig,
        attribute_decisions,
        generate_cpcv_folds,
        generate_folds,
        run_cpcv,
        run_walk_forward,
        sized_weights_from_trades,
    )

    provider = ohlcv_provider or default_ohlcv_provider
    episodes = load_cohort(
        ticker, cache_dir=cache_dir, workflow_version=workflow_version, source=source
    )
    report = EvalReport(ticker=ticker, n_episodes=len(episodes), validation=validation)
    if not episodes:
        report.error = "No episodes for this ticker/cohort."
        return report

    proposals = proposals_from_episodes(episodes)
    report.n_proposals = len(proposals)
    if not proposals:
        report.error = "All episodes had invalid proposals."
        return report

    counts: Dict[str, int] = {}
    for p in proposals:
        counts[p.side.value] = counts.get(p.side.value, 0) + 1
    report.sides = counts

    last_ts = max(p.decision_ts for p in proposals)
    try:
        end_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    ohlcv = provider(ticker, end_dt, 365 * lookback_years)
    if ohlcv is None or len(ohlcv) == 0:
        report.error = "No OHLCV history available for the backtest."
        return report
    ohlcv_by_symbol = {ticker: ohlcv}

    if validation == "cpcv":
        folds = generate_cpcv_folds(
            n_obs=len(proposals),
            config=CPCVConfig(n_groups=cpcv_groups, k_test=cpcv_k, embargo_size=embargo_size),
        )
        report.n_folds = len(folds)
        if not folds:
            report.error = (
                f"Not enough proposals ({len(proposals)}) for CPCV "
                f"n_groups={cpcv_groups}, k_test={cpcv_k}."
            )
            return report
        cp = run_cpcv(
            proposals=proposals, ohlcv_by_symbol=ohlcv_by_symbol,
            cpcv_folds=folds, cost_model_name=cost_model,
        )
        report.metrics = _metrics_dict(cp.aggregate_metrics)
        report.metrics["mean_oos_sharpe"] = _finite(cp.mean_oos_sharpe)
        report.metrics["median_oos_sharpe"] = _finite(cp.median_oos_sharpe)
        all_trades = [t for path in cp.paths for t in path.trades]
    else:
        folds = generate_folds(
            n_obs=len(proposals),
            config=WalkForwardConfig(
                train_size=train_size, embargo_size=embargo_size, test_size=test_size
            ),
        )
        report.n_folds = len(folds)
        if not folds:
            report.error = (
                f"Not enough proposals ({len(proposals)}) for "
                f"train={train_size}+embargo={embargo_size}+test={test_size}."
            )
            return report
        res = run_walk_forward(
            proposals=proposals, ohlcv_by_symbol=ohlcv_by_symbol,
            folds=folds, cost_model_name=cost_model,
        )
        report.metrics = _metrics_dict(res.overall_metrics)
        all_trades = [t for fr in res.folds for t in fr.trades]

    status_counts: Dict[str, int] = {}
    for t in all_trades:
        status_counts[t.status] = status_counts.get(t.status, 0) + 1
    report.trade_status_counts = status_counts
    report.n_filled = sum(
        1 for t in all_trades if "FILL" in str(t.status).upper()
    )

    try:
        attr = attribute_decisions(
            proposals, ohlcv_by_symbol,
            sized_weights=sized_weights_from_trades(all_trades),
            cost_model_name=cost_model,
        )
        report.attribution = attr.to_dict()
    except Exception:  # noqa: BLE001 — attribution is diagnostic, never fatal
        report.attribution = None

    return report


# ---------------------------------------------------------------------------
# Signal fusion for the *latest* decision (L3 PSO expectancy + LLM [+ RL])
# ---------------------------------------------------------------------------

def latest_fused_signal(
    ticker: str,
    *,
    cache_dir: Optional[str] = None,
    workflow_version: Optional[str] = None,
    source: Optional[str] = None,
    lookback_years: int = 5,
    cost_model: str = "standard",
    ohlcv_provider: Optional[OHLCVProvider] = None,
) -> Optional[dict]:
    """Fuse the most recent LLM decision with the L3 PSO-tuned expectancy and
    (if promoted) the RL policy recommendation into one guardrailed signal.

    Returns ``None`` when there is no usable decision to fuse.
    """
    from trading_crew.agentic.signals.fusion import (
        LLMSignal,
        PolicySignal,
        fuse_signals,
        pso_tuned_expectancy,
    )

    provider = ohlcv_provider or default_ohlcv_provider
    episodes = load_cohort(
        ticker, cache_dir=cache_dir, workflow_version=workflow_version, source=source
    )
    proposals = proposals_from_episodes(episodes)
    if not proposals:
        return None

    latest = proposals[-1]
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        end_dt = datetime.fromisoformat(
            max(p.decision_ts for p in proposals).replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        pass
    ohlcv = provider(ticker, end_dt, 365 * lookback_years)
    ohlcv_by_symbol = {ticker: ohlcv} if ohlcv is not None and len(ohlcv) else {}

    expectancy = None
    if ohlcv_by_symbol:
        try:
            expectancy = pso_tuned_expectancy(
                proposals, ohlcv_by_symbol, cost_model_name=cost_model
            )
        except Exception:  # noqa: BLE001
            expectancy = None

    policy = None
    if ohlcv_by_symbol:
        try:
            from trading_crew.agentic.rl.inference import load_policy

            client = load_policy(ticker)
            if client is not None:
                rec = client.recommend(ohlcv, as_of=latest.decision_ts)
                policy = PolicySignal(
                    target_weight=rec.best_target_weight, confidence=rec.confidence
                )
        except Exception:  # noqa: BLE001 — RL is optional; absence is not an error
            policy = None

    llm = LLMSignal(
        side=latest.side,
        target_weight=latest.target_weight,
        conviction=latest.conviction_score,
        expected_return_pct=latest.expected_return_pct,
    )
    fused = fuse_signals(llm, policy=policy, expectancy=expectancy, crowd=None)
    return {
        "ticker": ticker,
        "as_of": latest.decision_ts,
        "llm_side": latest.side.value,
        "policy_available": policy is not None,
        "expectancy_available": expectancy is not None,
        "fused": {
            "side": fused.side,
            "score": fused.score,
            "confidence": fused.confidence,
            "target_weight": fused.target_weight,
            "guardrails": list(fused.guardrails),
        },
    }
