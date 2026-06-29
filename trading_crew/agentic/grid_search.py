"""Walk-forward grid search — agentic training Level 3.

We treat the existing M6 walk-forward backtest as a *deterministic oracle*
that takes (proposals, sizing_config, gate_config) -> equity curve, then
sweep the config grid offline.  No LLM calls happen during the sweep
(that's the whole point — LLM calls are expensive and noisy; sizing/gate
parameters are cheap and reproducible).

What we tune
------------
We expose six knobs that materially change live behaviour:

- ``kelly_fraction`` — fraction of full-Kelly to use (caps over-betting).
- ``vol_target``     — per-position annualised vol cap.
- ``max_cvar_pct``   — max single-position 1d CVaR vs NAV.
- ``max_position_weight`` — hard concentration cap (sizing + gate).
- ``max_leverage``   — gross-leverage gate.
- ``drawdown_kill_threshold`` — kill-switch trigger.

Other parameters (``vol_lookback_days``, ``risk_mult_floor``,
``risk_mult_ceiling``) are left at defaults because changing them in a
backtest invalidates the historical proposals (the original sizing was
computed under one set of assumptions; changing them now would mean we're
comparing apples to oranges, not policy variants).

Why Deflated Sharpe is the chooser
----------------------------------
Bailey & López de Prado (2014) show that the more configurations you try,
the higher the *expected* Sharpe of the best one — even if every variant
is pure noise.  ``deflated_sharpe`` penalises this multiple-comparison
overfit; picking the highest *deflated* Sharpe (not raw) gives a
configuration that has a better-than-random chance of generalising
forward.  We pass ``n_trials`` = number of grid points so the deflation
is correctly calibrated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backtest import (
    BacktestMetrics,
    BacktestResult,
    Fold,
    WalkForwardConfig,
    compute_metrics,
    deflated_sharpe,
    generate_folds,
    run_walk_forward,
)
from .execution.contracts import ActionProposal
from .risk import GateConfig, SizingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid description + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridAxis:
    """One axis of the grid — a parameter name and the values to sweep."""

    name: str
    values: Tuple[float, ...]


@dataclass
class GridPoint:
    """One configuration in the grid + its measured metrics.

    ``rank_metric`` is what the UI sorts on by default; we set it to the
    Deflated Sharpe (falling back to raw Sharpe when DS can't be computed
    because the equity series is too short).
    """

    sizing_config: Dict[str, float]
    gate_config: Dict[str, float]
    n_folds: int
    metrics: BacktestMetrics
    rank_metric: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sizing_config": self.sizing_config,
            "gate_config": self.gate_config,
            "n_folds": self.n_folds,
            "metrics": {
                "total_return_pct": self.metrics.total_return_pct,
                "cagr": self.metrics.cagr,
                "annualised_vol": self.metrics.annualised_vol,
                "sharpe": self.metrics.sharpe,
                "sortino": self.metrics.sortino,
                "calmar": self.metrics.calmar,
                "max_drawdown": self.metrics.max_drawdown,
                "deflated_sharpe": self.metrics.deflated_sharpe,
                "n_periods": self.metrics.n_periods,
            } if self.metrics is not None else {},
            "rank_metric": self.rank_metric,
            "error": self.error,
        }


@dataclass
class GridSearchResult:
    """Top-level grid-search output."""

    ticker: str
    n_proposals: int
    n_points: int
    points: List[GridPoint] = field(default_factory=list)
    best_index: Optional[int] = None

    def best(self) -> Optional[GridPoint]:
        if self.best_index is None:
            return None
        return self.points[self.best_index]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_proposals": self.n_proposals,
            "n_points": self.n_points,
            "best_index": self.best_index,
            "best": self.best().to_dict() if self.best() is not None else None,
            "points": [p.to_dict() for p in self.points],
        }


# ---------------------------------------------------------------------------
# Default grid (kept small — Tier 3 = 3^3 = 27 points, ~seconds locally)
# ---------------------------------------------------------------------------


def default_grid() -> List[GridAxis]:
    """Defaults chosen to bracket what 'careful', 'standard', and 'punchy'
    institutional desks would actually run.  Tweak in the API call if you
    want a coarser/finer sweep.
    """
    return [
        GridAxis("kelly_fraction", (0.10, 0.25, 0.50)),
        GridAxis("vol_target", (0.07, 0.10, 0.15)),
        GridAxis("max_position_weight", (0.10, 0.20, 0.30)),
    ]


# Param-name -> which config it lives on. Used by the loop to build
# SizingConfig / GateConfig kwargs without case-bashing.
_SIZING_PARAMS = {
    "kelly_fraction",
    "vol_target",
    "max_cvar_pct",
    "max_position_weight",   # sizing also enforces it
}
_GATE_PARAMS = {
    "max_leverage",
    "drawdown_kill_threshold",
    "max_position_weight",   # gate enforces it too
    "max_position_cvar_pct",
}


def _build_configs(point: Dict[str, float]) -> Tuple[SizingConfig, GateConfig]:
    """Translate one grid point into (SizingConfig, GateConfig) instances.

    Parameters that appear in both configs (notably ``max_position_weight``)
    are mirrored so the sizer and the gate stay consistent.
    """
    sizing_kwargs = {k: v for k, v in point.items() if k in _SIZING_PARAMS}
    gate_kwargs = {k: v for k, v in point.items() if k in _GATE_PARAMS}
    return SizingConfig(**sizing_kwargs), GateConfig(**gate_kwargs)


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------


def run_grid_search(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, Any],
    walk_forward_config: Optional[WalkForwardConfig] = None,
    grid: Optional[Sequence[GridAxis]] = None,
    cost_model_name: str = "standard",
    rank_by: str = "deflated_sharpe",
) -> GridSearchResult:
    """Sweep ``grid`` against ``proposals`` via walk-forward backtest.

    Parameters
    ----------
    proposals : all logged ActionProposals (we'll fold them ourselves).
    ohlcv_by_symbol : dict ``{symbol: ohlcv DataFrame}`` covering the
        proposal window.  Reused across grid points; not refetched per config.
    walk_forward_config : if None we pick a small default (3/1/1) so even
        new users with few proposals get usable folds.
    grid : list of ``GridAxis``; defaults to ``default_grid()``.
    rank_by : "deflated_sharpe" (default — handles overfit) | "sharpe"
              (raw) | "sortino" | "calmar" | "total_return_pct".
    """
    if not proposals:
        return GridSearchResult(
            ticker=(next(iter(ohlcv_by_symbol)) if ohlcv_by_symbol else "?"),
            n_proposals=0, n_points=0,
        )

    grid = list(grid or default_grid())
    walk_forward_config = walk_forward_config or WalkForwardConfig(
        train_size=3, embargo_size=1, test_size=1,
    )

    # Build the full cartesian product up front so we know ``n_trials`` and
    # can pass it to ``compute_metrics`` for Deflated Sharpe deflation.
    axis_names = [ax.name for ax in grid]
    axis_values = [ax.values for ax in grid]
    grid_points_raw = [dict(zip(axis_names, combo)) for combo in product(*axis_values)]
    n_trials = max(1, len(grid_points_raw))

    folds: List[Fold] = generate_folds(n_obs=len(proposals), config=walk_forward_config)
    if not folds:
        result = GridSearchResult(
            ticker=next(iter(ohlcv_by_symbol)) if ohlcv_by_symbol else "?",
            n_proposals=len(proposals), n_points=0,
        )
        return result

    points: List[GridPoint] = []
    for raw in grid_points_raw:
        sizing_cfg, gate_cfg = _build_configs(raw)
        try:
            bt: BacktestResult = run_walk_forward(
                proposals=proposals,
                ohlcv_by_symbol=ohlcv_by_symbol,
                folds=folds,
                cost_model_name=cost_model_name,
                sizing_config=sizing_cfg,
                gate_config=gate_cfg,
            )
            # Recompute metrics with the correct ``n_trials`` so the
            # Deflated Sharpe is comparable across the grid.
            equity = bt.combined_equity
            corrected = compute_metrics(
                equity, n_trials=n_trials, periods_per_year=252,
            )
            rank_value = _extract_rank(corrected, rank_by)
            points.append(
                GridPoint(
                    sizing_config=_dict_of(sizing_cfg),
                    gate_config=_dict_of(gate_cfg),
                    n_folds=len(bt.folds),
                    metrics=corrected,
                    rank_metric=rank_value,
                )
            )
        except Exception as exc:
            logger.exception("Grid point failed: %s", raw)
            points.append(
                GridPoint(
                    sizing_config=_dict_of(sizing_cfg),
                    gate_config=_dict_of(gate_cfg),
                    n_folds=0,
                    metrics=None,  # type: ignore[arg-type]
                    rank_metric=float("-inf"),
                    error=str(exc),
                )
            )

    best_idx = _argmax(points)
    return GridSearchResult(
        ticker=next(iter(ohlcv_by_symbol)) if ohlcv_by_symbol else "?",
        n_proposals=len(proposals),
        n_points=len(points),
        points=points,
        best_index=best_idx,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dict_of(cfg) -> Dict[str, float]:
    """Serialise a frozen-dataclass config to a plain dict.  Drops
    private/derived fields by selecting only float/int attrs.
    """
    return {
        k: v for k, v in vars(cfg).items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    }


def _extract_rank(metrics: BacktestMetrics, rank_by: str) -> float:
    """Pull the ranking metric out of a ``BacktestMetrics`` bundle.

    Deflated Sharpe is preferred for overfit-resistance but isn't always
    available (short series).  We fall back to raw Sharpe only when DS is
    None, never silently to a more flattering metric.
    """
    if metrics is None:
        return float("-inf")
    v = getattr(metrics, rank_by, None)
    if v is None and rank_by == "deflated_sharpe":
        v = getattr(metrics, "sharpe", None)
    if v is None:
        return float("-inf")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    if f != f:  # NaN
        return float("-inf")
    return f


def _argmax(points: Sequence[GridPoint]) -> Optional[int]:
    """Return the index of the highest ``rank_metric``, or None if all -inf."""
    best_idx: Optional[int] = None
    best_val = float("-inf")
    for i, p in enumerate(points):
        if p.rank_metric > best_val:
            best_val = p.rank_metric
            best_idx = i
    if best_val == float("-inf"):
        return None
    return best_idx
