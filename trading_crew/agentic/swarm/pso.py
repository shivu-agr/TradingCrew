"""Particle Swarm Optimization over the sizing / gate parameter space.

This is the "swarm intelligence" that the Firefly / Fintrens references
put at the centre of their pitch, applied to the *tuning* problem the
project already solves with a brute-force grid (:mod:`agentic.grid_search`).

Why a swarm instead of a grid
-----------------------------
A grid pays ``k^d`` backtests for ``d`` parameters at ``k`` resolution —
the "fine" preset already costs ``5^3 = 125`` full walk-forwards.  A
particle swarm treats each candidate configuration as a *particle* that
remembers its own best position and is pulled toward the swarm's global
best (Kennedy & Eberhart 1995):

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x <- x + v         (clamped to the parameter bounds)

With a handful of particles over a handful of iterations the swarm
converges on the good region of the space in an order of magnitude fewer
backtests, while still exploring (the inertia + cognitive terms stop it
collapsing onto the first local optimum).

Anti-overfit chooser
---------------------
Exactly like the grid search, the objective is the **Deflated Sharpe
Ratio** (Bailey & López de Prado 2014), and we pass ``n_trials`` = the
total number of particle evaluations so the deflation honestly reflects
how many configurations the swarm tried before it reported a winner.

Determinism
-----------
The whole search is driven by a single seeded ``random.Random`` so a
given (proposals, ohlcv, seed) triple always produces the same swarm
trajectory — a hard requirement for the reproducible-backtest contract
(paper §8.3).  No LLM calls happen anywhere in here.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..backtest import (
    BacktestMetrics,
    CPCVConfig,
    WalkForwardConfig,
    compute_metrics,
    generate_cpcv_folds,
    generate_folds,
    run_cpcv,
    run_walk_forward,
)
from ..execution.contracts import ActionProposal
from ..grid_search import _build_configs, _dict_of, _extract_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


# Default search bounds for the sizing / gate knobs (mirrors the axes the
# grid search sweeps, but continuous).  Each entry is (low, high).
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "kelly_fraction": (0.05, 0.75),
    "vol_target": (0.05, 0.20),
    "max_position_weight": (0.05, 0.50),
}


@dataclass(frozen=True)
class PSOConfig:
    """Swarm hyper-parameters.

    - ``swarm_size``: number of particles.
    - ``iterations``: number of update steps (total backtests =
                      ``swarm_size * (iterations + 1)``).
    - ``inertia`` / ``cognitive`` / ``social``: the classic PSO weights
      ``w`` / ``c1`` / ``c2``.
    - ``vmax_frac``: velocity clamp as a fraction of each axis' range,
      to stop particles teleporting across the whole space in one step.
    - ``seed``: RNG seed for full reproducibility.
    """

    swarm_size: int = 12
    iterations: int = 8
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    vmax_frac: float = 0.3
    seed: int = 0

    def __post_init__(self):
        if self.swarm_size < 2:
            raise ValueError("swarm_size must be >= 2")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")


@dataclass
class SwarmPoint:
    """One evaluated configuration (a particle position at some step)."""

    params: Dict[str, float]
    score: float
    sizing_config: Dict[str, float] = field(default_factory=dict)
    gate_config: Dict[str, float] = field(default_factory=dict)
    metrics: Optional[BacktestMetrics] = None
    iteration: int = 0

    def to_dict(self) -> dict:
        m = self.metrics
        return {
            "params": self.params,
            "score": self.score,
            "iteration": self.iteration,
            "sizing_config": self.sizing_config,
            "gate_config": self.gate_config,
            "metrics": {
                "total_return_pct": m.total_return_pct,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "calmar": m.calmar,
                "max_drawdown": m.max_drawdown,
                "deflated_sharpe": m.deflated_sharpe,
                "win_rate": m.win_rate,
                "expectancy": m.expectancy,
                "payoff_ratio": m.payoff_ratio,
                "profit_factor": m.profit_factor,
                "alpha": m.alpha,
                "beta": m.beta,
                "n_periods": m.n_periods,
            } if m is not None else {},
        }


@dataclass
class PSOResult:
    """Top-level swarm-search output."""

    ticker: str
    n_proposals: int
    best_params: Dict[str, float]
    best_score: float
    best_metrics: Optional[BacktestMetrics]
    history: List[float] = field(default_factory=list)   # global-best per iteration
    evaluations: List[SwarmPoint] = field(default_factory=list)
    n_evaluations: int = 0
    validation: str = "walk_forward"

    def best(self) -> Optional[SwarmPoint]:
        best_pt: Optional[SwarmPoint] = None
        for pt in self.evaluations:
            if best_pt is None or pt.score > best_pt.score:
                best_pt = pt
        return best_pt

    def to_dict(self) -> dict:
        best_pt = self.best()
        return {
            "ticker": self.ticker,
            "n_proposals": self.n_proposals,
            "optimizer": "pso",
            "validation": self.validation,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "history": self.history,
            "n_evaluations": self.n_evaluations,
            "best": best_pt.to_dict() if best_pt is not None else None,
            "evaluations": [p.to_dict() for p in self.evaluations],
        }


# ---------------------------------------------------------------------------
# Generic particle swarm (objective-agnostic)
# ---------------------------------------------------------------------------


def particle_swarm_optimize(
    objective: Callable[[Dict[str, float]], float],
    bounds: Dict[str, Tuple[float, float]],
    config: PSOConfig = PSOConfig(),
) -> Tuple[Dict[str, float], float, List[float], List[Tuple[Dict[str, float], float]]]:
    """Maximise ``objective`` over the box ``bounds``.

    Returns ``(best_params, best_score, history, all_evaluations)`` where
    ``history`` is the global-best score after each iteration and
    ``all_evaluations`` is every ``(params, score)`` the swarm visited
    (so callers can surface the full search, not just the winner).

    The optimizer is deterministic given ``config.seed``.
    """
    rng = random.Random(config.seed)
    names = list(bounds.keys())
    lo = {n: bounds[n][0] for n in names}
    hi = {n: bounds[n][1] for n in names}
    span = {n: (hi[n] - lo[n]) for n in names}
    vmax = {n: config.vmax_frac * span[n] for n in names}

    def _clamp(n: str, v: float) -> float:
        return max(lo[n], min(hi[n], v))

    # Initialise particles uniformly in the box, velocities small-random.
    positions: List[Dict[str, float]] = []
    velocities: List[Dict[str, float]] = []
    for _ in range(config.swarm_size):
        pos = {n: rng.uniform(lo[n], hi[n]) for n in names}
        vel = {n: rng.uniform(-vmax[n], vmax[n]) for n in names}
        positions.append(pos)
        velocities.append(vel)

    all_evals: List[Tuple[Dict[str, float], float]] = []

    pbest = [dict(p) for p in positions]
    pbest_score = [float("-inf")] * config.swarm_size
    gbest: Dict[str, float] = dict(positions[0])
    gbest_score = float("-inf")

    # Evaluate the initial swarm (iteration 0).
    for i, pos in enumerate(positions):
        score = objective(pos)
        all_evals.append((dict(pos), score))
        pbest_score[i] = score
        pbest[i] = dict(pos)
        if score > gbest_score:
            gbest_score = score
            gbest = dict(pos)

    history: List[float] = [gbest_score]

    for _it in range(config.iterations):
        for i in range(config.swarm_size):
            for n in names:
                r1 = rng.random()
                r2 = rng.random()
                new_v = (
                    config.inertia * velocities[i][n]
                    + config.cognitive * r1 * (pbest[i][n] - positions[i][n])
                    + config.social * r2 * (gbest[n] - positions[i][n])
                )
                new_v = max(-vmax[n], min(vmax[n], new_v))
                velocities[i][n] = new_v
                positions[i][n] = _clamp(n, positions[i][n] + new_v)

            score = objective(positions[i])
            all_evals.append((dict(positions[i]), score))
            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = dict(positions[i])
            if score > gbest_score:
                gbest_score = score
                gbest = dict(positions[i])
        history.append(gbest_score)

    return gbest, gbest_score, history, all_evals


# ---------------------------------------------------------------------------
# Backtest-wired swarm search
# ---------------------------------------------------------------------------


def run_swarm_search(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, Any],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    pso_config: PSOConfig = PSOConfig(),
    walk_forward_config: Optional[WalkForwardConfig] = None,
    cpcv_config: Optional[CPCVConfig] = None,
    cost_model_name: str = "standard",
    rank_by: str = "deflated_sharpe",
    validation: str = "walk_forward",
) -> PSOResult:
    """Tune sizing / gate params with a particle swarm over backtests.

    The objective replays ``proposals`` through the M2/M5 pipeline under
    each candidate config and returns the ``rank_by`` metric (default
    Deflated Sharpe).  ``n_trials`` for the deflation is the *total*
    number of particle evaluations, so the reported winner is already
    penalised for the breadth of the search.
    """
    ticker = next(iter(ohlcv_by_symbol)) if ohlcv_by_symbol else "?"
    if not proposals:
        return PSOResult(
            ticker=ticker, n_proposals=0,
            best_params={}, best_score=float("-inf"), best_metrics=None,
            validation=validation,
        )

    bounds = dict(bounds or DEFAULT_BOUNDS)
    use_cpcv = validation == "cpcv"
    walk_forward_config = walk_forward_config or WalkForwardConfig(
        train_size=3, embargo_size=1, test_size=1,
    )
    cpcv_config = cpcv_config or CPCVConfig(n_groups=6, k_test=2, embargo_size=1)

    if use_cpcv:
        cpcv_folds = generate_cpcv_folds(n_obs=len(proposals), config=cpcv_config)
        if not cpcv_folds:
            return PSOResult(
                ticker=ticker, n_proposals=len(proposals),
                best_params={}, best_score=float("-inf"), best_metrics=None,
                validation=validation,
            )
        folds = []
    else:
        folds = generate_folds(n_obs=len(proposals), config=walk_forward_config)
        if not folds:
            return PSOResult(
                ticker=ticker, n_proposals=len(proposals),
                best_params={}, best_score=float("-inf"), best_metrics=None,
                validation=validation,
            )

    total_evals = pso_config.swarm_size * (pso_config.iterations + 1)
    evaluations: List[SwarmPoint] = []

    def _objective(params: Dict[str, float]) -> float:
        sizing_cfg, gate_cfg = _build_configs(params)
        try:
            if use_cpcv:
                res = run_cpcv(
                    proposals=proposals,
                    ohlcv_by_symbol=ohlcv_by_symbol,
                    cpcv_folds=cpcv_folds,
                    cost_model_name=cost_model_name,
                    sizing_config=sizing_cfg,
                    gate_config=gate_cfg,
                )
                equity = res.combined_equity
            else:
                res = run_walk_forward(
                    proposals=proposals,
                    ohlcv_by_symbol=ohlcv_by_symbol,
                    folds=folds,
                    cost_model_name=cost_model_name,
                    sizing_config=sizing_cfg,
                    gate_config=gate_cfg,
                )
                equity = res.combined_equity
            metrics = compute_metrics(equity, n_trials=total_evals, periods_per_year=252)
            score = _extract_rank(metrics, rank_by)
        except Exception as exc:  # noqa: BLE001 — one bad config shouldn't kill the swarm
            logger.warning("Swarm objective failed for %s: %s", params, exc)
            metrics = None
            score = float("-inf")
        evaluations.append(SwarmPoint(
            params=dict(params),
            score=score,
            sizing_config=_dict_of(sizing_cfg),
            gate_config=_dict_of(gate_cfg),
            metrics=metrics,
        ))
        return score

    best_params, best_score, history, _all = particle_swarm_optimize(
        _objective, bounds, pso_config,
    )

    # Tag each evaluation with the iteration it belongs to (swarm_size per step).
    for idx, pt in enumerate(evaluations):
        pt.iteration = idx // pso_config.swarm_size

    best_pt = None
    for pt in evaluations:
        if best_pt is None or pt.score > best_pt.score:
            best_pt = pt

    return PSOResult(
        ticker=ticker,
        n_proposals=len(proposals),
        best_params=best_params,
        best_score=best_score,
        best_metrics=best_pt.metrics if best_pt else None,
        history=history,
        evaluations=evaluations,
        n_evaluations=len(evaluations),
        validation=validation,
    )
