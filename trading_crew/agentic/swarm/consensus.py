"""Consensus ensemble — the "strategy swarm" whose collective vote trades.

Firefly frames swarm intelligence as *many* simple agents whose collective
behaviour produces a better decision than any single one.  Here the
"agents" are candidate strategy configurations (the top-K particles from
the PSO swarm, or the top-K grid points): each one wanted a particular
signed weight on each (timestamp, symbol) decision, and we fuse those
into a single **consensus weight** — weighted by each member's robustness
(Deflated Sharpe) — then backtest the consensus.

Two things fall out of this that a single "best" config never gives you:

1. **Robustness by construction.**  A weight that only the single
   IS-optimal config liked gets diluted; a weight the whole swarm agrees
   on survives.  This is the ensemble analogue of the PBO defence.
2. **Agreement diagnostics.**  We report the directional agreement (do
   the members even agree on the *sign*?) and the weight dispersion
   (how tightly do their sizes cluster?).  Low agreement is an honest
   "the swarm is confused here" signal.

Everything is deterministic and LLM-free; we reuse the M2/M5 replay so
the consensus is scored on exactly the same footing as its members.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..backtest import BacktestMetrics, run_backtest
from ..execution.contracts import ActionProposal, ActionSide
from ..risk import GateConfig, SizingConfig

logger = logging.getLogger(__name__)


# A pass-through sizing config: caps set so high that the *intent*
# (the consensus target weight) is the binding constraint, so the
# engine trades the consensus weight rather than re-sizing it (the
# member configs already did the sizing; re-applying Kelly here would
# double-count).  The hard cap stays at 1.0 as a final safety clamp.
_PASSTHROUGH_SIZING = SizingConfig(
    kelly_fraction=100.0,
    vol_target=100.0,
    max_cvar_pct=100.0,
    max_position_weight=1.0,
    risk_mult_floor=1.0,
    risk_mult_ceiling=1.0,
)


@dataclass(frozen=True)
class ConsensusConfig:
    """Ensemble parameters.

    - ``top_k``: how many members to fuse (0 / None = all supplied).
    - ``min_score``: drop members whose robustness score is below this
      (e.g. non-positive Deflated Sharpe).  ``None`` keeps all.
    - ``sign_threshold``: |consensus weight| below this rounds to 0 (dust).
    """

    top_k: int = 5
    min_score: Optional[float] = None
    sign_threshold: float = 0.005


@dataclass
class ConsensusResult:
    """Consensus-ensemble backtest output."""

    n_members: int
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    benchmark_curve: List[float] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    directional_agreement: float = 0.0
    weight_dispersion: float = 0.0
    member_scores: List[float] = field(default_factory=list)
    n_consensus_decisions: int = 0

    def to_dict(self) -> dict:
        m = self.metrics
        return {
            "n_members": self.n_members,
            "member_scores": self.member_scores,
            "directional_agreement": self.directional_agreement,
            "weight_dispersion": self.weight_dispersion,
            "n_consensus_decisions": self.n_consensus_decisions,
            "equity_curve": self.equity_curve,
            "timestamps": self.timestamps,
            "benchmark_curve": self.benchmark_curve,
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
                "sharpe_ci_low": m.sharpe_ci_low,
                "sharpe_ci_high": m.sharpe_ci_high,
                "n_periods": m.n_periods,
            } if m is not None else {},
        }


def _member_weights(
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, object],
    sizing_config: SizingConfig,
    gate_config: GateConfig,
    cost_model_name: str,
) -> Dict[Tuple[str, str], float]:
    """Replay one member config and return its signed sized weight per
    ``(decision_ts, symbol)`` decision.

    Uses the ``sized_weight`` the engine actually chose for the trade;
    rejected / no-data decisions contribute 0 (the member declined to
    take that decision at that size).
    """
    result = run_backtest(
        proposals=proposals,
        ohlcv_by_symbol=ohlcv_by_symbol,
        cost_model_name=cost_model_name,
        sizing_config=sizing_config,
        gate_config=gate_config,
    )
    out: Dict[Tuple[str, str], float] = {}
    for t in result.trades:
        key = (t.ts, t.symbol)
        # sized_weight is already signed via the proposal's side.
        w = t.sized_weight if t.status in ("FILLED", "PARTIAL_FILL") else 0.0
        out[key] = w
    return out


def run_consensus_backtest(
    *,
    proposals: Sequence[ActionProposal],
    ohlcv_by_symbol: Dict[str, object],
    members: Sequence[Tuple[Dict[str, float], Dict[str, float], float]],
    config: ConsensusConfig = ConsensusConfig(),
    cost_model_name: str = "standard",
) -> ConsensusResult:
    """Fuse ``members`` into a consensus book and backtest it.

    ``members`` is a sequence of ``(sizing_config_dict, gate_config_dict,
    robustness_score)`` — exactly what ``PSOResult`` / ``GridSearchResult``
    expose per candidate.  The robustness score (Deflated Sharpe) is the
    fusion weight; non-finite / below-``min_score`` members are dropped.

    Returns the consensus equity curve + metrics and the swarm agreement
    diagnostics.
    """
    # Rank + trim members by robustness.
    ranked = sorted(members, key=lambda m: m[2], reverse=True)
    if config.min_score is not None:
        ranked = [m for m in ranked if m[2] >= config.min_score]
    if config.top_k and config.top_k > 0:
        ranked = ranked[: config.top_k]
    ranked = [m for m in ranked if m[2] == m[2] and m[2] not in (float("inf"), float("-inf"))]

    if not ranked:
        return ConsensusResult(n_members=0)

    member_scores = [m[2] for m in ranked]
    # Shift scores to be non-negative so the fusion weights are valid even
    # when some Deflated Sharpes are negative; add a small floor so a
    # member never gets exactly zero say.
    min_s = min(member_scores)
    shift = (-min_s + 1e-6) if min_s < 0 else 0.0
    fusion_w = [s + shift + 1e-9 for s in member_scores]
    total_w = sum(fusion_w)
    fusion_w = [w / total_w for w in fusion_w]

    # Replay each member and collect its per-decision weights.
    per_member: List[Dict[Tuple[str, str], float]] = []
    for (sizing_d, gate_d, _score) in ranked:
        sizing_cfg = SizingConfig(**sizing_d) if sizing_d else SizingConfig()
        gate_cfg = GateConfig(**gate_d) if gate_d else GateConfig()
        per_member.append(_member_weights(
            proposals, ohlcv_by_symbol, sizing_cfg, gate_cfg, cost_model_name,
        ))

    # Union of all decision keys across members.
    keys = set()
    for mw in per_member:
        keys.update(mw.keys())

    consensus_w: Dict[Tuple[str, str], float] = {}
    agreements: List[float] = []
    dispersions: List[float] = []
    for key in keys:
        weights = [mw.get(key, 0.0) for mw in per_member]
        cw = sum(fusion_w[i] * weights[i] for i in range(len(weights)))
        consensus_w[key] = cw

        # Directional agreement: fraction of members on the majority sign
        # (among members that took a non-zero position).
        signs = [(1 if w > 0 else (-1 if w < 0 else 0)) for w in weights]
        non_zero = [s for s in signs if s != 0]
        if non_zero:
            pos = sum(1 for s in non_zero if s > 0)
            neg = sum(1 for s in non_zero if s < 0)
            agreements.append(max(pos, neg) / len(non_zero))
            mean_abs = sum(abs(w) for w in weights) / len(weights)
            if mean_abs > 0:
                var = sum((w - (sum(weights) / len(weights))) ** 2 for w in weights) / len(weights)
                dispersions.append((var ** 0.5) / mean_abs)

    directional_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    weight_dispersion = sum(dispersions) / len(dispersions) if dispersions else 0.0

    # Build consensus proposals from the original proposals, overriding the
    # target weight with the fused weight.  We use the original proposal as
    # the template so horizon / limit / tif etc. are preserved.
    template: Dict[Tuple[str, str], ActionProposal] = {}
    for p in proposals:
        template[(p.decision_ts, p.symbol)] = p

    consensus_proposals: List[ActionProposal] = []
    for key, cw in consensus_w.items():
        base = template.get(key)
        if base is None:
            continue
        if abs(cw) < config.sign_threshold:
            continue
        side = ActionSide.BUY if cw > 0 else ActionSide.SELL
        # expected_return_pct sign should track the side for the sizer /
        # allocator; keep the magnitude from the template.
        exp = abs(base.expected_return_pct) * (1.0 if cw > 0 else -1.0)
        consensus_proposals.append(base.model_copy(update={
            "side": side,
            "target_weight": cw,
            "expected_return_pct": exp,
        }))

    if not consensus_proposals:
        return ConsensusResult(
            n_members=len(ranked),
            member_scores=member_scores,
            directional_agreement=directional_agreement,
            weight_dispersion=weight_dispersion,
            n_consensus_decisions=0,
        )

    result = run_backtest(
        proposals=consensus_proposals,
        ohlcv_by_symbol=ohlcv_by_symbol,
        cost_model_name=cost_model_name,
        sizing_config=_PASSTHROUGH_SIZING,
        bootstrap=True,
    )
    return ConsensusResult(
        n_members=len(ranked),
        equity_curve=result.equity_curve,
        timestamps=result.timestamps,
        benchmark_curve=result.benchmark_curve,
        metrics=result.metrics,
        directional_agreement=directional_agreement,
        weight_dispersion=weight_dispersion,
        member_scores=member_scores,
        n_consensus_decisions=len(consensus_proposals),
    )
