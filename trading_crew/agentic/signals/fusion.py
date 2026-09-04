"""Signal fusion meta-layer (Phase 4).

The desk produces four *independent* readings of the same name.  Trading
any one of them alone throws away the others; naively averaging them
ignores that they measure different things and can conflict.  This module
combines them into a single directional signal + a bounded target weight,
with explicit **guardrails** so a disagreement or a crowded/edgeless setup
shrinks risk instead of silently averaging into a false confidence.

Roles of the four inputs
------------------------
- **LLM decision** (required primary) — direction + conviction from the
  crew.  If the LLM abstains/holds, fusion does **not** initiate a trade on
  the quant signals alone (``require_llm``): the LLM is the desk's mandate.
- **L4 RL policy** — a second *directional* opinion (sign of its target
  weight, weighted by its softmax confidence).
- **L3 tuned expectancy** — a *confidence gate*, not a direction: if the
  tuned M5 config has no positive per-trade edge, size is dampened.
- **Crowd positioning** — a *contrarian fade*: when the fused trade is on
  the same side as an already-extreme crowd, size is trimmed (late to a
  crowded trade → thinner edge, larger unwind risk).

Everything is pure Python, deterministic, and unit-testable — no I/O, no
LLM call.  The assembly of live inputs (loading the latest proposal, the
policy client, etc.) happens in the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trading_crew.agentic.swarm.crowd import CrowdSignal


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class LLMSignal:
    """The crew's decision view."""

    side: str  # BUY / SELL / HOLD / ABSTAIN
    conviction: float = 0.0  # 0..1
    expected_return_pct: float = 0.0
    target_weight: Optional[float] = None  # signed; unused for direction (side wins)


@dataclass
class PolicySignal:
    """The L4 PPO policy recommendation."""

    target_weight: float  # signed; sign is the policy's direction
    confidence: float = 0.0  # 0..1 (max softmax prob)
    value_estimate: float = 0.0


@dataclass
class ExpectancySignal:
    """The L3 tuned-config edge (magnitude matters, not sign)."""

    expectancy: float  # per-trade expected return of the tuned config
    win_rate: Optional[float] = None
    n_samples: int = 0


@dataclass(frozen=True)
class FusionConfig:
    """Fusion weights + guardrail thresholds (model hyperparameters)."""

    w_llm: float = 0.6
    w_policy: float = 0.4
    disagreement_penalty: float = 0.5   # shrink confidence when LLM & policy disagree
    min_confidence: float = 0.15        # below → ABSTAIN (stand aside)
    max_position_weight: float = 0.15   # cap on |fused target weight|
    require_llm: bool = True            # do not trade on quant signals if LLM abstains
    # L3 expectancy gate
    expectancy_floor: float = 0.0       # expectancy <= floor ⇒ no historical edge
    expectancy_dampen: float = 0.3      # size multiplier when edge is absent
    # Crowd contrarian fade
    use_crowd_fade: bool = True
    crowd_threshold: float = 0.5        # |crowd score| below this ⇒ no fade
    crowd_max_fade: float = 0.5         # smallest multiplier at maximal aligned crowding


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class FusedSignal:
    side: str                    # BUY / SELL / HOLD / ABSTAIN
    score: float                 # fused directional score in [-1, 1]
    confidence: float            # post-guardrail confidence in [0, 1]
    target_weight: float         # signed, bounded by max_position_weight
    crowd_multiplier: float = 1.0
    contributions: Dict[str, dict] = field(default_factory=dict)
    guardrails: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "score": self.score,
            "confidence": self.confidence,
            "target_weight": self.target_weight,
            "crowd_multiplier": self.crowd_multiplier,
            "contributions": self.contributions,
            "guardrails": self.guardrails,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pso_tuned_expectancy(
    proposals,
    ohlcv_by_symbol,
    *,
    cost_model_name: str = "standard",
    min_proposals: int = 5,
) -> Optional["ExpectancySignal"]:
    """Compute the L3 expectancy from the **PSO-tuned** M5 config.

    Runs a particle-swarm search over sizing/gate params (the swarm
    optimizer), replaying ``proposals`` through the M2/M5 backtest, and
    returns the winning config's per-trade expectancy as an
    :class:`ExpectancySignal`.  This is a stronger edge estimate than a raw
    mean of realised returns because the swarm searches for the config that
    actually maximises the risk-adjusted objective.

    Returns ``None`` when there are too few proposals to form folds or the
    search produced no valid config (no silent fallback — the caller then
    simply omits the expectancy gate).
    """
    if proposals is None or len(proposals) < min_proposals:
        return None
    try:
        from trading_crew.agentic.swarm.pso import run_swarm_search

        res = run_swarm_search(
            proposals=proposals,
            ohlcv_by_symbol=ohlcv_by_symbol,
            cost_model_name=cost_model_name,
        )
    except Exception:  # noqa: BLE001 — expectancy is an optional gate
        return None
    m = getattr(res, "best_metrics", None)
    if m is None:
        return None
    return ExpectancySignal(
        expectancy=float(getattr(m, "expectancy", 0.0) or 0.0),
        win_rate=getattr(m, "win_rate", None),
        n_samples=int(getattr(m, "n_periods", 0) or 0),
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _direction(side: str) -> float:
    s = (side or "").upper()
    if s == "BUY":
        return 1.0
    if s == "SELL":
        return -1.0
    return 0.0


def _crowd_fade(crowd_score: float, fused_side: str, config: FusionConfig) -> float:
    """Contrarian fade multiplier in ``[crowd_max_fade, 1.0]``.

    1.0 when the crowd is balanced or the trade is contrarian to it; shrinks
    toward ``crowd_max_fade`` as the trade aligns with an ever-more-extreme
    crowd.
    """
    if abs(crowd_score) < config.crowd_threshold:
        return 1.0
    d = _direction(fused_side)
    if d == 0.0:
        return 1.0
    crowd_is_long = crowd_score > 0
    trade_is_long = d > 0
    if crowd_is_long != trade_is_long:
        return 1.0  # contrarian to the crowd — no fade (Firefly-style edge)
    denom = max(1e-9, 1.0 - config.crowd_threshold)
    intensity = min(1.0, (abs(crowd_score) - config.crowd_threshold) / denom)
    return 1.0 - intensity * (1.0 - config.crowd_max_fade)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fuse_signals(
    llm: LLMSignal,
    *,
    policy: Optional[PolicySignal] = None,
    expectancy: Optional[ExpectancySignal] = None,
    crowd: Optional[CrowdSignal] = None,
    config: FusionConfig = FusionConfig(),
) -> FusedSignal:
    """Combine the four views into one guardrailed :class:`FusedSignal`."""
    contributions: Dict[str, dict] = {}
    guardrails: List[str] = []

    llm_dir = _direction(llm.side) * _clamp01(llm.conviction)
    contributions["llm"] = {
        "side": (llm.side or "").upper(),
        "score": llm_dir,
        "weight": config.w_llm,
        "conviction": _clamp01(llm.conviction),
    }

    pol_dir = 0.0
    have_policy = policy is not None
    if have_policy:
        pol_sign = 1.0 if policy.target_weight > 0 else (-1.0 if policy.target_weight < 0 else 0.0)
        pol_dir = pol_sign * _clamp01(policy.confidence)
        contributions["policy"] = {
            "target_weight": policy.target_weight,
            "score": pol_dir,
            "weight": config.w_policy,
            "confidence": _clamp01(policy.confidence),
        }

    # LLM is the mandate: no directional LLM view ⇒ stand aside.
    if config.require_llm and _direction(llm.side) == 0.0:
        guardrails.append("LLM has no directional view — standing aside (require_llm).")
        return FusedSignal(
            side="ABSTAIN" if (llm.side or "").upper() != "HOLD" else "HOLD",
            score=0.0, confidence=0.0, target_weight=0.0,
            contributions=contributions, guardrails=guardrails,
        )

    # Weighted directional blend over the sources actually present.
    pairs = [(llm_dir, config.w_llm)]
    if have_policy:
        pairs.append((pol_dir, config.w_policy))
    total_w = sum(w for _, w in pairs)
    fused = sum(s * w for s, w in pairs) / total_w if total_w > 0 else 0.0
    fused = _clamp(fused, -1.0, 1.0)

    side = "BUY" if fused > 0 else ("SELL" if fused < 0 else "HOLD")
    confidence = abs(fused)

    # Guardrail: LLM vs policy directional disagreement.
    if have_policy and llm_dir != 0.0 and pol_dir != 0.0 and (llm_dir > 0) != (pol_dir > 0):
        confidence *= (1.0 - config.disagreement_penalty)
        guardrails.append(
            f"LLM ({'long' if llm_dir > 0 else 'short'}) and RL policy "
            f"({'long' if pol_dir > 0 else 'short'}) disagree — confidence "
            f"×{1.0 - config.disagreement_penalty:.2f}."
        )

    # Guardrail: L3 expectancy edge gate.
    if expectancy is not None:
        contributions["expectancy"] = {
            "expectancy": expectancy.expectancy,
            "win_rate": expectancy.win_rate,
            "n_samples": expectancy.n_samples,
        }
        if expectancy.expectancy <= config.expectancy_floor:
            confidence *= config.expectancy_dampen
            guardrails.append(
                f"Tuned L3 config shows no positive expectancy "
                f"({expectancy.expectancy:+.4f}) — size ×{config.expectancy_dampen:.2f}."
            )

    # Guardrail: contrarian crowd fade.
    crowd_multiplier = 1.0
    if crowd is not None and config.use_crowd_fade:
        contributions["crowd"] = {
            "score": crowd.score,
            "label": crowd.label,
            "intensity": crowd.intensity,
        }
        crowd_multiplier = _crowd_fade(crowd.score, side, config)
        if crowd_multiplier < 1.0:
            guardrails.append(
                f"Trade aligned with a {crowd.label} (score {crowd.score:+.2f}) — "
                f"fading size ×{crowd_multiplier:.2f}."
            )

    confidence = _clamp01(confidence)

    # Below the floor ⇒ stand aside (do not trade a weak/contested signal).
    if side in ("BUY", "SELL") and confidence < config.min_confidence:
        guardrails.append(
            f"Fused confidence {confidence:.2f} < min {config.min_confidence:.2f} — standing aside."
        )
        return FusedSignal(
            side="ABSTAIN", score=fused, confidence=confidence, target_weight=0.0,
            crowd_multiplier=crowd_multiplier, contributions=contributions, guardrails=guardrails,
        )

    sign = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
    target_weight = sign * confidence * crowd_multiplier * config.max_position_weight
    target_weight = _clamp(target_weight, -config.max_position_weight, config.max_position_weight)

    return FusedSignal(
        side=side,
        score=fused,
        confidence=confidence,
        target_weight=target_weight,
        crowd_multiplier=crowd_multiplier,
        contributions=contributions,
        guardrails=guardrails,
    )
