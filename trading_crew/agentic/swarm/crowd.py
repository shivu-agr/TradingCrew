"""Crowd-positioning signal — "analyse crowd behaviour in the market".

The reference platforms describe swarm intelligence as *reading the crowd*:
inferring how the mass of participants is positioned and trading with (or
against) that flow.  This module distils the crowding of a name into a
single deterministic score in ``[-1, +1]``:

    +1  = the crowd is maximally positioned **long / bullish**
     0  = balanced / no clear consensus
    -1  = the crowd is maximally positioned **short / bearish**

The score fuses signals the desk already has tools for:

- **Options** — put/call ratio and IV skew (fear demand for downside).
- **Insiders** — net insider buying vs selling.
- **Analysts** — bullish rating fraction (herding of sell-side views).
- **Institutions** — FII / DII / promoter accumulation drift (India path).
- **COT** — commercial vs speculative net positioning (commodities).

Two uses (both opt-in, nothing changes unless a caller enables it):

1. A **fade-the-crowd sizing tilt** (:func:`crowd_size_multiplier`).  The
   probability-edge reading: when everyone is already crowded on your
   side, the marginal edge is smaller and the unwind risk is larger, so
   trim; when you are the contrarian against an extreme crowd, you may
   size up (bounded).  Disabled by default (``CrowdConfig.enabled=False``).
2. A logged **feature** on episodes so the backtest can segment
   performance by crowd regime.

All math is plain Python and side-effect free; parsing real tool output
into :class:`CrowdInputs` happens in the ``get_crowd_positioning`` tool,
which keeps this core unit-testable and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Inputs / config / result
# ---------------------------------------------------------------------------


@dataclass
class CrowdInputs:
    """Normalised positioning inputs.  Every field is optional — the
    signal is computed over whatever is available (missing components are
    simply dropped from the weighted average, never invented).

    - ``put_call_ratio``: total put OI / call OI.  > 1 = more puts
      (crowd hedged / bearish).
    - ``iv_skew``: put IV minus call IV (fraction, e.g. 0.03 = 3 vol pts).
      Positive = downside demand (fear) => bearish crowd.
    - ``insider_net_ratio``: (buys - sells) / (buys + sells) in [-1, 1].
    - ``analyst_bullish_frac``: fraction of ratings that are buy /
      strong-buy, in [0, 1].
    - ``institutional_net_drift``: net FII+DII+promoter change as a
      signed fraction (positive = accumulation).
    - ``cot_net_positioning``: net speculative positioning in [-1, 1]
      (commodities; positive = net long).
    """

    put_call_ratio: Optional[float] = None
    iv_skew: Optional[float] = None
    insider_net_ratio: Optional[float] = None
    analyst_bullish_frac: Optional[float] = None
    institutional_net_drift: Optional[float] = None
    cot_net_positioning: Optional[float] = None


@dataclass(frozen=True)
class CrowdConfig:
    """Crowd-signal weights + fade-the-crowd tilt parameters.

    - ``enabled``: master switch for the sizing tilt.  When ``False``
      :func:`crowd_size_multiplier` always returns ``1.0`` (no effect).
    - ``crowd_threshold``: |score| below this is treated as "balanced";
      no tilt is applied.
    - ``max_fade``: smallest multiplier when the trade is *aligned* with
      an extreme crowd (0.5 = halve the size at max crowding).
    - ``max_boost``: largest multiplier when the trade is *contrarian* to
      an extreme crowd (1.0 = no boost; raise to allow sizing up).
    - ``w_*``: per-component fusion weights.
    """

    enabled: bool = False
    crowd_threshold: float = 0.5
    max_fade: float = 0.5
    max_boost: float = 1.0
    w_put_call: float = 1.0
    w_iv_skew: float = 1.0
    w_insider: float = 1.0
    w_analyst: float = 1.0
    w_institutional: float = 1.0
    w_cot: float = 1.0


@dataclass
class CrowdSignal:
    """Fused crowd-positioning result."""

    score: float                 # [-1, +1]: + long crowd, - short crowd
    intensity: float             # |score|
    label: str                   # "crowded long" | "crowded short" | "balanced"
    components: Dict[str, float] = field(default_factory=dict)  # per-signal [-1,1]
    n_components: int = 0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "intensity": self.intensity,
            "label": self.label,
            "components": self.components,
            "n_components": self.n_components,
        }


# ---------------------------------------------------------------------------
# Signal fusion
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_crowd_signal(
    inputs: CrowdInputs,
    config: CrowdConfig = CrowdConfig(),
) -> CrowdSignal:
    """Fuse ``inputs`` into a single crowd-positioning score in [-1, 1].

    Each available component is mapped to a bullish(+)/bearish(-) axis and
    combined as a weighted average.  Components that are ``None`` are
    dropped from both the numerator and the denominator, so the score is
    always the mean over *observed* signals (no zero-filling bias).
    """
    contributions: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    if inputs.put_call_ratio is not None:
        # PCR > 1 => more puts => bearish crowd (negative).
        contributions["put_call"] = _clamp(-math.tanh(inputs.put_call_ratio - 1.0))
        weights["put_call"] = config.w_put_call
    if inputs.iv_skew is not None:
        # Positive skew (puts richer) => downside fear => bearish.
        contributions["iv_skew"] = _clamp(-math.tanh(inputs.iv_skew * 10.0))
        weights["iv_skew"] = config.w_iv_skew
    if inputs.insider_net_ratio is not None:
        contributions["insider"] = _clamp(inputs.insider_net_ratio)
        weights["insider"] = config.w_insider
    if inputs.analyst_bullish_frac is not None:
        # 0..1 fraction => center on 0.5 and scale to [-1, 1].
        contributions["analyst"] = _clamp((inputs.analyst_bullish_frac - 0.5) * 2.0)
        weights["analyst"] = config.w_analyst
    if inputs.institutional_net_drift is not None:
        contributions["institutional"] = _clamp(math.tanh(inputs.institutional_net_drift * 5.0))
        weights["institutional"] = config.w_institutional
    if inputs.cot_net_positioning is not None:
        contributions["cot"] = _clamp(inputs.cot_net_positioning)
        weights["cot"] = config.w_cot

    if not contributions:
        return CrowdSignal(score=0.0, intensity=0.0, label="balanced",
                           components={}, n_components=0)

    total_w = sum(weights[k] for k in contributions)
    if total_w <= 0:
        score = 0.0
    else:
        score = sum(contributions[k] * weights[k] for k in contributions) / total_w
    score = _clamp(score)
    intensity = abs(score)
    if score >= config.crowd_threshold:
        label = "crowded long"
    elif score <= -config.crowd_threshold:
        label = "crowded short"
    else:
        label = "balanced"

    return CrowdSignal(
        score=score,
        intensity=intensity,
        label=label,
        components=contributions,
        n_components=len(contributions),
    )


def crowd_size_multiplier(
    signal: CrowdSignal,
    side: str,
    config: CrowdConfig = CrowdConfig(),
) -> float:
    """Fade-the-crowd size multiplier in ``[max_fade, max_boost]``.

    ``side`` is the trade direction ("BUY"/"SELL", case-insensitive).

    - Disabled (``config.enabled=False``) or a balanced crowd
      (``|score| < crowd_threshold``) → ``1.0`` (no change).
    - Trade *aligned* with an extreme crowd → shrink toward ``max_fade``
      (you're late to a crowded trade; probability edge is thinner).
    - Trade *contrarian* to an extreme crowd → grow toward ``max_boost``.

    The magnitude scales linearly with how far past the threshold the
    crowd is, so a marginally-crowded name barely moves and an extreme
    one moves the full amount.
    """
    if not config.enabled:
        return 1.0
    score = signal.score
    if abs(score) < config.crowd_threshold:
        return 1.0
    s = side.upper()
    if s not in ("BUY", "SELL"):
        return 1.0

    denom = max(1e-9, 1.0 - config.crowd_threshold)
    intensity = min(1.0, (abs(score) - config.crowd_threshold) / denom)
    crowd_is_long = score > 0
    trade_is_long = s == "BUY"
    aligned = crowd_is_long == trade_is_long

    if aligned:
        # 1.0 at threshold -> max_fade at full crowding.
        return 1.0 - intensity * (1.0 - config.max_fade)
    # Contrarian: 1.0 at threshold -> max_boost at full crowding.
    return 1.0 + intensity * (config.max_boost - 1.0)
