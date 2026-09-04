"""Score a cohort against the editable desk-strategy success/failure bars.

Deterministic.  No LLM.  A single trade is never SUCCESS/FAILURE of the
book — those verdicts are cohort-level, which is what the self-learn
loop consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from trading_crew.agentic.outcome import DIRECTIONAL_SIDES, OutcomeLabel, normalize_side
from trading_crew.agentic.strategy import DeskStrategy, ObjectiveBand, load_desk_strategy


class BookVerdict:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    MIXED = "MIXED"
    INSUFFICIENT = "INSUFFICIENT"
    LOOKAHEAD_SUSPECT = "LOOKAHEAD_SUSPECT"


@dataclass
class HorizonMetrics:
    n: int
    wins: int
    losses: int
    hit_rate: Optional[float]
    mean_alpha: Optional[float]
    mean_drawdown: Optional[float]
    expectancy: Optional[float]
    profit_factor: Optional[float]
    mean_realised: Optional[float]


@dataclass
class ObjectiveScore:
    tier: str
    verdict: str
    metrics: HorizonMetrics
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        m = self.metrics
        return {
            "tier": self.tier,
            "verdict": self.verdict,
            "n": m.n,
            "wins": m.wins,
            "losses": m.losses,
            "hit_rate": m.hit_rate,
            "mean_alpha": m.mean_alpha,
            "mean_drawdown": m.mean_drawdown,
            "expectancy": m.expectancy,
            "profit_factor": m.profit_factor,
            "reasons": list(self.reasons),
        }


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def review_payload(episode: Any) -> Optional[Dict[str, Any]]:
    """Return the 30-day review dict when it has been written."""
    status = (getattr(episode, "review_status", None) or "").upper()
    blob = getattr(episode, "review", None)
    if status == "RESOLVED" and isinstance(blob, dict):
        return blob
    return None


def _trade_row(episode: Any, *, prefer_review: bool) -> Optional[Dict[str, Any]]:
    proposal = getattr(episode, "action_proposal", None) or {}
    side = normalize_side(proposal.get("side") or proposal.get("action"))
    if side not in DIRECTIONAL_SIDES:
        return None
    if prefer_review:
        rev = review_payload(episode)
        if rev is None:
            return None
        return {
            "side": side,
            "label": str(rev.get("outcome_label") or ""),
            "alpha": _float(rev.get("alpha_return")),
            "realised": _float(rev.get("realised_return")),
            "drawdown": _float(rev.get("max_drawdown")),
        }
    status = getattr(episode, "outcome_status", None)
    status_s = getattr(status, "value", status)
    if str(status_s).upper() != "RESOLVED":
        return None
    return {
        "side": side,
        "label": str(getattr(episode, "outcome_label", None) or ""),
        "alpha": _float(getattr(episode, "alpha_return", None)),
        "realised": _float(getattr(episode, "realised_return", None)),
        "drawdown": _float(getattr(episode, "max_drawdown", None)),
    }


def collect_horizon_metrics(
    episodes: Sequence[Any],
    *,
    prefer_review: bool,
) -> HorizonMetrics:
    wins = losses = 0
    alphas: List[float] = []
    dds: List[float] = []
    realised: List[float] = []
    for ep in episodes:
        row = _trade_row(ep, prefer_review=prefer_review)
        if row is None:
            continue
        label = row["label"].upper()
        if label == OutcomeLabel.WIN.value:
            wins += 1
        elif label == OutcomeLabel.LOSS.value:
            losses += 1
        else:
            continue
        if row["alpha"] is not None:
            alphas.append(row["alpha"])
        if row["drawdown"] is not None:
            dds.append(row["drawdown"])
        if row["realised"] is not None:
            realised.append(row["realised"])
    n = wins + losses
    hit = (wins / n) if n else None
    mean_a = (sum(alphas) / len(alphas)) if alphas else None
    mean_dd = (sum(dds) / len(dds)) if dds else None
    mean_r = (sum(realised) / len(realised)) if realised else None
    pos = sum(x for x in realised if x > 0)
    neg = sum(x for x in realised if x < 0)
    if neg < 0:
        pf = pos / abs(neg)
    elif pos > 0:
        pf = float("inf")
    else:
        pf = None
    return HorizonMetrics(
        n=n,
        wins=wins,
        losses=losses,
        hit_rate=hit,
        mean_alpha=mean_a,
        mean_drawdown=mean_dd,
        expectancy=mean_r,
        profit_factor=pf,
        mean_realised=mean_r,
    )


def score_band(metrics: HorizonMetrics, band: ObjectiveBand, *, tier: str) -> ObjectiveScore:
    reasons: List[str] = []
    la = band.look_ahead
    if (
        metrics.n > 0
        and metrics.n < int(la["min_n_for_trust"])
        and metrics.hit_rate is not None
        and metrics.mean_alpha is not None
        and metrics.hit_rate >= float(la["hit_rate_above"])
        and metrics.mean_alpha >= float(la["mean_alpha_above"])
    ):
        reasons.append(
            f"look-ahead suspect: hit-rate {metrics.hit_rate:.0%} and "
            f"mean α {metrics.mean_alpha:+.1%} on n={metrics.n} "
            f"< {int(la['min_n_for_trust'])}"
        )
        return ObjectiveScore(tier=tier, verdict=BookVerdict.LOOKAHEAD_SUSPECT, metrics=metrics, reasons=reasons)

    if metrics.n < int(band.min_n):
        reasons.append(f"n={metrics.n} < min_n={band.min_n}")
        return ObjectiveScore(tier=tier, verdict=BookVerdict.INSUFFICIENT, metrics=metrics, reasons=reasons)

    failed = False
    suc = band.success
    fail = band.failure
    if metrics.hit_rate is not None and metrics.hit_rate < fail["hit_rate_below"]:
        failed = True
        reasons.append(f"hit-rate {metrics.hit_rate:.0%} < {fail['hit_rate_below']:.0%}")
    if metrics.mean_alpha is not None and metrics.mean_alpha < fail["mean_alpha_below"]:
        failed = True
        reasons.append(f"mean α {metrics.mean_alpha:+.2%} < {fail['mean_alpha_below']:+.2%}")
    if metrics.mean_drawdown is not None and metrics.mean_drawdown < fail["mean_drawdown_below"]:
        failed = True
        reasons.append(f"mean DD {metrics.mean_drawdown:.1%} < {fail['mean_drawdown_below']:.1%}")
    if failed:
        return ObjectiveScore(tier=tier, verdict=BookVerdict.FAILURE, metrics=metrics, reasons=reasons)

    ok = True
    if metrics.hit_rate is None or metrics.hit_rate < suc["min_hit_rate"]:
        ok = False
        reasons.append(f"hit-rate {metrics.hit_rate} below success {suc['min_hit_rate']:.0%}")
    if metrics.mean_alpha is None or metrics.mean_alpha < suc["min_mean_alpha"]:
        ok = False
        reasons.append(f"mean α {metrics.mean_alpha} below success {suc['min_mean_alpha']:+.2%}")
    if metrics.mean_drawdown is None or metrics.mean_drawdown < suc["max_mean_drawdown"]:
        ok = False
        reasons.append(f"mean DD {metrics.mean_drawdown} worse than {suc['max_mean_drawdown']:.1%}")
    if metrics.expectancy is None or metrics.expectancy < suc["min_expectancy"]:
        ok = False
        reasons.append(f"expectancy {metrics.expectancy} below {suc['min_expectancy']:+.2%}")
    pf = metrics.profit_factor
    if pf is None or pf < suc["min_profit_factor"]:
        ok = False
        reasons.append(f"profit factor {pf} below {suc['min_profit_factor']:.2f}")
    if ok:
        reasons.append("all success bars cleared")
        return ObjectiveScore(tier=tier, verdict=BookVerdict.SUCCESS, metrics=metrics, reasons=reasons)
    return ObjectiveScore(tier=tier, verdict=BookVerdict.MIXED, metrics=metrics, reasons=reasons)


def score_episodes(
    episodes: Sequence[Any],
    *,
    tier: str = "short",
    strategy: Optional[DeskStrategy] = None,
    prefer_review: Optional[bool] = None,
) -> ObjectiveScore:
    s = strategy or load_desk_strategy()
    if tier not in s.objectives:
        raise KeyError(f"unknown objective tier {tier!r}")
    use_review = prefer_review if prefer_review is not None else (tier == "short")
    metrics = collect_horizon_metrics(episodes, prefer_review=use_review)
    return score_band(metrics, s.objectives[tier], tier=tier)
