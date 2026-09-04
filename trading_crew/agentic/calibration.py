"""Conviction-calibration measurement — is the desk's stated confidence honest?

The Portfolio Manager attaches a ``conviction_score`` in [0, 1] to every
directional call, and M5 sizing scales risk by it.  But nothing measured
whether that number is *calibrated*: when the desk says "0.8 conviction",
does it actually win ~80% of the time?  An overconfident desk oversizes its
losers; an underconfident one leaves edge on the table.

This module turns the resolved-episode store into a calibration report:

* **Brier score** — mean squared error between the predicted win-probability
  (the conviction score) and the realised binary outcome (did the directional
  call make money?).  Lower is better; 0.25 is the "always predict 0.5"
  baseline.
* **Reliability bins** — bucket predictions by conviction and compare the
  mean predicted probability to the realised hit-rate in each bucket.  A
  well-calibrated desk sits on the diagonal (predicted ≈ realised).
* **Over/under-confidence gap** — the signed mean of (predicted − realised),
  so a positive number means the desk is systematically overconfident.

It is measurement ONLY — deterministic, dependency-light (stdlib), and it
does not change sizing.  Once enough episodes have resolved, the gap can feed
a conviction-recalibration multiplier into M5, but that is a separate,
opt-in step (kept out of the hot loop until the data justifies it).

Only RESOLVED, directional (BUY/SELL) episodes contribute; HOLD/ABSTAIN carry
no directional bet to score.  ``use_alpha=True`` scores against benchmark-
relative alpha instead of raw return (did the call beat SPY, not just drift up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple


@dataclass
class CalibrationBin:
    """One reliability bucket."""

    lower: float          # bin lower edge (inclusive)
    upper: float          # bin upper edge (exclusive, except the last)
    n: int
    mean_predicted: float  # mean conviction in the bin
    hit_rate: float        # realised fraction of wins in the bin

    def to_dict(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "mean_predicted": self.mean_predicted,
            "hit_rate": self.hit_rate,
            "gap": self.mean_predicted - self.hit_rate,
        }


@dataclass
class CalibrationReport:
    """Aggregate calibration of the desk's conviction scores."""

    n: int = 0
    brier: Optional[float] = None
    hit_rate: Optional[float] = None            # overall realised win-rate
    mean_conviction: Optional[float] = None     # overall mean predicted
    overconfidence_gap: Optional[float] = None  # mean(predicted - realised)
    bins: List[CalibrationBin] = field(default_factory=list)
    basis: str = "return"                       # "return" | "alpha"

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier,
            "hit_rate": self.hit_rate,
            "mean_conviction": self.mean_conviction,
            "overconfidence_gap": self.overconfidence_gap,
            "bins": [b.to_dict() for b in self.bins],
            "basis": self.basis,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }


def brier_score(pairs: Sequence[Tuple[float, int]]) -> Optional[float]:
    """Mean squared error between predicted probability and binary outcome.

    ``pairs`` is a sequence of ``(predicted_prob, outcome)`` with outcome in
    {0, 1}.  Returns ``None`` for an empty input (no fabricated score).
    """
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def reliability_bins(
    pairs: Sequence[Tuple[float, int]], *, n_bins: int = 10
) -> List[CalibrationBin]:
    """Bucket ``(predicted, outcome)`` pairs into ``n_bins`` equal-width bins
    over [0, 1] and report the mean prediction vs realised hit-rate per bin.

    Empty bins are omitted so the report only shows buckets with evidence.
    """
    if not pairs or n_bins < 1:
        return []
    width = 1.0 / n_bins
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        idx = min(int(max(0.0, min(1.0, p)) / width), n_bins - 1)
        buckets[idx].append((p, o))
    out: List[CalibrationBin] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        preds = [p for p, _ in bucket]
        outs = [o for _, o in bucket]
        out.append(CalibrationBin(
            lower=round(i * width, 4),
            upper=round((i + 1) * width, 4),
            n=len(bucket),
            mean_predicted=sum(preds) / len(preds),
            hit_rate=sum(outs) / len(outs),
        ))
    return out


def calibration_from_pairs(
    pairs: Sequence[Tuple[float, int]],
    *,
    n_bins: int = 10,
    basis: str = "return",
) -> CalibrationReport:
    """Build a :class:`CalibrationReport` from ``(predicted, outcome)`` pairs."""
    if not pairs:
        return CalibrationReport(basis=basis)
    preds = [p for p, _ in pairs]
    outs = [o for _, o in pairs]
    n = len(pairs)
    mean_pred = sum(preds) / n
    hit = sum(outs) / n
    return CalibrationReport(
        n=n,
        brier=brier_score(pairs),
        hit_rate=hit,
        mean_conviction=mean_pred,
        overconfidence_gap=mean_pred - hit,
        bins=reliability_bins(pairs, n_bins=n_bins),
        basis=basis,
    )


def _pairs_from_rows(rows, *, use_alpha: bool) -> List[Tuple[float, int]]:
    """Extract ``(conviction, win)`` pairs from RLMF-style rows.

    Only directional (BUY/SELL) rows count.  ``win`` is 1 when the signed
    realised return (or alpha, if ``use_alpha``) is strictly positive.
    """
    pairs: List[Tuple[float, int]] = []
    for r in rows:
        action = (getattr(r, "action", "") or "").upper()
        if action not in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL"):
            continue
        conviction = getattr(r, "conviction_score", None)
        if conviction is None:
            continue
        if use_alpha:
            outcome_val = getattr(r, "alpha_return", None)
            if outcome_val is None:
                outcome_val = getattr(r, "realised_return", None)
        else:
            outcome_val = getattr(r, "realised_return", None)
        if outcome_val is None:
            continue
        pairs.append((max(0.0, min(1.0, float(conviction))), 1 if float(outcome_val) > 0 else 0))
    return pairs


def compute_calibration(
    *,
    symbol: Optional[str] = None,
    memory=None,
    n_bins: int = 10,
    use_alpha: bool = False,
    workflow_version: Optional[str] = None,
    source: Optional[str] = None,
) -> CalibrationReport:
    """Build a calibration report from the resolved-episode store.

    Reuses the RLMF dataset rows (already reward-ranked resolved episodes) so
    calibration is scored on exactly the same cohort as the learning loops.
    Cohort filters (``workflow_version`` / ``source``) mirror the Phase-0
    provenance filtering.  Returns an empty report (``n == 0``) when there are
    no resolved directional decisions yet.
    """
    from trading_crew.agentic.rlmf import build_dataset

    rows = build_dataset(
        memory=memory,
        symbol=symbol,
        workflow_version=workflow_version,
        source=source,
    )
    pairs = _pairs_from_rows(rows, use_alpha=use_alpha)
    return calibration_from_pairs(
        pairs, n_bins=n_bins, basis="alpha" if use_alpha else "return"
    )
