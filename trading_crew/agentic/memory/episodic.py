"""Episodic memory — (S_t, A_t, O_{t+k}, τ) tuples with embargo + decay.

Implements paper §4.2 "Episodic Memory" and §13.1.2 "Outcome Embargo":

- **Episodes** capture *what the agent saw* (``state_summary``),
  *what it decided* (``action`` as an ``ActionProposal`` reference),
  *what happened next* (``outcome`` with realised return / Fill /
  drawdown), and *when* (``decision_ts`` and ``outcome_ts``).
- **Outcome embargo** is the retrieval-time filter that prevents an
  episode from being used as evidence by a query whose timestamp falls
  *before* the episode's outcome materialised.  Without this, episodes
  with future-known outcomes leak into the retrieval set during
  backtests, silently inflating reported edge.
- **Time-aware retrieval** scores ``score = similarity − λ·exp(−decay·Δdays)``
  so older episodes contribute less.  Both ``λ`` and ``decay`` are
  config-exposed.
- **Regime tags** annotate each episode with a market regime
  (TREND/RANGE/HIGH_VOL/CRISIS).  The cascaded controller in M4 reads
  the tag of the *current* state and prefers episodes from the same regime.

Embedding strategy
------------------
We use a deterministic TF-IDF bag-of-words representation so the module has
zero ML-model dependencies and runs the same in tests, CI, and live.
Callers can swap in a richer embedding by passing ``embed_fn`` to the
``EpisodicMemory`` constructor; the default keeps us correct, fast, and
reproducible.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime / outcome enums
# ---------------------------------------------------------------------------


class Regime(str, Enum):
    """Coarse-grained market regime tag (paper §4.2 / §11.2).

    The four-tier taxonomy is sufficient for the cascaded controller's
    routing decisions.  Sub-classifying further (e.g. "early trend" vs
    "late trend") is left as a future extension because regime estimators
    in literature rarely agree on finer-grained labels.
    """

    TREND = "TREND"          # directional persistence, low realised vol
    RANGE = "RANGE"          # mean-reverting, low realised vol
    HIGH_VOL = "HIGH_VOL"    # elevated vol, no clear direction (legacy umbrella)
    # Phase 2E — split HIGH_VOL by trend direction so the cascaded
    # controller can route HIGH_VOL_RANGE to a risk-only mini-crew while
    # HIGH_VOL_TREND keeps the full 18-agent debate.
    HIGH_VOL_TREND = "HIGH_VOL_TREND"
    HIGH_VOL_RANGE = "HIGH_VOL_RANGE"
    CRISIS = "CRISIS"        # extreme vol + drawdown + correlated risk-off
    UNKNOWN = "UNKNOWN"      # default before regime detection runs


class OutcomeStatus(str, Enum):
    """Lifecycle of an episode's outcome.

    Episodes are written at decision time with ``PENDING`` outcomes.  The
    walk-forward backtest / live outcome loop transitions them to
    ``RESOLVED`` once ``horizon_days`` have elapsed.  ``ABANDONED`` is
    used when an episode's outcome can no longer be computed (e.g. ticker
    delisted, data missing) — these are still retained for replay but
    excluded from retrieval.
    """

    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    ABANDONED = "ABANDONED"


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


# Bump this when ``Episode`` gains a non-backward-compatible field. The
# in-memory migrator below patches older records on load so the disk
# format never needs an offline rewrite.  Mirrors ``PortfolioState.SCHEMA_VERSION``
# (paper §M7) so all on-disk schemas have a uniform "what version is
# this" identifier.
EPISODE_SCHEMA_VERSION = 1


@dataclass
class Episode:
    """A single (state, action, outcome, timestamp) tuple.

    Fields:

    - ``episode_id``       — stable identifier (e.g. ``f"{ticker}-{trade_date}"``).
    - ``symbol``           — ticker the episode concerns.
    - ``decision_ts``      — when the action was taken (ISO-8601).
    - ``state_summary``    — free-text summary of the inputs the agent had.
                             Used for similarity retrieval.  The full audit
                             trail lives in the run manifest (M6).
    - ``regime``           — regime tag *at decision time* (see ``Regime``).
    - ``action_proposal``  — JSON-serialised ``ActionProposal`` (from M1).
    - ``outcome_ts``       — when the outcome was realised (decision +
                             horizon_days, in trading days).
    - ``outcome_status``   — PENDING / RESOLVED / ABANDONED.
    - ``realised_return``  — signed return over the horizon (None if PENDING).
    - ``alpha_return``     — return less the benchmark (None if PENDING).
    - ``max_drawdown``     — peak-to-trough drawdown during the holding
                             period (None if PENDING).
    - ``reflection``       — optional post-hoc lesson learned (filled by
                             the M4 reflective critic during outcome loop).
    - ``embargo_days``     — extra calendar days the episode is hidden
                             after ``outcome_ts``.  Default 0; bump to >0
                             when the outcome's tape is still rolling
                             (e.g. earnings drift episodes).
    - ``schema_version``   — disk-format version. Auto-migrated on load
                             by :class:`EpisodicMemory` to the current
                             ``EPISODE_SCHEMA_VERSION``.
    - ``retrieval_count``  — how many times this episode has been
                             surfaced to a future run.  Used by
                             :meth:`EpisodicMemory.evict` to keep
                             frequently-cited lessons even when they're
                             old.
    """

    episode_id: str
    symbol: str
    decision_ts: str
    state_summary: str
    regime: Regime
    action_proposal: dict
    outcome_ts: str
    outcome_status: OutcomeStatus = OutcomeStatus.PENDING
    realised_return: Optional[float] = None
    alpha_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    reflection: Optional[str] = None
    embargo_days: int = 0
    created_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = EPISODE_SCHEMA_VERSION
    retrieval_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.value
        d["outcome_status"] = self.outcome_status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        # Backward-compat migrator: older episodes written before
        # SCHEMA_VERSION=1 lack ``schema_version`` / ``retrieval_count``.
        # We fill them in here so retrieval / eviction code can assume
        # every loaded episode has the latest shape.
        return cls(
            episode_id=data["episode_id"],
            symbol=data["symbol"],
            decision_ts=data["decision_ts"],
            state_summary=data["state_summary"],
            regime=Regime(data["regime"]),
            action_proposal=data["action_proposal"],
            outcome_ts=data["outcome_ts"],
            outcome_status=OutcomeStatus(data.get("outcome_status", "PENDING")),
            realised_return=data.get("realised_return"),
            alpha_return=data.get("alpha_return"),
            max_drawdown=data.get("max_drawdown"),
            reflection=data.get("reflection"),
            embargo_days=int(data.get("embargo_days", 0)),
            created_ts=data.get("created_ts", datetime.now(timezone.utc).isoformat()),
            schema_version=int(data.get("schema_version", 0)),
            retrieval_count=int(data.get("retrieval_count", 0)),
        )

    def effective_unembargoed_ts(self) -> datetime:
        """Earliest timestamp at which this episode becomes retrievable.

        Equal to ``outcome_ts + embargo_days``.  PENDING episodes are
        always embargoed (they have no realised outcome yet).
        """
        base = datetime.fromisoformat(self.outcome_ts.replace("Z", "+00:00"))
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return base + _days(self.embargo_days)


def _days(n: int):
    """Return ``timedelta(days=n)`` — module-local for clarity in retrieval code."""
    from datetime import timedelta
    return timedelta(days=n)


# ---------------------------------------------------------------------------
# Retrieved episode (enriched with retrieval diagnostics)
# ---------------------------------------------------------------------------


@dataclass
class RetrievedEpisode:
    """An episode plus the scoring components that ranked it.

    Diagnostic fields are surfaced to the UI's memory panel so users see
    *why* an episode was retrieved (high similarity?  recent?  same
    regime?).  Without them, retrieval is opaque and feedback loops are
    impossible.
    """

    episode: Episode
    similarity: float
    decay_factor: float
    score: float
    delta_days: float


# ---------------------------------------------------------------------------
# Default embedder — deterministic TF-IDF bag of words
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}")


def _tokenise(text: str) -> List[str]:
    """Lowercase + alphanumeric token split with a 2-character minimum.

    Deliberately simple: callers can swap in a model-based embedder via
    ``EpisodicMemory(embed_fn=...)``.  The default keeps tests deterministic
    and free of network/model dependencies.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _tf_vector(tokens: Sequence[str]) -> Dict[str, float]:
    """Sub-linear term frequency vector (log-tf normalisation)."""
    counts = Counter(tokens)
    return {term: 1.0 + math.log(c) for term, c in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Append-only episodic store with outcome-embargo retrieval.

    Storage layout (JSONL, one ``Episode`` per line) is chosen because:
    - It's append-friendly (no rewrite penalty per add).
    - It's grep/diff-able (debugging an audit trail is line-oriented).
    - It avoids a database dependency.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        decay_per_day: float = 0.005,
        decay_weight: float = 0.5,
        regime_match_bonus: float = 0.05,
        embed_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        path : path to the JSONL store. Created with parents on first write.
        decay_per_day : exponential decay rate per day (default 0.005 ≈ half-life ~140 days).
        decay_weight : λ in ``score = similarity + λ·exp(−decay·Δdays)``.
                       The decay factor itself is the *boost* for newer
                       episodes; higher λ amplifies the recency bias.
        regime_match_bonus : γ in ``score += γ·1[regime_match]``.  Adds a
                       small additive bonus to candidates whose regime
                       matches the query's.  Default 0.05.  Set to 0.0
                       to disable regime-aware retrieval.
        embed_fn : optional callable mapping text -> sparse vector. Used when
                   you want to swap the default TF-IDF for a richer model
                   embedding without touching the retrieval logic.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.decay_per_day = decay_per_day
        self.decay_weight = decay_weight
        self.regime_match_bonus = regime_match_bonus
        self._embed = embed_fn or self._default_embed

    @staticmethod
    def _default_embed(text: str) -> Dict[str, float]:
        return _tf_vector(_tokenise(text))

    # ------------------------------------------------------------- writes

    def add(self, episode: Episode) -> None:
        """Append a new episode (or replace one with the same ``episode_id``)."""
        existing = list(self.iter_all())
        replaced = False
        for i, ep in enumerate(existing):
            if ep.episode_id == episode.episode_id:
                existing[i] = episode
                replaced = True
                break
        if not replaced:
            existing.append(episode)
        self._rewrite(existing)

    def update_outcome(
        self,
        episode_id: str,
        *,
        realised_return: float,
        alpha_return: float,
        max_drawdown: float,
        reflection: Optional[str] = None,
        outcome_ts: Optional[str] = None,
    ) -> bool:
        """Transition an episode from PENDING to RESOLVED.

        Returns True on success, False when the episode_id was not found.
        Idempotent: re-resolving an already-RESOLVED episode updates its
        outcome but keeps the original ``created_ts``.
        """
        existing = list(self.iter_all())
        found = False
        for ep in existing:
            if ep.episode_id == episode_id:
                ep.realised_return = realised_return
                ep.alpha_return = alpha_return
                ep.max_drawdown = max_drawdown
                ep.outcome_status = OutcomeStatus.RESOLVED
                if reflection is not None:
                    ep.reflection = reflection
                if outcome_ts is not None:
                    ep.outcome_ts = outcome_ts
                found = True
                break
        if found:
            self._rewrite(existing)
        return found

    def abandon(self, episode_id: str, reason: str = "") -> bool:
        """Mark an episode ``ABANDONED`` — excluded from retrieval but retained for replay."""
        existing = list(self.iter_all())
        found = False
        for ep in existing:
            if ep.episode_id == episode_id:
                ep.outcome_status = OutcomeStatus.ABANDONED
                if reason:
                    ep.reflection = f"ABANDONED: {reason}"
                found = True
                break
        if found:
            self._rewrite(existing)
        return found

    # -------------------------------------------------------------- reads

    def iter_all(self) -> Iterable[Episode]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield Episode.from_dict(json.loads(line))

    def all_episodes(self) -> List[Episode]:
        return list(self.iter_all())

    def get(self, episode_id: str) -> Optional[Episode]:
        for ep in self.iter_all():
            if ep.episode_id == episode_id:
                return ep
        return None

    # -------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query_text: str,
        *,
        as_of: str,
        k: int = 5,
        symbol: Optional[str] = None,
        regime: Optional[Regime] = None,
    ) -> List[RetrievedEpisode]:
        """Return up to ``k`` most relevant episodes given the query.

        Filtering rules:

        1. Only ``RESOLVED`` episodes are eligible (PENDING leaks future
           info; ABANDONED is signal-noise).
        2. ``effective_unembargoed_ts`` must be ≤ ``as_of``.  This is the
           outcome-embargo step (paper §13.1.2).
        3. If ``symbol`` is given, prefer same-symbol episodes (we don't
           hard-filter — cross-ticker lessons are valuable per the paper —
           but we boost same-symbol by 10% in the score).
        4. If ``regime`` is given, same-regime episodes get a small boost.

        Score: ``cosine(query, episode) − decay_weight · exp(−decay·Δdays) · (Δdays - 0)`` — wait,
        we use ``cosine + decay_weight · exp(−decay·Δdays)`` where the
        decay factor itself is the *boost* for newer episodes.  Higher
        Δdays → smaller decay → smaller boost.  We add (not subtract)
        because the boost is already monotonically decreasing in Δdays;
        if we subtracted, infinite age would *grow* the score.
        """
        as_of_dt = _parse_ts(as_of)
        query_vec = self._embed(query_text)

        candidates: List[RetrievedEpisode] = []
        for ep in self.iter_all():
            if ep.outcome_status != OutcomeStatus.RESOLVED:
                continue
            if ep.effective_unembargoed_ts() > as_of_dt:
                continue  # OUTCOME EMBARGO — never include future-known outcomes

            ep_vec = self._embed(ep.state_summary)
            sim = _cosine(query_vec, ep_vec)
            if symbol and ep.symbol.upper() == symbol.upper():
                sim *= 1.10

            decision_dt = _parse_ts(ep.decision_ts)
            delta_days = max(0.0, (as_of_dt - decision_dt).total_seconds() / 86400.0)
            decay_factor = math.exp(-self.decay_per_day * delta_days)
            # Final score = cosine similarity + recency boost
            #             + γ * 1[regime_match].  The regime bonus is
            # an *additive* delta so it doesn't get crushed by very high
            # similarity scores (the previous 1.05x multiplier had the
            # opposite problem — irrelevant noise got boosted too).
            score = sim + self.decay_weight * decay_factor
            if regime and ep.regime == regime and self.regime_match_bonus:
                score += self.regime_match_bonus

            candidates.append(
                RetrievedEpisode(
                    episode=ep,
                    similarity=sim,
                    decay_factor=decay_factor,
                    score=score,
                    delta_days=delta_days,
                )
            )

        candidates.sort(key=lambda r: r.score, reverse=True)
        top = candidates[:k]

        # Retrieval-counter bookkeeping — increments on every successful
        # retrieve() call so evict() can preserve frequently-cited
        # lessons.  We persist via a single _rewrite() at the end (no
        # per-episode incremental write) to keep retrieval cheap.
        if top:
            touched = {r.episode.episode_id for r in top}
            episodes = list(self.iter_all())
            changed = False
            for ep in episodes:
                if ep.episode_id in touched:
                    ep.retrieval_count = int(getattr(ep, "retrieval_count", 0) or 0) + 1
                    changed = True
            if changed:
                try:
                    self._rewrite(episodes)
                except OSError:
                    # Persistence is best-effort — retrieval still
                    # succeeds even if disk is read-only.
                    pass

        return top

    def evict(
        self,
        *,
        max_records: Optional[int] = 10_000,
        max_age_days: Optional[int] = 365,
        min_retrieval_count: int = 0,
        keep_resolved: bool = True,
    ) -> int:
        """Prune the episodic store by age + cap on total records.

        Eviction rules — applied in order:

        1. Drop episodes older than ``max_age_days`` **unless** their
           ``retrieval_count`` is strictly above ``min_retrieval_count``
           (so frequently-cited lessons survive even when old).
        2. Drop ``ABANDONED`` episodes outright.  ``RESOLVED`` and
           ``PENDING`` are kept if ``keep_resolved=True`` (the default).
        3. If the survivor set still exceeds ``max_records``, drop the
           *oldest* survivors first (LRU by ``decision_ts``).

        Returns the number of episodes removed.  ``max_records=None`` or
        ``max_age_days=None`` disables that step individually so callers
        can use eviction as a one-shot age-only or count-only sweep.
        """
        episodes = list(self.iter_all())
        if not episodes:
            return 0
        now = datetime.now(timezone.utc)
        survivors: List[Episode] = []
        for ep in episodes:
            if ep.outcome_status == OutcomeStatus.ABANDONED:
                continue
            if max_age_days is not None and max_age_days > 0:
                try:
                    decision_dt = _parse_ts(ep.decision_ts)
                except ValueError:
                    decision_dt = now
                age_days = (now - decision_dt).total_seconds() / 86400.0
                rc = int(getattr(ep, "retrieval_count", 0) or 0)
                if age_days > max_age_days and rc <= min_retrieval_count:
                    continue
            survivors.append(ep)

        # Step 3 — cap on total records (LRU by decision_ts).
        if max_records is not None and len(survivors) > max_records:
            survivors.sort(key=lambda ep: ep.decision_ts, reverse=True)
            survivors = survivors[:max_records]

        removed = len(episodes) - len(survivors)
        if removed:
            self._rewrite(survivors)
        return removed

    # -------------------------------------------------------------- internals

    def _rewrite(self, episodes: Sequence[Episode]) -> None:
        """Atomic JSONL rewrite — same crash-safe pattern as state.py."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for ep in episodes:
                    fh.write(json.dumps(ep.to_dict(), sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("could not persist episodic memory to %s: %s", self.path, exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp, defaulting to UTC if naive.

    Tolerates the common "Z" suffix shorthand.  No silent fallback to
    ``datetime.now()`` — invalid input raises.
    """
    cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
