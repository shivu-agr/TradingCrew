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
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
EPISODE_SCHEMA_VERSION = 2


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
    # Structured L2 reflection: ``{desks: [...], error_class: str,
    # recommended_change: str, tag: str, confidence: float}``.  Drives
    # accurate cross-desk feedback routing (see ``agent_feedback``) and the
    # calibration analytics.  Optional + additive so older episodes (free-text
    # reflection only) still load with None.
    reflection_meta: Optional[Dict[str, Any]] = None
    embargo_days: int = 0
    created_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = EPISODE_SCHEMA_VERSION
    retrieval_count: int = 0
    # Swarm-intelligence crowd-positioning score at decision time, in
    # [-1, +1] (+ crowd long / - crowd short).  Optional + additive so
    # older episodes (written before the swarm layer) still load with None
    # and the backtest can segment performance by crowd regime.
    crowd_score: Optional[float] = None
    # Per-component crowd contributions at decision time (each in [-1, +1],
    # keyed by "put_call" / "iv_skew" / "insider" / "analyst" /
    # "institutional" / "cot").  Logged so the crowd-weight learner can fit
    # data-driven fusion weights from realised alpha.  Optional + additive.
    crowd_components: Optional[Dict[str, float]] = None
    # --- Trust / provenance (Phase 0) ------------------------------------
    # Which workflow version produced this decision.  Optional + additive
    # so pre-Phase-0 episodes load with None; the backtest / grid-search /
    # RLMF paths can filter to a single trustworthy cohort instead of
    # blindly replaying proposals from a drifting workflow.
    workflow_version: Optional[str] = None
    prompts_hash: Optional[str] = None
    code_git_sha: Optional[str] = None
    source: str = "live"  # "live" | "backfill"
    # Per-desk directional lean at decision time, keyed by crew agent_key
    # (``market_analyst`` → ``bullish``/``bearish``/``neutral``).  Optional so
    # older episodes still load.  Used after resolve to attribute a WIN/LOSS
    # to the steps that actually predicted the wrong way.
    desk_leans: Optional[Dict[str, str]] = None
    # Unsigned close-to-close market return over the holding window (not
    # side-signed).  Set on resolve so HOLD opportunity-cost and desk-vs-tape
    # alignment do not depend on the position PnL.
    market_return: Optional[float] = None
    # Canonical prediction-vs-outcome label (WIN/LOSS/FLAT/HOLD_*), set on
    # resolve.  See ``trading_crew.agentic.outcome``.
    outcome_label: Optional[str] = None
    # Resolve-day web briefing of *why* the tape moved.  Training-only
    # (Mass 1PO negatives).  Never written into a live or PIT decision.
    market_reason: Optional[str] = None
    # 30-day learning clock (desk_strategy.yaml).  Independent of
    # ``outcome_ts`` so a MEDIUM/LONG thesis can stay pending while the
    # short review teaches the desks.
    review_ts: Optional[str] = None
    review_status: Optional[str] = None  # PENDING / RESOLVED
    review_embargo_days: int = 0
    review: Optional[Dict[str, Any]] = None
    # Per-tier priced scores once each clock elapses: {short|medium|long: {...}}.
    horizon_scores: Optional[Dict[str, Any]] = None

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
            reflection_meta=data.get("reflection_meta"),
            embargo_days=int(data.get("embargo_days", 0)),
            created_ts=data.get("created_ts", datetime.now(timezone.utc).isoformat()),
            schema_version=int(data.get("schema_version", 0)),
            retrieval_count=int(data.get("retrieval_count", 0)),
            crowd_score=data.get("crowd_score"),
            crowd_components=data.get("crowd_components"),
            workflow_version=data.get("workflow_version"),
            prompts_hash=data.get("prompts_hash"),
            code_git_sha=data.get("code_git_sha"),
            source=data.get("source", "live"),
            desk_leans=data.get("desk_leans"),
            market_return=data.get("market_return"),
            outcome_label=data.get("outcome_label"),
            market_reason=data.get("market_reason"),
            review_ts=data.get("review_ts"),
            review_status=data.get("review_status"),
            review_embargo_days=int(data.get("review_embargo_days", 0) or 0),
            review=data.get("review"),
            horizon_scores=data.get("horizon_scores"),
        )

    def effective_unembargoed_ts(self) -> datetime:
        """Earliest timestamp at which this episode becomes retrievable.

        Equal to ``outcome_ts + embargo_days``.  PENDING episodes are
        always embargoed (they have no realised outcome yet) unless a
        30-day review has resolved — see ``effective_review_unembargoed_ts``.
        """
        base = datetime.fromisoformat(self.outcome_ts.replace("Z", "+00:00"))
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return base + _days(self.embargo_days)

    def effective_review_unembargoed_ts(self) -> Optional[datetime]:
        """When the 30-day learning review may be retrieved.

        ``None`` until ``review_ts`` is set.  Thesis numbers stay hidden
        until ``effective_unembargoed_ts`` — this clock only unlocks the
        short-review lesson.
        """
        if not self.review_ts:
            return None
        base = datetime.fromisoformat(self.review_ts.replace("Z", "+00:00"))
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return base + _days(int(self.review_embargo_days or 0))

    def review_retrievable(self, as_of: datetime) -> bool:
        if (self.review_status or "").upper() != "RESOLVED":
            return False
        open_at = self.effective_review_unembargoed_ts()
        return open_at is not None and open_at <= as_of


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


def _ep_create(conn: sqlite3.Connection) -> None:
    """Create the ``episodes`` table + indexes on ``conn`` (idempotent)."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
               episode_id      TEXT PRIMARY KEY,
               symbol          TEXT,
               decision_ts     TEXT,
               outcome_ts      TEXT,
               outcome_status  TEXT,
               source          TEXT,
               retrieval_count INTEGER NOT NULL DEFAULT 0,
               created_ts      TEXT,
               data_json       TEXT NOT NULL
           )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_status ON episodes(outcome_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_symbol ON episodes(symbol)")


class EpisodicMemory:
    """Episodic store with outcome-embargo retrieval, backed by SQLite.

    Storage is an ``episodes`` table inside the single application database
    (``prediction.db``) — not a separate file, and not the former append-only
    JSONL — because:
    - Upsert-by-id is O(1); the JSONL ``add`` did a full read-modify-write of
      the entire file per append (O(n²) over a 200-name batch).
    - Concurrent collector workers can write safely (WAL + busy-timeout).
    - It keeps *all* app state in one DB with multiple tables.

    The ``path`` argument still names the store.  The canonical production
    store (``<cache>/memory/episodes.*`` sitting next to ``prediction.db``)
    is transparently routed into ``prediction.db``'s ``episodes`` table,
    sharing that DB's single serialized connection.  Any other path (tests)
    opens its own isolated SQLite file, so per-test stores never collide.
    Legacy ``episodes.jsonl`` / standalone ``episodes.db`` data is imported
    once on first open.
    """

    _MIGRATED_DBS: set = set()  # store keys whose legacy data was already imported

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
        path : store location.  The canonical ``<cache>/memory/episodes.*``
               store is routed into the shared ``prediction.db``; any other
               ``.jsonl``/``.db`` path opens its own isolated SQLite file.
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
        resolved = self.path if self.path.suffix == ".db" else self.path.with_suffix(".db")

        # Route the canonical production store into the single app DB so all
        # state lives in one file (multiple tables), sharing its connection.
        self._shared = False
        self._pdb = None
        try:
            from web.backend import prediction_db as _pdb

            pred = _pdb.db_path()
            if (
                resolved.name == "episodes.db"
                and resolved.parent.name == "memory"
                and resolved.parent.parent == pred.parent
            ):
                self._shared = True
                self._pdb = _pdb
                resolved = pred
        except Exception:  # noqa: BLE001 — prediction_db unavailable ⇒ standalone
            pass

        self._db_path = resolved
        self.decay_per_day = decay_per_day
        self.decay_weight = decay_weight
        self.regime_match_bonus = regime_match_bonus
        self._embed = embed_fn or self._default_embed

        self._own_lock = threading.Lock()
        self._own_conn: Optional[sqlite3.Connection] = None
        if self._shared:
            conn, lock = self._acquire()
            with lock:
                _ep_create(conn)
                conn.commit()
            self._migrate_shared()
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._own_conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._own_conn.row_factory = sqlite3.Row
            self._own_conn.execute("PRAGMA journal_mode=WAL")
            self._own_conn.execute("PRAGMA busy_timeout=10000")
            _ep_create(self._own_conn)
            self._own_conn.commit()
            self._migrate_legacy_jsonl()

    @staticmethod
    def _default_embed(text: str) -> Dict[str, float]:
        return _tf_vector(_tokenise(text))

    # ------------------------------------------------------------- internals

    def _acquire(self):
        """Return the active ``(connection, lock)`` pair.

        Shared mode fetches ``prediction.db``'s connection fresh each call so
        it stays valid even if that singleton reopens (e.g. cache-dir change).
        """
        if self._shared:
            return self._pdb.shared_connection()
        return self._own_conn, self._own_lock

    def _bulk_insert(self, episodes: Sequence[Episode]) -> None:
        """INSERT OR IGNORE a batch (never clobbers a newer existing row)."""
        conn, lock = self._acquire()
        with lock:
            for ep in episodes:
                conn.execute(
                    "INSERT OR IGNORE INTO episodes "
                    "(episode_id, symbol, decision_ts, outcome_ts, outcome_status, "
                    " source, retrieval_count, created_ts, data_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    self._episode_row(ep),
                )
            conn.commit()

    def _import_jsonl(self, jsonl: Path) -> None:
        try:
            episodes: List[Episode] = []
            with jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        episodes.append(Episode.from_dict(json.loads(line)))
            self._bulk_insert(episodes)
            jsonl.rename(jsonl.with_suffix(".jsonl.migrated"))
            logger.info("imported %d legacy episodes from %s into %s",
                        len(episodes), jsonl, self._db_path)
        except Exception as exc:  # noqa: BLE001 — migration is best-effort
            logger.warning("legacy episode JSONL import failed for %s: %s", jsonl, exc)

    def _import_sqlite(self, src_path: Path) -> None:
        try:
            src = sqlite3.connect(str(src_path))
            src.row_factory = sqlite3.Row
            rows = src.execute("SELECT data_json FROM episodes").fetchall()
            src.close()
            episodes = [Episode.from_dict(json.loads(r["data_json"])) for r in rows]
            self._bulk_insert(episodes)
            src_path.rename(src_path.with_suffix(".db.migrated"))
            logger.info("imported %d episodes from standalone %s into %s",
                        len(episodes), src_path, self._db_path)
        except Exception as exc:  # noqa: BLE001 — migration is best-effort
            logger.warning("standalone episode DB import failed for %s: %s", src_path, exc)

    def _migrate_shared(self) -> None:
        """Import legacy episode stores from ``<cache>/memory/`` into the shared DB once."""
        key = "shared:" + str(self._db_path)
        if key in EpisodicMemory._MIGRATED_DBS:
            return
        EpisodicMemory._MIGRATED_DBS.add(key)
        memdir = self._db_path.parent / "memory"
        jsonl = memdir / "episodes.jsonl"
        if jsonl.is_file():
            self._import_jsonl(jsonl)
        standalone = memdir / "episodes.db"
        if standalone.is_file() and standalone != self._db_path:
            self._import_sqlite(standalone)

    def _migrate_legacy_jsonl(self) -> None:
        """Standalone mode: import a legacy JSONL at ``self.path`` once."""
        key = str(self._db_path)
        if key in EpisodicMemory._MIGRATED_DBS:
            return
        EpisodicMemory._MIGRATED_DBS.add(key)
        legacy = self.path
        if legacy.suffix == ".jsonl" and legacy.is_file():
            self._import_jsonl(legacy)

    @staticmethod
    def _episode_row(ep: Episode) -> tuple:
        return (
            ep.episode_id, ep.symbol, ep.decision_ts, ep.outcome_ts,
            ep.outcome_status.value, ep.source,
            int(getattr(ep, "retrieval_count", 0) or 0), ep.created_ts,
            json.dumps(ep.to_dict(), sort_keys=True),
        )

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode.from_dict(json.loads(row["data_json"]))

    def _upsert(self, ep: Episode) -> None:
        conn, lock = self._acquire()
        with lock:
            conn.execute(
                """INSERT INTO episodes
                     (episode_id, symbol, decision_ts, outcome_ts, outcome_status,
                      source, retrieval_count, created_ts, data_json)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(episode_id) DO UPDATE SET
                     symbol=excluded.symbol, decision_ts=excluded.decision_ts,
                     outcome_ts=excluded.outcome_ts, outcome_status=excluded.outcome_status,
                     source=excluded.source, retrieval_count=excluded.retrieval_count,
                     created_ts=excluded.created_ts, data_json=excluded.data_json""",
                self._episode_row(ep),
            )
            conn.commit()

    # ------------------------------------------------------------- writes

    def add(self, episode: Episode) -> None:
        """Insert a new episode (or replace one with the same ``episode_id``)."""
        self._upsert(episode)

    def update_outcome(
        self,
        episode_id: str,
        *,
        realised_return: float,
        alpha_return: float,
        max_drawdown: float,
        reflection: Optional[str] = None,
        reflection_meta: Optional[Dict[str, Any]] = None,
        outcome_ts: Optional[str] = None,
        market_return: Optional[float] = None,
        outcome_label: Optional[str] = None,
        market_reason: Optional[str] = None,
    ) -> bool:
        """Transition an episode from PENDING to RESOLVED.

        Returns True on success, False when the episode_id was not found.
        Idempotent: re-resolving an already-RESOLVED episode updates its
        outcome but keeps the original ``created_ts``.
        """
        ep = self.get(episode_id)
        if ep is None:
            return False
        ep.realised_return = realised_return
        ep.alpha_return = alpha_return
        ep.max_drawdown = max_drawdown
        ep.outcome_status = OutcomeStatus.RESOLVED
        if reflection is not None:
            ep.reflection = reflection
        if reflection_meta is not None:
            ep.reflection_meta = reflection_meta
        if outcome_ts is not None:
            ep.outcome_ts = outcome_ts
        if market_return is not None:
            ep.market_return = market_return
        if outcome_label is not None:
            ep.outcome_label = outcome_label
        if market_reason is not None:
            ep.market_reason = market_reason
        self._upsert(ep)
        return True

    def update_review(
        self,
        episode_id: str,
        *,
        review_ts: str,
        realised_return: float,
        alpha_return: float,
        max_drawdown: float,
        market_return: Optional[float],
        outcome_label: str,
        reflection: Optional[str] = None,
        reflection_meta: Optional[Dict[str, Any]] = None,
        embargo_days: int = 0,
    ) -> bool:
        """Write the 30-day learning review without flipping thesis status."""
        ep = self.get(episode_id)
        if ep is None:
            return False
        payload: Dict[str, Any] = {
            "realised_return": realised_return,
            "alpha_return": alpha_return,
            "max_drawdown": max_drawdown,
            "market_return": market_return,
            "outcome_label": outcome_label,
            "reflection": reflection,
            "reflection_meta": reflection_meta,
        }
        ep.review_ts = review_ts
        ep.review_status = "RESOLVED"
        ep.review_embargo_days = int(embargo_days)
        ep.review = payload
        scores = dict(ep.horizon_scores or {})
        scores["short"] = {
            "realised_return": realised_return,
            "alpha_return": alpha_return,
            "max_drawdown": max_drawdown,
            "outcome_label": outcome_label,
        }
        ep.horizon_scores = scores
        self._upsert(ep)
        return True

    def abandon(self, episode_id: str, reason: str = "") -> bool:
        """Mark an episode ``ABANDONED`` — excluded from retrieval but retained for replay."""
        ep = self.get(episode_id)
        if ep is None:
            return False
        ep.outcome_status = OutcomeStatus.ABANDONED
        if reason:
            ep.reflection = f"ABANDONED: {reason}"
        self._upsert(ep)
        return True

    # -------------------------------------------------------------- reads

    def iter_all(self) -> Iterable[Episode]:
        conn, lock = self._acquire()
        with lock:
            rows = conn.execute(
                "SELECT data_json FROM episodes ORDER BY rowid"
            ).fetchall()
        for row in rows:
            yield self._row_to_episode(row)

    def all_episodes(self) -> List[Episode]:
        return list(self.iter_all())

    def get(self, episode_id: str) -> Optional[Episode]:
        conn, lock = self._acquire()
        with lock:
            row = conn.execute(
                "SELECT data_json FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return self._row_to_episode(row) if row else None

    @property
    def db_path(self) -> Path:
        """The SQLite file backing this store (the shared prediction.db in production)."""
        return self._db_path

    def remove(self, episode_id: str) -> bool:
        """Delete one episode by id.  Returns True if a row was removed."""
        conn, lock = self._acquire()
        with lock:
            cur = conn.execute("DELETE FROM episodes WHERE episode_id=?", (episode_id,))
            conn.commit()
        return cur.rowcount > 0

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

        1. ``RESOLVED`` episodes whose ``effective_unembargoed_ts`` ≤
           ``as_of`` are eligible (the thesis-horizon embargo).
        2. ``PENDING`` episodes whose 30-day ``review`` has resolved and
           whose ``effective_review_unembargoed_ts`` ≤ ``as_of`` are also
           eligible — only the short-review lesson is visible, not the
           unfinished thesis outcome.
        3. ``ABANDONED`` is excluded.
        4. If ``symbol`` is given, prefer same-symbol episodes (we don't
           hard-filter — cross-ticker lessons are valuable per the paper —
           but we boost same-symbol by 10% in the score).
        5. If ``regime`` is given, same-regime episodes get a small boost.

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
            if ep.outcome_status == OutcomeStatus.ABANDONED:
                continue
            thesis_open = (
                ep.outcome_status == OutcomeStatus.RESOLVED
                and ep.effective_unembargoed_ts() <= as_of_dt
            )
            review_open = (not thesis_open) and ep.review_retrievable(as_of_dt)
            if not thesis_open and not review_open:
                continue  # OUTCOME / REVIEW EMBARGO — never include future-known tape

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
        # retrieve() call so evict() can preserve frequently-cited lessons.
        # A targeted UPDATE per touched id (keeps ``data_json`` in sync too).
        if top:
            try:
                conn, lock = self._acquire()
                with lock:
                    for r in top:
                        ep = r.episode
                        ep.retrieval_count = int(getattr(ep, "retrieval_count", 0) or 0) + 1
                        conn.execute(
                            "UPDATE episodes SET retrieval_count=?, data_json=? WHERE episode_id=?",
                            (ep.retrieval_count, json.dumps(ep.to_dict(), sort_keys=True), ep.episode_id),
                        )
                    conn.commit()
            except sqlite3.Error:
                # Persistence is best-effort — retrieval still succeeds.
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

        survivor_ids = {ep.episode_id for ep in survivors}
        removed_ids = [ep.episode_id for ep in episodes if ep.episode_id not in survivor_ids]
        if removed_ids:
            conn, lock = self._acquire()
            with lock:
                conn.executemany(
                    "DELETE FROM episodes WHERE episode_id=?", [(i,) for i in removed_ids]
                )
                conn.commit()
        return len(removed_ids)


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
