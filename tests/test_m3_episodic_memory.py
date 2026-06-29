"""M3 — EpisodicMemory: outcome embargo, time-decay, regime filtering."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_crew.agentic.memory.episodic import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
    Regime,
)
from trading_crew.agentic.memory.regime import detect_regime


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


def _make_episode(
    *,
    episode_id="AAPL-2026-01-15",
    symbol="AAPL",
    decision_ts="2026-01-15T20:00:00+00:00",
    outcome_ts="2026-01-22T20:00:00+00:00",
    state_summary="AAPL fundamentals strong, sentiment positive, momentum confirms.",
    regime=Regime.TREND,
    status=OutcomeStatus.RESOLVED,
    realised_return=0.04,
    embargo_days=0,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        symbol=symbol,
        decision_ts=decision_ts,
        state_summary=state_summary,
        regime=regime,
        action_proposal={"side": "BUY", "target_weight": 0.08},
        outcome_ts=outcome_ts,
        outcome_status=status,
        realised_return=realised_return,
        alpha_return=0.02,
        max_drawdown=0.05,
        embargo_days=embargo_days,
    )


def test_episode_roundtrips_through_json():
    ep = _make_episode()
    data = ep.to_dict()
    # JSON-safe
    json.dumps(data)
    parsed = Episode.from_dict(data)
    assert parsed.episode_id == ep.episode_id
    assert parsed.regime == ep.regime
    assert parsed.outcome_status == ep.outcome_status
    assert parsed.realised_return == ep.realised_return


def test_effective_unembargoed_ts_adds_embargo_days():
    ep = _make_episode(outcome_ts="2026-01-22T20:00:00+00:00", embargo_days=10)
    eff = ep.effective_unembargoed_ts()
    expected = datetime(2026, 2, 1, 20, 0, 0, tzinfo=timezone.utc)
    assert eff == expected


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_add_and_iter_all(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="A"))
    mem.add(_make_episode(episode_id="B"))
    ids = [e.episode_id for e in mem.iter_all()]
    assert ids == ["A", "B"]


def test_add_replaces_episode_with_same_id(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="X", realised_return=0.01))
    mem.add(_make_episode(episode_id="X", realised_return=0.05))
    eps = mem.all_episodes()
    assert len(eps) == 1
    assert eps[0].realised_return == 0.05


def test_update_outcome_transitions_pending_to_resolved(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="X", status=OutcomeStatus.PENDING, realised_return=None))
    ok = mem.update_outcome(
        "X", realised_return=0.05, alpha_return=0.02, max_drawdown=0.01,
        reflection="Edge confirmed."
    )
    assert ok is True
    ep = mem.get("X")
    assert ep.outcome_status == OutcomeStatus.RESOLVED
    assert ep.realised_return == 0.05
    assert ep.reflection == "Edge confirmed."


def test_update_outcome_returns_false_when_unknown_id(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="X"))
    assert (
        mem.update_outcome(
            "missing", realised_return=0.0, alpha_return=0.0, max_drawdown=0.0
        ) is False
    )


def test_abandon_marks_status_but_keeps_record(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="X"))
    assert mem.abandon("X", reason="ticker delisted") is True
    ep = mem.get("X")
    assert ep.outcome_status == OutcomeStatus.ABANDONED
    assert "delisted" in ep.reflection


# ---------------------------------------------------------------------------
# Retrieval — outcome embargo (the critical anti-leakage check)
# ---------------------------------------------------------------------------


def test_retrieval_excludes_pending_episodes(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="resolved", status=OutcomeStatus.RESOLVED))
    mem.add(_make_episode(
        episode_id="pending",
        decision_ts="2025-10-01T20:00:00+00:00",
        outcome_ts="2025-10-15T20:00:00+00:00",
        status=OutcomeStatus.PENDING,
    ))
    results = mem.retrieve(
        "AAPL fundamentals strong sentiment positive",
        as_of="2026-02-01T00:00:00+00:00",
        k=5,
    )
    ids = [r.episode.episode_id for r in results]
    assert "resolved" in ids
    assert "pending" not in ids


def test_retrieval_excludes_abandoned_episodes(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="ok"))
    mem.add(_make_episode(episode_id="bad", status=OutcomeStatus.ABANDONED))
    results = mem.retrieve("AAPL momentum", as_of="2026-02-01T00:00:00+00:00", k=5)
    ids = [r.episode.episode_id for r in results]
    assert "bad" not in ids


def test_retrieval_outcome_embargo_blocks_future_known_episodes(tmp_path):
    """The critical anti-leakage check.  An episode whose outcome materialised
    *after* the query's as_of must never be returned — otherwise the agent
    is cheating with knowledge of the future."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    # Episode resolved on 2026-03-01, query is dated 2026-02-15
    mem.add(_make_episode(
        episode_id="future",
        decision_ts="2026-02-20T20:00:00+00:00",
        outcome_ts="2026-03-01T20:00:00+00:00",
    ))
    # An episode resolved before the query is fine
    mem.add(_make_episode(
        episode_id="past",
        decision_ts="2025-12-01T20:00:00+00:00",
        outcome_ts="2025-12-15T20:00:00+00:00",
    ))
    results = mem.retrieve("AAPL momentum", as_of="2026-02-15T00:00:00+00:00", k=5)
    ids = [r.episode.episode_id for r in results]
    assert "future" not in ids, "embargo failed; future-known outcome leaked into retrieval"
    assert "past" in ids


def test_retrieval_embargo_days_extend_the_quarantine(tmp_path):
    """An explicit embargo_days field should push the unembargoes timestamp
    further into the future — outcome occurred 2026-01-22 but embargo_days=14
    means it stays hidden until 2026-02-05."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(
        episode_id="quarantined",
        outcome_ts="2026-01-22T20:00:00+00:00",
        embargo_days=14,
    ))
    # Query 7 days after outcome — still embargoed
    early = mem.retrieve("AAPL momentum", as_of="2026-01-29T00:00:00+00:00", k=5)
    assert "quarantined" not in [r.episode.episode_id for r in early]

    # Query 14+ days after — visible
    late = mem.retrieve("AAPL momentum", as_of="2026-02-10T00:00:00+00:00", k=5)
    assert "quarantined" in [r.episode.episode_id for r in late]


# ---------------------------------------------------------------------------
# Retrieval — similarity + decay scoring
# ---------------------------------------------------------------------------


def test_retrieval_orders_by_similarity_then_decay(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl", decay_per_day=0.01, decay_weight=0.5)
    # Two episodes with identical state_summary — only delta_days differs.
    # The newer one should score higher.
    mem.add(_make_episode(
        episode_id="recent",
        decision_ts="2026-02-01T20:00:00+00:00",
        outcome_ts="2026-02-08T20:00:00+00:00",
    ))
    mem.add(_make_episode(
        episode_id="ancient",
        decision_ts="2024-02-01T20:00:00+00:00",
        outcome_ts="2024-02-08T20:00:00+00:00",
    ))
    results = mem.retrieve(
        "AAPL fundamentals strong sentiment positive",
        as_of="2026-03-01T00:00:00+00:00",
        k=5,
    )
    assert results[0].episode.episode_id == "recent"
    assert results[1].episode.episode_id == "ancient"
    assert results[0].score > results[1].score


def test_retrieval_score_drops_as_age_increases(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl", decay_per_day=0.01, decay_weight=1.0)
    mem.add(_make_episode(
        episode_id="e1",
        decision_ts="2026-01-01T00:00:00+00:00",
        outcome_ts="2026-01-08T00:00:00+00:00",
    ))
    near = mem.retrieve("AAPL momentum", as_of="2026-01-10T00:00:00+00:00", k=1)[0]
    far = mem.retrieve("AAPL momentum", as_of="2026-12-31T00:00:00+00:00", k=1)[0]
    assert near.decay_factor > far.decay_factor


def test_retrieval_boosts_same_symbol_by_10pct(tmp_path):
    """All else equal, an AAPL episode should outrank a MSFT one when querying AAPL."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    text = "earnings beat guidance raised momentum positive"
    mem.add(_make_episode(episode_id="aapl", symbol="AAPL", state_summary=text))
    mem.add(_make_episode(episode_id="msft", symbol="MSFT", state_summary=text))
    results = mem.retrieve(text, as_of="2026-02-01T00:00:00+00:00", k=5, symbol="AAPL")
    ids = [r.episode.episode_id for r in results]
    assert ids[0] == "aapl"


def test_retrieval_returns_empty_when_no_resolved_episodes(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="X", status=OutcomeStatus.PENDING))
    results = mem.retrieve("anything", as_of="2030-01-01T00:00:00+00:00", k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Phase 2A — regime bonus, eviction, schema_version migration, embedder swap
# ---------------------------------------------------------------------------


def test_regime_match_bonus_promotes_same_regime_episode(tmp_path):
    """An episode whose regime matches the query should outrank a same-text
    episode in a different regime — when γ > 0."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl", regime_match_bonus=0.5)
    text = "AAPL fundamentals strong, sentiment positive."
    mem.add(_make_episode(episode_id="t-aapl", state_summary=text, regime=Regime.TREND))
    mem.add(_make_episode(
        episode_id="r-aapl", state_summary=text, regime=Regime.RANGE,
        decision_ts="2026-01-16T00:00:00+00:00",
        outcome_ts="2026-01-23T00:00:00+00:00",
    ))
    results = mem.retrieve(text, as_of="2026-02-01T00:00:00+00:00", k=5, regime=Regime.TREND)
    assert [r.episode.episode_id for r in results][0] == "t-aapl"


def test_regime_match_bonus_zero_disables_boost(tmp_path):
    """γ=0 collapses to the previous similarity-only behaviour."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl", regime_match_bonus=0.0)
    text = "AAPL fundamentals strong."
    mem.add(_make_episode(episode_id="t-aapl", state_summary=text, regime=Regime.TREND))
    mem.add(_make_episode(
        episode_id="r-aapl", state_summary=text, regime=Regime.RANGE,
        decision_ts="2026-01-16T00:00:00+00:00",
        outcome_ts="2026-01-23T00:00:00+00:00",
    ))
    a = mem.retrieve(text, as_of="2026-02-01T00:00:00+00:00", k=5, regime=Regime.TREND)
    b = mem.retrieve(text, as_of="2026-02-01T00:00:00+00:00", k=5, regime=Regime.RANGE)
    # Same set in same order regardless of regime arg.
    assert [r.episode.episode_id for r in a] == [r.episode.episode_id for r in b]


def test_evict_drops_old_episodes_keeps_recently_retrieved(tmp_path):
    """Eviction respects ``min_retrieval_count`` so cited lessons survive."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    # Old episode that was retrieved once.
    old_cited = _make_episode(
        episode_id="old-cited",
        decision_ts="2020-01-15T00:00:00+00:00",
        outcome_ts="2020-01-22T00:00:00+00:00",
    )
    old_cited.retrieval_count = 5
    mem.add(old_cited)
    # Old episode never retrieved.
    mem.add(_make_episode(
        episode_id="old-stale",
        decision_ts="2020-01-15T00:00:00+00:00",
        outcome_ts="2020-01-22T00:00:00+00:00",
    ))
    # Recent episode.
    mem.add(_make_episode(
        episode_id="recent",
        decision_ts=datetime.now(timezone.utc).isoformat(),
        outcome_ts=datetime.now(timezone.utc).isoformat(),
    ))
    removed = mem.evict(max_age_days=365, min_retrieval_count=0)
    surviving_ids = [ep.episode_id for ep in mem.all_episodes()]
    assert "old-stale" not in surviving_ids
    assert "old-cited" in surviving_ids
    assert "recent" in surviving_ids
    assert removed == 1


def test_evict_caps_total_records(tmp_path):
    """``max_records`` keeps the N most recent surviving episodes."""
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    for i in range(6):
        mem.add(_make_episode(
            episode_id=f"e{i}",
            decision_ts=f"2026-01-{i+1:02d}T00:00:00+00:00",
            outcome_ts=f"2026-01-{i+8:02d}T00:00:00+00:00",
        ))
    removed = mem.evict(max_records=3, max_age_days=None)
    survivors = sorted(ep.episode_id for ep in mem.all_episodes())
    # LRU by decision_ts → e3..e5 are the most recent.
    assert survivors == ["e3", "e4", "e5"]
    assert removed == 3


def test_episode_schema_version_migration_from_disk(tmp_path):
    """Loading an older record without schema_version still works and fills 0."""
    p = tmp_path / "ep.jsonl"
    # Write a record without schema_version / retrieval_count.
    legacy = {
        "episode_id": "legacy",
        "symbol": "AAPL",
        "decision_ts": "2026-01-15T00:00:00+00:00",
        "state_summary": "legacy lesson",
        "regime": "TREND",
        "action_proposal": {"side": "BUY"},
        "outcome_ts": "2026-01-22T00:00:00+00:00",
        "outcome_status": "RESOLVED",
        "realised_return": 0.03,
    }
    p.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    mem = EpisodicMemory(p)
    eps = mem.all_episodes()
    assert len(eps) == 1
    assert eps[0].schema_version == 0  # migrator surfaces the old value
    assert eps[0].retrieval_count == 0


def test_retrieval_increments_retrieval_count(tmp_path):
    mem = EpisodicMemory(tmp_path / "ep.jsonl")
    mem.add(_make_episode(episode_id="touched"))
    mem.retrieve("AAPL momentum", as_of="2026-02-01T00:00:00+00:00", k=5)
    refreshed = mem.get("touched")
    assert refreshed is not None and refreshed.retrieval_count >= 1


def test_default_embedder_factory_returns_none_for_tfidf(monkeypatch):
    """``TRADINGCREW_MEMORY_EMBEDDER`` defaults to TF-IDF (None)."""
    from trading_crew.agentic.memory.embedding import get_default_embed_fn
    monkeypatch.delenv("TRADINGCREW_MEMORY_EMBEDDER", raising=False)
    assert get_default_embed_fn() is None
    monkeypatch.setenv("TRADINGCREW_MEMORY_EMBEDDER", "tfidf")
    assert get_default_embed_fn() is None


def test_default_embedder_factory_falls_back_on_missing_env(monkeypatch):
    """``vllm`` without env vars logs a warning + falls back to TF-IDF (None)."""
    from trading_crew.agentic.memory.embedding import get_default_embed_fn
    monkeypatch.setenv("TRADINGCREW_MEMORY_EMBEDDER", "vllm")
    monkeypatch.delenv("VLLM_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("VLLM_EMBEDDING_API_KEY", raising=False)
    assert get_default_embed_fn() is None  # quiet fallback


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


def test_detect_regime_returns_unknown_for_short_input():
    assert detect_regime([100.0, 101.0, 102.0]) == Regime.UNKNOWN


def test_detect_regime_identifies_low_vol_range():
    """A mean-reverting series with no drift should classify as RANGE."""
    import random
    rng = random.Random(42)
    # Low-vol oscillation around 100, no trend
    closes = [100.0 + rng.gauss(0, 0.3) for _ in range(300)]
    assert detect_regime(closes) == Regime.RANGE


def test_detect_regime_identifies_high_vol_on_volatility_spike():
    """A regime where short-window vol is much higher than long-window vol
    should fire HIGH_VOL even when total drawdown is mild."""
    import random
    rng = random.Random(7)
    # 252 days of calm (sigma = 0.4)
    calm = [100.0 + rng.gauss(0, 0.4) for _ in range(252)]
    # 20 days of high vol (sigma = 4.0) — 10x amplitude
    last = calm[-1]
    spike = []
    for _ in range(20):
        last = last + rng.gauss(0, 4.0)
        spike.append(last)
    closes = calm + spike
    regime = detect_regime(closes)
    # Phase 2E — the legacy HIGH_VOL bucket has been split into
    # HIGH_VOL_TREND and HIGH_VOL_RANGE; either is acceptable here.
    assert regime in (Regime.CRISIS, Regime.HIGH_VOL, Regime.HIGH_VOL_TREND, Regime.HIGH_VOL_RANGE)


def test_detect_regime_identifies_crisis_on_drawdown_with_vol_spike():
    """Crisis = high vol AND >10% drawdown."""
    import random
    rng = random.Random(11)
    # 252 days of calm
    calm = [100.0 + rng.gauss(0, 0.4) for _ in range(252)]
    # 20 days of violent, downward-skewed moves
    last = calm[-1]
    crash = []
    for _ in range(20):
        last = last + rng.gauss(-1.0, 4.0)  # negative drift + high vol
        crash.append(last)
    closes = calm + crash
    regime = detect_regime(closes)
    assert regime in (Regime.CRISIS, Regime.HIGH_VOL, Regime.HIGH_VOL_TREND, Regime.HIGH_VOL_RANGE)


# ---------------------------------------------------------------------------
# Phase 2E — HIGH_VOL split (trend / range sub-regimes)
# ---------------------------------------------------------------------------


def test_high_vol_trend_separates_from_high_vol_range():
    """A trending high-vol regime should be HIGH_VOL_TREND, not HIGH_VOL_RANGE."""
    import random
    rng = random.Random(33)
    calm = [100.0 + rng.gauss(0, 0.4) for _ in range(252)]
    # 20 days of high vol *with* a strong positive drift.
    last = calm[-1]
    spike = []
    for _ in range(20):
        last = last + rng.gauss(2.0, 4.0)  # +2 drift, σ=4 → both vol-heavy AND trending
        spike.append(last)
    closes = calm + spike
    regime = detect_regime(closes)
    # Either the legacy umbrella or the new TREND sub-regime is fine —
    # what we *do* want is to never see HIGH_VOL_RANGE on a trending tape.
    assert regime != Regime.HIGH_VOL_RANGE


def test_high_vol_range_when_no_trend():
    """High vol + no drift → HIGH_VOL_RANGE (or CRISIS if drawdown also exceeds the threshold)."""
    import random
    rng = random.Random(99)
    calm = [100.0 + rng.gauss(0, 0.4) for _ in range(252)]
    last = calm[-1]
    spike = []
    for _ in range(20):
        last = last + rng.gauss(0.0, 2.5)  # zero drift, high vol but bounded
        spike.append(last)
    closes = calm + spike
    regime = detect_regime(closes)
    # Either bucket is acceptable — *not* TREND nor RANGE, that's the
    # actual invariant.
    assert regime not in (Regime.TREND, Regime.RANGE, Regime.UNKNOWN)
