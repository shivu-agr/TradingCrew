"""Phase 0 — workflow-version stamping + cohort filtering."""

from __future__ import annotations

from trading_crew.agentic import versioning
from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime


def _episode(episode_id: str, *, wf=None, prompts=None, source="live") -> Episode:
    return Episode(
        episode_id=episode_id,
        symbol="AAA",
        decision_ts="2024-01-01T00:00:00+00:00",
        state_summary="s",
        regime=Regime.UNKNOWN,
        action_proposal={"side": "BUY", "target_weight": 0.1, "horizon_days": 30},
        outcome_ts="2024-02-01T00:00:00+00:00",
        outcome_status=OutcomeStatus.PENDING,
        workflow_version=wf,
        prompts_hash=prompts,
        source=source,
    )


def test_prompts_hash_deterministic_and_workflow_version_shape():
    versioning.reset_cache()
    h1 = versioning.prompts_hash()
    h2 = versioning.prompts_hash()
    assert h1 == h2
    wf = versioning.workflow_version()
    # "{git8}-{prompts8}" — two 8-char segments joined by a hyphen.
    assert wf.count("-") == 1
    left, right = wf.split("-")
    assert len(left) == 8 and len(right) == 8


def test_episode_roundtrip_preserves_provenance_fields():
    ep = _episode("AAA-1", wf="abc-def", prompts="deadbeef", source="backfill")
    restored = Episode.from_dict(ep.to_dict())
    assert restored.workflow_version == "abc-def"
    assert restored.prompts_hash == "deadbeef"
    assert restored.source == "backfill"


def test_legacy_episode_loads_with_live_defaults():
    # A pre-Phase-0 record has none of the provenance keys.
    legacy = {
        "episode_id": "AAA-legacy",
        "symbol": "AAA",
        "decision_ts": "2023-01-01T00:00:00+00:00",
        "state_summary": "s",
        "regime": "UNKNOWN",
        "action_proposal": {"side": "BUY"},
        "outcome_ts": "2023-02-01T00:00:00+00:00",
    }
    ep = Episode.from_dict(legacy)
    assert ep.workflow_version is None
    assert ep.prompts_hash is None
    assert ep.source == "live"


def test_cohort_filter_matches_and_excludes_unstamped():
    eps = [
        _episode("m1", wf="v1"),
        _episode("m2", wf="v1"),
        _episode("m3", wf="v2"),
        _episode("legacy", wf=None),
    ]
    # No filter → everything.
    assert len(versioning.filter_episodes_by_cohort(eps)) == 4
    # Specific version → only that cohort; unstamped excluded.
    v1 = versioning.filter_episodes_by_cohort(eps, workflow_version="v1")
    assert {e.episode_id for e in v1} == {"m1", "m2"}


def test_cohort_filter_by_source():
    eps = [
        _episode("live1", source="live"),
        _episode("bf1", source="backfill"),
        _episode("bf2", source="backfill"),
    ]
    bf = versioning.filter_episodes_by_cohort(eps, source="backfill")
    assert {e.episode_id for e in bf} == {"bf1", "bf2"}
