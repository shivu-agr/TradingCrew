"""Unit tests for the FinGPT-inspired RLMF module.

We test the reward function against hand-picked outcomes (so it's
obvious why a long-tier + high-conviction win must beat a short-tier
+ low-conviction moonshot), the dataset builder against a synthetic
memory bank, and the few-shot pool for correct ordering + framing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from trading_crew.agentic.memory.episodic import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
    Regime,
)
from trading_crew.agentic.rlmf import (
    RLMFRow,
    _conviction_weight,
    _episode_to_row,
    build_dataset,
    build_few_shot_pool,
    compute_reward,
    export_dataset,
    summarise_dataset,
)
from trading_crew.horizons import Horizon


# ---------------------------------------------------------------------------
# Fixtures — build a small in-memory episode bank
# ---------------------------------------------------------------------------


def _make_episode(
    *,
    episode_id: str,
    symbol: str = "RELIANCE.NS",
    horizon_days: int = 90,
    side: str = "BUY",
    conviction: float = 0.7,
    realised: float = 0.10,
    alpha: float = 0.05,
    status: OutcomeStatus = OutcomeStatus.RESOLVED,
    reflection: str = "",
) -> Episode:
    """Build an ``Episode`` with only the fields the RLMF loop reads."""
    decision_ts = datetime(2025, 1, 15, tzinfo=timezone.utc).isoformat()
    outcome_ts = (
        datetime(2025, 1, 15, tzinfo=timezone.utc)
        + timedelta(days=horizon_days)
    ).isoformat()
    return Episode(
        episode_id=episode_id,
        symbol=symbol,
        decision_ts=decision_ts,
        state_summary=f"Synthetic state for {episode_id}",
        regime=Regime.TREND,
        action_proposal={
            "side": side,
            "conviction_score": conviction,
            "horizon_days": horizon_days,
            "target_weight": 0.02,
        },
        outcome_ts=outcome_ts,
        outcome_status=status,
        realised_return=realised,
        alpha_return=alpha,
        max_drawdown=-0.03,
        reflection=reflection or None,
    )


@pytest.fixture
def memory(tmp_path: Path) -> EpisodicMemory:
    """A memory bank with a curated mix of tier / conviction / outcome."""
    m = EpisodicMemory(tmp_path / "episodes.jsonl")
    episodes = [
        # LONG + high conviction + high alpha => should be the top win
        _make_episode(
            episode_id="win_long_hi",
            horizon_days=1260,
            conviction=0.85,
            realised=0.60, alpha=0.35,
        ),
        # MEDIUM + medium conviction + medium alpha => solid MEDIUM win
        _make_episode(
            episode_id="win_med_mid",
            horizon_days=180,
            conviction=0.60,
            realised=0.15, alpha=0.08,
        ),
        # SHORT + low conviction + tiny alpha => weak SHORT win
        _make_episode(
            episode_id="win_short_lo",
            horizon_days=21,
            conviction=0.35,
            realised=0.04, alpha=0.01,
        ),
        # LONG + high conviction + big negative alpha => top loss
        _make_episode(
            episode_id="loss_long_hi",
            horizon_days=1260,
            conviction=0.85,
            side="SELL",
            realised=-0.40, alpha=-0.30,
        ),
        # PENDING — must be excluded from the dataset entirely
        _make_episode(
            episode_id="pending_medium",
            horizon_days=180,
            status=OutcomeStatus.PENDING,
            realised=0.0, alpha=0.0,
        ),
    ]
    for ep in episodes:
        m.add(ep)
    return m


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_conviction_weight_monotone_in_score():
    """A higher conviction score must produce a larger reward weight —
    the LLM's high-conviction wins should be more informative than
    barely-differentiated abstentions.
    """
    weights = [_conviction_weight(x) for x in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert weights == sorted(weights)
    assert _conviction_weight(0.5) == pytest.approx(1.0)


def test_compute_reward_returns_none_for_pending():
    ep = _make_episode(
        episode_id="pending",
        status=OutcomeStatus.PENDING,
        realised=0.0, alpha=0.0,
    )
    assert compute_reward(ep) is None


def test_compute_reward_prefers_alpha_over_realised():
    """A +5% return during a +8% rally is a *negative* alpha — the
    reward must be signed by alpha, not raw realised return.
    """
    ep = _make_episode(
        episode_id="rally_underperform",
        realised=0.05,
        alpha=-0.03,
    )
    r = compute_reward(ep)
    assert r is not None
    assert r < 0, "underperforming a hot market should be negative reward"


def test_compute_reward_scales_by_horizon_tier():
    """LONG-tier hits get more credit than SHORT-tier hits for the
    same alpha — matches the desk's mid-to-long-term bias.
    """
    short_ep = _make_episode(
        episode_id="s", horizon_days=21, alpha=0.05, realised=0.05,
    )
    medium_ep = _make_episode(
        episode_id="m", horizon_days=180, alpha=0.05, realised=0.05,
    )
    long_ep = _make_episode(
        episode_id="l", horizon_days=1260, alpha=0.05, realised=0.05,
    )
    r_s = compute_reward(short_ep)
    r_m = compute_reward(medium_ep)
    r_l = compute_reward(long_ep)
    assert r_s is not None and r_m is not None and r_l is not None
    assert r_s < r_m < r_l, (
        f"expected SHORT<MEDIUM<LONG rewards, got {r_s} / {r_m} / {r_l}"
    )


def test_compute_reward_clips_extreme_returns():
    """A 500% moonshot must not blow up the reward — the winsorization
    keeps the case-book balanced across ordinary vs freak outcomes.
    """
    normal = _make_episode(
        episode_id="normal", realised=0.10, alpha=0.10,
    )
    moon = _make_episode(
        episode_id="moon", realised=5.0, alpha=5.0,
    )
    r_normal = compute_reward(normal)
    r_moon = compute_reward(moon)
    assert r_normal is not None and r_moon is not None
    assert r_moon < 20.0, "moonshot should be winsorised, not proportional"


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def test_build_dataset_excludes_pending_episodes(memory):
    rows = build_dataset(memory=memory)
    ids = {r.episode_id for r in rows}
    assert "pending_medium" not in ids


def test_build_dataset_sorted_descending_by_reward(memory):
    rows = build_dataset(memory=memory)
    # The high-alpha LONG win must be first; the high-conviction LONG
    # loss must be last (largest-magnitude negative reward).
    assert rows[0].episode_id == "win_long_hi"
    assert rows[-1].episode_id == "loss_long_hi"
    # And the ranking must be monotone descending
    rewards = [r.reward for r in rows]
    assert rewards == sorted(rewards, reverse=True)


def test_build_dataset_filters_by_symbol(memory, tmp_path):
    # Add an AAPL episode; ensure symbol filter keeps only RELIANCE.NS.
    mem2 = EpisodicMemory(tmp_path / "episodes2.jsonl")
    for ep in memory.all_episodes():
        mem2.add(ep)
    mem2.add(_make_episode(
        episode_id="win_us", symbol="AAPL", alpha=0.20, realised=0.30,
    ))
    all_rows = build_dataset(memory=mem2)
    reliance_rows = build_dataset(memory=mem2, symbol="RELIANCE.NS")
    aapl_rows = build_dataset(memory=mem2, symbol="AAPL")
    assert len(all_rows) == len(reliance_rows) + len(aapl_rows)
    assert {r.symbol for r in reliance_rows} == {"RELIANCE.NS"}
    assert {r.symbol for r in aapl_rows} == {"AAPL"}


def test_build_dataset_filters_by_horizon_tier(memory):
    long_rows = build_dataset(memory=memory, horizon_tier=Horizon.LONG)
    for r in long_rows:
        assert r.horizon == "LONG"
    assert {r.episode_id for r in long_rows} == {"win_long_hi", "loss_long_hi"}


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------


def test_export_dataset_is_fingpt_compatible(memory, tmp_path):
    rows = build_dataset(memory=memory)
    path = tmp_path / "rlmf.jsonl"
    export_dataset(rows, path)

    import json
    lines = path.read_text().splitlines()
    assert len(lines) == len(rows)
    first = json.loads(lines[0])
    # FinGPT-Forecaster's JSONL schema requires ``prompt`` + ``response``
    # — these are what the LoRA fine-tune notebook reads.
    assert "prompt" in first
    assert "response" in first
    # Our extra analytics fields must ride alongside without breaking
    # the compat.
    for k in ("horizon", "action", "reward", "realised_return"):
        assert k in first


# ---------------------------------------------------------------------------
# Few-shot exemplar pool
# ---------------------------------------------------------------------------


def test_build_few_shot_pool_returns_empty_on_cold_start(tmp_path):
    """A fresh install with no resolved episodes must produce an
    empty string so the PM prompt doesn't get polluted with a bare
    "PAST WINS" heading over an empty list.
    """
    m = EpisodicMemory(tmp_path / "cold.jsonl")
    assert build_few_shot_pool(memory=m) == ""


def test_build_few_shot_pool_covers_wins_and_losses(memory):
    pool = build_few_shot_pool(memory=memory, k_wins=2, k_losses=1)
    assert "PAST WINS TO EMULATE" in pool
    assert "PAST LOSSES TO AVOID" in pool
    # The prefix should reference at least the top winner and the
    # top loser explicitly.
    assert "win_long_hi" in pool or "RELIANCE.NS" in pool
    assert "loss_long_hi" in pool or "RELIANCE.NS" in pool


def test_few_shot_pool_respects_symbol_filter(memory, tmp_path):
    mem2 = EpisodicMemory(tmp_path / "e.jsonl")
    for ep in memory.all_episodes():
        mem2.add(ep)
    mem2.add(_make_episode(
        episode_id="aapl_win", symbol="AAPL", alpha=0.20,
    ))
    pool_all = build_few_shot_pool(memory=mem2)
    pool_reliance = build_few_shot_pool(memory=mem2, symbol="RELIANCE.NS")
    assert "AAPL" in pool_all
    assert "AAPL" not in pool_reliance


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def test_summarise_dataset_shape(memory):
    rows = build_dataset(memory=memory)
    summary = summarise_dataset(rows)
    assert summary["n_resolved"] == len(rows)
    # We expect all three tiers to be represented in the fixture.
    assert set(summary["by_tier"].keys()) >= {"SHORT", "MEDIUM", "LONG"}
    for tier, bucket in summary["by_tier"].items():
        assert "n" in bucket and "mean_reward" in bucket
    assert isinstance(summary["top_wins"], list)
    assert isinstance(summary["top_losses"], list)


def test_summarise_dataset_empty_is_stable():
    """The web UI polls this even before any resolutions — must return
    a well-formed empty payload, not raise / return None.
    """
    summary = summarise_dataset([])
    assert summary["n_resolved"] == 0
    assert summary["by_tier"] == {}
    assert summary["top_wins"] == []
