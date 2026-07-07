"""Tests for the web-backend RLMF controller + runtime LLM presets.

Covers the read-only surfaces the "expose training data" UI needs
(``list_episodes`` / ``dataset_summary`` / ``export_dataset``) and the
runtime-preset registration that lets a server-launched (base / base +
LoRA) model show up in the run-config LLM dropdown for inference.

The heavy LoRA / vLLM-serving paths are intentionally *not* exercised
here — they need torch + a GPU + a vLLM binary.  We assert the thin
control surface (single-tenant guard, empty-dataset rejection) instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_crew.agentic.memory.episodic import (
    Episode,
    EpisodicMemory,
    OutcomeStatus,
    Regime,
)


def _episode(
    *,
    episode_id: str,
    symbol: str = "RELIANCE.NS",
    status: OutcomeStatus = OutcomeStatus.RESOLVED,
    realised: float = 0.10,
    alpha: float = 0.05,
    horizon_days: int = 180,
) -> Episode:
    decision_ts = datetime(2025, 1, 15, tzinfo=timezone.utc).isoformat()
    outcome_ts = (
        datetime(2025, 1, 15, tzinfo=timezone.utc) + timedelta(days=horizon_days)
    ).isoformat()
    return Episode(
        episode_id=episode_id,
        symbol=symbol,
        decision_ts=decision_ts,
        state_summary=f"Synthetic state for {episode_id}",
        regime=Regime.TREND,
        action_proposal={
            "side": "BUY",
            "conviction_score": 0.7,
            "horizon_days": horizon_days,
            "target_weight": 0.02,
        },
        outcome_ts=outcome_ts,
        outcome_status=status,
        realised_return=realised if status == OutcomeStatus.RESOLVED else None,
        alpha_return=alpha if status == OutcomeStatus.RESOLVED else None,
        max_drawdown=-0.03,
    )


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the RLMF/episode store at a temp dir and seed episodes."""
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    store = tmp_path / "memory" / "episodes.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    m = EpisodicMemory(store)
    m.add(_episode(episode_id="r1", symbol="RELIANCE.NS"))
    m.add(_episode(episode_id="r2", symbol="RELIANCE.NS"))
    m.add(_episode(episode_id="p1", symbol="TCS.NS", status=OutcomeStatus.PENDING))
    return tmp_path


# ---------------------------------------------------------------------------
# Episodes / dataset read surfaces
# ---------------------------------------------------------------------------


def test_list_episodes_counts_and_filters(cache: Path) -> None:
    from web.backend.rlmf_runner import list_episodes

    allrows = list_episodes()
    assert allrows["total"] == 3
    assert allrows["counts"]["RESOLVED"] == 2
    assert allrows["counts"]["PENDING"] == 1
    assert allrows["store_exists"] is True

    # symbol filter
    reliance = list_episodes(symbol="reliance.ns")
    assert {e["symbol"] for e in reliance["episodes"]} == {"RELIANCE.NS"}

    # status filter
    pending = list_episodes(status="PENDING")
    assert [e["episode_id"] for e in pending["episodes"]] == ["p1"]
    # state_summary is truncated but present for the expand-on-click view
    assert pending["episodes"][0]["state_summary"].startswith("Synthetic state")


def test_dataset_summary_only_resolved(cache: Path) -> None:
    from web.backend.rlmf_runner import dataset_summary

    ds = dataset_summary()
    # Only the 2 RESOLVED RELIANCE episodes become training rows.
    assert ds["summary"]["n_resolved"] == 2
    assert len(ds["rows"]) == 2
    assert all("prompt" in r and "reward" in r for r in ds["rows"])


def test_export_dataset_writes_jsonl(cache: Path) -> None:
    from web.backend.rlmf_runner import export_dataset

    out = export_dataset()
    assert out["rows"] == 2
    p = Path(out["path"])
    assert p.exists()
    assert len(p.read_text().strip().splitlines()) == 2


def test_seed_and_clear_sample_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.rlmf_runner import (
        clear_sample_episodes,
        dataset_summary,
        list_episodes,
        seed_sample_episodes,
    )

    seeded = seed_sample_episodes(count=8)
    assert seeded["seeded"] == 8

    listing = list_episodes()
    assert listing["total"] == 8
    assert listing["sample_count"] == 8
    # All sample episodes are RESOLVED, so they become RLMF training rows.
    assert dataset_summary()["summary"]["n_resolved"] == 8

    cleared = clear_sample_episodes()
    assert cleared["removed"] == 8
    assert list_episodes()["total"] == 0


def test_clear_sample_preserves_real_episodes(cache: Path) -> None:
    # ``cache`` seeds 3 real episodes; add samples on top, then clear
    # should remove ONLY the samples.
    from web.backend.rlmf_runner import (
        clear_sample_episodes,
        list_episodes,
        seed_sample_episodes,
    )

    seed_sample_episodes(count=4)
    assert list_episodes()["total"] == 7
    clear_sample_episodes()
    after = list_episodes()
    assert after["total"] == 3
    assert after["sample_count"] == 0


def test_lora_start_rejects_empty_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh cache dir → no episodes → fine-tune must refuse rather than
    # train on an empty corpus.
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    from web.backend.rlmf_runner import start_lora_training

    with pytest.raises(ValueError, match="No resolved episodes"):
        start_lora_training(base_model="Qwen/Qwen2.5-1.5B-Instruct")


# ---------------------------------------------------------------------------
# Runtime LLM presets (inference with/without LoRA)
# ---------------------------------------------------------------------------


def test_runtime_preset_registration_and_resolution() -> None:
    from trading_crew.llm_presets import (
        LlmPreset,
        clear_runtime_presets,
        list_presets,
        register_runtime_preset,
        set_active,
    )

    clear_runtime_presets()
    register_runtime_preset(
        LlmPreset(
            id="served-lora",
            label="Served + LoRA",
            kind="open-source",
            provider="hosted_vllm",
            model="rlmf",
            base_url="http://localhost:8081/v1",
        )
    )
    try:
        presets = list_presets()
        ids = [p["id"] for p in presets]
        assert "served-lora" in ids
        # Runtime presets float to the front and are always "configured".
        assert presets[0]["id"] == "served-lora"
        served = next(p for p in presets if p["id"] == "served-lora")
        assert served["runtime"] is True
        assert served["api_key_configured"] is True
        # set_active resolves runtime presets, not just built-ins.
        assert set_active("served-lora").model == "rlmf"
    finally:
        set_active(None)
        clear_runtime_presets()

    assert "served-lora" not in [p["id"] for p in list_presets()]
