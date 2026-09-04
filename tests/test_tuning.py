"""Persisted L3 sizing/gate tunes composed with Hermes size_scale."""

from __future__ import annotations

import pytest

from trading_crew.agentic import tuning
from trading_crew.agentic.risk import GateConfig, SizingConfig
from trading_crew.agentic.self_learn import save_journal


def test_load_returns_none_without_file(tmp_path):
    assert tuning.load_tuned_config("AAPL", cache_dir=str(tmp_path)) is None


def test_save_then_load_roundtrip(tmp_path):
    tuning.save_tuned_config(
        "aapl",
        sizing_config={"kelly_fraction": 0.5, "vol_target": 0.15, "max_position_weight": 0.30},
        gate_config={"max_leverage": 2.0, "max_position_weight": 0.30},
        meta={"rank_metric": 1.23, "n_proposals": 40},
        cache_dir=str(tmp_path),
    )
    loaded = tuning.load_tuned_config("AAPL", cache_dir=str(tmp_path))  # case-insensitive
    assert loaded is not None
    sizing, gate = loaded
    assert isinstance(sizing, SizingConfig) and isinstance(gate, GateConfig)
    assert sizing.kelly_fraction == 0.5
    assert sizing.vol_target == 0.15
    assert sizing.max_position_weight == 0.30
    assert gate.max_leverage == 2.0

    entry = tuning.load_tuned_entry("AAPL", cache_dir=str(tmp_path))
    assert entry["meta"]["n_proposals"] == 40


def test_unknown_kwargs_are_ignored(tmp_path):
    tuning.save_tuned_config(
        "MSFT",
        sizing_config={"kelly_fraction": 0.1, "bogus_param": 999},
        gate_config={"max_leverage": 1.0, "also_bogus": "x"},
        cache_dir=str(tmp_path),
    )
    loaded = tuning.load_tuned_config("MSFT", cache_dir=str(tmp_path))
    assert loaded is not None
    sizing, gate = loaded
    assert sizing.kelly_fraction == 0.1
    assert gate.max_leverage == 1.0


def test_per_ticker_isolation(tmp_path):
    tuning.save_tuned_config("AAA", sizing_config={"kelly_fraction": 0.2}, gate_config={}, cache_dir=str(tmp_path))
    tuning.save_tuned_config("BBB", sizing_config={"kelly_fraction": 0.6}, gate_config={}, cache_dir=str(tmp_path))
    a, _ = tuning.load_tuned_config("AAA", cache_dir=str(tmp_path))
    b, _ = tuning.load_tuned_config("BBB", cache_dir=str(tmp_path))
    assert a.kelly_fraction == 0.2
    assert b.kelly_fraction == 0.6
    # Saving BBB must not have clobbered AAA.
    assert tuning.load_tuned_entry("AAA", cache_dir=str(tmp_path)) is not None


def _open_journal(path, *, size_scale=None, knob="size_scale_down"):
    exp = {
        "knob": knob,
        "apply": "prompt",
        "status": "open",
        "hypothesis": "test",
    }
    if size_scale is not None:
        exp["size_scale"] = size_scale
    save_journal(
        {
            "experiment": exp,
            "history": [],
            "used_knobs": [knob],
            "last_score": None,
        },
        path,
    )


def test_resolve_none_without_tune_or_scale(tmp_path):
    journal = tmp_path / "journal.json"
    sizing, gate, meta = tuning.resolve_live_sizing(
        "AAPL", cache_dir=str(tmp_path), journal_file=journal
    )
    assert sizing is None and gate is None
    assert meta["l3_applied"] is False
    assert meta["hermes_size_scale"] is None


def test_resolve_open_experiment_without_size_scale_does_not_invent(tmp_path):
    journal = tmp_path / "journal.json"
    _open_journal(journal, knob="debate_rounds_2")
    sizing, gate, meta = tuning.resolve_live_sizing(
        "AAPL", cache_dir=str(tmp_path), journal_file=journal
    )
    assert sizing is None and gate is None
    assert meta["hermes_size_scale"] is None


def test_resolve_stale_size_scale_down_uses_yaml_knob(tmp_path):
    """Live journals opened before size_scale was stamped still use YAML."""
    journal = tmp_path / "journal.json"
    _open_journal(journal, knob="size_scale_down")
    sizing, gate, meta = tuning.resolve_live_sizing(
        "AAPL", cache_dir=str(tmp_path), journal_file=journal
    )
    assert meta["hermes_size_scale"] == 0.75
    assert sizing is not None
    assert sizing.kelly_fraction == pytest.approx(SizingConfig().kelly_fraction * 0.75)


def test_resolve_hermes_only_scales_defaults(tmp_path):
    journal = tmp_path / "journal.json"
    _open_journal(journal, size_scale=0.75)
    sizing, gate, meta = tuning.resolve_live_sizing(
        "AAPL", cache_dir=str(tmp_path), journal_file=journal
    )
    assert meta["l3_applied"] is False
    assert meta["hermes_size_scale"] == 0.75
    default = SizingConfig()
    assert sizing is not None and gate is not None
    assert sizing.kelly_fraction == pytest.approx(default.kelly_fraction * 0.75)
    assert sizing.max_position_weight == pytest.approx(default.max_position_weight * 0.75)
    assert sizing.vol_target == pytest.approx(default.vol_target)
    assert gate.max_position_weight == pytest.approx(0.20 * 0.75)


def test_resolve_l3_times_hermes_scale(tmp_path):
    journal = tmp_path / "journal.json"
    _open_journal(journal, size_scale=0.75)
    tuning.save_tuned_config(
        "AAPL",
        sizing_config={"kelly_fraction": 0.40, "vol_target": 0.12, "max_position_weight": 0.30},
        gate_config={"max_position_weight": 0.30},
        cache_dir=str(tmp_path),
    )
    sizing, gate, meta = tuning.resolve_live_sizing(
        "AAPL", cache_dir=str(tmp_path), journal_file=journal
    )
    assert meta["l3_applied"] is True
    assert meta["hermes_size_scale"] == 0.75
    assert sizing.kelly_fraction == pytest.approx(0.30)
    assert sizing.vol_target == pytest.approx(0.12)
    assert sizing.max_position_weight == pytest.approx(0.225)
    assert gate.max_position_weight == pytest.approx(0.225)


def test_hermes_size_scale_rejects_non_positive(tmp_path):
    journal = tmp_path / "journal.json"
    _open_journal(journal, size_scale=0)
    with pytest.raises(ValueError, match="size_scale"):
        tuning.hermes_size_scale(journal)
