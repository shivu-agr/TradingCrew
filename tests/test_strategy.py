"""Editable desk_strategy.yaml — load, validate, render, no silent defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_crew.agentic.strategy import (
    StrategyError,
    load_desk_strategy,
    packaged_strategy_path,
    parse_desk_strategy,
    render_prompt_block,
    revert_live_strategy,
    save_desk_strategy,
    strategy_path,
    strategy_source,
    strategy_to_raw,
)


def test_package_strategy_loads_and_has_required_clocks():
    s = load_desk_strategy(packaged_strategy_path())
    assert s.learning.calendar_days == 30
    assert s.learning.trading_days == 21
    assert s.thesis["medium"].calendar_days == 180
    assert s.thesis["long"].trading_days == 504
    assert s.objectives["short"].min_n >= 1
    assert s.knobs
    assert "equity" in s.asset_classes and "commodity" in s.asset_classes
    size = next(k for k in s.knobs if k.id == "size_scale_down")
    assert size.size_scale == 0.75
    assert all(hasattr(k, "enabled") for k in s.knobs)


def test_strategy_path_points_at_editable_yaml():
    p = strategy_path()
    assert p.name == "desk_strategy.yaml"
    assert p.is_file()


def test_missing_file_raises(tmp_path):
    with pytest.raises(StrategyError, match="not found"):
        load_desk_strategy(tmp_path / "missing.yaml")


def test_missing_required_key_raises():
    raw = yaml.safe_load(Path(strategy_path()).read_text())
    del raw["objectives"]
    with pytest.raises(StrategyError, match="objectives"):
        parse_desk_strategy(raw)


def test_render_includes_success_bars_and_does_not_collapse_thesis():
    text = render_prompt_block(book="equity")
    assert "30" in text
    assert "SUCCESS" in text
    assert "FAILURE" in text
    assert "MEDIUM" in text and "LONG" in text
    assert "Do not shrink" in text


def test_commodity_render_uses_curve_rules():
    text = render_prompt_block(book="commodity")
    assert "Curve first" in text or "curve" in text.lower()


def test_knob_missing_enabled_raises():
    raw = yaml.safe_load(packaged_strategy_path().read_text())
    del raw["knobs"][0]["enabled"]
    with pytest.raises(StrategyError, match="enabled"):
        parse_desk_strategy(raw)


def test_next_unused_skips_disabled():
    raw = strategy_to_raw(load_desk_strategy(packaged_strategy_path()))
    raw["knobs"][0]["enabled"] = False
    s = parse_desk_strategy(raw)
    nxt = s.next_unused_knob([])
    assert nxt is not None
    assert nxt.id != raw["knobs"][0]["id"]
    assert nxt.enabled


def test_save_blocked_when_strategy_path_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_STRATEGY_PATH", str(packaged_strategy_path()))
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    raw = strategy_to_raw(load_desk_strategy(packaged_strategy_path()))
    with pytest.raises(StrategyError, match="TRADINGCREW_STRATEGY_PATH"):
        save_desk_strategy(raw)


def test_save_and_revert_live_strategy(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGCREW_STRATEGY_PATH", raising=False)
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    raw = strategy_to_raw(load_desk_strategy(packaged_strategy_path()))
    raw["mandate"] = "UI-edited mandate for the cash book."
    raw["scientific_method"]["min_reviews_to_change"] = 15
    saved = save_desk_strategy(raw)
    assert saved.mandate.startswith("UI-edited")
    assert saved.min_reviews_to_change == 15
    assert strategy_source() == "live"
    assert load_desk_strategy().mandate.startswith("UI-edited")
    assert revert_live_strategy() is True
    assert strategy_source() == "packaged"
    assert load_desk_strategy().mandate == load_desk_strategy(packaged_strategy_path()).mandate
