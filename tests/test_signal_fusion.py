"""Phase 4 — signal fusion meta-layer + guardrails."""

from __future__ import annotations

from trading_crew.agentic.signals import (
    ExpectancySignal,
    FusionConfig,
    LLMSignal,
    PolicySignal,
    fuse_signals,
    pso_tuned_expectancy,
)
from trading_crew.agentic.swarm.crowd import CrowdSignal


def test_pso_tuned_expectancy_needs_min_proposals():
    assert pso_tuned_expectancy(["p1", "p2"], {}, min_proposals=5) is None
    assert pso_tuned_expectancy(None, {}) is None


def test_pso_tuned_expectancy_reads_best_metrics(monkeypatch):
    class _Metrics:
        expectancy = 0.031
        win_rate = 0.6
        n_periods = 8

    class _Res:
        best_metrics = _Metrics()

    monkeypatch.setattr(
        "trading_crew.agentic.swarm.pso.run_swarm_search",
        lambda **kw: _Res(),
    )
    sig = pso_tuned_expectancy(list(range(6)), {"X": object()})
    assert isinstance(sig, ExpectancySignal)
    assert sig.expectancy == 0.031
    assert sig.win_rate == 0.6
    assert sig.n_samples == 8


def test_llm_and_policy_agree_boosts_confidence():
    f = fuse_signals(
        LLMSignal(side="BUY", conviction=0.8),
        policy=PolicySignal(target_weight=0.1, confidence=0.7),
    )
    assert f.side == "BUY"
    assert f.target_weight > 0
    assert not any("disagree" in g for g in f.guardrails)
    assert f.confidence > 0.5


def test_disagreement_halves_confidence():
    agree = fuse_signals(
        LLMSignal(side="BUY", conviction=0.8),
        policy=PolicySignal(target_weight=0.1, confidence=0.8),
    )
    disagree = fuse_signals(
        LLMSignal(side="BUY", conviction=0.8),
        policy=PolicySignal(target_weight=-0.1, confidence=0.8),
    )
    assert any("disagree" in g for g in disagree.guardrails)
    assert disagree.confidence < agree.confidence


def test_require_llm_abstains_when_llm_holds():
    f = fuse_signals(
        LLMSignal(side="HOLD", conviction=0.0),
        policy=PolicySignal(target_weight=0.2, confidence=0.9),
    )
    assert f.side in ("HOLD", "ABSTAIN")
    assert f.target_weight == 0.0
    assert any("no directional view" in g for g in f.guardrails)


def test_no_expectancy_edge_dampens_size():
    cfg = FusionConfig()
    strong = fuse_signals(
        LLMSignal(side="BUY", conviction=0.9),
        expectancy=ExpectancySignal(expectancy=0.02, win_rate=0.6, n_samples=50),
        config=cfg,
    )
    edgeless = fuse_signals(
        LLMSignal(side="BUY", conviction=0.9),
        expectancy=ExpectancySignal(expectancy=-0.01, win_rate=0.4, n_samples=50),
        config=cfg,
    )
    assert abs(edgeless.target_weight) < abs(strong.target_weight)
    assert any("no positive expectancy" in g for g in edgeless.guardrails)


def test_crowd_fade_when_aligned_with_extreme_crowd():
    crowded_long = CrowdSignal(score=0.9, intensity=0.9, label="crowded long", n_components=4)
    f_long_into_crowd = fuse_signals(
        LLMSignal(side="BUY", conviction=0.9), crowd=crowded_long,
    )
    f_contrarian = fuse_signals(
        LLMSignal(side="SELL", conviction=0.9), crowd=crowded_long,
    )
    assert f_long_into_crowd.crowd_multiplier < 1.0
    assert any("fading size" in g for g in f_long_into_crowd.guardrails)
    # Contrarian to the crowd → no fade.
    assert f_contrarian.crowd_multiplier == 1.0


def test_low_confidence_stands_aside():
    # Weak LLM + disagreeing policy pushes fused confidence under the floor.
    f = fuse_signals(
        LLMSignal(side="BUY", conviction=0.2),
        policy=PolicySignal(target_weight=-0.2, confidence=0.2),
        config=FusionConfig(min_confidence=0.3),
    )
    assert f.side == "ABSTAIN"
    assert f.target_weight == 0.0


def test_target_weight_bounded_by_max():
    cfg = FusionConfig(max_position_weight=0.1)
    f = fuse_signals(
        LLMSignal(side="BUY", conviction=1.0),
        policy=PolicySignal(target_weight=0.5, confidence=1.0),
        config=cfg,
    )
    assert abs(f.target_weight) <= 0.1 + 1e-9
