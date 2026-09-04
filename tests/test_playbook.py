"""Topology playbook credit/debit from resolved WIN/LOSS."""

from __future__ import annotations

from trading_crew.agentic.graph.playbook import (
    playbook_key,
    preferred_action,
    record_outcome,
)
from trading_crew.agentic.outcome import OutcomeLabel


def test_record_win_loss_and_default_action(tmp_path):
    path = tmp_path / "pb.json"
    key = playbook_key(market="IN", has_filings=True)
    record_outcome(key, OutcomeLabel.WIN, path=path)
    row = record_outcome(key, OutcomeLabel.LOSS, path=path)
    assert row["wins"] == 1
    assert row["losses"] == 1
    assert row["win_rate"] == 0.5
    # No promoted_action yet — live DAG stays default.
    assert preferred_action(key, path=path) == "default"


def test_preferred_action_honours_promoted_field(tmp_path):
    from trading_crew.agentic.graph.playbook import load_playbook, save_playbook

    path = tmp_path / "pb.json"
    key = playbook_key(market="US")
    record_outcome(key, OutcomeLabel.WIN, path=path)
    data = load_playbook(path)
    data["keys"][key]["promoted_action"] = "skip_reports"
    save_playbook(data, path)
    assert preferred_action(key, path=path) == "skip_reports"
