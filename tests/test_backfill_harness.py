"""Phase 2 — backfill harness: resumability + backfill-episode recording."""

from __future__ import annotations

from datetime import datetime, timezone

from trading_crew.agentic.backfill import (
    BackfillConfig,
    generate_cells,
    harness,
    job_id_for,
    load_state,
)
from trading_crew.agentic.memory import EpisodicMemory, OutcomeStatus


_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _mem(tmp_path):
    return EpisodicMemory(tmp_path / "memory" / "episodes.jsonl")


class _FakeProposal:
    rationale = "backfill test rationale that is reasonably long"
    horizon_days = 30

    def model_dump(self):
        return {"side": "BUY", "target_weight": 0.1, "horizon_days": 30, "conviction_score": 0.6}


def test_generate_cells_drops_unresolvable():
    cfg = BackfillConfig(
        tickers=["AAA"], start_date="2023-11-01", end_date="2024-06-01",
        spacing_days=30, horizon_days=90, only_resolvable=True,
    )
    cells = generate_cells(cfg, now=_NOW)
    # Only decision dates whose outcome (+90d) is before 2024-01-01 survive.
    for c in cells:
        as_of = datetime.strptime(c.as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        assert as_of.replace() and (as_of.timestamp() + 90 * 86400) < _NOW.timestamp()


def test_record_backfill_episode_stamps_source_and_version(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    mem = _mem(tmp_path)
    ep_id = harness._record_backfill_episode(mem, "AAA", _FakeProposal(), "2023-01-01T00:00:00+00:00")
    ep = mem.get(ep_id)
    assert ep is not None
    assert ep.source == "backfill"
    assert ep.outcome_status == OutcomeStatus.PENDING
    assert ep.workflow_version is not None
    # outcome_ts = decision + horizon_days (30d).
    assert ep.outcome_ts.startswith("2023-01-31")


def test_resume_skips_done_cells(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    cfg = BackfillConfig(
        tickers=["AAA"], start_date="2023-01-01", end_date="2023-03-02",
        spacing_days=30, horizon_days=30, resolve=False, cell_gap_seconds=0.0,
    )
    cells = generate_cells(cfg, now=_NOW)
    assert len(cells) == 3

    calls: list[str] = []

    def _fake_run(cell, config, memory):
        calls.append(cell.cell_id)
        return f"{cell.ticker}-{cell.as_of}"

    monkeypatch.setattr(harness, "_run_one_cell", _fake_run)

    # First run: cooperatively stop after the first cell completes.
    counter = {"n": 0}

    def _fake_run_stop(cell, config, memory):
        calls.append(cell.cell_id)
        counter["n"] += 1
        return f"{cell.ticker}-{cell.as_of}"

    monkeypatch.setattr(harness, "_run_one_cell", _fake_run_stop)
    s1 = harness.run_backfill(
        cfg, should_stop=lambda: counter["n"] >= 1, now=_NOW, memory=_mem(tmp_path),
    )
    assert s1["stopped"] is True
    assert len(calls) == 1  # only one cell ran before the stop took effect

    job_id = job_id_for(cfg)
    state = load_state(job_id)
    done_ids = [cid for cid, v in state.items() if v.get("status") == "done"]
    assert len(done_ids) == 1

    # Resume: no stop. The already-done cell must be skipped.
    calls.clear()
    monkeypatch.setattr(harness, "_run_one_cell", _fake_run)
    s2 = harness.run_backfill(cfg, should_stop=lambda: False, now=_NOW, memory=_mem(tmp_path))
    assert s2["stopped"] is False
    assert done_ids[0] not in calls  # the finished cell was not re-run
    assert len(calls) == 2  # the two remaining cells ran
    assert s2["done"] == 3
