"""Per-session web-search metrics + trace-id propagation.

Covers the observability plumbing added so backend health is visible per
batch/session (``trace_id``) without grepping logs:

* ``prediction_db.bump_search_stat`` / ``get_search_stats`` aggregation.
* ``record_search`` attributes to the active trace and is a no-op without one.
* the ``trace_id`` contextvar rides through ``asyncio.to_thread`` +
  ``contextvars.copy_context().run`` — the exact path CrewAI's
  ``Task.execute_async`` uses to run a parallel analyst — so a search made
  deep inside a spawned worker thread is still counted against its session.
* ``TraceLogFilter`` stamps records with the active trace id.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading

import pytest

from trading_crew.observability import trace as tr


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_bump_and_get_search_stats(isolated_cache):
    from web.backend import prediction_db as db

    for _ in range(3):
        db.bump_search_stat("sess-A", "searxng", "success")
    db.bump_search_stat("sess-A", "searxng", "empty")
    db.bump_search_stat("sess-A", "yahoo", "success", n=2)
    db.bump_search_stat("sess-A", "google", "throttled")
    # A different session must stay isolated.
    db.bump_search_stat("sess-B", "searxng", "success")

    stats = db.get_search_stats("sess-A")
    assert stats["total_calls"] == 7
    assert stats["backends"]["searxng"] == {
        "success": 3, "empty": 1, "throttled": 0, "error": 0, "total": 4,
    }
    assert stats["backends"]["yahoo"]["success"] == 2
    assert stats["backends"]["google"]["throttled"] == 1
    assert stats["totals"] == {"success": 5, "empty": 1, "throttled": 1, "error": 0}

    assert db.get_search_stats("sess-B")["total_calls"] == 1

    ids = {row["trace_id"] for row in db.list_search_trace_ids()}
    assert {"sess-A", "sess-B"} <= ids


def test_record_search_noop_without_trace(isolated_cache):
    from web.backend import prediction_db as db

    # No active trace → nothing recorded.
    tr.record_search("searxng", "success")
    assert db.get_search_stats("")["total_calls"] == 0

    with tr.trace_scope("sess-C"):
        tr.record_search("searxng", "success")
        tr.record_search("google", "empty")
    assert db.get_search_stats("sess-C")["total_calls"] == 2


def test_trace_propagates_through_copy_context_thread(isolated_cache):
    """Mirrors runner→CrewAI: to_thread copies context, then execute_async
    copies again before spawning the analyst's thread."""
    from web.backend import prediction_db as db

    async def main() -> None:
        with tr.trace_scope("sess-THREAD"):
            def analyst_thread() -> None:
                # search made deep inside the spawned worker
                tr.record_search("yahoo", "success")

            def kickoff_thread() -> None:
                ctx = contextvars.copy_context()  # what Task.execute_async does
                t = threading.Thread(target=ctx.run, args=(analyst_thread,))
                t.start()
                t.join()

            await asyncio.to_thread(kickoff_thread)

    asyncio.run(main())
    assert db.get_search_stats("sess-THREAD")["backends"]["yahoo"]["success"] == 1


def test_trace_log_filter_stamps_record():
    f = tr.TraceLogFilter()
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", (), None)

    token = tr.set_trace_id("sess-LOG")
    try:
        assert f.filter(record) is True
        assert record.trace_id == "sess-LOG"
    finally:
        tr.reset_trace_id(token)

    # Outside a trace the field still resolves (to "-") so "%(trace_id)s"
    # never blows up formatting.
    record2 = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", (), None)
    f.filter(record2)
    assert record2.trace_id == "-"
