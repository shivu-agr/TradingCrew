"""Phase 1 — point-in-time context + leakage guard + snapshot store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_crew.data_sources import pit


def test_live_mode_is_noop():
    # No context set → live: every guard passes, nothing is filtered.
    assert pit.cutoff() is None
    assert pit.is_backfill() is False
    assert pit.is_before_cutoff(None) is True
    assert pit.is_before_cutoff("2099-01-01") is True
    pit.assert_before_cutoff("2099-01-01", what="x")  # must not raise


def test_as_of_scope_sets_and_restores_context():
    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill") as ctx:
        assert ctx.is_backfill
        assert pit.cutoff() == as_of
        assert pit.is_backfill() is True
    # Restored to live outside the scope.
    assert pit.cutoff() is None
    assert pit.is_backfill() is False


@pytest.mark.parametrize(
    "value,expected_year",
    [
        ("2024-06-24", 2024),
        ("2024-06-24T09:30:00Z", 2024),
        ("06/24/2024, 07:00 AM, +0000 UTC", 2024),
        ("06/24/2024", 2024),
    ],
)
def test_parse_publish_date_formats(value, expected_year):
    dt = pit.parse_publish_date(value)
    assert dt is not None and dt.year == expected_year


def test_parse_publish_date_rejects_relative_and_garbage():
    assert pit.parse_publish_date("15h ago") is None
    assert pit.parse_publish_date("") is None
    assert pit.parse_publish_date(None) is None


def test_backfill_strict_blocks_future_and_unknown():
    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        assert pit.is_before_cutoff("2024-08-01") is True   # before cutoff → ok
        assert pit.is_before_cutoff("2024-12-01") is False  # after cutoff → blocked
        assert pit.is_before_cutoff(None) is False          # unknown, strict → blocked
        with pytest.raises(pit.LeakageError):
            pit.assert_before_cutoff("2024-12-01", what="future bar")


def test_backfill_nonstrict_allows_unknown():
    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=False):
        assert pit.is_before_cutoff(None) is True
        assert pit.is_before_cutoff("2024-12-01") is False  # still blocks known-future


def test_filter_before_cutoff_drops_future_rows():
    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    rows = [
        {"title": "old", "published": "2024-06-01"},
        {"title": "future", "published": "2025-01-01"},
        {"title": "undated", "published": None},
    ]
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        kept = pit.filter_before_cutoff(rows, key="published")
    assert [r["title"] for r in kept] == ["old"]


def test_snapshot_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    assert pit.load_snapshot("screener_html", "RELIANCE") is None
    pit.save_snapshot("screener_html", "RELIANCE", {"html": "<html>x</html>", "mode": "consolidated"})
    loaded = pit.load_snapshot("screener_html", "RELIANCE")
    assert loaded is not None and loaded["html"] == "<html>x</html>"
