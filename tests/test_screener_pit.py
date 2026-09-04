"""Phase 1 — screener.in point-in-time document filtering."""

from __future__ import annotations

from datetime import datetime, timezone

from trading_crew.data_sources import pit, screener_source
from trading_crew.data_sources.screener_source import DocumentLink


def test_parse_document_date_absolute():
    dt = screener_source.parse_document_date("Board Meeting Outcome 24 Jun 2024")
    assert dt is not None and (dt.year, dt.month, dt.day) == (2024, 6, 24)


def test_parse_document_date_relative_is_now():
    dt = screener_source.parse_document_date("Announcement 15h")
    assert dt is not None
    # "15h" anchors at now → clearly after any historical cutoff.
    assert dt.year >= 2024


def test_parse_document_date_ambiguous_returns_none():
    # "24 Jun" with no year is ambiguous → None (fail-closed downstream).
    assert screener_source.parse_document_date("Filing 24 Jun") is None


def test_pit_filter_documents_live_mode_passthrough():
    docs = [DocumentLink(title="X 24 Jun 2024", url="https://x/1")]
    # No PIT context → unchanged.
    assert screener_source.pit_filter_documents(docs) == docs


def test_classify_document_categories():
    c = screener_source.classify_document
    assert c("Q2 FY24 Earnings Concall Transcript 24 Jul 2023", "https://x/concall") == "concall"
    assert c("Annual Report 2023", "https://x/annual-reports/2023") == "annual_report"
    assert c("CRISIL reaffirms AA rating 12 Aug 2023", "https://x/rating") == "credit_rating"
    assert c("Board Meeting Outcome 24 Jun 2023", "https://x/ann") == "announcement"


def test_render_documents_report_groups_by_kind():
    from trading_crew.data_sources.screener_source import CompanyPage, TopRatios

    page = CompanyPage(
        symbol="AAA", source_url="u", mode="consolidated", long_name="AAA Ltd",
        about="", top=TopRatios(),
        documents=[
            DocumentLink(title="Board Meeting 01 Feb 2023", url="https://x/a", kind="announcement"),
            DocumentLink(title="Annual Report 2022", url="https://x/ar", kind="annual_report"),
            DocumentLink(title="Q3 Concall Transcript 05 Jan 2023", url="https://x/cc", kind="concall"),
        ],
    )
    md = screener_source.render_documents_report(page)
    assert "Annual reports" in md
    assert "Concall transcripts" in md
    assert md.count("[source:") == 3


def test_pit_filter_documents_backfill_drops_future_and_undated():
    docs = [
        DocumentLink(title="Past filing 01 Feb 2024", url="https://x/past"),
        DocumentLink(title="Future filing 01 Dec 2024", url="https://x/future"),
        DocumentLink(title="Recent filing 3d", url="https://x/recent"),  # now → future vs cutoff
        DocumentLink(title="Ambiguous 24 Jun", url="https://x/amb"),     # undated → dropped strict
    ]
    as_of = datetime(2024, 9, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        kept = screener_source.pit_filter_documents(docs)
    assert [d.url for d in kept] == ["https://x/past"]
