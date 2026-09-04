"""Tests for the SEC EDGAR client (US primary-filing feed for get_report_insights).

EDGAR is the US analog of screener.in's dated ``#documents`` list: the official,
free, keyless submissions feed of 10-K / 10-Q reports and earnings 8-Ks.  These
tests stub the two JSON endpoints (``company_tickers.json`` and the per-company
``submissions`` feed) plus the filing-folder index so they run fully offline,
and verify:

* ticker -> CIK resolution (incl. class-share ``.``/``-`` spelling) and the
  ``is_edgar_ticker`` gate;
* the submissions feed is mapped to the right ``kind`` (10-K -> annual_report,
  10-Q -> quarterly_report, earnings 8-K [item 2.02] -> earnings) and generic
  8-Ks / other forms are dropped;
* exact-date PIT filtering drops filings filed after the cutoff;
* an earnings 8-K resolves to its Exhibit 99.1 press-release document.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_crew.data_sources import sec_edgar_source as se


_CIK_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc"},
}

# A trimmed submissions.recent payload with one of each relevant form + noise.
_SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "8-K", "10-K", "4"],
            "filingDate": ["2026-05-01", "2026-04-30", "2026-04-20",
                           "2025-10-31", "2025-10-15"],
            "accessionNumber": ["0000320193-26-000013", "0000320193-26-000011",
                                 "0001140361-26-015711", "0000320193-25-000079",
                                 "0000320193-25-000070"],
            "primaryDocument": ["aapl-20260328.htm", "aapl-20260430.htm",
                                 "ef20071035_8k.htm", "aapl-20250927.htm", "x.htm"],
            "items": ["", "2.02,9.01", "5.02", "", ""],
        }
    },
}

# Folder index for the earnings 8-K (acc 0000320193-26-000011).
_INDEX_JSON = {
    "directory": {
        "item": [
            {"name": "aapl-20260430.htm", "size": "37639"},   # thin cover (primary)
            {"name": "a8-kex991q2.htm", "size": "168815"},    # Ex-99.1 press release
            {"name": "aapl-20260430_g1.jpg", "size": "1264"},
        ]
    }
}


@pytest.fixture(autouse=True)
def _reset_and_stub(monkeypatch):
    """Reset caches and stub the network JSON layer for every test."""
    se.clear_cache()
    monkeypatch.setenv("TRADINGCREW_EDGAR_ENABLED", "1")

    def _fake_get_json(url: str):
        if url == se._CIK_MAP_URL:
            return _CIK_MAP
        if "/submissions/CIK" in url:
            return _SUBMISSIONS
        if url.endswith("/index.json"):
            return _INDEX_JSON
        raise AssertionError(f"unexpected URL fetched: {url}")

    monkeypatch.setattr(se, "_get_json", _fake_get_json)
    yield
    se.clear_cache()


def test_lookup_cik_and_gate():
    assert se.lookup_cik("AAPL") == (320193, "Apple Inc.")
    # lookup is case-insensitive; unknown ticker -> None.
    assert se.lookup_cik("aapl") == (320193, "Apple Inc.")
    assert se.lookup_cik("NOPE") is None
    assert se.is_edgar_ticker("AAPL") is True
    assert se.is_edgar_ticker("NOPE") is False


def test_class_share_spelling_variant():
    """yfinance 'BRK.B' must resolve to EDGAR's 'BRK-B'."""
    assert se.lookup_cik("BRK.B") == (1067983, "Berkshire Hathaway Inc")


def test_disabled_flag_gate(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_EDGAR_ENABLED", "0")
    assert se.is_edgar_ticker("AAPL") is False


def test_fetch_filings_maps_kinds_and_drops_noise():
    company = se.fetch_filings("AAPL")
    assert company.name == "Apple Inc." and company.cik == 320193
    kinds = [(f.form, f.kind) for f in company.filings]
    # 10-Q -> quarterly_report, earnings 8-K -> earnings, 10-K -> annual_report.
    # The item-5.02 8-K and the Form 4 are dropped.
    assert kinds == [
        ("10-Q", "quarterly_report"),
        ("8-K", "earnings"),
        ("10-K", "annual_report"),
    ]
    # Primary URL is built from the accession folder + primary document.
    q = company.filings[0]
    assert q.primary_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019326000013/aapl-20260328.htm"
    )


def test_pit_filter_drops_post_cutoff():
    from trading_crew.data_sources import pit

    company = se.fetch_filings("AAPL")
    with pit.as_of_scope(datetime(2025, 12, 31, tzinfo=timezone.utc), mode="backfill"):
        filtered = se.pit_filter_documents(company)
    # Only the 2025-10-31 10-K survives a 2025-12-31 cutoff.
    assert [f.form for f in filtered.filings] == ["10-K"]


def test_pit_filter_noop_in_live_mode():
    company = se.fetch_filings("AAPL")
    assert len(se.pit_filter_documents(company).filings) == len(company.filings)


def test_resolve_text_url_picks_ex99_for_earnings():
    company = se.fetch_filings("AAPL")
    docs = se.to_document_links(company)
    earnings = next(d for d in docs if d.kind == "earnings")
    resolved = se.resolve_text_url(earnings.url)
    assert resolved.endswith("/a8-kex991q2.htm"), resolved
    # Non-earnings docs are returned unchanged (no folder fetch).
    tenk = next(d for d in docs if d.kind == "annual_report")
    assert se.resolve_text_url(tenk.url) == tenk.url


def test_to_document_links_shape():
    company = se.fetch_filings("AAPL")
    docs = se.to_document_links(company)
    assert all(d.url.startswith("https://www.sec.gov/Archives/") for d in docs)
    assert {d.kind for d in docs} == {"quarterly_report", "earnings", "annual_report"}
    assert any("Earnings release" in d.title for d in docs)
