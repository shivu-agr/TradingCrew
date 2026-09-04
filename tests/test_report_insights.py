"""Tests for the ``get_report_insights`` tool (Reports Analyst desk).

Background
----------
Nothing in the workflow read the company's OWN primary filings — concall /
earnings-call transcripts, investor presentations, annual-report MD&A and
credit-rating rationales — where management states explicit forward TARGETS
and where a prior target can be checked against what was DELIVERED.  That is
exactly the "target / target-vs-actual" signal the user found missing from
predictions (they ran NTNX and only got a generic sell justification).

``get_report_insights`` fills the gap for Indian (NSE/BSE) names via
screener.in's dated ``#documents`` feed: it selects the guidance-rich
filings, crawls the (large) PDFs, and has the workflow LLM summarise them
BEFORE the digest reaches the agent — so the analyst never ingests a 200-page
PDF.  These tests stub the network + LLM so they run offline and verify:

* the tool is registered in ``ALL_TOOLS`` and wired to the Reports desk;
* non-Indian tickers get a clear not-available note (no fabricated targets);
* the Indian happy path crawls the selected filings, LLM-summarises them, and
  surfaces the digest + a "Filings read" provenance list;
* concall / annual-report filings are prioritised over generic announcements;
* an empty crawl degrades to a filings listing instead of crashing.
"""

from __future__ import annotations

import pytest

from trading_crew.data_sources.screener_source import CompanyPage, DocumentLink, TopRatios


@pytest.fixture(autouse=True)
def _no_tool_cache(monkeypatch):
    """Disable the per-process TTL cache so each test sees a fresh call
    rather than a memoised result from a sibling test.

    Also neutralise the deterministic financial-trend block by default so
    filing-path tests stay offline (it reads yfinance/screener quarters);
    the dedicated computed-signal test re-enables it explicitly."""
    monkeypatch.setenv("TRADINGCREW_TOOL_CACHE_TTL_SECONDS", "0")
    monkeypatch.setenv("TRADINGCREW_REPORT_INSIGHTS_CACHE", "0")
    from trading_crew import tools

    monkeypatch.setattr(tools, "_computed_trend_block", lambda *a, **k: "", raising=True)


def _page(documents):
    return CompanyPage(
        symbol="TCS",
        source_url="https://www.screener.in/company/TCS/consolidated/",
        mode="consolidated",
        long_name="Tata Consultancy Services Ltd",
        about="",
        top=TopRatios(),
        documents=documents,
    )


def test_tool_is_registered_and_wired_to_reports_desk():
    from trading_crew.tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS

    assert "get_report_insights" in ALL_TOOLS
    assert "get_report_insights" in DEFAULT_AGENT_TOOLS["reports_analyst"]


def test_name_in_neither_feed_returns_not_available_note(monkeypatch):
    """A name in neither feed (not Indian AND not in EDGAR — e.g. a London
    listing) must say so plainly — never fabricate management guidance."""
    from trading_crew import tools

    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: False)
    monkeypatch.setattr(tools.sec_edgar_source, "is_edgar_ticker", lambda t: False)

    out = tools.get_report_insights.func("BP.L")  # type: ignore[attr-defined]

    assert "in neither feed" in out
    assert "get_forward_estimates" in out


def test_us_edgar_happy_path(monkeypatch):
    """US path: EDGAR filings → crawl → LLM-summarise → digest tagged SEC EDGAR,
    using EDGAR's own company name (no yfinance round-trip)."""
    from trading_crew import tools
    from trading_crew.data_sources import sec_edgar_source as se

    docs = [
        DocumentLink(title="Earnings release (8-K) — filed 2026-01-30",
                     url="https://sec/ex99.htm", kind="earnings"),
        DocumentLink(title="Annual report (10-K) — filed 2025-10-31",
                     url="https://sec/10k.htm", kind="annual_report"),
    ]
    company = se.EdgarCompany(ticker="AAPL", cik=320193, name="Apple Inc.", filings=[])
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: False)
    monkeypatch.setattr(tools.sec_edgar_source, "is_edgar_ticker", lambda t: True)
    monkeypatch.setattr(tools.sec_edgar_source, "fetch_filings", lambda t, **k: company)
    monkeypatch.setattr(tools.sec_edgar_source, "pit_filter_documents", lambda c: c)
    monkeypatch.setattr(tools.sec_edgar_source, "to_document_links", lambda c: docs)
    monkeypatch.setattr(tools.sec_edgar_source, "resolve_text_url", lambda u: u)
    # EDGAR uses its OWN SEC-compliant fetcher (not web_research.crawl).
    monkeypatch.setattr(
        tools.sec_edgar_source, "fetch_document_text",
        lambda url, **kw: (
            f"RAW {url}: Q1 revenue $124B; guides Q2 revenue up mid-single digits. "
            + "Services set an all-time record and gross margin expanded. " * 12
        ),
    )
    monkeypatch.setattr(
        tools, "_summarize_report_docs",
        lambda name, t, crawled: (
            "1. GUIDANCE: Q2 revenue up mid-single digits "
            "[source: https://sec/ex99.htm]"
        ),
    )

    out = tools.get_report_insights.func("AAPL")  # type: ignore[attr-defined]

    assert "Management guidance & filings" in out
    assert "Apple Inc." in out
    assert "SEC EDGAR" in out
    assert "mid-single digits" in out
    assert "https://sec/ex99.htm" in out


def test_us_earnings_8k_crawls_resolved_press_release(monkeypatch):
    """The crawl must hit the URL returned by resolve_text_url — i.e. the
    earnings 8-K's Exhibit 99.1 press release, not the thin cover doc."""
    from trading_crew import tools
    from trading_crew.data_sources import sec_edgar_source as se

    docs = [DocumentLink(title="Earnings release (8-K)",
                         url="https://sec/cover-8k.htm", kind="earnings")]
    company = se.EdgarCompany(ticker="MSFT", cik=789019, name="Microsoft Corp", filings=[])
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_DOCS", "1")
    crawled: list[str] = []
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: False)
    monkeypatch.setattr(tools.sec_edgar_source, "is_edgar_ticker", lambda t: True)
    monkeypatch.setattr(tools.sec_edgar_source, "fetch_filings", lambda t, **k: company)
    monkeypatch.setattr(tools.sec_edgar_source, "pit_filter_documents", lambda c: c)
    monkeypatch.setattr(tools.sec_edgar_source, "to_document_links", lambda c: docs)
    monkeypatch.setattr(
        tools.sec_edgar_source, "resolve_text_url",
        lambda u: "https://sec/ex-99_1.htm" if "cover" in u else u,
    )

    def _fetch(url, **kw):
        crawled.append(url)
        return "raw"

    monkeypatch.setattr(tools.sec_edgar_source, "fetch_document_text", _fetch)
    monkeypatch.setattr(tools, "_summarize_report_docs", lambda *a, **k: "digest")

    tools.get_report_insights.func("MSFT")  # type: ignore[attr-defined]

    assert crawled == ["https://sec/ex-99_1.htm"], (
        f"earnings 8-K should be crawled at its resolved Ex-99.1 url; got {crawled}"
    )


def test_indian_happy_path_crawls_and_summarises(monkeypatch):
    """The Indian path: fetch filings → crawl PDFs → LLM-summarise → digest
    with a 'Filings read' provenance list."""
    from trading_crew import tools

    docs = [
        DocumentLink(title="Concall transcript Q2FY24", url="https://s3/concall.pdf", kind="concall"),
        DocumentLink(title="Annual Report FY23", url="https://s3/ar.pdf", kind="annual_report"),
    ]
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile",
        lambda t: type("P", (), {"long_name": "Tata Consultancy Services Ltd"})(),
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))

    from trading_crew.data_sources import web_research

    monkeypatch.setattr(
        web_research, "crawl",
        lambda url, **kw: (
            f"RAW TEXT for {url}: management targets 15% revenue growth. "
            + "Management reiterated the outlook and demand commentary. " * 12
        ),
    )
    # Stub the summariser LLM so the test is offline + deterministic.
    monkeypatch.setattr(
        tools, "_summarize_report_docs",
        lambda name, ticker, crawled: (
            "1. GUIDANCE: 15% revenue growth guided for FY24 "
            "[source: https://s3/concall.pdf]"
        ),
    )

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert "Management guidance & filings" in out
    assert "15% revenue growth" in out
    # Provenance list of what was actually read + LLM-summarised.
    assert "Filings read" in out
    assert "https://s3/concall.pdf" in out
    assert "https://s3/ar.pdf" in out


def test_prioritises_concall_over_announcement(monkeypatch):
    """With max_docs small, guidance-rich concalls/annual reports must be
    selected ahead of generic announcements."""
    from trading_crew import tools

    docs = [
        DocumentLink(title="Board meeting outcome", url="https://s3/ann.pdf", kind="announcement"),
        DocumentLink(title="Concall transcript", url="https://s3/concall.pdf", kind="concall"),
    ]
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_DOCS", "1")
    crawled_urls: list[str] = []
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile",
        lambda t: type("P", (), {"long_name": "TCS"})(),
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))

    from trading_crew.data_sources import web_research

    def _crawl(url, **kw):
        crawled_urls.append(url)
        return "raw"

    monkeypatch.setattr(web_research, "crawl", _crawl)
    monkeypatch.setattr(tools, "_summarize_report_docs", lambda *a, **k: "digest")

    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert crawled_urls == ["https://s3/concall.pdf"], (
        "The concall (guidance-rich) must be picked before the generic "
        f"announcement; crawled {crawled_urls}"
    )


def test_empty_crawl_degrades_to_listing(monkeypatch):
    """If every PDF crawl returns nothing (e.g. no Firecrawl key / gated
    links), the tool must list the filings on record instead of crashing or
    fabricating a summary."""
    from trading_crew import tools

    docs = [DocumentLink(title="Concall Q1", url="https://s3/c.pdf", kind="concall")]
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "TCS"})()
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))

    from trading_crew.data_sources import web_research

    monkeypatch.setattr(web_research, "crawl", lambda url, **kw: None)

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert "Could not extract text" in out
    assert "https://s3/c.pdf" in out


def test_big_pdf_is_chunked_not_truncated(monkeypatch):
    """A PDF larger than the chunk size must be summarised chunk-by-chunk
    (map) then consolidated (reduce) — NOT truncated to the head — so the tail
    (analyst Q&A / late guidance) is read.  We assert one map call per chunk
    plus one final reduce call, and that a fact only present in the LAST chunk
    survives into the digest."""
    from trading_crew import tools
    from trading_crew.data_sources import web_research

    # Force multi-chunk map-reduce: the doc must exceed the raw-passthrough
    # threshold (docs below it feed reduce whole, skipping map) AND be several
    # chunks long at the configured chunk size.
    monkeypatch.setenv("TRADINGCREW_REPORT_DOC_CHARS", "4000")
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_CHUNKS", "8")

    # 3 chunks of 4000 chars; the guidance figure lives ONLY in the tail so a
    # head-only truncation would miss it entirely.
    tail = "TAIL-GUIDANCE: management targets 22% EBITDA margin in FY25."
    content = ("f" * 11000) + tail  # len 11059 -> 3 chunks at size 4000

    calls: list[str] = []

    class _FakeLLM:
        def call(self, messages):
            prompt = messages[0]["content"]
            calls.append(prompt)
            # MAP calls quote the EXCERPT; echo any tail-guidance found so the
            # reduce has it.  REDUCE call gets the SOURCES block.
            if "EXCERPT:" in prompt and "TAIL-GUIDANCE" in prompt:
                return "- targets 22% EBITDA margin in FY25 [source: https://s3/big.pdf]"
            if "EXCERPT:" in prompt:
                return "NONE"
            return "1. GUIDANCE: 22% EBITDA margin FY25 [source: https://s3/big.pdf]"

    monkeypatch.setattr("trading_crew._common.get_report_process_llm", lambda **kw: _FakeLLM())
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "Big Co"})()
    )
    docs = [DocumentLink(title="Concall Q4", url="https://s3/big.pdf", kind="concall")]
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))
    monkeypatch.setattr(web_research, "crawl", lambda url, **kw: content)

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    map_calls = [c for c in calls if "EXCERPT:" in c]
    reduce_calls = [c for c in calls if "EXCERPT:" not in c]
    assert len(map_calls) == 3, f"expected 3 map (chunk) calls, got {len(map_calls)}"
    assert len(reduce_calls) == 1, "expected exactly one reduce consolidation call"
    # The tail-only guidance made it through map-reduce into the final digest.
    assert "22% EBITDA margin" in out


def test_announcement_pdf_is_mapped_in_full(monkeypatch):
    """Announcement / credit PDFs used to be chunk-capped. Every chunk maps."""
    from trading_crew import tools
    from trading_crew.data_sources import web_research

    monkeypatch.setenv("TRADINGCREW_REPORT_DOC_CHARS", "3000")
    content = ("x" * 14000) + "TAIL-ONLY: order book 45k crore."
    calls: list[str] = []

    class _FakeLLM:
        def call(self, messages):
            prompt = messages[0]["content"]
            calls.append(prompt)
            if "EXCERPT:" in prompt and "TAIL-ONLY" in prompt:
                return "- order book 45k crore"
            if "EXCERPT:" in prompt:
                return "NONE"
            return "1. GUIDANCE: order book 45k crore"

    monkeypatch.setattr("trading_crew._common.get_report_process_llm", lambda **kw: _FakeLLM())
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "Co"})()
    )
    docs = [DocumentLink(title="Announcement", url="https://s3/a.pdf", kind="announcement")]
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))
    monkeypatch.setattr(web_research, "crawl", lambda url, **kw: content)

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    map_calls = [c for c in calls if "EXCERPT:" in c]
    assert len(map_calls) == 5, f"expected 5 map calls for full announcement, got {len(map_calls)}"
    assert "45k crore" in out


def test_full_document_fetched_not_head_sliced(monkeypatch):
    """The crawler must be asked for the whole PDF (no char / table-page cap)."""
    from trading_crew import tools
    from trading_crew.data_sources import web_research

    seen: list[dict] = []

    def _crawl(url, **kw):
        seen.append(kw)
        return "raw"

    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "Co"})()
    )
    docs = [DocumentLink(title="AR", url="https://s3/ar.pdf", kind="annual_report")]
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))
    monkeypatch.setattr(web_research, "crawl", _crawl)
    monkeypatch.setattr(tools, "_summarize_report_docs", lambda *a, **k: "digest")

    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert seen, "expected a crawl"
    assert seen[0].get("crawler") == "http"
    assert seen[0].get("max_chars") is None
    assert seen[0].get("extract_tables_pages") is None


def test_no_documents_returns_note(monkeypatch):
    from trading_crew import tools

    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "TCS"})()
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page([]))

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert "No dated primary filings" in out


def test_select_dedupes_by_url_and_diversifies(monkeypatch):
    """A screener feed lists the SAME concall under several dated titles; the
    selector must de-dupe by URL AND spread the budget across kinds so the
    annual report / credit rating aren't starved by a pile of concalls."""
    from trading_crew import tools

    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_DOCS", "5")
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_PER_KIND", "2")
    docs = [
        DocumentLink(title="Concall Q4 · 24 Jun", url="https://x/c1.pdf", kind="concall"),
        DocumentLink(title="Concall Q4 (relative-date dup)", url="https://x/c1.pdf", kind="concall"),
        DocumentLink(title="Concall Q3", url="https://x/c2.pdf", kind="concall"),
        DocumentLink(title="Concall Q2", url="https://x/c3.pdf", kind="concall"),
        DocumentLink(title="Annual Report FY24", url="https://x/ar.pdf", kind="annual_report"),
        DocumentLink(title="CRISIL rating rationale", url="https://x/cr.pdf", kind="credit_rating"),
    ]
    sel = tools._select_report_documents(docs, 5)
    urls = [d.url for d in sel]

    assert len(urls) == len(set(urls)), f"URLs must be deduped; got {urls}"
    assert urls.count("https://x/c1.pdf") == 1, "the duplicate concall URL must appear once"
    # Diversification: the annual report + credit rating are reached even though
    # four concalls precede them in the feed.
    assert "https://x/ar.pdf" in urls
    assert "https://x/cr.pdf" in urls
    # Per-kind cap (2) means concalls don't monopolise pass 1.
    assert [d.kind for d in sel[:2]].count("concall") == 2


def test_select_keeps_latest_annual_report_only(monkeypatch):
    """Newest annual report is enough; leftover MAX_DOCS slots must not
    back-fill older ARs. Concalls still keep multiple vintages."""
    from trading_crew import tools

    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_DOCS", "8")
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_PER_KIND", "3")
    monkeypatch.delenv("TRADINGCREW_REPORT_MAX_ANNUAL_REPORTS", raising=False)
    docs = [
        DocumentLink(title="Concall Q4", url="https://x/c4.pdf", kind="concall"),
        DocumentLink(title="Concall Q3", url="https://x/c3.pdf", kind="concall"),
        DocumentLink(title="Concall Q2", url="https://x/c2.pdf", kind="concall"),
        DocumentLink(title="Annual Report 2025", url="https://x/ar25.pdf", kind="annual_report"),
        DocumentLink(title="Annual Report 2024", url="https://x/ar24.pdf", kind="annual_report"),
        DocumentLink(title="Annual Report 2023", url="https://x/ar23.pdf", kind="annual_report"),
        DocumentLink(title="CRISIL", url="https://x/cr.pdf", kind="credit_rating"),
    ]
    sel = tools._select_report_documents(docs, 8)
    ar_urls = [d.url for d in sel if d.kind == "annual_report"]
    assert ar_urls == ["https://x/ar25.pdf"]
    assert [d.kind for d in sel].count("concall") == 3
    assert "https://x/cr.pdf" in [d.url for d in sel]


def test_report_doc_chars_from_context_tokens(monkeypatch):
    from trading_crew import tools

    monkeypatch.delenv("TRADINGCREW_REPORT_DOC_CHARS", raising=False)
    monkeypatch.setenv("REPORT_PROCESS_CONTEXT_TOKENS", "128000")
    reserve_128 = min(24000, max(4000, 128000 // 6))
    assert tools._report_doc_chars() == (128000 - reserve_128) * 4

    monkeypatch.setenv("REPORT_PROCESS_CONTEXT_TOKENS", "256000")
    reserve_256 = min(24000, max(4000, 256000 // 6))
    assert tools._report_doc_chars() == (256000 - reserve_256) * 4

    monkeypatch.setenv("TRADINGCREW_REPORT_DOC_CHARS", "4000")
    assert tools._report_doc_chars() == 4000


def test_map_prompt_keeps_broader_briefing():
    from trading_crew import tools

    text = tools._REPORT_MAP_PROMPT
    for needle in (
        "order book",
        "MD&A",
        "material risks",
        "reported actuals",
        "cover letter",
    ):
        assert needle in text, f"map prompt must keep {needle!r}"
    assert "[source: {url}]" in text
    assert "four sections" not in tools._REPORT_INSIGHT_PROMPT
    assert "ORDER BOOK" in tools._REPORT_INSIGHT_PROMPT
    assert "8000 chars" in tools._REPORT_INSIGHT_PROMPT
    assert "exact URL" in tools._REPORT_INSIGHT_PROMPT


def test_retag_stamps_filing_url_on_untagged_briefing_numbers():
    """Reduce often emits a prose Source column instead of [source: url].
    Those valid figures must still leave the tool tagged."""
    from trading_crew import tools

    body = (
        "SIGNAL | DELIVERY_TRACK_RECORD | INSUFFICIENT_DATA | conf=0.00 | "
        "No prior guide.\n"
        "| PAT | **187.82 cr** | Q1 FY26 call |\n"
        "FY25 turnover **4527.84 cr** | Annual report |\n"
        "Steel 5% rise would cut profit (CRISIL rating note)\n"
        "AAA reaffirmed [source: https://www.crisil.com/note]\n"
        "Order book ~21000 cr with no publisher label.\n"
    )
    filings = [
        {"kind": "concall", "title": "Transcript",
         "url": "https://www.bseindia.com/q1.pdf"},
        {"kind": "annual_report", "title": "Annual Report 2025",
         "url": "https://www.bseindia.com/ar.pdf"},
        {"kind": "credit_rating", "title": "Rating update",
         "url": "https://www.crisil.com/note"},
    ]
    out = tools._render_report_insights(
        "Cochin Shipyard", "COCHINSHIP.NS", body, filings, "",
        "Report insights", "catalog-hit",
    )
    insufficient = [
        ln for ln in out.splitlines() if "INSUFFICIENT_DATA" in ln
    ][0]
    assert "[source:" not in insufficient
    pat = [ln for ln in out.splitlines() if "187.82 cr" in ln][0]
    assert "https://www.bseindia.com/q1.pdf" in pat
    annual = [ln for ln in out.splitlines() if "4527.84 cr" in ln][0]
    assert "https://www.bseindia.com/ar.pdf" in annual
    aaa = [ln for ln in out.splitlines() if "AAA reaffirmed" in ln][0]
    assert aaa.count("[source:") == 1
    steel = [ln for ln in out.splitlines() if "Steel 5%" in ln][0]
    assert "https://www.crisil.com/note" in steel
    assert "[source: get_report_insights · COCHINSHIP.NS]" in out
    again = tools._retag_numeric_claims(
        tools._retag_numeric_claims(
            body, default_source="get_report_insights · COCHINSHIP.NS",
            citations=filings,
        ),
        default_source="get_report_insights · COCHINSHIP.NS",
        citations=filings,
    )
    assert again == tools._retag_numeric_claims(
        body, default_source="get_report_insights · COCHINSHIP.NS",
        citations=filings,
    )


def test_chunk_report_text_keeps_every_chunk():
    """Map-reduce must not drop the tail of a long filing."""
    from trading_crew import tools

    text = "a" * 10000
    chunks = tools._chunk_report_text(text, 4000)
    assert len(chunks) == 3
    assert "".join(chunks) == text


def test_computed_trend_block_appended_to_digest(monkeypatch):
    """The deterministic financial-trend card is appended under the LLM's
    filing signals so both feed one ReportSignals set downstream."""
    from trading_crew import tools

    docs = [DocumentLink(title="Concall Q1", url="https://s3/c.pdf", kind="concall")]
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile", lambda t: type("P", (), {"long_name": "TCS"})()
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))

    from trading_crew.data_sources import web_research

    monkeypatch.setattr(
        web_research, "crawl",
        lambda url, **kw: "Concall transcript: revenue grew. " * 40,
    )
    monkeypatch.setattr(
        tools, "_summarize_report_docs",
        lambda name, t, crawled: "## REPORT TREND SIGNALS\nSIGNAL | OUTLOOK_TONE | BULLISH | conf=0.7 | strong",
    )
    # Re-enable the computed block (the autouse fixture stubs it off) with a
    # deterministic payload so we don't touch yfinance/screener quarters.
    monkeypatch.setattr(
        tools, "_computed_trend_block",
        lambda ticker, quarters: (
            "## COMPUTED FINANCIAL TREND SIGNALS\n"
            "SIGNAL | REVENUE_TREND | ACCELERATING | conf=0.80 | YoY +18% "
            "[source: screener.in #quarters (computed)]"
        ),
    )

    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert "COMPUTED FINANCIAL TREND SIGNALS" in out
    assert "REVENUE_TREND | ACCELERATING" in out
    assert "OUTLOOK_TONE | BULLISH" in out


def _indian_docs_harness(monkeypatch, docs, crawl_fn, summarize_fn=None):
    from trading_crew import tools
    from trading_crew.data_sources import web_research

    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t)
    monkeypatch.setattr(tools, "_screener_available_for", lambda t: True)
    monkeypatch.setattr(
        tools, "get_market_profile",
        lambda t: type("P", (), {"long_name": "Tata Consultancy Services Ltd"})(),
    )
    monkeypatch.setattr(tools.screener_source, "fetch_company", lambda t: _page(docs))
    monkeypatch.setattr(web_research, "crawl", crawl_fn)
    if summarize_fn is not None:
        monkeypatch.setattr(tools, "_summarize_report_docs", summarize_fn)
    return tools


def test_indian_filings_use_http_crawler_not_firecrawl(monkeypatch):
    """Screener PDFs must not wait on Firecrawl before REPORT_PROCESS LLM."""
    from trading_crew import tools

    seen: list[str] = []

    def _crawl(url, *, crawler="auto", max_chars=8000, **kw):
        seen.append(crawler)
        return ("management targets 12% revenue growth. ") * 20

    docs = [
        DocumentLink(title="Concall Q2 24 Jun 2024", url="https://s3/concall.pdf", kind="concall"),
    ]
    _indian_docs_harness(
        monkeypatch, docs, _crawl,
        lambda name, t, crawled: "1. GUIDANCE: 12% [source: https://s3/concall.pdf]",
    )
    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert seen == ["http"]


def test_catalog_hit_skips_crawl_and_llm(monkeypatch, tmp_path):
    """Same listing metadata (url + title + kind) reuses the digest."""
    from trading_crew import tools

    monkeypatch.setenv("TRADINGCREW_REPORT_INSIGHTS_CACHE", "1")
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    docs = [
        DocumentLink(title="Concall Q2 24 Jun 2024", url="https://s3/concall.pdf", kind="concall"),
    ]
    crawls = {"n": 0}
    summaries = {"n": 0}

    def _crawl(url, **kw):
        crawls["n"] += 1
        return ("RAW TEXT for " + url + ": management targets 15% revenue growth. ") * 8

    def _sum(name, ticker, crawled):
        summaries["n"] += 1
        return "1. GUIDANCE: 15% [source: https://s3/concall.pdf]"

    _indian_docs_harness(monkeypatch, docs, _crawl, _sum)
    first = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert "15%" in first
    assert crawls["n"] == 1
    assert summaries["n"] == 1
    assert list((tmp_path / "report_insights").glob("TCS.NS.json"))

    second = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert "catalog-hit" in second
    assert "15%" in second
    assert crawls["n"] == 1
    assert summaries["n"] == 1


def test_head_hash_hit_skips_llm_when_title_drifts(monkeypatch, tmp_path):
    """Same URL + same first-page hash, even if the listing title changed."""
    from trading_crew import tools

    monkeypatch.setenv("TRADINGCREW_REPORT_INSIGHTS_CACHE", "1")
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    body = ("same filing body for hash " * 40)
    crawls = {"n": 0}
    summaries = {"n": 0}

    def _crawl(url, **kw):
        crawls["n"] += 1
        return body

    def _sum(name, ticker, crawled):
        summaries["n"] += 1
        return "1. GUIDANCE: unchanged [source: https://s3/concall.pdf]"

    first_docs = [
        DocumentLink(title="Concall 15h", url="https://s3/concall.pdf", kind="concall"),
    ]
    _indian_docs_harness(monkeypatch, first_docs, _crawl, _sum)
    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert crawls["n"] == 1 and summaries["n"] == 1

    later_docs = [
        DocumentLink(title="Concall 1d", url="https://s3/concall.pdf", kind="concall"),
    ]
    _indian_docs_harness(monkeypatch, later_docs, _crawl, _sum)
    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert "head-hash-hit" in out
    assert summaries["n"] == 1
    assert crawls["n"] == 2


def test_new_filing_resummarises(monkeypatch, tmp_path):
    """A new URL on the listing must run map-reduce again."""
    from trading_crew import tools

    monkeypatch.setenv("TRADINGCREW_REPORT_INSIGHTS_CACHE", "1")
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    summaries = {"n": 0}

    def _crawl(url, **kw):
        return ("RAW " + url + " management target text. ") * 10

    def _sum(name, ticker, crawled):
        summaries["n"] += 1
        return f"digest-{summaries['n']} urls={len(crawled)}"

    docs1 = [DocumentLink(title="Concall A", url="https://s3/a.pdf", kind="concall")]
    _indian_docs_harness(monkeypatch, docs1, _crawl, _sum)
    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    docs2 = [
        DocumentLink(title="Concall A", url="https://s3/a.pdf", kind="concall"),
        DocumentLink(title="Annual FY25", url="https://s3/ar.pdf", kind="annual_report"),
    ]
    _indian_docs_harness(monkeypatch, docs2, _crawl, _sum)
    out = tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]
    assert summaries["n"] == 2
    assert "digest-2" in out
    assert "crawl + LLM-summarise" in out


def test_selected_filings_extracted_in_parallel(monkeypatch):
    """Selected filings extract concurrently; crawled order stays catalog order."""
    import threading

    from trading_crew import tools

    n = 3
    barrier = threading.Barrier(n, timeout=5)
    order: list[str] = []

    def _crawl(url, **kw):
        barrier.wait()
        return ("RAW TEXT for " + url + " management targets 15% growth. ") * 12

    def _sum(name, ticker, crawled):
        order.extend(c["url"] for c in crawled)
        return "digest"

    docs = [
        DocumentLink(title="Concall Q4", url="https://s3/c4.pdf", kind="concall"),
        DocumentLink(title="Concall Q3", url="https://s3/c3.pdf", kind="concall"),
        DocumentLink(title="Concall Q2", url="https://s3/c2.pdf", kind="concall"),
    ]
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_DOCS", "3")
    monkeypatch.setenv("TRADINGCREW_REPORT_MAX_PER_KIND", "3")
    _indian_docs_harness(monkeypatch, docs, _crawl, _sum)
    tools.get_report_insights.func("TCS.NS")  # type: ignore[attr-defined]

    assert order == [
        "https://s3/c4.pdf",
        "https://s3/c3.pdf",
        "https://s3/c2.pdf",
    ], f"parallel extract must keep catalog order, got {order}"
