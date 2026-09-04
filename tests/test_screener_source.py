"""Unit tests for the screener.in HTML parsers + markdown renderers.

We deliberately keep the network layer *fully* mocked: the tests feed a
known-good fixture HTML through the parsers directly.  A separate,
network-guarded test at the bottom exercises the fetch layer against
``requests`` via a monkey-patched ``session.get`` so we prove the URL
fallback (``consolidated/`` -> ``/``) and rate-limit path without
hitting the live site.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from trading_crew.data_sources import screener_source as ss


FIXTURE = Path(__file__).parent / "fixtures" / "screener" / "reliance.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE.read_text()


@pytest.fixture
def soup(fixture_html: str) -> BeautifulSoup:
    return BeautifulSoup(fixture_html, "html.parser")


@pytest.fixture(autouse=True)
def _clear_screener_cache():
    """Every test starts with an empty page cache + session flag."""
    ss.clear_cache()
    yield
    ss.clear_cache()


# ---------------------------------------------------------------------------
# _parse_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("₹ 17,64,643 Cr.", 1764643.0),   # Indian lakh/crore grouping
        ("22.7", 22.7),
        ("8.91 %", 8.91),
        ("-3.2", -3.2),
        ("₹ 1,304", 1304.0),
        ("0.46 %", 0.46),
        ("", None),
        ("n/a", None),
        ("--", None),
    ],
)
def test_parse_number(raw, expected):
    assert ss._parse_number(raw) == expected


# ---------------------------------------------------------------------------
# parse_top_ratios
# ---------------------------------------------------------------------------


def test_parse_top_ratios_extracts_named_ratios(soup):
    top = ss.parse_top_ratios(soup.select_one("#top"))
    # A representative slice of ratios (screener.in strips vary run to
    # run — we assert the ones the fundamentals report actually surfaces).
    assert top.raw["Market Cap"] == "₹ 17,64,643 Cr."
    assert top.raw["Stock P/E"] == "22.7"
    assert top.raw["ROCE"] == "10.3 %"
    assert top.raw["ROE"] == "8.91 %"
    # numeric mirror is populated only where a number is unambiguous
    assert top.numeric["Stock P/E"] == pytest.approx(22.7)
    assert top.numeric["ROE"] == pytest.approx(8.91)


def test_parse_top_ratios_handles_missing_section():
    # A None #top should give an empty (but valid) TopRatios, not raise
    empty = ss.parse_top_ratios(None)
    assert empty.raw == {}
    assert empty.numeric == {}


# ---------------------------------------------------------------------------
# _parse_financial_table (P&L / BS / CF)
# ---------------------------------------------------------------------------


def test_parse_profit_loss_table(soup):
    pl = ss._parse_financial_table(soup.select_one("#profit-loss"), "profit-loss")
    assert pl is not None
    assert pl.kind == "profit-loss"
    # First column ("Mar 2022") sits after the metric label
    assert pl.columns[0] == "Mar 2022"
    assert pl.columns[-1] == "TTM"
    # Row order preserved
    metrics = [row[0] for row in pl.rows]
    assert metrics == ["Sales", "Net Profit", "EPS in Rs"]
    # Value grid stays aligned with columns
    sales_row = dict(zip(pl.columns, pl.rows[0][1]))
    assert sales_row["Mar 2022"] == "792,756"
    assert sales_row["TTM"] == "981,736"


def test_parse_balance_sheet_and_cash_flow(soup):
    bs = ss._parse_financial_table(soup.select_one("#balance-sheet"), "balance-sheet")
    cf = ss._parse_financial_table(soup.select_one("#cash-flow"), "cash-flow")
    assert bs is not None and cf is not None
    assert [r[0] for r in bs.rows][0] == "Equity Capital"
    assert [r[0] for r in cf.rows] == [
        "Cash from Operating Activity",
        "Cash from Investing Activity",
        "Cash from Financing Activity",
        "Net Cash Flow",
    ]


def test_parse_financial_table_returns_none_for_missing_section():
    assert ss._parse_financial_table(None, "profit-loss") is None


def test_parse_financial_table_pads_short_rows():
    """A row with fewer <td> cells than the header must right-pad with
    empty strings so the LLM sees a rectangular grid.
    """
    html = """
    <section id="quarters">
      <table class="data-table">
        <thead><tr><th></th><th>Q1</th><th>Q2</th><th>Q3</th></tr></thead>
        <tbody>
          <tr><td>Sales</td><td>100</td><td>200</td><td>300</td></tr>
          <tr><td>Short</td><td>7</td></tr>
        </tbody>
      </table>
    </section>
    """
    soup = BeautifulSoup(html, "html.parser")
    t = ss._parse_financial_table(soup.select_one("#quarters"), "quarters")
    assert t is not None
    _, short_values = t.rows[1]
    assert short_values == ["7", "", ""]


# ---------------------------------------------------------------------------
# parse_shareholding
# ---------------------------------------------------------------------------


def test_parse_shareholding_splits_quarterly_and_yearly(soup):
    sh = ss.parse_shareholding(soup.select_one("#shareholding"))
    assert sh is not None
    assert sh.quarterly is not None and sh.yearly is not None
    q_metrics = [r[0] for r in sh.quarterly.rows]
    assert "Promoters" in q_metrics
    assert "FIIs" in q_metrics
    assert "DIIs" in q_metrics
    assert "Public" in q_metrics
    # Latest column reflects the FII taper the fixture encodes.
    fii_row = dict(zip(sh.quarterly.columns, sh.quarterly.rows[q_metrics.index("FIIs")][1]))
    assert fii_row["Mar 2025"] == "22.13%"
    assert fii_row["Dec 2025"] == "20.85%"
    # Yearly table is separate
    y_metrics = [r[0] for r in sh.yearly.rows]
    assert y_metrics == q_metrics


def test_parse_shareholding_handles_single_table():
    """Some small caps only publish a quarterly split — the parser must
    still surface it as ``.quarterly`` and leave ``.yearly`` as None.
    """
    html = """
    <section id="shareholding">
      <table class="data-table">
        <thead><tr><th></th><th>Mar 2025</th></tr></thead>
        <tbody><tr><td>Promoters</td><td>75%</td></tr></tbody>
      </table>
    </section>
    """
    soup = BeautifulSoup(html, "html.parser")
    sh = ss.parse_shareholding(soup.select_one("#shareholding"))
    assert sh is not None
    assert sh.quarterly is not None
    assert sh.yearly is None


# ---------------------------------------------------------------------------
# parse_documents (announcements)
# ---------------------------------------------------------------------------


def test_parse_documents_extracts_filings_and_drops_all_link(soup):
    docs = ss.parse_documents(soup.select_one("#documents"))
    # 4 filings in the fixture; the "All" summary link is dropped.
    assert len(docs) == 4
    titles = [d.title for d in docs]
    # The buyback resolution is exactly the kind of signal the previous
    # BAJAJ-AUTO run missed — the announcement parser is what closes
    # that gap.
    assert any("Buyback Approval" in t for t in titles)
    assert any("Credit Rating" in t for t in titles)
    assert all(d.url.startswith("http") for d in docs)


# ---------------------------------------------------------------------------
# parse_pros_cons
# ---------------------------------------------------------------------------


def test_parse_pros_cons_uses_class_hints(soup):
    pc = ss.parse_pros_cons(soup.select_one("#analysis"))
    assert len(pc.pros) == 2
    assert len(pc.cons) == 2
    assert "Debt reduction" in pc.pros[0]
    assert "low return on equity" in pc.cons[0]


def test_parse_pros_cons_positional_fallback():
    """When the ULs have no ``.pros`` / ``.cons`` class the parser must
    fall back to positional order (first UL = pros, second = cons).
    """
    html = """
    <section id="analysis">
      <ul><li>Good thing</li></ul>
      <ul><li>Bad thing</li></ul>
    </section>
    """
    soup = BeautifulSoup(html, "html.parser")
    pc = ss.parse_pros_cons(soup.select_one("#analysis"))
    assert pc.pros == ["Good thing"]
    assert pc.cons == ["Bad thing"]


# ---------------------------------------------------------------------------
# _parse_company_html (integration of all parsers)
# ---------------------------------------------------------------------------


def test_parse_company_html_full_snapshot(fixture_html):
    page = ss._parse_company_html(
        "RELIANCE",
        "https://www.screener.in/company/RELIANCE/consolidated/",
        "consolidated",
        fixture_html,
    )
    assert page.symbol == "RELIANCE"
    assert page.mode == "consolidated"
    assert page.long_name == "Reliance Industries Ltd"
    assert page.about.startswith("Reliance Industries Ltd is engaged in oil")
    assert page.top.raw["Market Cap"].startswith("₹")
    assert page.quarters is not None
    assert page.profit_loss is not None
    assert page.balance_sheet is not None
    assert page.cash_flow is not None
    assert page.shareholding is not None
    assert len(page.documents) == 4
    assert len(page.pros_cons.pros) == 2 and len(page.pros_cons.cons) == 2


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def test_render_fundamentals_report_includes_key_lines(fixture_html):
    page = ss._parse_company_html("RELIANCE", "u", "consolidated", fixture_html)
    md = ss.render_fundamentals_report(page)
    assert "# Reliance Industries Ltd" in md
    assert "## Key ratios" in md
    assert "Market Cap" in md
    assert "## Pros" in md and "## Cons" in md


def test_render_shareholding_report_covers_quarterly_and_yearly(fixture_html):
    page = ss._parse_company_html("RELIANCE", "u", "consolidated", fixture_html)
    md = ss.render_shareholding_report(page)
    assert "Quarterly" in md
    assert "Yearly" in md
    assert "Promoters" in md


def test_render_announcements_report_limits_and_tags_source(fixture_html):
    page = ss._parse_company_html("RELIANCE", "u", "consolidated", fixture_html)
    md = ss.render_announcements_report(page, limit=2)
    assert md.count("[source:") == 2, "each surfaced filing must carry a provenance tag"


def test_render_financial_statement_rejects_unknown_kind(fixture_html):
    page = ss._parse_company_html("RELIANCE", "u", "consolidated", fixture_html)
    with pytest.raises(ValueError, match="Unknown statement"):
        ss.render_financial_statement_report(page, "made-up-statement")


@pytest.mark.parametrize(
    "kind",
    ["profit-loss", "balance-sheet", "cash-flow", "quarters"],
)
def test_render_financial_statement_all_kinds(fixture_html, kind):
    page = ss._parse_company_html("RELIANCE", "u", "consolidated", fixture_html)
    md = ss.render_financial_statement_report(page, kind)
    # Every statement should render a title line + a markdown table header.
    assert "# Reliance Industries Ltd" in md
    assert "| Metric |" in md


# ---------------------------------------------------------------------------
# fetch_company — URL fallback + rate limiting
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, text: str = ""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    return r


def test_fetch_company_hits_consolidated_first(fixture_html, monkeypatch):
    """The primary URL is ``/consolidated/`` — a success there must not
    trigger a second GET for the standalone fallback.
    """
    session = MagicMock()
    session.get.return_value = _mock_response(200, fixture_html)
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    page = ss.fetch_company("RELIANCE.NS")
    assert page.symbol == "RELIANCE"
    assert page.mode == "consolidated"
    # Only one URL was fetched
    assert session.get.call_count == 1
    (url,), _ = session.get.call_args
    assert url.endswith("/consolidated/")


def test_fetch_company_falls_back_to_standalone_on_404(fixture_html, monkeypatch):
    """Group-less names (single-entity businesses) don't publish a
    consolidated page — the parser must retry ``/`` on a 404 and
    surface ``mode="standalone"``.
    """
    session = MagicMock()
    session.get.side_effect = [
        _mock_response(404),
        _mock_response(200, fixture_html),
    ]
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    page = ss.fetch_company("SMALLCAP.NS")
    assert page.mode == "standalone"
    assert session.get.call_count == 2


def test_fetch_company_falls_back_when_consolidated_quarters_empty(fixture_html, monkeypatch):
    """A name with no consolidated financials (e.g. COLPAL) serves a
    label-only #quarters SKELETON on ``/consolidated/`` (HTTP 200, no period
    columns) while its ANNUAL P&L is still populated with stale history.  The
    fetch must not stop at that page — it must fetch ``/`` (standalone), which
    carries real quarterly numbers, and surface ``mode="standalone"``."""
    skeleton = (
        "<html><body><h1>Colgate Palmolive</h1>"
        "<section id='quarters'><table class='data-table'><tbody>"
        "<tr><td class='text'>Sales +</td></tr>"
        "<tr><td class='text'>Net Profit +</td></tr>"
        "</tbody></table></section>"
        # a populated annual P&L, to prove the P&L is NOT enough to keep the page
        "<section id='profit-loss'><table class='data-table'>"
        "<thead><tr><th></th><th>Mar 2022</th><th>Mar 2023</th></tr></thead>"
        "<tbody><tr><td class='text'>Sales</td><td>100</td><td>110</td></tr></tbody>"
        "</table></section></body></html>"
    )
    session = MagicMock()
    session.get.side_effect = [
        _mock_response(200, skeleton),      # /consolidated/ — empty quarters
        _mock_response(200, fixture_html),  # /             — real quarters
    ]
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    page = ss.fetch_company("COLPALTEST.NS")
    assert page.mode == "standalone"
    assert session.get.call_count == 2
    assert ss._table_has_values(page.quarters)


def test_fetch_company_raises_on_rate_limit(monkeypatch):
    """A 429 must NOT be papered over with a yfinance fallback (project
    rule) — it propagates so the tool layer can emit a clear line.
    """
    session = MagicMock()
    session.get.return_value = _mock_response(429)
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    with pytest.raises(ss.ScreenerRateLimitError):
        ss.fetch_company("RELIANCE.NS")


def test_fetch_company_raises_not_found_on_double_404(monkeypatch):
    session = MagicMock()
    session.get.return_value = _mock_response(404)
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    with pytest.raises(ss.ScreenerNotFoundError):
        ss.fetch_company("BOGUS.NS")


def test_fetch_company_resolves_listing_slug_via_search(fixture_html, monkeypatch):
    """Yahoo root SBI is not the NSE listing (SBIN). Search on the same
    source must resolve the slug instead of raising not-found."""
    search = MagicMock()
    search.status_code = 200
    search.json.return_value = [
        {"name": "State Bank of India", "url": "/company/SBIN/consolidated/"},
    ]

    def _get(url, **kwargs):
        if "api/company/search" in str(url):
            return search
        if "/company/SBIN/" in str(url):
            return _mock_response(200, fixture_html)
        return _mock_response(404)

    session = MagicMock()
    session.get.side_effect = _get
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    page = ss.fetch_company("SBI.NS")
    assert page.symbol == "SBIN"
    assert page.mode == "consolidated"


def test_search_company_slug_prefers_name_or_code_match():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [
        {"name": "SBI Cards", "url": "/company/SBICARD/consolidated/"},
        {"name": "State Bank of India", "url": "/company/SBIN/consolidated/"},
    ]
    session.get.return_value = resp
    slug = ss.search_company_slug("SBI", session=session)
    assert slug == "SBIN"


def test_fetch_company_is_cached(fixture_html, monkeypatch):
    """Two calls in the same 15-min window must produce one HTTP GET.

    This is the mechanism that keeps 5 parallel analysts from each
    scraping the same company page — the memory analyst, market
    context, and the various fundamentals tools all share this cache.
    """
    session = MagicMock()
    session.get.return_value = _mock_response(200, fixture_html)
    monkeypatch.setattr(ss, "_get_session", lambda: session)
    monkeypatch.setattr(ss, "_maybe_login", lambda s: None)

    p1 = ss.fetch_company("RELIANCE.NS")
    p2 = ss.fetch_company("RELIANCE.NS")
    assert p1 is p2, "cache must return the same parsed object"
    assert session.get.call_count == 1
