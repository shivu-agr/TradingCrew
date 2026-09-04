"""Search → crawl → LLM-summarize web-research pipeline + Firecrawl rotation."""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone

import pytest

from trading_crew.data_sources import news_providers, pit, web_research


class _FakeLLM:
    def __init__(self, reply="- Fact [source: https://x.com/a]"):
        self.reply = reply
        self.prompts = []

    def call(self, messages):
        self.prompts.append(messages[0]["content"])
        return self.reply


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_KEYS",
        "WEBSEARCH_BACKENDS",
        "WEBSEARCH_SH_PATH",
        "GEMINI_WEB_SEARCH",
        "GOOGLE_WEB_SEARCH",
        "GEMINI_SEARCH_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    web_research.reset_gemini_module()
    news_providers.reset_rotation()
    yield
    news_providers.reset_rotation()
    web_research.reset_gemini_module()


# --------------------------------------------------------------------------- search
def test_search_links_uses_first_nonempty_backend(monkeypatch):
    calls = []

    def _empty(q, n):
        calls.append("empty")
        return []

    def _hit(q, n):
        calls.append("hit")
        return [{"title": "A", "url": "https://reuters.com/a"},
                {"title": "B", "url": "https://reuters.com/a"},  # dupe URL
                {"title": "C", "url": "https://ft.com/c"}]

    monkeypatch.setitem(web_research._SEARCH_FUNCS, "websearch", _empty)
    monkeypatch.setitem(web_research._SEARCH_FUNCS, "ddg", _hit)
    out = web_research.search_links("q", max_results=5, backends=["websearch", "ddg", "brave"])
    assert [l["url"] for l in out] == ["https://reuters.com/a", "https://ft.com/c"]
    assert calls == ["empty", "hit"]  # stops at first non-empty


def test_search_websearch_sh_parses_stdout(monkeypatch, tmp_path):
    script = tmp_path / "websearch.sh"
    script.write_text("#!/usr/bin/env bash\n")
    monkeypatch.setenv("WEBSEARCH_SH_PATH", str(script))

    class _Proc:
        stdout = "https://reuters.com/a\nhttps://ft.com/c\n[brave] diagnostic\n"
        stderr = ""

    monkeypatch.setattr(web_research.subprocess, "run", lambda *a, **k: _Proc())
    rows = web_research._search_websearch_sh("q", 5)
    assert [r["url"] for r in rows] == ["https://reuters.com/a", "https://ft.com/c"]


def test_search_websearch_sh_missing_script_returns_empty(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_SH_PATH", "/nonexistent/websearch.sh")
    assert web_research._search_websearch_sh("q", 5) == []


def test_default_websearch_sh_is_vendored_in_repo():
    """The default script path must point at the in-repo copy and exist."""
    assert web_research._DEFAULT_WEBSEARCH_SH.endswith("scripts/websearch/websearch.sh")
    assert os.path.isfile(web_research._DEFAULT_WEBSEARCH_SH)


# --------------------------------------------------------------------------- PIT
def test_pit_bounded_query_appends_before_operator():
    as_of = datetime(2023, 6, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        bounded = web_research._pit_bounded_query("TCS results")
    assert "before:2023-06-02" in bounded          # cutoff + 1 day (inclusive)
    assert bounded.startswith("TCS results")


def test_pit_bounded_query_noop_in_live_mode():
    assert web_research._pit_bounded_query("TCS results") == "TCS results"


def test_search_links_date_bounds_query_under_as_of(monkeypatch):
    seen = {}

    def _capture(q, n):
        seen["q"] = q
        return [{"title": "t", "url": "https://reuters.com/a"}]

    monkeypatch.setitem(web_research._SEARCH_FUNCS, "websearch", _capture)
    as_of = datetime(2023, 6, 1, tzinfo=timezone.utc)
    with pit.as_of_scope(as_of, mode="backfill", strict=True):
        web_research.search_links("TCS results", backends=["websearch"])
    assert "before:2023-06-02" in seen["q"]


def test_default_searxng_engines_include_working_mix():
    mix = web_research._DEFAULT_SEARXNG_ENGINES
    for name in (
        "duckduckgo web", "privacywall", "infospace", "abcnyheter",
        "bing news", "google cse", "duckduckgo news", "gmx",
        "bing", "startpage", "duckduckgo", "yahoo",
    ):
        assert name in mix
    assert mix.index("duckduckgo web") < mix.index("bing")


def test_parse_relevance_verdict_reads_last_line_only():
    assert web_research.parse_relevance_verdict("ok\nRELEVANT") is True
    assert web_research.parse_relevance_verdict("no\nIRRELEVANT") is False
    assert web_research.parse_relevance_verdict("VERDICT: IRRELEVANT") is False
    assert web_research.parse_relevance_verdict("") is None
    assert web_research.parse_relevance_verdict("maybe later") is None


def test_links_relevant_to_query_fail_closed_on_missing_verdict():
    llm = _FakeLLM("I am not sure about these pages.")
    assert web_research.links_relevant_to_query(
        "Bharat Forge Q1",
        [{"title": "Suzuki", "url": "https://suzuki.co.jp/x"}],
        llm=llm,
    ) is False


def test_links_relevant_to_query_accepts_relevant(monkeypatch):
    llm = _FakeLLM("The titles match the issuer.\nRELEVANT")
    assert web_research.links_relevant_to_query(
        "Bharat Forge Q1 FY27",
        [{"title": "Bharat Forge Q1 FY27", "url": "https://cnbctv18.com/bf"}],
        llm=llm,
    ) is True
    assert "Bharat Forge Q1 FY27" in llm.prompts[0]


def test_ring_rejects_offtopic_searxng_waits_and_retries(monkeypatch):
    """Off-topic Bing dump must not win; next SearXNG node is tried after wait."""
    monkeypatch.setattr(
        web_research, "_PROVIDER_RING",
        deque(["searxng:bing", "searxng:duckduckgo web"]),
    )
    calls: list[str] = []
    waits: list[int] = []

    def _run(name, query, limit):
        calls.append(name)
        if name == "searxng:bing":
            return [{"title": "Suzuki Okinawa", "url": "https://suzuki.co.jp/dealer/1"}]
        return [{"title": "Bharat Forge Q1 FY27", "url": "https://cnbctv18.com/bharat-forge"}]

    def _judge(query, links, llm=None):
        return any("bharat forge" in (l.get("title") or "").lower() for l in links)

    monkeypatch.setattr(web_research, "_run_provider", _run)
    monkeypatch.setattr(web_research, "links_relevant_to_query", _judge)
    monkeypatch.setattr(web_research, "_acquire_search_slot", lambda: waits.append(1))
    out = web_research._search_links_ring("Bharat Forge Q1 FY27 defence", 5)
    assert calls == ["searxng:bing", "searxng:duckduckgo web"]
    assert waits == [1]
    assert [l["url"] for l in out] == ["https://cnbctv18.com/bharat-forge"]


def test_ring_google_headless_skips_llm_gate(monkeypatch):
    """Non-SearXNG ring nodes (google-headless) are not LLM-judged."""
    monkeypatch.setattr(web_research, "_PROVIDER_RING", deque(["google"]))
    judged: list[int] = []

    def _run(name, query, limit):
        return [{"title": "Anything", "url": "https://example.com/x"}]

    monkeypatch.setattr(web_research, "_run_provider", _run)
    monkeypatch.setattr(
        web_research, "links_relevant_to_query",
        lambda *a, **k: judged.append(1) or False,
    )
    out = web_research._search_links_ring("q", 3)
    assert judged == []
    assert out[0]["url"] == "https://example.com/x"


def test_search_links_skips_failing_backend(monkeypatch):
    def _boom(q, n):
        raise RuntimeError("blocked")

    monkeypatch.setitem(web_research._SEARCH_FUNCS, "ddg", _boom)
    monkeypatch.setitem(web_research._SEARCH_FUNCS, "brave",
                        lambda q, n: [{"title": "ok", "url": "https://x.com/1"}])
    out = web_research.search_links("q", backends=["ddg", "brave"])
    assert [l["url"] for l in out] == ["https://x.com/1"]
    assert out[0]["title"] == "ok"


def test_ddg_parser_extracts_and_resolves(monkeypatch):
    html = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg='
        'https%3A%2F%2Freuters.com%2Fstory">Reuters Story</a>'
        '<a class="result__a" href="https://ft.com/x">FT</a>'
    )
    monkeypatch.setattr(web_research, "_scrape_html", lambda url, timeout=12: html)
    rows = web_research._search_ddg("q", 5)
    urls = [r["url"] for r in rows]
    assert "https://reuters.com/story" in urls
    assert "https://ft.com/x" in urls


# --------------------------------------------------------------------------- crawl
def test_crawl_prefers_firecrawl(monkeypatch):
    monkeypatch.setattr(news_providers, "firecrawl_scrape", lambda url, max_chars=8000: "# Markdown body")
    assert web_research.crawl("https://x.com", crawler="auto") == "# Markdown body"


def test_crawl_firecrawl_only_returns_none_when_empty(monkeypatch):
    monkeypatch.setattr(news_providers, "firecrawl_scrape", lambda url, max_chars=8000: None)
    assert web_research.crawl("https://x.com", crawler="firecrawl") is None


def test_crawl_http_backend(monkeypatch):
    html = b"<html><body><script>x</script><p>Hello world</p></body></html>"
    monkeypatch.setattr(web_research, "_fetch_url_bytes", lambda url, **kw: (html, "text/html"))
    out = web_research.crawl("https://x.com", crawler="http")
    assert "Hello world" in out
    assert "x" not in out.split()  # script stripped


def test_crawl_http_does_not_call_firecrawl(monkeypatch):
    called = {"n": 0}

    def _scrape(url, max_chars=8000):
        called["n"] += 1
        return "# should not run"

    monkeypatch.setattr(news_providers, "firecrawl_scrape", _scrape)
    monkeypatch.setattr(
        web_research, "_fetch_url_bytes",
        lambda url, **kw: (b"<html><body><p>local</p></body></html>", "text/html"),
    )
    out = web_research.crawl("https://example.com/page", crawler="http")
    assert called["n"] == 0
    assert "local" in out


def test_crawl_http_routes_pdf_to_local_parser(monkeypatch):
    """A PDF response (by content-type / magic bytes) must be parsed with the
    local pdfplumber parser, not tag-stripped as HTML."""
    monkeypatch.setattr(
        web_research, "_fetch_url_bytes",
        lambda url, **kw: (b"%PDF-1.7 ...binary...", "application/pdf"),
    )
    monkeypatch.setattr(
        web_research, "extract_pdf_text",
        lambda content, max_chars=8000, **kw: "PARSED PDF TEXT",
    )
    out = web_research.crawl("https://s3/report.pdf", crawler="http")
    assert out == "PARSED PDF TEXT"


# --------------------------------------------------------------------------- summarize
def test_summarize_builds_prompt_and_returns(monkeypatch):
    llm = _FakeLLM()
    docs = [{"title": "T", "url": "https://x.com/a", "content": "Company beat estimates."}]
    out = web_research.summarize("earnings", docs, llm=llm)
    assert out == "- Fact [source: https://x.com/a]"
    assert "https://x.com/a" in llm.prompts[0]
    assert "earnings" in llm.prompts[0]


def test_summarize_empty_when_no_content():
    assert web_research.summarize("q", [{"title": "t", "url": "u", "content": ""}], llm=_FakeLLM()) == ""


def test_research_and_summarize_end_to_end(monkeypatch):
    monkeypatch.setattr(web_research, "search_links",
                        lambda q, max_results=5, backends=None, **kw: [{"title": "T", "url": "https://x.com/a"}])
    monkeypatch.setattr(web_research, "crawl",
                        lambda url, crawler="auto", max_chars=8000: "raw page text")
    res = web_research.research_and_summarize("q", llm=_FakeLLM("SUMMARY"))
    assert res["summary"] == "SUMMARY"
    assert res["n_crawled"] == 1
    assert res["sources"] == [{"title": "T", "url": "https://x.com/a"}]


# --------------------------------------------------------------------------- firecrawl rotation
def test_firecrawl_scrape_rotates_on_failure(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_ENABLE_FIRECRAWL", "1")
    monkeypatch.setenv("FIRECRAWL_API_KEYS", "k1,k2")
    used = []

    class _Resp:
        def __init__(self, ok):
            self._ok = ok

        def raise_for_status(self):
            if not self._ok:
                raise RuntimeError("429")

        def json(self):
            return {"data": {"markdown": "content-from-k2"}}

    def _post(url, headers=None, json=None, timeout=None):
        key = headers["Authorization"].split()[-1]
        used.append(key)
        return _Resp(ok=(key == "k2"))

    monkeypatch.setattr(news_providers.requests, "post", _post)
    out = news_providers.firecrawl_scrape("https://x.com/a")
    assert out == "content-from-k2"
    assert used == ["k1", "k2"]  # tried k1, rolled over to k2


def test_firecrawl_scrape_none_without_key():
    assert news_providers.firecrawl_scrape("https://x.com/a") is None


def test_gemini_research_skips_searxng(monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_SEARCH", "1")

    def _gem(query):
        return {
            "summary": "Beacon is a Cincinnati orthopedic practice.",
            "sources": [{"title": "Beacon", "url": "https://www.beaconortho.com/"}],
        }

    def boom(*_a, **_k):
        raise AssertionError("SearXNG must not run when Gemini returns a briefing")

    monkeypatch.setattr(web_research, "_gemini_research", _gem)
    monkeypatch.setattr(web_research, "search_links", boom)
    out = web_research.research_and_summarize("Beacon Orthopaedics")
    assert out["backend"] == "gemini"
    assert "Cincinnati" in out["summary"]
    assert out["sources"][0]["url"] == "https://www.beaconortho.com/"


def test_gemini_off_does_not_call_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_WEB_SEARCH", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("Gemini must stay dark when GEMINI_WEB_SEARCH is off")

    monkeypatch.setattr(web_research, "_gemini_research", boom)
    monkeypatch.setattr(
        web_research,
        "search_links",
        lambda *a, **k: [{"title": "A", "url": "https://reuters.com/a"}],
    )
    monkeypatch.setattr(web_research, "crawl", lambda *a, **k: "body")
    monkeypatch.setattr(web_research, "summarize", lambda *a, **k: "- Fact")
    out = web_research.research_and_summarize("q")
    assert out["status"] == "ok"
    assert out.get("backend") != "gemini"


def test_gemini_failure_falls_through_to_searxng(monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_SEARCH", "1")
    seen = {}

    def _fail(query):
        raise RuntimeError("quota")

    def _links(query, max_results=6, backends=None, skip_gemini=False):
        seen["skip_gemini"] = skip_gemini
        return [{"title": "A", "url": "https://reuters.com/a"}]

    monkeypatch.setattr(web_research, "_gemini_research", _fail)
    monkeypatch.setattr(web_research, "search_links", _links)
    monkeypatch.setattr(web_research, "crawl", lambda *a, **k: "body")
    monkeypatch.setattr(web_research, "summarize", lambda *a, **k: "- Fact")
    out = web_research.research_and_summarize("q")
    assert seen["skip_gemini"] is True
    assert out["status"] == "ok"
    assert out.get("backend") != "gemini"


def test_search_links_gemini_first_then_ring(monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_SEARCH", "1")
    monkeypatch.setattr(
        web_research,
        "_search_gemini",
        lambda q, n: [{"title": "G", "url": "https://gemini.test/a"}],
    )

    def boom(*_a, **_k):
        raise AssertionError("SearXNG ring must not run when Gemini returns links")

    monkeypatch.setattr(web_research, "_search_links_ring", boom)
    monkeypatch.setattr(web_research, "_acquire_search_slot", lambda: None)
    out = web_research.search_links("q")
    assert [l["url"] for l in out] == ["https://gemini.test/a"]


def test_ordered_keys_rotates(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEYS", "a,b,c")
    first = news_providers.ordered_keys("serpapi")
    second = news_providers.ordered_keys("serpapi")
    assert sorted(first) == ["a", "b", "c"]
    assert first[0] != second[0]  # round-robin advanced the start
