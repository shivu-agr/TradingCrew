"""Guard for the deterministic citation sanitizer.

The root cause of sound theses being downgraded to HOLD was the LLM
inventing an ``example.com`` URL inside its own prose to satisfy the
``[source: …]`` provenance rule.  The tool layer already rejects fake
URLs it *returns*, but it can't stop a model from typing one into free
text.  ``sanitize_citations`` fixes that at the runner boundary: a
fabricated ``[source: <placeholder-url>]`` tag is rewritten to the
honest ``[analyst judgement]`` label, while real URLs and real tool
identifiers are left untouched.
"""

from __future__ import annotations

from trading_crew.tools import sanitize_citations, _is_fabricated_identifier


def test_fabricated_full_url_is_relabelled():
    text = "BEL order book is strong [source: https://example.com/bel-orders]."
    out = sanitize_citations(text)
    assert "[analyst judgement]" in out
    assert "example.com" not in out


def test_fabricated_schemeless_placeholder_is_relabelled():
    text = "Defence capex up [source: example.com/defence]."
    out = sanitize_citations(text)
    assert "[analyst judgement]" in out
    assert "example.com" not in out


def test_multiple_fabricated_tags_all_relabelled():
    text = (
        "A [source: http://example.org/a] and "
        "B [source: https://test.com/b] and C [source: placeholder.com/c]."
    )
    out = sanitize_citations(text)
    assert out.count("[analyst judgement]") == 3
    for bad in ("example.org", "test.com", "placeholder.com"):
        assert bad not in out


def test_real_url_is_preserved():
    text = "RBI held rates [source: https://www.moneycontrol.com/news/rbi]."
    out = sanitize_citations(text)
    assert "https://www.moneycontrol.com/news/rbi" in out
    assert "[analyst judgement]" not in out


def test_real_tool_identifier_is_preserved():
    # Non-URL tool identifiers must NOT be treated as fabricated.
    text = "P/E is 24 [source: yfinance .info BEL.NS (fundamentals)]."
    out = sanitize_citations(text)
    assert "[source: yfinance .info BEL.NS (fundamentals)]" in out


def test_screener_identifier_is_preserved():
    text = "ROCE trend up [source: screener.in BEL.NS #ratios]."
    out = sanitize_citations(text)
    assert "[source: screener.in BEL.NS #ratios]" in out


def test_existing_analyst_judgement_label_untouched():
    text = "Durable pricing power [analyst judgement]."
    assert sanitize_citations(text) == text


def test_empty_and_none_are_safe():
    assert sanitize_citations("") == ""
    assert sanitize_citations(None) is None


def test_is_fabricated_identifier_unit():
    assert _is_fabricated_identifier("https://example.com/x") is True
    assert _is_fabricated_identifier("example.com/x") is True
    assert _is_fabricated_identifier("https://reuters.com/x") is False
    assert _is_fabricated_identifier("yfinance option_chain BEL.NS") is False
