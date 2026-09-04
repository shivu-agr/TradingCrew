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

from trading_crew.tools import (
    sanitize_citations,
    _is_fabricated_identifier,
    normalize_source_identifier,
    normalize_source_providers,
)


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
    # Host-token match — do not treat a real host as fabricated because
    # it merely contains a placeholder substring (test.com / domain.com).
    assert _is_fabricated_identifier("https://contest.com/x") is False
    assert _is_fabricated_identifier("https://www.subdomain.com/page") is False
    assert _is_fabricated_identifier("www.reuters.com/markets/rbi") is False


def test_mixed_compound_tag_keeps_the_real_url():
    text = (
        "Stake sale [source: https://scanx.trade/deal, https://example.com/fake]."
    )
    out = sanitize_citations(text)
    assert "https://scanx.trade/deal" in out
    assert "example.com" not in out
    assert "[analyst judgement]" not in out


def test_compound_all_placeholders_become_judgement():
    text = "X [source: https://example.com/a, https://test.com/b]."
    out = sanitize_citations(text)
    assert out == "X [analyst judgement]."


# ---------------------------------------------------------------------------
# Source-provider normalization ([source: profit-loss] -> screener.in · …)
# ---------------------------------------------------------------------------

def test_screener_sections_get_provider_prefix():
    assert normalize_source_providers("[source: profit-loss]") == "[source: screener.in · profit-loss]"
    assert normalize_source_providers("[source: balance-sheet]") == "[source: screener.in · balance-sheet]"
    assert normalize_source_providers("[source: cash-flow]") == "[source: screener.in · cash-flow]"


def test_yahoo_fields_get_provider_prefix():
    assert normalize_source_providers("[source: eps_revisions]") == "[source: yahoo · eps_revisions]"
    assert normalize_source_providers("[source: analyst_price_targets]") == "[source: yahoo · analyst_price_targets]"
    assert normalize_source_providers("[source: sector_peers]") == "[source: yahoo · sector_peers]"


def test_explicit_provider_token_is_condensed():
    assert normalize_source_providers(
        "[source: yfinance eps_revisions RELIANCE.NS]"
    ) == "[source: yahoo · eps_revisions]"
    assert normalize_source_providers(
        "[source: screener.in https://www.screener.in/company/RELIANCE/ #ratios]"
    ) == "[source: screener.in · ratios]"


def test_compound_tag_collapses_same_provider():
    assert normalize_source_providers(
        "[source: profit-loss; ratios]"
    ) == "[source: screener.in · profit-loss, ratios]"


def test_ambiguous_thesis_needs_explicit_provider():
    # Bare ambiguous id is left untouched (both screener.in and yfinance paths exist).
    assert normalize_source_providers("[source: long_term_thesis]") == "[source: long_term_thesis]"
    # With an explicit provider token it is attributed.
    assert normalize_source_providers(
        "[source: long_term_thesis via yfinance AAPL]"
    ) == "[source: yahoo · long_term_thesis]"


def test_urls_and_unknowns_untouched():
    assert normalize_source_providers(
        "[source: https://www.moneycontrol.com/x]"
    ) == "[source: https://www.moneycontrol.com/x]"


def test_normalization_is_idempotent():
    prose = (
        "Margins 9% [source: profit-loss; ratios]. Target [source: analyst_price_targets]. "
        "Debt [source: balance-sheet]."
    )
    once = normalize_source_providers(prose)
    assert normalize_source_providers(once) == once


def test_sources_list_entry_normalization():
    assert normalize_source_identifier("profit-loss") == "screener.in · profit-loss"
    assert normalize_source_identifier("yfinance eps_revisions RELIANCE.NS") == "yahoo · eps_revisions"
    assert normalize_source_identifier("https://x.com/a") == "https://x.com/a"
