"""Static bias audits — does any prompt / tool / agent definition hard-
code a specific ticker, sector, or country?

These are CI-friendly checks (no LLM calls, no network) that catch the
most common form of bias: a prompt that mentions ``NTNX`` or ``NVDA``
verbatim, an agent backstory that says "specialise in US tech", or a
tool that branches its behaviour on a US-vs-Indian ticker suffix.

The companion script ``scripts/sector_bias_basket.py`` does the live
basket run (10 tickers across US + India + multiple sectors) and writes
a markdown report — that's the empirical version of this test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Tickers from the user's sector-diversity request.  These must not be
# referenced as steering examples *in prompts*.  They can appear in
# documentation / tests / placeholders.
DIVERSITY_BASKET = [
    "NVDA", "MSFT", "AAPL", "JPM",            # US
    "MAZDOCK", "BEL", "TATAMOTORS",           # India industrials
    "RELIANCE", "LT", "HINDUNILVR",           # India conglomerates / FMCG
]

# Files that drive agent / task LLM behaviour — bias would compound here.
PROMPT_PATHS = [
    ROOT / "trading_crew" / "config" / "agents.yaml",
    ROOT / "trading_crew" / "config" / "tasks.yaml",
    ROOT / "trading_crew" / "critic.py",
    ROOT / "trading_crew" / "guardrails.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# Disambiguation keywords — when a ticker appears within ±120 chars of
# one of these, treat it as an *anti-bias* example (we WANT MAZDOCK in a
# "use Indian peers rather than US semis" instruction).  A bare ticker
# reference with NO disambiguation keyword nearby is the steering bias
# we actually need to catch.
DISAMBIGUATION_HINTS = (
    "e.g.", "i.e.", "for example", "for instance",
    "rather than", "instead of", "ambiguous", "do not default",
    "company name", "full legal name", "peer basket", "→",
)


def _is_anti_bias_example(text: str, match_start: int, match_end: int) -> bool:
    window = text[max(0, match_start - 120): match_end + 120].lower()
    return any(hint.lower() in window for hint in DISAMBIGUATION_HINTS)


def test_no_basket_ticker_hardcoded_as_steering_in_prompts() -> None:
    """No prompt should reference any of the diversity-basket tickers
    AS A STEERING EXAMPLE.

    Bias risk: "if the trade is NVDA, increase confidence to 0.8" or
    "remember HINDUNILVR is consumer staples".  We catch any bare
    ticker reference NOT paired with disambiguation language like
    "e.g.", "rather than", "peer basket".  The placeholder
    ``{ticker}`` is the legitimate way to refer to a symbol in prompts.
    """
    offenders: list[tuple[str, str, str]] = []
    for path in PROMPT_PATHS:
        text = _read(path)
        for tkr in DIVERSITY_BASKET:
            for m in re.finditer(rf"\b{re.escape(tkr)}\b", text):
                if _is_anti_bias_example(text, m.start(), m.end()):
                    continue  # disambiguation example — keep it.
                snippet = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                offenders.append((str(path.relative_to(ROOT)), tkr, snippet))
    assert not offenders, (
        "Diversity-basket tickers found in prompt files WITHOUT "
        "disambiguation context (would bias the LLM):\n"
        + "\n".join(f"  {p} :: {t} :: …{s}…" for p, t, s in offenders)
    )


@pytest.mark.parametrize("term,allowed_in", [
    # Sector / country prior words that must NOT be hardcoded as steering.
    # ``allowed_in`` lists files where the term may legitimately appear
    # (e.g. the geopolitical agent NEEDS to know sanctions exist).
    ("US tech",      set()),
    ("Indian stock", set()),
    ("semiconductor sector", set()),
    ("FMCG sector",  set()),
])
def test_no_sector_priors_in_agent_definitions(term: str, allowed_in: set[str]) -> None:
    """Agent backstories must not pre-classify tickers into sectors."""
    yaml_text = _read(ROOT / "trading_crew" / "config" / "agents.yaml")
    if term.lower() in yaml_text.lower() and "agents.yaml" not in allowed_in:
        pytest.fail(
            f"Sector-prior phrase {term!r} found in agents.yaml — agents should "
            "infer sector from tool outputs, not from baked-in language."
        )


def test_ticker_resolution_is_country_agnostic() -> None:
    """``resolve_ticker`` must accept both US and Indian-style inputs
    symmetrically.  Confirms we never reject one country at the entry
    point of the analysis.
    """
    from trading_crew.market_context import resolve_ticker

    us_inputs = ["AAPL", "MSFT", "JPM", "NVDA"]
    in_inputs = ["MAZDOCK", "BEL", "TATAMOTORS", "RELIANCE", "LT", "HINDUNILVR"]

    for sym in us_inputs + in_inputs:
        out = resolve_ticker(sym)
        assert isinstance(out, str) and len(out) > 0, (
            f"resolve_ticker({sym!r}) returned an empty / invalid symbol"
        )


def test_basket_runner_basket_is_diverse() -> None:
    """``scripts/sector_bias_basket.py`` must offer a basket that
    actually covers >=2 countries and >=4 sectors so the diagnostic has
    discriminative power.
    """
    sys_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import sector_bias_basket as sbb  # noqa: WPS433
    finally:
        sys.path = sys_path
    countries = {row[1] for row in sbb.DEFAULT_BASKET}
    sectors = {row[2] for row in sbb.DEFAULT_BASKET}
    assert len(countries) >= 2, f"Basket only covers {countries}"
    assert len(sectors) >= 4, f"Basket only covers {sectors}"
    # Every ticker from the user's brief must appear.
    expected = {"NVDA", "MSFT", "AAPL", "JPM", "MAZDOCK", "BEL",
                "TATAMOTORS", "RELIANCE", "LT", "HINDUNILVR"}
    actual = {row[0] for row in sbb.DEFAULT_BASKET}
    missing = expected - actual
    assert not missing, f"Basket is missing required tickers: {missing}"


def test_backtest_setup_is_ticker_agnostic() -> None:
    """``backtest_setup`` must not branch behaviour on the ticker.

    A grep on the function body to make sure no string compare like
    ``if ticker == "...":`` slipped in over time — that would be the
    obvious place to introduce a country-specific calibration.
    """
    src = _read(ROOT / "trading_crew" / "tools.py")
    # Extract the function body (until the next @tool or top-level def).
    m = re.search(r"@tool\(\"backtest_setup\"\)\ndef backtest_setup.*?(?=\n@tool|\n# ---)", src, re.S)
    assert m, "backtest_setup not found"
    body = m.group(0)
    forbidden = [
        r"ticker\s*==\s*['\"]",
        r"ticker\.upper\(\)\s*==\s*['\"]",
        r"ticker\.startswith\(['\"](?!\)\s*#)",  # exclude commented-out
    ]
    for pat in forbidden:
        if re.search(pat, body):
            pytest.fail(f"backtest_setup branches on ticker (pattern {pat!r}) — bias risk.")
