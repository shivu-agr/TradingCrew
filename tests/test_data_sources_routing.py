"""Unit tests for the ``trading_crew.data_sources.routing`` module.

The routing layer is the single decision point for "is this an Indian
ticker AND is the alternate source enabled?" — every tool call that
could go through screener.in or upstox reads it, so a regression here
would leak into every fundamentals / OHLCV path.
"""

from __future__ import annotations

import pytest

from trading_crew.data_sources import routing


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Every routing test starts with a fresh env snapshot.

    The config is frozen for the lifetime of the process by design (so
    a mid-run env flip doesn't split two parallel analysts across
    different sources), but the tests need to iterate over
    permutations rapidly.  We clear before and after so we don't leak
    a mocked config into an unrelated test.
    """
    routing.reset_config_cache()
    yield
    routing.reset_config_cache()


@pytest.mark.parametrize(
    "sym,expected",
    [
        ("RELIANCE.NS", True),
        ("500325.BO", True),
        ("BAJAJ-AUTO.NS", True),
        ("reliance.ns", True),           # case-insensitive
        ("AAPL", False),
        ("0700.HK", False),
        ("CL=F", False),
        ("", False),
        ("  RELIANCE.NS  ", True),       # whitespace-tolerant
    ],
)
def test_is_indian_ticker(sym, expected):
    assert routing.is_indian_ticker(sym) is expected


@pytest.mark.parametrize(
    "sym,expected",
    [
        ("RELIANCE.NS", "RELIANCE"),
        ("500325.BO", "500325"),
        ("BAJAJ-AUTO.NS", "BAJAJ-AUTO"),
        ("AAPL", "AAPL"),                # no suffix -> pass through
        ("reliance.ns", "RELIANCE"),     # uppercased
        ("", ""),
    ],
)
def test_strip_exchange_suffix(sym, expected):
    assert routing.strip_exchange_suffix(sym) == expected


def test_screener_enabled_defaults_to_on(monkeypatch):
    """A blank env should give the screener path — it needs no auth."""
    monkeypatch.delenv("TRADINGCREW_SCREENER_ENABLED", raising=False)
    monkeypatch.delenv("SCREENER_EMAIL", raising=False)
    monkeypatch.delenv("SCREENER_PASSWORD", raising=False)
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    cfg = routing.get_config()
    assert cfg.screener_enabled is True
    assert cfg.screener_email is None
    assert cfg.screener_password is None


def test_screener_disabled_via_env(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", "0")
    assert routing.get_config().screener_enabled is False


@pytest.mark.parametrize("val", ["false", "no", "off", "0"])
def test_screener_disabled_accepts_several_falsy_strings(monkeypatch, val):
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", val)
    routing.reset_config_cache()
    assert routing.get_config().screener_enabled is False


def test_upstox_needs_token_to_be_enabled(monkeypatch):
    """Even with the flag on, upstox is disabled unless a token is set."""
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "1")
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    cfg = routing.get_config()
    assert cfg.upstox_enabled is False
    assert cfg.upstox_analytics_token is None


def test_upstox_enabled_when_token_present(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "1")
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "test-token-abc")
    cfg = routing.get_config()
    assert cfg.upstox_enabled is True
    assert cfg.upstox_analytics_token == "test-token-abc"


def test_upstox_flag_off_disables_even_with_token(monkeypatch):
    """The explicit disable flag must always win over the token."""
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "0")
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "test-token-abc")
    cfg = routing.get_config()
    assert cfg.upstox_enabled is False
    assert cfg.upstox_analytics_token == "test-token-abc"


def test_config_is_cached_across_calls(monkeypatch):
    """Reading env once per process is deliberate — a mid-run flip
    would split two parallel analyst tasks across different sources.
    """
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", "1")
    first = routing.get_config()
    monkeypatch.setenv("TRADINGCREW_SCREENER_ENABLED", "0")
    second = routing.get_config()
    assert first is second, "get_config() must return the cached snapshot"
    assert second.screener_enabled is True, (
        "mid-run env flip must not affect the cached config"
    )
