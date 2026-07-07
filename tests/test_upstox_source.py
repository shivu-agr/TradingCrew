"""Unit tests for the upstox v3 client.

Every HTTP interaction is mocked at the ``requests.get`` boundary.
This gives us confidence that:

* the instrument-catalog CSV is parsed correctly (equity/index only,
  F&O rows filtered out) and cached per process,
* the historical-candle endpoint URL, auth header, and payload shape
  are consumed correctly,
* the option-chain endpoint pulls both call and put legs plus Greeks,
* the ``is_indian_and_upstox_available`` gate honours both the ticker
  suffix and the env-var config.
"""

from __future__ import annotations

import gzip
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_crew.data_sources import routing
from trading_crew.data_sources import upstox_source as us


# ---------------------------------------------------------------------------
# Shared fixtures — clean routing config & catalog on every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Force upstox to be *enabled* + the catalog cache to be empty.

    The routing module reads env vars once and caches, so we need to
    prime the env before ``get_config()`` runs for the first time in
    this test.
    """
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "1")
    monkeypatch.setenv("UPSTOX_ANALYTICS_TOKEN", "fake-token-for-tests")
    routing.reset_config_cache()
    us.clear_catalog_cache()
    yield
    us.clear_catalog_cache()
    routing.reset_config_cache()


# ---------------------------------------------------------------------------
# Instrument catalog
# ---------------------------------------------------------------------------


def _make_catalog_response(rows: list) -> MagicMock:
    """Build a mocked ``requests.get`` response for the catalog CSV.

    Upstox serves the catalog as a gzipped CSV; we mimic that end to
    end so the parser exercises the ``gzip.decompress`` path.
    """
    header = "tradingsymbol,instrument_key,instrument_type,name"
    body = "\n".join(
        [header] + [",".join(str(x) for x in row) for row in rows]
    )
    gz = gzip.compress(body.encode("utf-8"))
    r = MagicMock()
    r.status_code = 200
    r.content = gz
    return r


def test_load_exchange_catalog_filters_to_equity_and_index():
    """The catalog CSV has ~20k rows across equity/index/option/future
    — we only want equity + index.
    """
    rows = [
        ("RELIANCE", "NSE_EQ|INE002A01018", "EQUITY", "Reliance"),
        ("NIFTY", "NSE_INDEX|Nifty 50", "INDEX", "Nifty 50"),
        ("RELIANCE24DEC1400CE", "NSE_FO|123456", "CE", "Reliance Dec 1400 CE"),
        ("RELIANCE24DECFUT", "NSE_FO|999999", "FUT", "Reliance Dec fut"),
    ]
    with patch.object(us, "requests") as mock_requests:
        mock_requests.get.return_value = _make_catalog_response(rows)
        mapping = us._load_exchange_catalog("NSE")
    assert set(mapping.keys()) == {"RELIANCE", "NIFTY"}
    assert mapping["RELIANCE"] == "NSE_EQ|INE002A01018"


def test_load_exchange_catalog_raises_on_non_200():
    r = MagicMock()
    r.status_code = 503
    with patch.object(us, "requests") as mock_requests:
        mock_requests.get.return_value = r
        with pytest.raises(us.UpstoxError, match="HTTP 503"):
            us._load_exchange_catalog("NSE")


def test_resolve_instrument_key_hits_nse_by_default():
    with patch.object(us, "_load_exchange_catalog") as mock_load:
        mock_load.side_effect = lambda ex: {"RELIANCE": f"{ex}_EQ|INE002A01018"} if ex == "NSE" else {}
        key = us.resolve_instrument_key("RELIANCE.NS")
    assert key == "NSE_EQ|INE002A01018"


def test_resolve_instrument_key_hits_bse_on_bo_suffix():
    with patch.object(us, "_load_exchange_catalog") as mock_load:
        mock_load.side_effect = lambda ex: (
            {"500325": "BSE_EQ|INE002A01018"} if ex == "BSE" else {}
        )
        key = us.resolve_instrument_key("500325.BO")
    assert key == "BSE_EQ|INE002A01018"


def test_resolve_instrument_key_normalises_hyphen():
    """``BAJAJ-AUTO`` on upstox is stored without the hyphen — the
    resolver must fall back to the dash-stripped variant automatically.
    """
    with patch.object(us, "_load_exchange_catalog") as mock_load:
        mock_load.side_effect = lambda ex: (
            {"BAJAJAUTO": "NSE_EQ|INE917I01010"} if ex == "NSE" else {}
        )
        key = us.resolve_instrument_key("BAJAJ-AUTO.NS")
    assert key == "NSE_EQ|INE917I01010"


def test_resolve_instrument_key_raises_when_unknown():
    with patch.object(us, "_load_exchange_catalog", return_value={}):
        with pytest.raises(us.UpstoxNotFoundError):
            us.resolve_instrument_key("NOSUCH.NS")


def test_ensure_catalog_survives_one_exchange_failing():
    """If BSE fetch fails but NSE succeeds we should still be able to
    resolve NSE tickers — the client must not fail closed on a
    partial catalog outage.
    """
    def fake(exchange):
        if exchange == "BSE":
            raise us.UpstoxError("bse down")
        return {"RELIANCE": "NSE_EQ|INE002A01018"}
    with patch.object(us, "_load_exchange_catalog", side_effect=fake):
        key = us.resolve_instrument_key("RELIANCE.NS")
    assert key.startswith("NSE_EQ|")


# ---------------------------------------------------------------------------
# Historical candles
# ---------------------------------------------------------------------------


def _candle_payload(candles):
    """Wrap a list of candle rows in the upstox v3 envelope shape."""
    return {"status": "success", "data": {"candles": candles}}


def test_fetch_daily_candles_returns_ohlcv_dataframe():
    candles = [
        ["2026-06-30T00:00:00+05:30", 100.0, 105.0, 99.0, 104.0, 1000, 500],
        ["2026-07-01T00:00:00+05:30", 104.0, 108.0, 103.5, 107.0, 1200, 550],
    ]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _candle_payload(candles)

    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|INE002A01018"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            # Preserve utils namespace on the mock — production code calls
            # ``requests.utils.quote``.
            import requests as real_requests
            mock_requests.utils = real_requests.utils
            cf = us.fetch_daily_candles("RELIANCE.NS", lookback_days=30)

    assert isinstance(cf.df, pd.DataFrame)
    assert list(cf.df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(cf.df) == 2
    # Rows sorted ascending by date
    assert cf.df["Close"].iloc[-1] == pytest.approx(107.0)
    assert cf.instrument_key == "NSE_EQ|INE002A01018"


def test_fetch_daily_candles_raises_on_empty_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "data": {"candles": []}}
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|X"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            import requests as real_requests
            mock_requests.utils = real_requests.utils
            with pytest.raises(us.UpstoxError, match="empty candle list"):
                us.fetch_daily_candles("RELIANCE.NS")


def test_fetch_daily_candles_raises_auth_error_on_401():
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "Unauthorized"
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|X"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            import requests as real_requests
            mock_requests.utils = real_requests.utils
            with pytest.raises(us.UpstoxAuthError):
                us.fetch_daily_candles("RELIANCE.NS")


def test_fetch_daily_candles_raises_on_5xx():
    resp = MagicMock()
    resp.status_code = 503
    resp.text = "Service unavailable"
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|X"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            import requests as real_requests
            mock_requests.utils = real_requests.utils
            with pytest.raises(us.UpstoxError, match="HTTP 503"):
                us.fetch_daily_candles("RELIANCE.NS")


def test_fetch_daily_candles_sends_bearer_token():
    """Every historical-candle call must send the analytics token as a
    Bearer header — this is the entire security perimeter.
    """
    candles = [["2026-07-01T00:00:00+05:30", 1, 1, 1, 1, 1, 0]]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _candle_payload(candles)
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|X"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            import requests as real_requests
            mock_requests.utils = real_requests.utils
            us.fetch_daily_candles("RELIANCE.NS")
    kwargs = mock_requests.get.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer fake-token-for-tests"


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------


def _option_chain_payload():
    return {
        "status": "success",
        "data": [
            {
                "strike_price": 1300,
                "underlying_spot_price": 1305.5,
                "call_options": {
                    "market_data": {"ltp": 45.0, "oi": 3000, "volume": 12345},
                    "option_greeks": {"iv": 0.28, "delta": 0.6},
                },
                "put_options": {
                    "market_data": {"ltp": 12.5, "oi": 4500, "volume": 8000},
                    "option_greeks": {"iv": 0.30, "delta": -0.35},
                },
            },
            {
                "strike_price": 1350,
                "underlying_spot_price": 1305.5,
                "call_options": {
                    "market_data": {"ltp": 15.0, "oi": 5000, "volume": 20000},
                    "option_greeks": {"iv": 0.31, "delta": 0.3},
                },
                "put_options": {
                    "market_data": {"ltp": 42.0, "oi": 1500, "volume": 3000},
                    "option_greeks": {"iv": 0.32, "delta": -0.7},
                },
            },
        ],
    }


def test_fetch_option_chain_parses_puts_calls_and_spot():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _option_chain_payload()
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|INE002A01018"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            chain = us.fetch_option_chain("RELIANCE.NS", "2026-07-31")

    assert chain.underlying == "RELIANCE"
    assert chain.expiry == "2026-07-31"
    assert chain.spot == pytest.approx(1305.5)
    # Sorted ascending by strike
    assert [r.strike for r in chain.rows] == [1300.0, 1350.0]
    atm = chain.rows[0]
    assert atm.call_ltp == pytest.approx(45.0)
    assert atm.put_ltp == pytest.approx(12.5)
    assert atm.call_iv == pytest.approx(0.28)
    assert atm.put_delta == pytest.approx(-0.35)


def test_fetch_option_chain_raises_on_400():
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "Not F&O enabled"
    with patch.object(us, "resolve_instrument_key", return_value="NSE_EQ|X"):
        with patch.object(us, "requests") as mock_requests:
            mock_requests.get.return_value = resp
            with pytest.raises(us.UpstoxError, match="HTTP 400"):
                us.fetch_option_chain("SMALLCAP.NS", "2026-07-31")


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def test_render_candle_summary_contains_key_metrics():
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 108.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [104.0, 105.0, 107.0],
            "Volume": [1000, 1200, 1500],
        },
        index=pd.to_datetime(["2026-06-29", "2026-06-30", "2026-07-01"], utc=True),
    )
    cf = us.CandleFrame(
        df=df,
        instrument_key="NSE_EQ|X",
        unit="days",
        interval="1",
        from_date="2026-06-29",
        to_date="2026-07-01",
    )
    md = us.render_candle_summary(cf, "Reliance Industries Ltd", "RELIANCE.NS")
    assert "upstox v3, exchange-clean" in md
    assert "NSE_EQ|X" in md
    assert "Cumulative return" in md
    assert "| Date |" in md


def test_render_option_chain_reports_pcr_and_atm_strikes():
    chain = us.OptionChain(
        underlying="RELIANCE",
        expiry="2026-07-31",
        spot=1305.5,
        rows=[
            us.OptionChainRow(strike=1250, call_oi=1000, put_oi=2000),
            us.OptionChainRow(strike=1300, call_oi=3000, put_oi=4500),
            us.OptionChainRow(strike=1350, call_oi=5000, put_oi=1500),
        ],
    )
    md = us.render_option_chain_summary(chain, "Reliance Industries Ltd")
    assert "Put/Call OI ratio" in md
    assert "1300" in md
    assert "@ 2026-07-31" in md


# ---------------------------------------------------------------------------
# Gate: is_indian_and_upstox_available
# ---------------------------------------------------------------------------


def test_is_indian_and_upstox_available_true_when_configured():
    assert us.is_indian_and_upstox_available("RELIANCE.NS") is True
    assert us.is_indian_and_upstox_available("500325.BO") is True


def test_is_indian_and_upstox_available_false_for_non_indian():
    assert us.is_indian_and_upstox_available("AAPL") is False
    assert us.is_indian_and_upstox_available("0700.HK") is False


def test_is_indian_and_upstox_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_UPSTOX_ENABLED", "0")
    routing.reset_config_cache()
    assert us.is_indian_and_upstox_available("RELIANCE.NS") is False


def test_is_indian_and_upstox_available_false_when_token_missing(monkeypatch):
    monkeypatch.delenv("UPSTOX_ANALYTICS_TOKEN", raising=False)
    routing.reset_config_cache()
    assert us.is_indian_and_upstox_available("RELIANCE.NS") is False
