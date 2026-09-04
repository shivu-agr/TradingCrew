"""get_sector_peers must accept a missing ticker on a live run.

The desk repeatedly called ``{}`` or ``{peers: ""}`` (AXISBANK,
BHARATFORG). CrewAI then rejected the call before the screen ran.
Omitted ticker binds the kickoff symbol; no run board → fail closed.
"""

from __future__ import annotations

from trading_crew.agentic.graph import blackboard, scratchpad
from trading_crew.tools import _resolve_run_ticker, get_sector_peers


def _clear_run_boards() -> None:
    blackboard.end_run()
    scratchpad.end_run()


def test_explicit_ticker_wins_over_board():
    _clear_run_boards()
    blackboard.begin_run("BHARATFORG.NS")
    try:
        assert _resolve_run_ticker("AXISBANK.NS") == "AXISBANK.NS"
    finally:
        _clear_run_boards()


def test_empty_ticker_binds_blackboard():
    _clear_run_boards()
    blackboard.begin_run("BHARATFORG.NS")
    try:
        assert _resolve_run_ticker("") == "BHARATFORG.NS"
        assert _resolve_run_ticker("   ") == "BHARATFORG.NS"
    finally:
        _clear_run_boards()


def test_empty_ticker_binds_scratchpad_when_board_absent():
    _clear_run_boards()
    scratchpad.begin_run("AXISBANK.NS")
    try:
        assert _resolve_run_ticker("") == "AXISBANK.NS"
    finally:
        _clear_run_boards()


def test_empty_ticker_without_run_is_fail_closed():
    _clear_run_boards()
    assert _resolve_run_ticker("") == ""
    assert _resolve_run_ticker("   ") == ""


def test_tool_empty_args_without_run_does_not_screen():
    _clear_run_boards()
    out = get_sector_peers.run()
    assert "needs ticker" in out
    assert "No peer screen was run" in out


def test_tool_empty_args_on_live_run_uses_kickoff_symbol(monkeypatch):
    _clear_run_boards()
    blackboard.begin_run("BHARATFORG.NS")
    seen: dict[str, str] = {}

    def _fake(ticker: str, peers: str = "") -> str:
        seen["ticker"] = ticker
        seen["peers"] = peers
        return f"PEER SCREEN {ticker}"

    monkeypatch.setattr("trading_crew.tools._get_sector_peers_impl", _fake)
    try:
        out = get_sector_peers.run()
        assert out == "PEER SCREEN BHARATFORG.NS"
        assert seen == {"ticker": "BHARATFORG.NS", "peers": ""}
        out2 = get_sector_peers.run(peers="")
        assert out2 == "PEER SCREEN BHARATFORG.NS"
    finally:
        _clear_run_boards()
