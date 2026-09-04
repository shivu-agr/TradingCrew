"""Shared bull/bear scratch pad — compact debate memory, no report redump."""

from __future__ import annotations

from trading_crew.agentic.graph.scratchpad import (
    ScratchPad,
    begin_run,
    compact_excerpt,
    current,
    end_run,
)


def test_compact_excerpt_prefers_tldr_and_stops_at_desk_bias():
    body = "TLDR:\n- catalyst A\nDESK_BIAS: BULLISH\n\n" + ("x" * 2000)
    assert compact_excerpt(body) == "- catalyst A"
    assert "x" * 50 not in compact_excerpt(body)


def test_compact_excerpt_falls_back_to_head():
    assert compact_excerpt("plain body " * 200, limit=20) == ("plain body " * 200)[:20]


def test_seed_upsert_and_debate_turns():
    pad = ScratchPad(ticker="NTNX")
    pad.seed_analyst("News Analyst", "bearish", "TLDR:\n- bad print\nDESK_BIAS: BEARISH\n\nlong")
    pad.seed_analyst("News Analyst", "bearish", "TLDR:\n- updated\nDESK_BIAS: BEARISH")
    seeds = [e for e in pad.entries if e.kind == "seed"]
    assert len(seeds) == 1
    assert "updated" in seeds[0].text
    assert "BEARISH" in seeds[0].text

    pad.append_debate("Bullish Researcher", 1, "long thesis")
    pad.append_debate("Bullish Researcher", 1, "revised thesis")
    debates = [e for e in pad.entries if e.kind == "debate"]
    assert len(debates) == 1
    assert debates[0].text == "revised thesis"

    pad.write_note("bull", "new 8-K [source: sec.gov]")
    rendered = pad.render()
    assert "DEBATE SCRATCH PAD (NTNX)" in rendered
    assert "updated" in rendered
    assert "revised thesis" in rendered
    assert "new 8-K" in rendered


def test_contextvar_begin_end():
    end_run()
    assert current() is None
    pad = begin_run("AAPL")
    assert current() is pad
    pad.write_note("bear", "note")
    assert "note" in current().render()
    end_run()
    assert current() is None


def test_scratchpad_tools_roundtrip():
    from trading_crew.tools import read_scratchpad, write_scratchpad

    end_run()
    begin_run("NTNX")
    try:
        empty = read_scratchpad.func()  # type: ignore[attr-defined]
        assert "Empty" in empty or "empty" in empty.lower() or "Analyst TLDRs" not in empty
        msg = write_scratchpad.func("fresh claim", author="bull")  # type: ignore[attr-defined]
        assert "updated" in msg.lower()
        out = read_scratchpad.func()  # type: ignore[attr-defined]
        assert "fresh claim" in out
    finally:
        end_run()
