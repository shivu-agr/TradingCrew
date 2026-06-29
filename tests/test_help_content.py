"""Audit: every ``data-help`` anchor used in the frontend has a definition.

Walks ``web/frontend/index.html`` and ``web/frontend/diagram.js``,
extracts every ``data-help="..."`` (or ``data-help-tip="..."``) anchor,
plus every ``glossary.*`` token referenced inside ``auto_glossary.js``,
and verifies each one resolves to an entry in ``help_content.js``.

Catches the common mistake of adding a new info-icon to the UI without
remembering to also add the matching help entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


FRONTEND = Path(__file__).resolve().parent.parent / "web" / "frontend"


def _extract_anchor_keys() -> set[str]:
    """Read all keys defined in ``help_content.js``."""
    src = (FRONTEND / "help_content.js").read_text()
    # Match top-level entries:   "tab.workflow": { … },
    return set(re.findall(r'"([a-z][a-zA-Z0-9_.]*)"\s*:\s*\{', src))


def _data_help_in(path: Path) -> set[str]:
    text = path.read_text() if path.exists() else ""
    # data-help="…" and data-help-tip="…" (single quotes too, just in case)
    return set(re.findall(r'data-help(?:-tip)?\s*=\s*["\']([a-z][a-zA-Z0-9_.]+)["\']', text))


def _glossary_anchors_in_auto_glossary() -> set[str]:
    p = FRONTEND / "auto_glossary.js"
    if not p.exists():
        return set()
    txt = p.read_text()
    # Tokens like glossary.kelly that the auto-annotator emits.
    return set(re.findall(r'glossary\.[a-z0-9_]+', txt))


def _agent_help_anchors_in_diagram() -> set[str]:
    p = FRONTEND / "diagram.js"
    if not p.exists():
        return set()
    txt = p.read_text()
    # Pairs in AGENT_HELP_ANCHORS like:   market_analyst: "agent.market_analyst"
    return set(re.findall(r'"((?:agent|panel)\.[a-zA-Z0-9_]+)"', txt))


def test_every_data_help_anchor_has_a_help_entry():
    defined = _extract_anchor_keys()
    used: set[str] = set()
    used |= _data_help_in(FRONTEND / "index.html")
    used |= _data_help_in(FRONTEND / "app.js")
    used |= _glossary_anchors_in_auto_glossary()
    used |= _agent_help_anchors_in_diagram()

    missing = sorted(used - defined)
    assert not missing, (
        f"{len(missing)} data-help anchor(s) referenced in the frontend "
        f"have no entry in help_content.js: {missing}"
    )


def test_help_content_entries_have_required_fields():
    """Every entry should at least carry a title + short blurb.

    We check by anchor key: for each defined key, find the block that
    starts with that key and confirm it contains both ``title:`` and
    ``short:`` declarations within the next ~2KB of source.
    """
    src = (FRONTEND / "help_content.js").read_text()
    for key in _extract_anchor_keys():
        idx = src.find(f'"{key}"')
        assert idx != -1, f"Anchor key {key} missing from source"
        window = src[idx:idx + 2000]
        assert "title:" in window, f"Missing title in entry for {key}"
        assert "short:" in window, f"Missing short blurb in entry for {key}"
