"""M3 — Semantic KB: upsert, removal, retrieval with provenance."""

from __future__ import annotations

import pytest

from trading_crew.agentic.memory.semantic import (
    KnowledgeDoc,
    SemanticKnowledgeBase,
)


def _doc(**over) -> KnowledgeDoc:
    base = {
        "doc_id": "doc-001",
        "title": "Post-earnings drift effect",
        "body": "Stocks that beat earnings tend to drift upward for 30-60 days.",
        "source_url": "https://example.com/paper.pdf",
        "version_ts": "2025-12-01T00:00:00+00:00",
        "tags": ["earnings", "drift"],
    }
    base.update(over)
    return KnowledgeDoc(**base)


def test_upsert_writes_doc_and_iter_returns_it(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc())
    docs = kb.all_docs()
    assert len(docs) == 1
    assert docs[0].doc_id == "doc-001"


def test_upsert_replaces_existing_doc_by_id(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc(title="v1"))
    kb.upsert(_doc(title="v2"))
    assert len(kb.all_docs()) == 1
    assert kb.all_docs()[0].title == "v2"


def test_remove_returns_true_when_doc_existed(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc())
    assert kb.remove("doc-001") is True
    assert kb.all_docs() == []


def test_remove_returns_false_when_doc_absent(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    assert kb.remove("never-existed") is False


def test_retrieval_ranks_relevant_docs_first(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc(doc_id="drift", title="Post-earnings drift",
                   body="Stocks that beat tend to drift upward for weeks."))
    kb.upsert(_doc(doc_id="vol", title="Volatility clustering",
                   body="High-vol days cluster together in time series.",
                   tags=["vol"]))
    results = kb.retrieve("earnings beat momentum drift")
    assert results[0].doc.doc_id == "drift"


def test_retrieval_filters_by_tag(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc(doc_id="d1", tags=["earnings"]))
    kb.upsert(_doc(doc_id="d2", tags=["macro"]))
    macro_only = kb.retrieve("earnings drift effect", tags=["macro"])
    # d1 has high keyword overlap but wrong tag — should be filtered out
    assert all(r.doc.doc_id != "d1" for r in macro_only)


def test_retrieved_doc_carries_provenance(tmp_path):
    kb = SemanticKnowledgeBase(tmp_path / "kb.jsonl")
    kb.upsert(_doc())
    results = kb.retrieve("earnings drift")
    assert len(results) == 1
    citation = results[0].citation()
    assert "doc-001" in citation
    assert "2025-12-01" in citation
    assert "https://example.com/paper.pdf" in citation
