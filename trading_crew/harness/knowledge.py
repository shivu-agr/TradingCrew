"""Query ingested book chunks. Empty corpus is reported, not invented."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from trading_crew.agentic.memory.semantic import SemanticKnowledgeBase

from .ingest import chunk_store_path, ingest_corpus


def query_knowledge(
    query: str,
    *,
    cfg: Dict[str, Any],
    tags: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> str:
    q = (query or "").strip()
    if not q:
        return "query_knowledge: query is required"
    corpus: Path = cfg["corpus_dir"]
    store = chunk_store_path(corpus)
    if not store.is_file():
        ingest_corpus(corpus)
    if not store.is_file():
        return (
            f"query_knowledge: corpus at {corpus} has no ingested chunks. "
            "Place books or text under corpus_dir and re-run ingest."
        )
    kb = SemanticKnowledgeBase(store)
    top = kb.retrieve(q, k=int(k or cfg["knowledge_k"]), tags=tags)
    if not top:
        return f"query_knowledge: no chunk matched {q!r}"
    lines = [f"query_knowledge hits for {q!r}:"]
    for row in top:
        lines.append(row.citation())
        lines.append(row.doc.body.strip())
        lines.append("")
    return "\n".join(lines).strip()
