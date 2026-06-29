"""Semantic knowledge base — paper §4.3 (curated + provenance).

Episodic memory captures *what happened*.  Semantic memory captures *what
the agent is supposed to know* — risk policy, market microstructure
reference data, named "playbooks" for known patterns (earnings drift,
post-FOMC reactions, etc.).

The single most important property of a semantic KB for agentic trading is
**provenance**: every retrieved fact must trace back to an identifiable
document with a version timestamp, so a downstream auditor can verify
whether the LLM hallucinated a number or genuinely cited a source.  That
is the gap paper §13.1.1 ("Hallucination Risk") highlights.

This module is a deliberately small first step:

- Documents are stored as JSONL with ``(doc_id, version_ts, source_url,
  title, body, tags)``.
- Retrieval is the same TF-IDF cosine used by episodic memory.
- Retrieved chunks return *with* their provenance metadata so the calling
  agent can cite them in its reasoning.

Future work (left for a later milestone): chunking long documents,
embedding-based retrieval, scheduled re-ingestion with diff detection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .episodic import _cosine, _tf_vector, _tokenise

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDoc:
    """A single curated knowledge document with provenance metadata."""

    doc_id: str
    title: str
    body: str
    source_url: str
    version_ts: str
    tags: List[str] = field(default_factory=list)
    ingested_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeDoc":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            body=data["body"],
            source_url=data["source_url"],
            version_ts=data["version_ts"],
            tags=list(data.get("tags", [])),
            ingested_ts=data.get("ingested_ts", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class RetrievedDoc:
    """A document plus the similarity score that ranked it."""

    doc: KnowledgeDoc
    similarity: float

    def citation(self) -> str:
        """Render a short citation line for inclusion in agent prompts."""
        return f"[{self.doc.doc_id} @ {self.doc.version_ts}] {self.doc.title} — {self.doc.source_url}"


class SemanticKnowledgeBase:
    """JSONL-backed semantic store with TF-IDF retrieval."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- writes

    def upsert(self, doc: KnowledgeDoc) -> None:
        """Insert or replace a document by ``doc_id``."""
        docs = list(self.iter_all())
        replaced = False
        for i, d in enumerate(docs):
            if d.doc_id == doc.doc_id:
                docs[i] = doc
                replaced = True
                break
        if not replaced:
            docs.append(doc)
        self._rewrite(docs)

    def remove(self, doc_id: str) -> bool:
        docs = list(self.iter_all())
        new_docs = [d for d in docs if d.doc_id != doc_id]
        if len(new_docs) == len(docs):
            return False
        self._rewrite(new_docs)
        return True

    # -------------------------------------------------------------- reads

    def iter_all(self) -> Iterable[KnowledgeDoc]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield KnowledgeDoc.from_dict(json.loads(line))

    def all_docs(self) -> List[KnowledgeDoc]:
        return list(self.iter_all())

    # -------------------------------------------------------------- retrieval

    def retrieve(self, query: str, *, k: int = 3, tags: Optional[Sequence[str]] = None) -> List[RetrievedDoc]:
        """Return top-``k`` matching documents, optionally filtered by tag."""
        query_vec = _tf_vector(_tokenise(query))

        scored: List[RetrievedDoc] = []
        for d in self.iter_all():
            if tags and not (set(tags) & set(d.tags)):
                continue
            doc_vec = _tf_vector(_tokenise(d.title + "\n" + d.body))
            sim = _cosine(query_vec, doc_vec)
            if sim <= 0:
                continue
            scored.append(RetrievedDoc(doc=d, similarity=sim))

        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:k]

    # -------------------------------------------------------------- internals

    def _rewrite(self, docs: Sequence[KnowledgeDoc]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for d in docs:
                    fh.write(json.dumps(d.to_dict(), sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("could not persist semantic KB to %s: %s", self.path, exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise
