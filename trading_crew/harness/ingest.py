"""Chunk operator-fed books into the semantic store with provenance."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from trading_crew.agentic.memory.semantic import KnowledgeDoc, SemanticKnowledgeBase

_TEXT_SUFFIX = {".txt", ".md", ".text"}
_CHUNK = 1200
_INGEST_LOCK = threading.Lock()


class HarnessIngestError(ValueError):
    """Corpus path is missing or unreadable."""


def chunk_store_path(corpus_dir: Path) -> Path:
    return corpus_dir / ".chunks.jsonl"


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise HarnessIngestError(f"corpus_dir does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIX and path.name != ".chunks.jsonl":
            yield path


def _chunks(text: str, size: int = _CHUNK) -> List[str]:
    body = (text or "").strip()
    if not body:
        return []
    parts: List[str] = []
    buf: List[str] = []
    n = 0
    for para in body.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if n + len(p) > size and buf:
            parts.append("\n\n".join(buf))
            buf = [p]
            n = len(p)
        else:
            buf.append(p)
            n += len(p) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def ingest_corpus(corpus_dir: Path) -> int:
    """Chunk every book in ``corpus_dir`` into the shared JSONL store.

    Parallel harness kickoffs share this directory. The lock serialises
    the read-modify-write so one run cannot delete another's temp file
    (``[Errno 2] ... .chunks.jsonl.tmp``).
    """
    with _INGEST_LOCK:
        return _ingest_corpus_unlocked(corpus_dir)


def _ingest_corpus_unlocked(corpus_dir: Path) -> int:
    root = Path(corpus_dir)
    store = chunk_store_path(root)
    kb = SemanticKnowledgeBase(store)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for path in _iter_files(root):
        raw = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
        tag = path.stem.lower().replace(" ", "_")
        for i, chunk in enumerate(_chunks(raw)):
            doc_id = f"{path.stem}-{digest}-{i:03d}"
            kb.upsert(
                KnowledgeDoc(
                    doc_id=doc_id,
                    title=f"{path.name}#{i}",
                    body=chunk,
                    source_url=str(path.resolve()),
                    version_ts=now,
                    tags=[tag, "corpus"],
                )
            )
            n += 1
    return n
