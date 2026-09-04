"""Session tracing + per-backend web-search metrics.

A single ``trace_id`` identifies one logical unit of work — a batch/stream
(all its tickers share the batch ``job_id``) or a standalone interactive run
(its ``run_id``).  It lets us:

* group every log line for a session (``[<trace_id>]`` prefix, see
  :class:`TraceLogFilter` and ``web/backend/logging_setup.py``), and
* attribute every web-search call to the session that made it, counted per
  backend in the ``search_stats`` DB table (see :func:`record_search`).

Propagation is free
===================
``trace_id`` is a :class:`contextvars.ContextVar`.  It is set once when a run
begins (see ``SessionManager._drive``) and rides through every layer without
manual threading:

* ``asyncio.create_task`` copies the current context, so the dispatcher task
  created inside ``_drive`` inherits it.
* ``asyncio.to_thread`` copies the context into the worker thread, so the
  CrewAI kickoff sees it.
* CrewAI's ``Task.execute_async`` runs each parallel analyst on a thread it
  spawns via ``contextvars.copy_context().run(...)`` — so a search made deep
  inside a parallel analyst tool still reports the run's ``trace_id``.

No fallback / process-global is needed: each concurrent run keeps its own
``trace_id`` in its own context.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_TRACE_ID: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "trace_id", default=""
)

# The outcomes we count per (trace_id, backend).  Kept small + fixed so the
# dashboard can render a stable set of columns.
SEARCH_OUTCOMES = ("success", "empty", "throttled", "error")


def get_trace_id() -> str:
    """Return the active trace id, or ``""`` when running outside a session."""
    return _TRACE_ID.get()


def set_trace_id(trace_id: str) -> "contextvars.Token[str]":
    """Set the active trace id; returns a token for :func:`reset_trace_id`."""
    return _TRACE_ID.set(trace_id or "")


def reset_trace_id(token: "contextvars.Token[str]") -> None:
    try:
        _TRACE_ID.reset(token)
    except (ValueError, LookupError):
        # Token from a different context (e.g. reset on a different thread) —
        # nothing we can safely do; leave the var as-is.
        pass


@contextmanager
def trace_scope(trace_id: str) -> Iterator[str]:
    """Bind ``trace_id`` for the duration of the ``with`` block."""
    token = set_trace_id(trace_id)
    try:
        yield get_trace_id()
    finally:
        reset_trace_id(token)


class TraceLogFilter(logging.Filter):
    """Stamps every log record with the active ``trace_id``.

    Attached to the root handlers so the format string can reference
    ``%(trace_id)s`` without third-party records (which never set the attr)
    blowing up formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = get_trace_id() or "-"
        return True


def record_search(backend: str, outcome: str, n: int = 1) -> None:
    """Increment the per-session web-search counter for ``backend``/``outcome``.

    A no-op when there is no active trace (e.g. an ad-hoc CLI search) or when
    the DB write fails — metrics must never break a search.
    """
    trace_id = get_trace_id()
    if not trace_id or not backend or not outcome:
        return
    try:
        from web.backend import prediction_db as db

        db.bump_search_stat(trace_id, backend, outcome, n)
    except Exception:  # noqa: BLE001 — telemetry is best-effort
        logger.debug("record_search(%s, %s) failed", backend, outcome, exc_info=True)
