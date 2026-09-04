"""Cross-cutting observability helpers (session tracing, search metrics)."""

from .trace import (  # noqa: F401
    TraceLogFilter,
    get_trace_id,
    record_search,
    reset_trace_id,
    set_trace_id,
    trace_scope,
)
