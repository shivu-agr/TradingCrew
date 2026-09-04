"""Process memory hygiene for the long-running batch.

Everything in this app runs inside one Python process (FastAPI + the Kafka
workflow consumers + up to ``WORKERS`` concurrent CrewAI crews, or the nightly
``scale collect`` CLI).  Over a multi-hour 200-name batch, freed Python objects
do not reliably hand their pages back to the OS: CPython keeps object arenas,
and glibc's per-thread malloc arenas fragment under the crews' thread pools.
The net effect is a resident set that climbs monotonically until the VM OOMs.

``release_memory()`` is the cheap counter-measure to call once a run/name
finishes: force a GC pass so the arenas are actually empty, then ask glibc to
trim the top of the heap back to the OS via ``malloc_trim(0)``.  Pair it with
``MALLOC_ARENA_MAX``/``MALLOC_TRIM_THRESHOLD_`` (set in run_web.sh)
which bound how badly the heap can fragment in the first place.

Everything here is strictly best-effort — a failure must never affect a run.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging

logger = logging.getLogger(__name__)

# Resolve libc.malloc_trim once (Linux/glibc only).  ``None`` elsewhere (macOS,
# musl) — the GC pass still runs, we just skip the trim.
_malloc_trim = None
try:  # pragma: no cover - platform dependent
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name)
    if hasattr(_libc, "malloc_trim"):
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
        _malloc_trim = _libc.malloc_trim
except Exception:  # noqa: BLE001 - trim is optional
    _malloc_trim = None


def release_memory(*, collect: bool = True) -> None:
    """Return freed heap to the OS.  Best-effort; safe to call anywhere."""
    try:
        if collect:
            gc.collect()
        if _malloc_trim is not None:
            _malloc_trim(0)
    except Exception:  # noqa: BLE001 - never break a run over cleanup
        logger.debug("release_memory failed (non-fatal)", exc_info=True)


def close_crew_memory(crew) -> None:
    """Tear down a finished crew's CrewAI Memory so a batch doesn't leak
    per-run resources.

    ``trading_crew.crew`` builds a fresh ``Memory`` (LanceDB-backed) *and*
    replaces its 1-worker save pool with an 8-worker ``ThreadPoolExecutor``
    for every run.  Neither is shut down when the run ends, so over a 200-name
    batch the process accumulates hundreds of live ``memory-save`` threads
    (each pinning the run's submitted work + results) plus LanceDB handles —
    the RSS creep that eventually OOMs the VM.  Explicitly shutting the pool
    and calling ``Memory.close()`` releases them; ``crew.memory`` then GCs.

    Best-effort and idempotent — any failure here must never affect the run.
    """
    if crew is None:
        return
    mem = getattr(crew, "memory", None) or getattr(crew, "_memory", None)
    if mem is None:
        return
    # 1) Our custom parallel save pool (set in crew.py).  cancel_futures so we
    #    don't block on any late analyze() work still queued.
    pool = getattr(mem, "_save_pool", None)
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # older Pythons lack cancel_futures
            try:
                pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    # 2) CrewAI Memory.close() releases the LanceDB storage handles.
    for meth in ("close",):
        fn = getattr(mem, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                logger.debug("crew memory %s() failed (non-fatal)", meth, exc_info=True)
