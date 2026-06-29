"""Logging configuration for the TradingCrew web server.

Centralises the logger wiring so every entry point (the FastAPI app,
ad-hoc scripts that import the runner, the SDK harness) gets the same
behaviour: stdout for tail-friendly live monitoring, and a rotating
file under ``logs/`` for after-the-fact triage.

Why a rotating file
===================
A long-running ``uvicorn`` process can easily produce hundreds of MB of
output across a day of debate / risk / memory analyze calls.  Tailing
``screen``/``stdout`` works for the most recent few minutes but loses
older context as the scrollback rolls.  Persisting to disk with size-
based rotation means:

* The latest activity is always visible in ``logs/web.log``.
* Historical context is kept in ``logs/web.log.1`` … ``logs/web.log.N``
  with N = ``backup_count``.
* Total disk footprint is bounded by ``(N+1) * max_bytes`` so it can't
  grow without limit even if the user forgets about it.

What we attach to
=================
We install one ``RotatingFileHandler`` (plus a stdout ``StreamHandler``
if one isn't already there) on the **root** logger so every module
that uses ``logging.getLogger(__name__)`` picks it up automatically.
Uvicorn ships with non-propagating loggers — ``uvicorn``,
``uvicorn.error``, ``uvicorn.access`` — so we flip their ``propagate``
flag back on and clear their default stdout handler.  That way
uvicorn's request log and the app's own logger both end up in the
same file with consistent formatting.

Env knobs
=========
* ``TRADINGCREW_LOG_DIR``        — directory for log files (default: ``logs/``
  next to the project root).
* ``TRADINGCREW_LOG_LEVEL``      — root log level (default: ``INFO``).
* ``TRADINGCREW_LOG_MAX_BYTES``  — per-file size cap before rotation
  (default: 20 MB).
* ``TRADINGCREW_LOG_BACKUP_COUNT`` — how many rotated backups to keep
  (default: 5 → up to ~120 MB total with the default size).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DEFAULT_LEVEL = "INFO"
_DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_DEFAULT_BACKUP_COUNT = 5
_DEFAULT_FILENAME = "web.log"

# Module-level flag so repeated calls (e.g. uvicorn --reload re-importing
# ``web.backend.app``) don't keep stacking duplicate handlers on the
# root logger.  Each successive process gets a fresh handler set;
# subsequent calls inside the same process are no-ops.
_configured = False


def configure_logging(
    log_dir: Optional[str | os.PathLike] = None,
    level: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    filename: str = _DEFAULT_FILENAME,
) -> Path:
    """Install stdout + rotating-file logging on the root logger.

    Idempotent within a single process.  Returns the absolute path of
    the active log file so callers can echo it on startup (handy for
    the user to know where to ``tail -f``).
    """
    global _configured

    log_dir = Path(
        log_dir
        or os.environ.get("TRADINGCREW_LOG_DIR")
        or (_project_root() / "logs")
    ).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = (
        level
        or os.environ.get("TRADINGCREW_LOG_LEVEL")
        or _DEFAULT_LEVEL
    ).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    max_bytes = int(
        max_bytes
        or os.environ.get("TRADINGCREW_LOG_MAX_BYTES")
        or _DEFAULT_MAX_BYTES
    )
    backup_count = int(
        backup_count
        if backup_count is not None
        else os.environ.get(
            "TRADINGCREW_LOG_BACKUP_COUNT", _DEFAULT_BACKUP_COUNT
        )
    )

    log_path = log_dir / filename
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    root = logging.getLogger()
    root.setLevel(level_value)

    if not _configured:
        # First time in this process — install our handlers.
        # We deliberately replace whatever ``logging.basicConfig`` left
        # behind so we own the formatter and don't double-print every
        # line (basicConfig adds a plain StreamHandler).
        for handler in list(root.handlers):
            root.removeHandler(handler)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_value)
        root.addHandler(file_handler)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level_value)
        root.addHandler(stream_handler)

        _reattach_uvicorn_loggers(level_value)
        _configured = True
    else:
        # Subsequent call — only update the level if the caller asked
        # for a different one (e.g. tests bumping to DEBUG).
        for handler in root.handlers:
            handler.setLevel(level_value)

    return log_path


def _reattach_uvicorn_loggers(level: int) -> None:
    """Make uvicorn's loggers propagate up to root.

    By default ``uvicorn`` and ``uvicorn.access`` each ship a stdout
    handler with ``propagate=False`` — which means our root handler
    never sees their messages and the rotated file would be missing
    request logs / startup banners.  Clearing their handlers and
    enabling propagation routes everything through the root logger,
    so the same file picks up app logs AND uvicorn's lifecycle output.
    """
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
        lg.propagate = True
        lg.setLevel(level)


def _project_root() -> Path:
    """Walk up from this file (web/backend/logging_setup.py → project root)."""
    return Path(__file__).resolve().parent.parent.parent
