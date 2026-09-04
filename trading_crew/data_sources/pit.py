"""Point-in-time (PIT) context + leakage guard (Phase 1).

The offline backfill harness re-runs the crew "as of" a historical
decision date so a decision dated 2024-09-01 sees **only** information
that was public on 2024-09-01.  For that to be valid, every data tool
must know the ``as_of`` cutoff and must refuse to hand back anything
published after it.

This module provides three things:

1. ``AsOfContext`` + a ``contextvars`` slot so the cutoff can be set once
   (per backfill cell / per request) and read by any tool deep in the
   call stack without threading it through every signature.
2. ``LeakageGuard`` helpers — ``assert_before_cutoff`` (fail-closed) and
   ``is_before_cutoff`` (soft filter) — that tools call on the publish
   timestamp of any datum before returning it.
3. A tiny on-disk **snapshot store** so an expensive scrape (screener.in)
   is fetched once and reused across every backfill cell for that symbol
   rather than re-hitting the site.

Live mode (``as_of=None``) makes every guard a no-op, so wiring this into
the tool layer has zero behavioural change for normal runs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Raised in backfill mode when a datum cannot be proven PIT-safe.

    Fail-closed: if we cannot establish that a piece of information was
    public on or before ``as_of``, we treat it as a leak and refuse it,
    rather than silently letting future information into a historical
    decision.
    """


@dataclass(frozen=True)
class AsOfContext:
    """The point-in-time cutoff for the current unit of work.

    - ``as_of``: only information public strictly-before or at this
      instant may be used.  ``None`` = live mode (no restriction).
    - ``mode``: ``"live"`` | ``"backfill"``.  In ``"backfill"`` the
      guards fail closed; in ``"live"`` they are no-ops.
    - ``strict``: when True (default in backfill) a datum with an unknown
      publish date is rejected.  Kept configurable so a caller can opt
      into a looser policy explicitly (never implicitly).
    """

    as_of: Optional[datetime] = None
    mode: str = "live"
    strict: bool = True

    @property
    def is_backfill(self) -> bool:
        return self.mode == "backfill" and self.as_of is not None


_LIVE = AsOfContext(as_of=None, mode="live", strict=False)
_ctx: contextvars.ContextVar[AsOfContext] = contextvars.ContextVar("as_of_ctx", default=_LIVE)


def get_context() -> AsOfContext:
    """Return the current PIT context (live by default)."""
    return _ctx.get()


def set_context(ctx: AsOfContext) -> contextvars.Token:
    """Set the PIT context, returning a token to restore it later."""
    return _ctx.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


class as_of_scope:
    """Context manager that installs an ``AsOfContext`` for its body.

    Usage::

        with as_of_scope(datetime(2024, 9, 1), mode="backfill"):
            run_crew(...)   # every tool inside sees the cutoff
    """

    def __init__(self, as_of: Optional[datetime], *, mode: str = "backfill", strict: bool = True):
        self._ctx = AsOfContext(as_of=_normalize_dt(as_of), mode=mode, strict=strict)
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> AsOfContext:
        self._token = set_context(self._ctx)
        return self._ctx

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            reset_context(self._token)


def is_backfill() -> bool:
    return get_context().is_backfill


def cutoff() -> Optional[datetime]:
    """Return the current as_of cutoff (tz-aware UTC) or ``None`` in live mode."""
    return get_context().as_of


# ---------------------------------------------------------------------------
# Datetime normalisation
# ---------------------------------------------------------------------------


def _normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# A permissive set of publish-date formats seen across news providers and
# scraped feeds.  Anything we can't parse is treated as "unknown" and, in
# strict backfill mode, rejected.
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?")


def parse_publish_date(value) -> Optional[datetime]:
    """Best-effort parse of a provider-supplied publish timestamp.

    Returns a tz-aware UTC ``datetime`` or ``None`` when the value can't
    be interpreted.  Handles ISO-8601, ``MM/DD/YYYY[, HH:MM ...]`` (SerpAPI
    Google-News), and epoch seconds.  Deliberately conservative: an
    ambiguous or relative value ("15h ago") returns ``None`` so the guard
    can fail closed rather than guess.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_dt(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # ISO-8601 (Tavily ``published_date`` is typically ISO).
    try:
        iso = s.replace("Z", "+00:00")
        return _normalize_dt(datetime.fromisoformat(iso))
    except ValueError:
        pass
    m = _ISO_RE.search(s)
    if m:
        frag = m.group(0).replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return _normalize_dt(datetime.strptime(frag, fmt))
            except ValueError:
                continue
    # SerpAPI Google-News: "06/24/2024, 07:00 AM, +0000 UTC" or "06/24/2024".
    m2 = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m2:
        mm, dd, yyyy = (int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        try:
            return datetime(yyyy, mm, dd, tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def is_before_cutoff(published) -> bool:
    """True if ``published`` is safe to use under the current context.

    Live mode → always True.  Backfill mode → True only when the parsed
    publish date is on/before the cutoff.  An unknown date is unsafe in
    strict mode and safe otherwise.
    """
    ctx = get_context()
    if not ctx.is_backfill:
        return True
    dt = published if isinstance(published, datetime) else parse_publish_date(published)
    dt = _normalize_dt(dt)
    if dt is None:
        return not ctx.strict
    return dt <= ctx.as_of


def assert_before_cutoff(published, *, what: str = "datum") -> None:
    """Raise :class:`LeakageError` if ``published`` is not PIT-safe.

    No-op in live mode.  Use for hard boundaries (e.g. an OHLCV bar) where
    a leak must abort rather than be silently dropped.
    """
    if is_before_cutoff(published):
        return
    ctx = get_context()
    raise LeakageError(
        f"PIT leak blocked: {what} (published={published!r}) is after the "
        f"as_of cutoff {ctx.as_of.isoformat() if ctx.as_of else None}."
    )


def filter_before_cutoff(rows, *, key: str = "published"):
    """Return only the ``rows`` whose ``row[key]`` is PIT-safe.

    Soft version of :func:`assert_before_cutoff` for list-shaped feeds
    (news, filings).  Live mode returns the rows unchanged.
    """
    if not get_context().is_backfill:
        return rows
    return [r for r in rows if is_before_cutoff(r.get(key) if isinstance(r, dict) else None)]


# ---------------------------------------------------------------------------
# On-disk snapshot store (avoid re-scraping the same source per backfill cell)
# ---------------------------------------------------------------------------


# Snapshots are stored in the ``pit_snapshots`` table of the shared
# ``prediction.db`` (keyed by namespace+key), not on the filesystem.
_LEGACY_PIT_MIGRATED = False


def _normalize_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key.strip().upper())


def _legacy_snapshot_root() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "pit_snapshots"


def _migrate_legacy_snapshots() -> None:
    """Import legacy on-disk snapshot JSON files into the DB once, then move
    the tree aside so it's no longer read."""
    global _LEGACY_PIT_MIGRATED
    if _LEGACY_PIT_MIGRATED:
        return
    _LEGACY_PIT_MIGRATED = True
    root = _legacy_snapshot_root()
    if not root.is_dir():
        return
    try:
        from web.backend import prediction_db as db

        count = 0
        for path in root.glob("*/*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ns = record.get("namespace") or path.parent.name
            key = record.get("key") or path.stem
            payload = record.get("payload")
            if payload is not None:
                db.import_pit_snapshot(
                    ns, _normalize_key(key), record.get("captured_ts"), payload
                )
                count += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root.rename(root.parent / f"pit_snapshots.migrated-{ts}")
        if count:
            logger.info("imported %d PIT snapshots into the DB", count)
    except Exception as exc:  # noqa: BLE001 — migration is best-effort
        logger.warning("legacy PIT snapshot import failed: %s", exc)


def save_snapshot(namespace: str, key: str, payload: dict) -> None:
    """Persist a scraped payload keyed by ``(namespace, key)`` in the DB."""
    from web.backend import prediction_db as db

    db.save_pit_snapshot(namespace, _normalize_key(key), payload)


def load_snapshot(namespace: str, key: str, *, max_age_days: Optional[float] = None) -> Optional[dict]:
    """Load a previously-saved snapshot payload from the DB, or ``None``.

    ``max_age_days`` optionally rejects a snapshot older than the bound
    (used in live mode where staleness matters); in backfill the caller
    typically passes ``None`` because the whole point is to reuse one
    capture across every historical cell for that symbol.
    """
    from web.backend import prediction_db as db

    _migrate_legacy_snapshots()
    record = db.load_pit_snapshot(namespace, _normalize_key(key))
    if not record:
        return None
    if max_age_days is not None:
        captured = parse_publish_date(record.get("captured_ts"))
        if captured is not None:
            age_days = (datetime.now(timezone.utc) - captured).total_seconds() / 86400.0
            if age_days > max_age_days:
                return None
    return record.get("payload")
