"""SQLite store for the Prediction dashboard.

This is the persistence layer behind the multi-page ``/predict`` UI. It owns:

  * the **editable universe** — sector ``categories`` and their ``companies``
    (ticker + display name + homepage + wikipedia link), seeded once from the
    curated ``universe/in100.txt`` / ``us100.txt`` files and thereafter mutated
    from the UI (search-to-add / remove);
  * a per-day **prediction index** — one ``predictions`` row per
    ``(ticker, run_date)`` holding the full decision (``decision_json``), a
    rendered markdown ``report_json`` and lazily-cached ``charts_json`` (the
    "sqlite_full" model chosen for this build — the DB is self-contained);
  * a ``run_days`` roll-up so the front page can list historical run dates and
    open a prediction-run review of any past day.

The DB lives at ``$TRADINGCREW_CACHE_DIR/prediction.db`` (defaults to
``~/.trading_crew/prediction.db``). One process-wide connection is shared with
a coarse lock — write volume is tiny (a handful of rows per crew run) so this
is simpler and safe against the scale job's worker threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# repo root is ``<repo>/web/backend/prediction_db.py`` -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNIVERSE_DIR = _REPO_ROOT / "universe"

_LOCK = threading.RLock()
_CONN: Optional[sqlite3.Connection] = None
_CONN_PATH: Optional[str] = None


def _cache_dir() -> Path:
    return Path(os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew"))


def _db_path() -> Path:
    return _cache_dir() / "prediction.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prediction_day() -> str:
    """Local calendar date for the live Prediction index.

    Analysis date is always the operator's local today (not UTC
    ``toISOString()``). The server uses its own local timezone — no
    invented offset.
    """
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _today() -> str:
    return prediction_day()


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    market      TEXT NOT NULL,               -- 'IN' | 'US'
    sort_order  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(name, market)
);

CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    homepage    TEXT,
    wikipedia   TEXT,
    market      TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    enriched    INTEGER NOT NULL DEFAULT 0,  -- 1 once name/links resolved
    added_ts    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    run_date        TEXT NOT NULL,           -- YYYY-MM-DD
    run_id          TEXT,
    status          TEXT NOT NULL,           -- done | failed | running
    action          TEXT,                    -- BUY | SELL | HOLD | ...
    conviction      REAL,
    size            REAL,                    -- target_weight
    expected_return REAL,
    error           TEXT,
    decision_json   TEXT,                    -- full ActionProposal dump
    report_json     TEXT,                    -- rendered markdown
    signals_json    TEXT,                    -- structured report trend signals
    charts_json     TEXT,                    -- lazily cached chart payload
    created_ts      TEXT NOT NULL,
    updated_ts      TEXT NOT NULL,
    UNIQUE(ticker, run_date)
);

CREATE TABLE IF NOT EXISTS run_days (
    run_date    TEXT PRIMARY KEY,            -- YYYY-MM-DD
    started_ts  TEXT,
    finished_ts TEXT,
    total       INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    source      TEXT
);

-- Forward-collection resumability counter, SEGREGATED BY DATE (one row per
-- (day, ticker)).  Replaces the old ``~/.trading_crew/forward/collect-*.json``
-- checkpoint files so all collection state lives in the DB, not the filesystem.
CREATE TABLE IF NOT EXISTS collect_state (
    day         TEXT NOT NULL,               -- YYYY-MM-DD
    ticker      TEXT NOT NULL,               -- upper-cased symbol
    status      TEXT NOT NULL,               -- done | failed
    episode_id  TEXT,                        -- CLI collector (forward episode id)
    run_id      TEXT,                        -- UI/Kafka runner (session id)
    elapsed_sec REAL,
    error       TEXT,
    ts          TEXT NOT NULL,
    PRIMARY KEY(day, ticker)
);

-- UI run history (full audit record per kickoff) — replaces the old
-- ``~/.trading_crew/runs/{TICKER}/{run_id}.json`` files + ``index.jsonl``.
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    status         TEXT,
    final_action   TEXT,
    final_size_pct REAL,
    cascade_route  TEXT,
    data_json      TEXT NOT NULL,               -- RunRecord header (report bodies live in run_reports)
    PRIMARY KEY(run_id, ticker)
);

-- Per-agent report bodies, exploded out of runs.data_json so the UI can
-- load a run shell (diagram + decision) without shipping every analyst
-- write-up on first paint. Fetched per role when a tile is clicked.
CREATE TABLE IF NOT EXISTS run_reports (
    run_id       TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    role         TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    round        INTEGER,
    total_rounds INTEGER,
    output       TEXT NOT NULL DEFAULT '',
    degraded     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, ticker, role, seq),
    FOREIGN KEY (run_id, ticker) REFERENCES runs(run_id, ticker) ON DELETE CASCADE
);

-- Point-in-time scrape snapshots — replaces the on-disk
-- ``~/.trading_crew/pit_snapshots/{namespace}/{key}.json`` files.
CREATE TABLE IF NOT EXISTS pit_snapshots (
    namespace    TEXT NOT NULL,
    key          TEXT NOT NULL,
    captured_ts  TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(namespace, key)
);

-- Per-session web-search counters — one row per (trace_id, backend, outcome).
-- ``trace_id`` is a batch job_id (all its tickers share it) or an interactive
-- run_id, so the dashboard can show search-backend health per batch/session
-- without grepping logs.  See trading_crew/observability/trace.py.
CREATE TABLE IF NOT EXISTS search_stats (
    trace_id    TEXT NOT NULL,
    backend     TEXT NOT NULL,               -- google | searxng | yahoo | ...
    outcome     TEXT NOT NULL,               -- success | empty | throttled | error
    count       INTEGER NOT NULL DEFAULT 0,
    first_ts    TEXT NOT NULL,
    updated_ts  TEXT NOT NULL,
    PRIMARY KEY(trace_id, backend, outcome)
);

CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(run_date);
CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_comp_cat ON companies(category_id);
CREATE INDEX IF NOT EXISTS idx_collect_day ON collect_state(day);
CREATE INDEX IF NOT EXISTS idx_runs_ticker ON runs(ticker);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_run_reports_role ON run_reports(run_id, ticker, role);
CREATE INDEX IF NOT EXISTS idx_search_trace ON search_stats(trace_id);
CREATE INDEX IF NOT EXISTS idx_search_updated ON search_stats(updated_ts);
"""


def _conn() -> sqlite3.Connection:
    global _CONN, _CONN_PATH
    with _LOCK:
        path = _db_path()
        # Reopen if the target DB path changed (e.g. TRADINGCREW_CACHE_DIR was
        # repointed — happens per-test, and keeps the singleton honest in prod).
        if _CONN is not None and _CONN_PATH != str(path):
            try:
                _CONN.close()
            except Exception:  # noqa: BLE001
                pass
            _CONN = None
        if _CONN is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL + a generous busy-timeout so the episodic-memory writer
            # (which shares this same DB file, see EpisodicMemory) and the
            # concurrent batch collectors don't trip over each other.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
            _CONN = conn
            _CONN_PATH = str(path)
            _seed_if_empty(conn)
        return _CONN


def db_path() -> Path:
    """Absolute path of the single application SQLite DB."""
    return _db_path()


def shared_connection() -> "tuple[sqlite3.Connection, threading.RLock]":
    """Return the process-wide DB connection + its lock.

    Lets other stores (e.g. episodic memory) keep all their tables inside
    the single ``prediction.db`` file, sharing one serialized connection
    rather than opening a second one.
    """
    return _conn(), _LOCK


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema to an existing DB.

    ``CREATE TABLE IF NOT EXISTS`` won't add new columns to a table that
    already exists, so newer columns are patched in here idempotently.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "signals_json" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN signals_json TEXT")
    _explode_existing_run_bodies(conn)


def init_db() -> None:
    """Idempotently create the schema and seed the universe on first use."""
    _conn()


# ---------------------------------------------------------------------------
# Universe seeding (from the curated universe/*.txt files)
# ---------------------------------------------------------------------------

_SECTION_FILES = [
    ("IN", _UNIVERSE_DIR / "in100.txt"),
    ("US", _UNIVERSE_DIR / "us100.txt"),
]


def _parse_universe_file(path: Path) -> List[Dict[str, str]]:
    """Parse a curated universe file into ``[{ticker, sector}]``.

    Sectors are the ``# --- <Name> (n) --- `` comment headers; everything until
    the next header inherits it. Bare (non-comment) lines are tickers.
    """
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    sector = "Other"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if body.startswith("---") and body.endswith("---"):
                inner = body.strip("-").strip()
                # drop a trailing "(12)" count if present
                if "(" in inner:
                    inner = inner[: inner.rfind("(")].strip()
                if inner:
                    sector = inner
            continue
        ticker = line.split("#", 1)[0].strip()
        if ticker:
            rows.append({"ticker": ticker, "sector": sector})
    return rows


def _humanize(ticker: str) -> str:
    base = ticker.split(".", 1)[0]
    return base.replace("-", " ").replace("_", " ").title()


def _seed_if_empty(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT COUNT(*) AS n FROM companies")
    if cur.fetchone()["n"] > 0:
        return
    now = _now()
    for market, path in _SECTION_FILES:
        order = 0
        cat_ids: Dict[str, int] = {}
        for row in _parse_universe_file(path):
            sector = row["sector"]
            if sector not in cat_ids:
                order += 1
                conn.execute(
                    "INSERT OR IGNORE INTO categories(name, market, sort_order) VALUES(?,?,?)",
                    (sector, market, order),
                )
                cid = conn.execute(
                    "SELECT id FROM categories WHERE name=? AND market=?", (sector, market)
                ).fetchone()["id"]
                cat_ids[sector] = cid
            conn.execute(
                """INSERT OR IGNORE INTO companies
                   (ticker, name, homepage, wikipedia, market, category_id, enriched, added_ts)
                   VALUES(?,?,?,?,?,?,0,?)""",
                (row["ticker"], _humanize(row["ticker"]), None, None, market, cat_ids[sector], now),
            )
    conn.commit()


# ---------------------------------------------------------------------------
# Read: universe with per-date prediction status
# ---------------------------------------------------------------------------


def list_universe(run_date: Optional[str] = None) -> Dict[str, Any]:
    """Return categories (grouped by market) with their companies + today's
    (or ``run_date``'s) prediction status folded in for tile colouring."""
    run_date = run_date or _today()
    conn = _conn()
    with _LOCK:
        cats = conn.execute(
            "SELECT * FROM categories ORDER BY market, sort_order, name"
        ).fetchall()
        comps = conn.execute("SELECT * FROM companies").fetchall()
        preds = conn.execute(
            "SELECT ticker, status, action, conviction, size, run_id FROM predictions WHERE run_date=?",
            (run_date,),
        ).fetchall()

    pred_by_ticker = {p["ticker"]: dict(p) for p in preds}
    comps_by_cat: Dict[int, List[Dict[str, Any]]] = {}
    for c in comps:
        p = pred_by_ticker.get(c["ticker"])
        comps_by_cat.setdefault(c["category_id"], []).append({
            "ticker": c["ticker"],
            "name": c["name"],
            "homepage": c["homepage"],
            "wikipedia": c["wikipedia"],
            "market": c["market"],
            "enriched": bool(c["enriched"]),
            "status": (p or {}).get("status"),
            "action": _five_tier_action((p or {}).get("action"), (p or {}).get("conviction")),
            "conviction": (p or {}).get("conviction"),
        })

    markets: Dict[str, List[Dict[str, Any]]] = {}
    for cat in cats:
        children = sorted(comps_by_cat.get(cat["id"], []), key=lambda x: x["name"].lower())
        markets.setdefault(cat["market"], []).append({
            "id": cat["id"],
            "name": cat["name"],
            "market": cat["market"],
            "count": len(children),
            "done": sum(1 for x in children if x["status"] == "done"),
            "companies": children,
        })

    total = len(comps)
    done = sum(1 for p in preds if p["status"] == "done")
    failed = sum(1 for p in preds if p["status"] == "failed")
    return {
        "run_date": run_date,
        "markets": markets,
        "progress": {"total": total, "done": done, "failed": failed},
    }


def get_company(ticker: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    with _LOCK:
        row = conn.execute("SELECT * FROM companies WHERE ticker=?", (ticker,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Write: add / remove / enrich companies
# ---------------------------------------------------------------------------


def _resolve_category(conn: sqlite3.Connection, sector: str, market: str) -> int:
    row = conn.execute(
        "SELECT id FROM categories WHERE name=? AND market=?", (sector, market)
    ).fetchone()
    if row:
        return row["id"]
    order_row = conn.execute(
        "SELECT COALESCE(MAX(sort_order),0)+1 AS o FROM categories WHERE market=?", (market,)
    ).fetchone()
    conn.execute(
        "INSERT INTO categories(name, market, sort_order) VALUES(?,?,?)",
        (sector, market, order_row["o"]),
    )
    return conn.execute(
        "SELECT id FROM categories WHERE name=? AND market=?", (sector, market)
    ).fetchone()["id"]


def add_company(
    *,
    ticker: str,
    name: str,
    sector: str,
    market: str,
    homepage: Optional[str] = None,
    wikipedia: Optional[str] = None,
) -> Dict[str, Any]:
    ticker = ticker.strip()
    if not ticker:
        raise ValueError("ticker is required")
    market = (market or "US").upper()
    # Indian listings must carry the exchange suffix for yfinance/the crew;
    # append .NS when an unsuffixed symbol is added under the Indian market.
    if market == "IN" and "." not in ticker:
        ticker = f"{ticker.upper()}.NS"
    sector = sector.strip() or "Other"
    conn = _conn()
    with _LOCK:
        existing = conn.execute("SELECT id FROM companies WHERE ticker=?", (ticker,)).fetchone()
        if existing:
            raise ValueError(f"{ticker} is already in the universe")
        cid = _resolve_category(conn, sector, market)
        conn.execute(
            """INSERT INTO companies
               (ticker, name, homepage, wikipedia, market, category_id, enriched, added_ts)
               VALUES(?,?,?,?,?,?,?,?)""",
            (ticker, name or _humanize(ticker), homepage, wikipedia, market, cid,
             1 if (homepage or wikipedia) else 0, _now()),
        )
        conn.commit()
    return get_company(ticker)  # type: ignore[return-value]


def remove_company(ticker: str) -> bool:
    conn = _conn()
    with _LOCK:
        cur = conn.execute("DELETE FROM companies WHERE ticker=?", (ticker,))
        conn.commit()
        return cur.rowcount > 0


def set_company_meta(ticker: str, *, name: Optional[str] = None,
                     homepage: Optional[str] = None, wikipedia: Optional[str] = None) -> None:
    conn = _conn()
    with _LOCK:
        row = conn.execute("SELECT name FROM companies WHERE ticker=?", (ticker,)).fetchone()
        if not row:
            return
        conn.execute(
            "UPDATE companies SET name=COALESCE(?,name), homepage=COALESCE(?,homepage), "
            "wikipedia=COALESCE(?,wikipedia), enriched=1 WHERE ticker=?",
            (name or None, homepage or None, wikipedia or None, ticker),
        )
        conn.commit()


def companies_needing_meta(limit: int = 500) -> List[Dict[str, str]]:
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            "SELECT ticker, name FROM companies WHERE enriched=0 LIMIT ?", (limit,)
        ).fetchall()
    return [{"ticker": r["ticker"], "name": r["name"]} for r in rows]


# ---------------------------------------------------------------------------
# Write: predictions (called by the scale job on each completed name)
# ---------------------------------------------------------------------------


def upsert_prediction(
    *,
    ticker: str,
    run_date: Optional[str] = None,
    status: str,
    run_id: Optional[str] = None,
    action: Optional[str] = None,
    conviction: Optional[float] = None,
    size: Optional[float] = None,
    expected_return: Optional[float] = None,
    error: Optional[str] = None,
    decision: Optional[Dict[str, Any]] = None,
    report: Optional[str] = None,
    signals: Optional[Dict[str, Any]] = None,
) -> None:
    run_date = run_date or _today()
    now = _now()
    conn = _conn()
    with _LOCK:
        conn.execute(
            """INSERT INTO predictions
                 (ticker, run_date, run_id, status, action, conviction, size,
                  expected_return, error, decision_json, report_json, signals_json,
                  created_ts, updated_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, run_date) DO UPDATE SET
                 run_id=excluded.run_id, status=excluded.status, action=excluded.action,
                 conviction=excluded.conviction, size=excluded.size,
                 expected_return=excluded.expected_return, error=excluded.error,
                 decision_json=COALESCE(excluded.decision_json, predictions.decision_json),
                 report_json=COALESCE(excluded.report_json, predictions.report_json),
                 signals_json=COALESCE(excluded.signals_json, predictions.signals_json),
                 updated_ts=excluded.updated_ts""",
            (
                ticker, run_date, run_id, status, action, conviction, size,
                expected_return, error,
                json.dumps(decision) if decision is not None else None,
                report,
                json.dumps(signals) if signals is not None else None,
                now, now,
            ),
        )
        conn.commit()


def set_prediction_charts(ticker: str, charts: Dict[str, Any], run_date: Optional[str] = None) -> None:
    run_date = run_date or _today()
    conn = _conn()
    with _LOCK:
        conn.execute(
            "UPDATE predictions SET charts_json=?, updated_ts=? WHERE ticker=? AND run_date=?",
            (json.dumps(charts), _now(), ticker, run_date),
        )
        conn.commit()


def _row_to_prediction(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("decision_json", "report_json", "signals_json", "charts_json"):
        val = d.pop(key, None)
        short = key.replace("_json", "")
        if key == "report_json":
            d["report"] = val
        elif val:
            try:
                d[short] = json.loads(val)
            except (ValueError, TypeError):
                d[short] = None
        else:
            d[short] = None
    return d


def get_prediction(ticker: str, run_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = _conn()
    with _LOCK:
        if run_date:
            row = conn.execute(
                "SELECT * FROM predictions WHERE ticker=? AND run_date=?", (ticker, run_date)
            ).fetchone()
        else:
            # Latest *usable* prediction: skip a ``failed`` row whenever a
            # successful (``done``) one exists for the ticker, so a transient
            # failure doesn't masquerade as the ticker's current forward
            # prediction on the Workflow tab (e.g. AXISBANK.NS 08-02 failed
            # while 07-28 succeeded).  A failed row is only shown if it is the
            # ONLY row for the ticker (nothing successful to fall back to).
            row = conn.execute(
                "SELECT * FROM predictions WHERE ticker=? AND status<>'failed' "
                "ORDER BY run_date DESC LIMIT 1",
                (ticker,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM predictions WHERE ticker=? "
                    "ORDER BY run_date DESC LIMIT 1",
                    (ticker,),
                ).fetchone()
    return _row_to_prediction(row) if row else None


def done_prediction(ticker: str, run_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the ``(ticker, day)`` row only when ``status == 'done'``.

    A missing row or a ``failed`` row is not a cache hit — the crew
    must run. ``run_date`` defaults to :func:`prediction_day`.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    pred = get_prediction(ticker, run_date or prediction_day())
    if pred and pred.get("status") == "done":
        return pred
    return None


def done_tickers(run_date: Optional[str] = None) -> set:
    """Uppercased tickers with a ``done`` prediction on ``run_date``."""
    run_date = run_date or prediction_day()
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            "SELECT ticker FROM predictions WHERE run_date=? AND status='done'",
            (run_date,),
        ).fetchall()
    return {(r["ticker"] or "").upper() for r in rows if r["ticker"]}


def cached_session_payload(ticker: str, run_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """JSON body for a same-day cache hit. ``run_id`` is omitted when
    the stored row has none — callers must not invent one."""
    ticker = (ticker or "").strip().upper()
    pred = done_prediction(ticker, run_date)
    if not pred:
        return None
    run_id = pred.get("run_id") or ""
    run = get_run(run_id, ticker) if run_id else None
    return {
        "cached": True,
        "ticker": ticker,
        "status": "completed",
        "run_id": run_id,
        "run_date": pred.get("run_date"),
        "prediction": pred,
        "run": run,
    }


def _side_from_run(data: Dict[str, Any]) -> Optional[str]:
    proposal = data.get("action_proposal") or {}
    if isinstance(proposal, dict):
        side = proposal.get("side")
        if isinstance(side, dict):
            side = side.get("value")
        if side:
            return _canon_action(side)
    decision = data.get("final_decision") or {}
    if isinstance(decision, dict):
        act = decision.get("action")
        if act:
            return _canon_action(act)
    return None


def _render_prediction_report(
    ticker: str,
    *,
    side: Optional[str],
    conviction: Any,
    size: Any,
    expected_return: Any,
    rationale: str = "",
) -> str:
    lines = [f"# {ticker} — forward prediction", "", f"- **Action**: {str(side or 'N/A').upper()}"]
    if conviction is not None:
        try:
            lines.append(f"- **Conviction**: {float(conviction):.2f}")
        except (TypeError, ValueError):
            lines.append(f"- **Conviction**: {conviction}")
    if size is not None:
        try:
            lines.append(f"- **Target weight**: {float(size):+.2%}")
        except (TypeError, ValueError):
            lines.append(f"- **Target weight**: {size}")
    if expected_return is not None:
        try:
            lines.append(f"- **Expected return**: {float(expected_return):+.2%}")
        except (TypeError, ValueError):
            lines.append(f"- **Expected return**: {expected_return}")
    rationale = (rationale or "").strip()
    if rationale:
        lines += ["", "## Rationale", "", rationale]
    return "\n".join(lines)


def record_from_run(data: Dict[str, Any]) -> None:
    """Index a stock ``RunRecord`` into today's (or ``trade_date``'s)
    Prediction run. Commodity records are ignored. ``running`` snapshots
    are skipped. A completed run with no side is stored as ``failed``
    and is not a cache hit.
    """
    cfg = data.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    if str(cfg.get("asset_class") or "stock").lower() == "commodity":
        return
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return
    status = str(data.get("status") or "").lower()
    if status == "running":
        return
    trade_date = str(cfg.get("trade_date") or "")[:10]
    day = trade_date or prediction_day()
    proposal = data.get("action_proposal") if isinstance(data.get("action_proposal"), dict) else {}
    decision = data.get("final_decision") if isinstance(data.get("final_decision"), dict) else {}
    side = _side_from_run(data)
    conviction = proposal.get("conviction_score")
    if conviction is None:
        conviction = decision.get("confidence")
    size = proposal.get("target_weight")
    expected_return = proposal.get("expected_return_pct")
    if expected_return is None:
        expected_return = decision.get("expected_return_pct")
    pred_status = "done" if status == "completed" and side else "failed"
    error = data.get("error")
    if pred_status == "failed" and not error:
        error = "completed with no decision" if status == "completed" else (status or "failed")
    report = data.get("action_proposal_markdown") or None
    if not report:
        report = _render_prediction_report(
            ticker,
            side=side,
            conviction=conviction,
            size=size,
            expected_return=expected_return,
            rationale=str(proposal.get("rationale") or decision.get("rationale") or ""),
        )
    signals = data.get("report_signals")
    if not isinstance(signals, dict):
        signals = None
    start_run_day(day, total=1, source="single")
    upsert_prediction(
        ticker=ticker,
        run_date=day,
        status=pred_status,
        run_id=data.get("run_id") or None,
        action=side,
        conviction=conviction,
        size=size,
        expected_return=expected_return,
        error=error if pred_status == "failed" else None,
        decision=proposal or decision or None,
        report=report,
        signals=signals,
    )
    touch_run_day(day)
    cell: Dict[str, Any] = {"status": pred_status, "ts": _now()}
    if data.get("run_id"):
        cell["run_id"] = data["run_id"]
    if pred_status == "failed" and error:
        cell["error"] = error
    upsert_collect_cell(day, ticker, cell)


# ---------------------------------------------------------------------------
# Forward-collection state (resumability counter) — DB-backed, per date
# ---------------------------------------------------------------------------

_COLLECT_COLS = ("status", "episode_id", "run_id", "elapsed_sec", "error", "ts")


def _cell_to_row(day: str, ticker: str, cell: Dict[str, Any]) -> tuple:
    return (
        day, ticker.upper(),
        cell.get("status"), cell.get("episode_id"), cell.get("run_id"),
        cell.get("elapsed_sec"), cell.get("error"), cell.get("ts") or _now(),
    )


_COLLECT_UPSERT = """INSERT INTO collect_state
        (day, ticker, status, episode_id, run_id, elapsed_sec, error, ts)
      VALUES(?,?,?,?,?,?,?,?)
      ON CONFLICT(day, ticker) DO UPDATE SET
        status=excluded.status, episode_id=excluded.episode_id, run_id=excluded.run_id,
        elapsed_sec=excluded.elapsed_sec, error=excluded.error, ts=excluded.ts"""


def load_collect_state(day: str) -> Dict[str, Dict[str, Any]]:
    """Return ``{TICKER: {status, ...}}`` for ``day`` (the shape the collector
    used to read from its JSON checkpoint), reading only non-null fields."""
    conn = _conn()
    with _LOCK:
        rows = conn.execute("SELECT * FROM collect_state WHERE day=?", (day,)).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cell: Dict[str, Any] = {"status": r["status"], "ts": r["ts"]}
        for k in ("episode_id", "run_id", "elapsed_sec", "error"):
            if r[k] is not None:
                cell[k] = r[k]
        out[r["ticker"]] = cell
    return out


def upsert_collect_cell(day: str, ticker: str, cell: Dict[str, Any]) -> None:
    """Upsert one ``(day, ticker)`` collection cell."""
    conn = _conn()
    with _LOCK:
        conn.execute(_COLLECT_UPSERT, _cell_to_row(day, ticker, cell))
        conn.commit()


def save_collect_state(day: str, state: Dict[str, Dict[str, Any]]) -> None:
    """Persist a whole ``{TICKER: cell}`` map for ``day`` in one transaction
    (preserves the collector's load-mutate-save call pattern)."""
    conn = _conn()
    with _LOCK:
        for ticker, cell in state.items():
            conn.execute(_COLLECT_UPSERT, _cell_to_row(day, ticker, cell))
        conn.commit()


# ---------------------------------------------------------------------------
# Historical run days
# ---------------------------------------------------------------------------


def start_run_day(run_date: str, *, total: int, source: Optional[str] = None) -> None:
    conn = _conn()
    with _LOCK:
        conn.execute(
            """INSERT INTO run_days(run_date, started_ts, total, source)
               VALUES(?,?,?,?)
               ON CONFLICT(run_date) DO UPDATE SET
                 started_ts=COALESCE(run_days.started_ts, excluded.started_ts),
                 total=MAX(run_days.total, excluded.total),
                 source=COALESCE(run_days.source, excluded.source)""",
            (run_date, _now(), total, source),
        )
        conn.commit()


def touch_run_day(run_date: str) -> None:
    """Recompute done/failed counts for a day from the predictions table."""
    conn = _conn()
    with _LOCK:
        agg = conn.execute(
            "SELECT SUM(status='done') AS d, SUM(status='failed') AS f FROM predictions WHERE run_date=?",
            (run_date,),
        ).fetchone()
        conn.execute(
            "UPDATE run_days SET done=?, failed=?, finished_ts=? WHERE run_date=?",
            (agg["d"] or 0, agg["f"] or 0, _now(), run_date),
        )
        conn.commit()


def list_run_days() -> List[Dict[str, Any]]:
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            """SELECT rd.run_date, rd.started_ts, rd.finished_ts, rd.total, rd.source,
                      SUM(p.status='done')   AS done,
                      SUM(p.status='failed') AS failed,
                      COUNT(p.id)            AS predicted
                 FROM run_days rd
                 LEFT JOIN predictions p ON p.run_date = rd.run_date
                GROUP BY rd.run_date
                ORDER BY rd.run_date DESC""",
        ).fetchall()
        # also surface any prediction dates that predate run_days bookkeeping
        extra = conn.execute(
            """SELECT run_date,
                      SUM(status='done')   AS done,
                      SUM(status='failed') AS failed,
                      COUNT(id)            AS predicted
                 FROM predictions
                WHERE run_date NOT IN (SELECT run_date FROM run_days)
                GROUP BY run_date ORDER BY run_date DESC""",
        ).fetchall()
    out = [dict(r) for r in rows]
    for r in extra:
        d = dict(r)
        d.update({"started_ts": None, "finished_ts": None, "total": d["predicted"], "source": None})
        out.append(d)
    out.sort(key=lambda x: x["run_date"], reverse=True)
    return out


def _canon_action(action: Any) -> str:
    """BUY/SELL/HOLD (or STRONG_*) from a stored action, including enum reprs."""
    a = str(action or "").strip()
    if "." in a:
        a = a.rsplit(".", 1)[-1]
    a = a.upper()
    return {"OVERWEIGHT": "BUY", "UNDERWEIGHT": "SELL", "NEUTRAL": "HOLD"}.get(a, a)


def _five_tier_action(action: Any, conviction: Any) -> Any:
    """Reconstruct the retail 5-tier action (STRONG_BUY / BUY / HOLD / SELL /
    STRONG_SELL) from the stored ``side`` + conviction score.

    The PM emits the 5-tier label, but the M1 bridge splits it into ``side`` +
    ``conviction_tier`` for storage (see trading_crew/agentic/bridge.py), so the
    persisted ``action`` column is just BUY/SELL/HOLD.  We re-derive STRONG the
    same way the bridge did: STRONG == HIGH conviction, i.e. score >=
    ``CONVICTION_HIGH_FLOOR``.  Legacy 3-tier strings are coerced first.
    """
    if not action:
        return action
    from trading_crew.agentic.execution.contracts import CONVICTION_HIGH_FLOOR

    a = _canon_action(action)
    try:
        strong = conviction is not None and float(conviction) >= CONVICTION_HIGH_FLOOR
    except (TypeError, ValueError):
        strong = False
    if strong and a == "BUY":
        return "STRONG_BUY"
    if strong and a == "SELL":
        return "STRONG_SELL"
    return a


def get_day_snapshot(run_date: str) -> Dict[str, Any]:
    """Historical read: all predictions for ``run_date`` grouped by sector."""
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            """SELECT p.ticker, p.status, p.action, p.conviction, p.size,
                      p.expected_return, p.run_id,
                      c.name, c.homepage, c.wikipedia, c.market, c.category_id
                 FROM predictions p
                 LEFT JOIN companies c ON c.ticker = p.ticker
                WHERE p.run_date=?
                ORDER BY c.market, p.action""",
            (run_date,),
        ).fetchall()
        cat_names = {
            r["id"]: {"name": r["name"], "market": r["market"]}
            for r in conn.execute("SELECT id, name, market FROM categories").fetchall()
        }
    sectors: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = dict(r)
        cat = cat_names.get(d.get("category_id") or -1, {"name": "Other", "market": d.get("market") or "US"})
        key = f"{cat['market']}::{cat['name']}"
        bucket = sectors.setdefault(key, {"sector": cat["name"], "market": cat["market"], "companies": []})
        bucket["companies"].append({
            "ticker": d["ticker"],
            "name": d.get("name") or _humanize(d["ticker"]),
            "status": d["status"],
            "action": _five_tier_action(d["action"], d["conviction"]),
            "conviction": d["conviction"],
            "size": d["size"],
            "expected_return": d["expected_return"],
            "run_id": d.get("run_id") or "",
            "homepage": d.get("homepage"),
            "wikipedia": d.get("wikipedia"),
        })
    counts: Dict[str, int] = {}
    for r in rows:
        act = _five_tier_action(r["action"], r["conviction"]) or "N/A"
        act = str(act).upper()
        counts[act] = counts.get(act, 0) + 1
    return {
        "run_date": run_date,
        "sectors": list(sectors.values()),
        "action_counts": counts,
        "total": len(rows),
        "done": sum(1 for r in rows if r["status"] == "done"),
        "failed": sum(1 for r in rows if r["status"] == "failed"),
    }


# ---------------------------------------------------------------------------
# UI run history (index + header JSON; report bodies in run_reports)
# ---------------------------------------------------------------------------

_RUN_INDEX_COLS = (
    "run_id", "ticker", "started_at", "completed_at", "status",
    "final_action", "final_size_pct", "cascade_route",
)


def _iter_report_rows(data: Dict[str, Any]) -> List[tuple]:
    """Yield ``(role, seq, round, total_rounds, output, degraded)``."""
    by_round = data.get("reports_by_round") or {}
    reports = data.get("reports") or {}
    rows: List[tuple] = []
    if isinstance(by_round, dict) and by_round:
        for role, entries in by_round.items():
            for seq, entry in enumerate(entries or []):
                if not isinstance(entry, dict):
                    continue
                rows.append((
                    role,
                    seq,
                    int(entry.get("round") or seq + 1),
                    int(entry.get("total_rounds") or 1),
                    entry.get("output") or "",
                    1 if entry.get("degraded") else 0,
                ))
        return rows
    if isinstance(reports, dict):
        for seq, (role, output) in enumerate(reports.items()):
            rows.append((role, 0, 1, 1, output or "", 0))
    return rows


def _has_inline_bodies(data: Dict[str, Any]) -> bool:
    for body in (data.get("reports") or {}).values():
        if body:
            return True
    for entries in (data.get("reports_by_round") or {}).values():
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("output"):
                return True
    return False


def _header_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``data`` with report bodies stripped and ``has_body`` stubs."""
    header = dict(data)
    by_round = data.get("reports_by_round") or {}
    reports = data.get("reports") or {}
    lite_reports: Dict[str, str] = {}
    lite_by_round: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(by_round, dict) and by_round:
        for role, entries in by_round.items():
            stubs: List[Dict[str, Any]] = []
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                output = entry.get("output") or ""
                stubs.append({
                    "round": entry.get("round") or 1,
                    "total_rounds": entry.get("total_rounds") or 1,
                    "output": "",
                    "degraded": bool(entry.get("degraded")),
                    "has_body": bool(output) or bool(entry.get("has_body")),
                })
            lite_by_round[role] = stubs
            lite_reports[role] = ""
    elif isinstance(reports, dict):
        for role, output in reports.items():
            lite_reports[role] = ""
            lite_by_round[role] = [{
                "round": 1,
                "total_rounds": 1,
                "output": "",
                "degraded": False,
                "has_body": bool(output),
            }]
    header["reports"] = lite_reports
    header["reports_by_round"] = lite_by_round
    return header


def _replace_run_reports(
    conn: sqlite3.Connection, run_id: str, ticker: str, data: Dict[str, Any],
) -> None:
    ticker = (ticker or "").upper()
    conn.execute("DELETE FROM run_reports WHERE run_id=? AND ticker=?", (run_id, ticker))
    rows = [
        (run_id, ticker, role, seq, rnd, total, output, deg)
        for role, seq, rnd, total, output, deg in _iter_report_rows(data)
    ]
    if rows:
        conn.executemany(
            """INSERT INTO run_reports
                 (run_id, ticker, role, seq, round, total_rounds, output, degraded)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )


def _attach_report_bodies(
    conn: sqlite3.Connection,
    data: Dict[str, Any],
    run_id: str,
    ticker: str,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    ticker = (ticker or "").upper()
    if role:
        rows = conn.execute(
            """SELECT role, seq, round, total_rounds, output, degraded
               FROM run_reports WHERE run_id=? AND ticker=? AND role=?
               ORDER BY seq""",
            (run_id, ticker, role),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT role, seq, round, total_rounds, output, degraded
               FROM run_reports WHERE run_id=? AND ticker=?
               ORDER BY role, seq""",
            (run_id, ticker),
        ).fetchall()
    if not rows:
        return data
    by_round: Dict[str, List[Dict[str, Any]]] = {}
    reports: Dict[str, str] = {}
    for row in rows:
        entry = {
            "round": row["round"],
            "total_rounds": row["total_rounds"],
            "output": row["output"] or "",
            "degraded": bool(row["degraded"]),
        }
        by_round.setdefault(row["role"], []).append(entry)
        reports[row["role"]] = entry["output"]
    out = dict(data)
    if role:
        merged_reports = dict(data.get("reports") or {})
        merged_by_round = dict(data.get("reports_by_round") or {})
        merged_reports[role] = reports.get(role, "")
        merged_by_round[role] = by_round.get(role, [])
        out["reports"] = merged_reports
        out["reports_by_round"] = merged_by_round
    else:
        out["reports"] = reports
        out["reports_by_round"] = by_round
    return out


def _explode_existing_run_bodies(conn: sqlite3.Connection) -> None:
    """One-time: move inline report text out of existing ``data_json`` rows."""
    try:
        rows = conn.execute("SELECT run_id, ticker, data_json FROM runs").fetchall()
    except sqlite3.Error:
        return
    for row in rows:
        try:
            data = json.loads(row["data_json"])
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or not _has_inline_bodies(data):
            continue
        _replace_run_reports(conn, row["run_id"], row["ticker"], data)
        header = _header_from_data(data)
        conn.execute(
            "UPDATE runs SET data_json=? WHERE run_id=? AND ticker=?",
            (json.dumps(header, default=str), row["run_id"], row["ticker"]),
        )


def upsert_run(data: Dict[str, Any]) -> None:
    """Persist a ``RunRecord.to_dict()`` payload.

    Indexed columns stay on ``runs``. Report bodies are written to
    ``run_reports``; ``data_json`` keeps the header (empty outputs +
    ``has_body`` stubs) so listing and first-paint loads stay small.
    """
    fd = data.get("final_decision") or {}
    header = _header_from_data(data)
    run_id = data.get("run_id")
    ticker = (data.get("ticker") or "").upper()
    row = (
        run_id, ticker,
        data.get("started_at"), data.get("completed_at"), data.get("status"),
        fd.get("action"), fd.get("size_pct_of_book"), data.get("cascade_route"),
        json.dumps(header, default=str),
    )
    conn = _conn()
    with _LOCK:
        conn.execute(
            """INSERT INTO runs
                 (run_id, ticker, started_at, completed_at, status,
                  final_action, final_size_pct, cascade_route, data_json)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id, ticker) DO UPDATE SET
                 started_at=excluded.started_at, completed_at=excluded.completed_at,
                 status=excluded.status, final_action=excluded.final_action,
                 final_size_pct=excluded.final_size_pct, cascade_route=excluded.cascade_route,
                 data_json=excluded.data_json""",
            row,
        )
        _replace_run_reports(conn, run_id, ticker, data)
        conn.commit()


def list_runs(limit: int = 20, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent run-index rows (newest first)."""
    conn = _conn()
    cols = ", ".join(_RUN_INDEX_COLS)
    with _LOCK:
        if ticker:
            rows = conn.execute(
                f"SELECT {cols} FROM runs WHERE ticker=? ORDER BY started_at DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def find_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Index row for ``run_id`` (newest ticker match if the id is reused)."""
    conn = _conn()
    cols = ", ".join(_RUN_INDEX_COLS)
    with _LOCK:
        row = conn.execute(
            f"SELECT {cols} FROM runs WHERE run_id=? ORDER BY started_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def get_run(run_id: str, ticker: str, full: bool = False) -> Optional[Dict[str, Any]]:
    """Load a run header, or the full record with report bodies when ``full``.

    Default is the lite header (empty report outputs, ``has_body`` stubs)
    so page load does not pull every analyst write-up.
    """
    ticker = (ticker or "").upper()
    conn = _conn()
    with _LOCK:
        row = conn.execute(
            "SELECT data_json FROM runs WHERE run_id=? AND ticker=?",
            (run_id, ticker),
        ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["data_json"])
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if full:
            return _attach_report_bodies(conn, data, run_id, ticker)
        if _has_inline_bodies(data):
            return _header_from_data(data)
        return data


def get_run_reports(
    run_id: str, ticker: str, role: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return report bodies for one role, or every role when ``role`` is empty."""
    ticker = (ticker or "").upper()
    conn = _conn()
    with _LOCK:
        exists = conn.execute(
            "SELECT 1 FROM runs WHERE run_id=? AND ticker=?",
            (run_id, ticker),
        ).fetchone()
        if not exists:
            return None
        row = conn.execute(
            "SELECT data_json FROM runs WHERE run_id=? AND ticker=?",
            (run_id, ticker),
        ).fetchone()
        try:
            data = json.loads(row["data_json"]) if row else {}
        except (ValueError, TypeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        attached = _attach_report_bodies(conn, data, run_id, ticker, role=role)
        reports = attached.get("reports") or {}
        by_round = attached.get("reports_by_round") or {}
        if role:
            return {
                "run_id": run_id,
                "ticker": ticker,
                "role": role,
                "reports": {role: reports.get(role, "")},
                "reports_by_round": {role: by_round.get(role, [])},
            }
        return {
            "run_id": run_id,
            "ticker": ticker,
            "role": None,
            "reports": reports,
            "reports_by_round": by_round,
        }


def clear_runs(ticker: Optional[str] = None) -> int:
    """Delete run records (all, or one ticker's).  Returns the count removed."""
    conn = _conn()
    with _LOCK:
        if ticker:
            key = ticker.upper()
            conn.execute("DELETE FROM run_reports WHERE ticker=?", (key,))
            cur = conn.execute("DELETE FROM runs WHERE ticker=?", (key,))
        else:
            conn.execute("DELETE FROM run_reports")
            cur = conn.execute("DELETE FROM runs")
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# Point-in-time scrape snapshots
# ---------------------------------------------------------------------------


def save_pit_snapshot(namespace: str, key: str, payload: Dict[str, Any]) -> str:
    """Upsert a scraped payload keyed by ``(namespace, key)``; returns captured_ts."""
    captured = _now()
    conn = _conn()
    with _LOCK:
        conn.execute(
            """INSERT INTO pit_snapshots (namespace, key, captured_ts, payload_json)
               VALUES(?,?,?,?)
               ON CONFLICT(namespace, key) DO UPDATE SET
                 captured_ts=excluded.captured_ts, payload_json=excluded.payload_json""",
            (namespace, key, captured, json.dumps(payload, ensure_ascii=False, default=str)),
        )
        conn.commit()
    return captured


def import_pit_snapshot(namespace: str, key: str, captured_ts: str, payload: Dict[str, Any]) -> None:
    """Insert a *legacy* snapshot preserving its original ``captured_ts``.

    Uses INSERT OR IGNORE so a one-time migration never clobbers a fresher
    snapshot already written by the running app.
    """
    conn = _conn()
    with _LOCK:
        conn.execute(
            "INSERT OR IGNORE INTO pit_snapshots (namespace, key, captured_ts, payload_json) "
            "VALUES(?,?,?,?)",
            (namespace, key, captured_ts or _now(),
             json.dumps(payload, ensure_ascii=False, default=str)),
        )
        conn.commit()


def load_pit_snapshot(namespace: str, key: str) -> Optional[Dict[str, Any]]:
    """Return ``{captured_ts, payload}`` for a snapshot, or ``None``."""
    conn = _conn()
    with _LOCK:
        row = conn.execute(
            "SELECT captured_ts, payload_json FROM pit_snapshots WHERE namespace=? AND key=?",
            (namespace, key),
        ).fetchone()
    if not row:
        return None
    try:
        return {"captured_ts": row["captured_ts"], "payload": json.loads(row["payload_json"])}
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Per-session web-search counters
# ---------------------------------------------------------------------------

_SEARCH_OUTCOMES = ("success", "empty", "throttled", "error")


def bump_search_stat(trace_id: str, backend: str, outcome: str, n: int = 1) -> None:
    """Increment the counter for one (trace_id, backend, outcome) by ``n``."""
    if not trace_id or not backend or not outcome or n <= 0:
        return
    conn = _conn()
    now = _now()
    with _LOCK:
        conn.execute(
            "INSERT INTO search_stats(trace_id, backend, outcome, count, first_ts, updated_ts) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(trace_id, backend, outcome) DO UPDATE SET "
            "count = count + excluded.count, updated_ts = excluded.updated_ts",
            (trace_id, backend, outcome, int(n), now, now),
        )
        conn.commit()


def get_search_stats(trace_id: str) -> Dict[str, Any]:
    """Aggregate the web-search counters for one session.

    Returns ``{trace_id, backends: {name: {success, empty, throttled, error,
    total}}, totals: {...}, total_calls, first_ts, last_ts}``.
    """
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            "SELECT backend, outcome, count, first_ts, updated_ts "
            "FROM search_stats WHERE trace_id=?",
            (trace_id,),
        ).fetchall()

    def _blank() -> Dict[str, int]:
        d = {o: 0 for o in _SEARCH_OUTCOMES}
        d["total"] = 0
        return d

    backends: Dict[str, Dict[str, int]] = {}
    totals = {o: 0 for o in _SEARCH_OUTCOMES}
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    total_calls = 0
    for r in rows:
        b = backends.setdefault(r["backend"], _blank())
        oc = r["outcome"] if r["outcome"] in totals else "error"
        cnt = int(r["count"] or 0)
        b[oc] += cnt
        b["total"] += cnt
        # ``searxng/<engine>`` rows are a *breakdown* of SearXNG's own internal
        # fanout, not separate top-level search calls — keep them in `backends`
        # (so the dashboard can expand them) but don't double-count them in the
        # session totals.
        if "/" not in r["backend"]:
            totals[oc] += cnt
            total_calls += cnt
        if r["first_ts"] and (first_ts is None or r["first_ts"] < first_ts):
            first_ts = r["first_ts"]
        if r["updated_ts"] and (last_ts is None or r["updated_ts"] > last_ts):
            last_ts = r["updated_ts"]
    return {
        "trace_id": trace_id,
        "backends": backends,
        "totals": totals,
        "total_calls": total_calls,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def list_search_trace_ids(limit: int = 50) -> List[Dict[str, Any]]:
    """Recent sessions that recorded searches, newest activity first."""
    conn = _conn()
    with _LOCK:
        rows = conn.execute(
            "SELECT trace_id, SUM(count) AS total_calls, "
            "MIN(first_ts) AS first_ts, MAX(updated_ts) AS last_ts "
            "FROM search_stats GROUP BY trace_id "
            "ORDER BY last_ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [
        {
            "trace_id": r["trace_id"],
            "total_calls": int(r["total_calls"] or 0),
            "first_ts": r["first_ts"],
            "last_ts": r["last_ts"],
        }
        for r in rows
    ]
