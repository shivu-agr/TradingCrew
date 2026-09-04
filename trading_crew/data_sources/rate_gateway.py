"""Kafka-backed rate-limit gateway for outbound web calls (search + scraping).

Why
---
Web search and screener.in scraping both have to respect upstream rate limits.
Historically each enforced its own *in-process* throttle (a slot scheduler in
``web_research`` and a "sleep since last request" gap in ``screener_source``).
Those work, but they're invisible and per-process.  This module funnels all
that demand through a **Kafka queue** so the rate limiting is centralised,
observable in the Redpanda Console, and consistent with the workflow pub-sub.

Model — a slot dispenser (not request/reply of results)
------------------------------------------------------
For each *domain* (``search``, ``scrape``) there are two single-partition
topics: ``ratelimit.<domain>.requests`` and ``ratelimit.<domain>.grants``.  A
single **pacer** consumes requests strictly in order and emits one **grant**
every ``gap`` (+ jitter) seconds — so the global rate is at most ``1/gap``.  A
caller (running on a CrewAI worker thread) publishes a request and blocks until
its grant arrives, then performs the actual fetch itself.  Keeping execution in
the caller means we don't move blocking HTTP onto the event loop, and the
existing backend-rotation / cooldown logic is untouched.

The queue depth (pending requests) and grant cadence are both visible in the
Console, which is exactly "the searches/scrapes going through a queue that
respects rate limits".

Lifecycle
---------
The web server calls :func:`start` from its FastAPI lifespan (on the event
loop).  ``trading_crew`` library code (``web_research`` / ``screener_source``)
calls :func:`is_enabled` + :func:`acquire` from worker threads; when the gateway
was never started (e.g. the headless ``main.py`` / cron path) ``is_enabled()``
is ``False`` and callers fall back to their own in-process throttle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_loop: Optional[asyncio.AbstractEventLoop] = None
_producer: Any = None
_enabled = False
_stopped = False
_pending: Dict[str, "asyncio.Future"] = {}
_consumers: List[Any] = []
_tasks: List[asyncio.Task] = []
_domains: Dict[str, Dict[str, float]] = {}


def _req_topic(domain: str) -> str:
    return f"ratelimit.{domain}.requests"


def _grant_topic(domain: str) -> str:
    return f"ratelimit.{domain}.grants"


def bootstrap_default() -> str:
    return os.environ.get("TRADINGCREW_KAFKA_BOOTSTRAP", "localhost:19092").strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def default_domains() -> Dict[str, Dict[str, float]]:
    """Per-domain pacing pulled from the same env knobs the in-process throttles
    used, so behaviour is identical — just enforced via Kafka."""
    search_gap = _env_float("TRADINGCREW_SEARCH_MIN_GAP_SECONDS", 0.0)
    search_jitter = _env_float("TRADINGCREW_SEARCH_GAP_JITTER_SECONDS", 0.0)
    scrape_gap = _env_float("SCREENER_MIN_REQUEST_GAP_SECONDS", 1.5)
    return {
        "search": {
            "gap": search_gap,
            "jitter": search_jitter,
            "timeout": _env_float("TRADINGCREW_SEARCH_SLOT_TIMEOUT_SECONDS", 300.0),
        },
        "scrape": {
            "gap": scrape_gap,
            # screener_source historically added up to 25% jitter on top.
            "jitter": scrape_gap * 0.25,
            "timeout": _env_float("TRADINGCREW_SCRAPE_SLOT_TIMEOUT_SECONDS", 120.0),
        },
    }


def is_enabled() -> bool:
    return _enabled


def domains() -> Dict[str, Dict[str, float]]:
    return dict(_domains)


# ---------------------------------------------------------------------------
# Lifecycle (called from the web-server event loop)
# ---------------------------------------------------------------------------

async def _ensure_topics(bootstrap: str) -> None:
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        want = []
        for d in _domains:
            # Single partition per topic → one global pacer, strict FIFO.
            want.append(NewTopic(_req_topic(d), num_partitions=1, replication_factor=1))
            want.append(NewTopic(_grant_topic(d), num_partitions=1, replication_factor=1))
        try:
            await admin.create_topics(want)
        except TopicAlreadyExistsError:
            pass
    finally:
        await admin.close()


async def _make_consumer(bootstrap: str, topic: str, group: str, offset_reset: str):
    from aiokafka import AIOKafkaConsumer

    c = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group,
        enable_auto_commit=False,
        auto_offset_reset=offset_reset,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    await c.start()
    return c


async def start(bootstrap: Optional[str] = None,
                domain_cfg: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    """Start the producer, pacers and grant routers.  Idempotent."""
    global _loop, _producer, _enabled, _stopped, _domains
    if _enabled:
        return
    from aiokafka import AIOKafkaProducer

    _loop = asyncio.get_running_loop()
    _stopped = False
    bootstrap = bootstrap or bootstrap_default()
    _domains = domain_cfg or default_domains()

    await _ensure_topics(bootstrap)

    _producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
    )
    await _producer.start()

    for d in _domains:
        # One durable pacer per domain (a shared group id means only one pacer
        # is ever active even across processes/restarts).
        pacer = await _make_consumer(bootstrap, _req_topic(d),
                                     group=f"ratelimit-pacer-{d}", offset_reset="earliest")
        _consumers.append(pacer)
        _tasks.append(asyncio.create_task(_pacer_loop(d, pacer), name=f"rl-pacer-{d}"))
        # Grant router is per-process (unique group) and only cares about new
        # grants for futures pending in *this* process.
        router = await _make_consumer(bootstrap, _grant_topic(d),
                                      group=f"ratelimit-router-{d}-{uuid.uuid4().hex[:8]}",
                                      offset_reset="latest")
        _consumers.append(router)
        _tasks.append(asyncio.create_task(_router_loop(router), name=f"rl-router-{d}"))

    _enabled = True
    logger.info("rate_gateway started: domains=%s bootstrap=%s", 
                {k: (v["gap"], v["jitter"]) for k, v in _domains.items()}, bootstrap)


async def stop() -> None:
    global _enabled, _stopped
    _stopped = True
    _enabled = False
    for t in _tasks:
        t.cancel()
    for t in _tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    for c in _consumers:
        try:
            await c.stop()
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if _producer is not None:
        try:
            await _producer.stop()
        except Exception:  # noqa: BLE001
            pass
    _tasks.clear()
    _consumers.clear()
    # fail any still-pending waiters so caller threads don't hang
    for corr, fut in list(_pending.items()):
        if not fut.done():
            fut.set_result(False)
    _pending.clear()


# ---------------------------------------------------------------------------
# Pacer + router loops (event loop)
# ---------------------------------------------------------------------------

async def _pacer_loop(domain: str, consumer: Any) -> None:
    gap = float(_domains[domain].get("gap", 0.0))
    jitter = float(_domains[domain].get("jitter", 0.0))
    next_slot = 0.0
    logger.info("rate_gateway pacer up for '%s' (gap=%.1fs jitter=%.1fs)", domain, gap, jitter)
    while not _stopped:
        try:
            batch = await consumer.getmany(timeout_ms=1000, max_records=1)
        except Exception:  # noqa: BLE001
            logger.exception("rate_gateway pacer '%s' poll error", domain)
            await asyncio.sleep(1.0)
            continue
        for _tp, msgs in batch.items():
            for msg in msgs:
                if _stopped:
                    return
                corr = (msg.value or {}).get("corr")
                # Reserve the next slot (uniform cadence, no bursts).
                g = gap + (random.uniform(0.0, jitter) if jitter > 0 else 0.0)
                now = _loop.time()
                slot = max(now, next_slot)
                next_slot = slot + g
                wait = slot - now
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    await _producer.send_and_wait(_grant_topic(domain), {"corr": corr}, key=corr)
                except Exception:  # noqa: BLE001
                    logger.exception("rate_gateway: failed to publish grant (%s)", domain)
                try:
                    await consumer.commit()
                except Exception:  # noqa: BLE001
                    logger.exception("rate_gateway: pacer commit failed (%s)", domain)


async def _router_loop(consumer: Any) -> None:
    while not _stopped:
        try:
            batch = await consumer.getmany(timeout_ms=1000, max_records=200)
        except Exception:  # noqa: BLE001
            logger.exception("rate_gateway router poll error")
            await asyncio.sleep(1.0)
            continue
        for _tp, msgs in batch.items():
            for msg in msgs:
                corr = (msg.value or {}).get("corr")
                fut = _pending.pop(corr, None) if corr else None
                if fut is not None and not fut.done():
                    fut.set_result(True)


# ---------------------------------------------------------------------------
# Client (called from CrewAI worker threads)
# ---------------------------------------------------------------------------

async def _acquire_async(domain: str, timeout: float) -> bool:
    corr = uuid.uuid4().hex
    fut: asyncio.Future = _loop.create_future()
    _pending[corr] = fut
    try:
        await _producer.send_and_wait(_req_topic(domain), {"corr": corr, "ts": time.time()}, key=corr)
        await asyncio.wait_for(fut, timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning("rate_gateway: slot grant timed out for '%s' after %.0fs", domain, timeout)
        return False
    except Exception:  # noqa: BLE001
        logger.exception("rate_gateway: acquire failed for '%s'", domain)
        return False
    finally:
        _pending.pop(corr, None)


def acquire(domain: str, timeout: Optional[float] = None) -> bool:
    """Block (on the calling worker thread) until a rate-limit slot is granted.

    Returns ``True`` when granted, ``False`` on timeout / error (the caller then
    proceeds anyway — it has already waited, and blocking the workflow forever
    would be worse).  A no-op returning ``False`` if the gateway isn't running.
    """
    if not _enabled or _loop is None:
        return False
    to = timeout if timeout is not None else float(_domains.get(domain, {}).get("timeout", 300.0))
    try:
        cf = asyncio.run_coroutine_threadsafe(_acquire_async(domain, to), _loop)
        return cf.result(to + 5.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rate_gateway.acquire(%s) bridge failed: %s", domain, exc)
        return False
