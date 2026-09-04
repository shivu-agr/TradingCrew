"""In-process Kafka consumer pool that runs workflow requests.

Design
------
Kafka caps parallelism at the partition level (one partition is processed in
order), so we pin **one consumer per partition** via *manual assignment*
(``assign()`` rather than ``subscribe()``).  Manual assignment means there is no
consumer-group rebalance dance at all — every consumer just owns its partition
for the life of the process, which keeps the logs clean and startup instant.

Actual concurrency is decoupled from the partition count by a single process-
wide :class:`asyncio.Semaphore`: a workflow only starts once it can acquire a
slot, so exactly ``concurrency`` runs execute at a time no matter how the work
is spread across partitions.  The producer (``scale_runner.start_scale``)
round-robins ticker → partition so a burst of N names lands on N partitions and
genuinely runs ``min(N, concurrency)`` at once — the continuous "pick the next
the instant a slot frees" behaviour the old single-batch semaphore couldn't do.

Offsets are committed only after a run finishes (at-least-once); a name
interrupted by a crash or a shrink is simply re-delivered and — because the
per-day collect-state marks completed names — never actually re-run.

Control surface (used by the /api/scale/* endpoints):
  * :func:`pause` / :func:`resume` — stop / restart pulling new names.  A pause
    also rewinds any name already fetched but not yet started (so it is
    re-delivered on resume); only workflows *actively running* at pause time
    finish.  In-flight runs are cancelled only by stop/kill.
  * :func:`set_concurrency` — change how many workflows run at once (UI workers).
  * :func:`discard_backlog` — drop everything still queued (hard stop/kill).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from . import kafka_bus

logger = logging.getLogger(__name__)

_consumers: List[Any] = []
_tasks: List[asyncio.Task] = []
_sem: Optional[asyncio.Semaphore] = None
_concurrency = 0
_paused = False
_halted = False   # hard stop: skip already-fetched messages waiting on a slot
_stopped = False  # process shutdown
_started = False


def is_running() -> bool:
    return _started and not _stopped


def is_paused() -> bool:
    return _paused


def consumer_count() -> int:
    return len(_consumers)


def concurrency() -> int:
    return _concurrency


def pause() -> None:
    """Stop pulling new names.  Names already fetched but not yet started are
    rewound (re-delivered on resume); only the workflows *actively running* at
    pause time finish.  Cancelling those mid-run is the job of stop/kill."""
    global _paused
    _paused = True
    logger.info("workflow consumers paused (new pulls stopped; active runs finish)")


def resume() -> None:
    global _paused, _halted
    _paused = False
    _halted = False
    logger.info("workflow consumers resumed")


def halt() -> None:
    """Hard-stop: any message already fetched and waiting for a slot is skipped
    (not run) once the slot frees.  Cleared by :func:`resume`."""
    global _halted
    _halted = True


def set_concurrency(n: int) -> None:
    """Resize the global concurrency limit (UI worker count)."""
    global _sem, _concurrency
    n = max(1, int(n))
    if n == _concurrency and _sem is not None:
        return
    _concurrency = n
    # asyncio.Semaphore isn't resizable; swap it.  In-flight tasks hold a
    # reference to the old object and release into it harmlessly; new starts
    # gate on the new one.
    _sem = asyncio.Semaphore(n)
    logger.info("workflow concurrency set to %d", n)


async def _make_consumer(partition: int):
    from aiokafka import AIOKafkaConsumer, TopicPartition

    topic = kafka_bus.workflow_requests_topic()
    c = AIOKafkaConsumer(
        bootstrap_servers=kafka_bus.bootstrap_servers(),
        group_id=kafka_bus.workflow_group(),
        enable_auto_commit=False,
        value_deserializer=lambda b: __import__("json").loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b is not None else None,
        # A workflow can run for many minutes; manual assignment means this only
        # affects fetch bookkeeping, not group membership.
        max_poll_interval_ms=3_600_000,
    )
    await c.start()
    tp = TopicPartition(topic, partition)
    c.assign([tp])
    # Resume from the group's committed offset (durable across restarts); if
    # none, start at the earliest so queued work is not skipped.
    committed = await c.committed(tp)
    if committed is not None:
        c.seek(tp, committed)
    else:
        await c.seek_to_beginning(tp)
    return c, tp


def _rewind(consumer: Any, tp: Any, offset: int, partition: int) -> None:
    """Seek back to ``offset`` so an unstarted (paused) message is re-delivered
    on resume instead of being consumed-and-dropped."""
    try:
        consumer.seek(tp, offset)
    except Exception:  # noqa: BLE001
        logger.exception("rewind (seek-back) failed on p%d", partition)


async def _loop(consumer: Any, tp: Any, idx: int) -> None:
    from . import scale_runner

    while not _stopped:
        if _paused:
            await asyncio.sleep(0.5)
            continue
        try:
            batch = await consumer.getmany(timeout_ms=1000, max_records=1)
        except Exception:  # noqa: BLE001
            logger.exception("consumer p%d poll error", tp.partition)
            await asyncio.sleep(1.0)
            continue
        if not batch:
            continue
        for _tp, msgs in batch.items():
            for msg in msgs:
                if _stopped:
                    return
                # A pause may have landed between the getmany() above and now.
                # Do NOT start this name: rewind so it is re-delivered on resume
                # (halt, used by stop/kill, just skips — discard_backlog seeks
                # past it).  This is what makes Pause actually stop pulling work
                # instead of draining every already-fetched partition.
                if _halted:
                    break
                if _paused:
                    _rewind(consumer, _tp, msg.offset, tp.partition)
                    break
                payload = msg.value if isinstance(msg.value, dict) else {}
                sem = _sem  # capture current limiter
                async with sem:  # gate: exactly `concurrency` run at once
                    # Re-check after acquiring the slot: pause / hard-stop may
                    # have fired while we waited for a free slot.
                    if _stopped:
                        return
                    if _halted:
                        break
                    if _paused:
                        _rewind(consumer, _tp, msg.offset, tp.partition)
                        break
                    try:
                        await scale_runner.consume_workflow(payload)
                    except Exception:  # noqa: BLE001
                        logger.exception("workflow run failed for %s", payload.get("ticker"))
                    # Commit only after a real run → at-least-once delivery.
                    try:
                        await consumer.commit()
                    except Exception:  # noqa: BLE001
                        logger.exception("offset commit failed on p%d", tp.partition)


async def _partition_count() -> int:
    """How many partitions the requests topic actually has (via the producer,
    so we don't spin up a throwaway consumer just to read metadata)."""
    try:
        parts = await kafka_bus.get_producer().partitions_for(
            kafka_bus.workflow_requests_topic()
        )
        if parts:
            return len(parts)
    except Exception:  # noqa: BLE001
        logger.exception("could not read partition count; falling back to config")
    return kafka_bus.num_partitions()


async def start_consumers(concurrency_limit: int) -> None:
    """Start one consumer per partition and set the concurrency limit.

    The pool starts **paused** so a server (re)start never auto-resumes the
    durable queued backlog — prediction work only starts when the user
    triggers it from the UI (Start/Resume → :func:`resume`).  Without this,
    restarting after ``stop.sh`` / ``start.sh`` made the consumers immediately
    pull and run every still-queued workflow request, bypassing the UI trigger
    and ignoring the previously-selected concurrency (it fell back to the
    ``TRADINGCREW_WORKFLOW_CONSUMERS`` env default).
    """
    global _started, _stopped, _paused
    if _started:
        set_concurrency(concurrency_limit)
        return
    _stopped = False
    _paused = True
    set_concurrency(concurrency_limit)
    p = await _partition_count()
    for i in range(p):
        consumer, tp = await _make_consumer(i)
        _consumers.append(consumer)
        _tasks.append(asyncio.create_task(_loop(consumer, tp, i), name=f"wf-consumer-p{i}"))
    _started = True
    logger.info(
        "workflow consumer pool up: %d partitions on '%s', concurrency=%d, group=%s",
        p, kafka_bus.workflow_requests_topic(), _concurrency, kafka_bus.workflow_group(),
    )


async def discard_backlog() -> int:
    """Skip everything still queued (used by a hard stop/kill).

    Must be called while paused so no consumer is mid-``getmany``; we pause, let
    the loops go idle, then seek every partition to the log end so the
    unprocessed backlog is never delivered.
    """
    global _paused
    was_paused = _paused
    _paused = True
    await asyncio.sleep(1.2)  # let in-flight getmany() calls return + loops idle
    skipped = 0
    for c in _consumers:
        try:
            assignment = c.assignment()
            if not assignment:
                continue
            end_offsets = await c.end_offsets(list(assignment))
            for tp in assignment:
                try:
                    pos = await c.position(tp)
                    skipped += max(0, end_offsets.get(tp, pos) - pos)
                except Exception:  # noqa: BLE001
                    pass
            await c.seek_to_end(*assignment)
            await c.commit()
        except Exception:  # noqa: BLE001
            logger.exception("discard_backlog: seek/commit failed")
    if not was_paused:
        _paused = False
    logger.info("workflow backlog discarded (~%d messages skipped)", skipped)
    return skipped


async def stop_consumers() -> None:
    """Tear the pool down (app shutdown)."""
    global _started, _stopped
    _stopped = True
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
            # aiokafka's coordinator close() can surface a CancelledError; it's
            # harmless during teardown.
            pass
    _tasks.clear()
    _consumers.clear()
    _started = False
