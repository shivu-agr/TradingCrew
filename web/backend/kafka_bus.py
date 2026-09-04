"""Kafka (Redpanda) pub-sub bus for the trading dashboard.

Why pub-sub
-----------
The old scale run created *all* workflow tasks up-front and gated them behind
one ``asyncio.Semaphore``: a single active job at a time, and no way to enqueue
more names until it drained.  This module replaces that with a durable Kafka
topic (``workflow.requests``) that ``scale_runner.start_scale`` *publishes* to
and that an in-process consumer pool (``workflow_consumer``) continuously pulls
from — so a free worker picks up the next name immediately and you can submit
new work at any time.

The broker is a local **Redpanda** (Kafka-API compatible, so aiokafka talks to
it unchanged).  Start it with ``scripts/redpanda.sh start`` and watch topics,
messages and consumer-group lag in the Console at ``http://localhost:18090``.

This module owns the *shared* pieces:
  * configuration (bootstrap servers, topic names, partition count),
  * a single process-wide :class:`~aiokafka.AIOKafkaProducer`, and
  * idempotent topic creation.

The consumer side lives in :mod:`web.backend.workflow_consumer`.

No fallback: if the broker is unreachable the bus fails loudly and the scale
endpoints report it — there is no silent in-process queue substitute.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (env-driven; .env is authoritative via load_dotenv override)
# ---------------------------------------------------------------------------

def bootstrap_servers() -> str:
    return os.environ.get("TRADINGCREW_KAFKA_BOOTSTRAP", "localhost:19092").strip()


def workflow_requests_topic() -> str:
    return os.environ.get("TRADINGCREW_KAFKA_WORKFLOW_TOPIC", "workflow.requests").strip()


def workflow_events_topic() -> str:
    return os.environ.get("TRADINGCREW_KAFKA_WORKFLOW_EVENTS_TOPIC", "workflow.events").strip()


def workflow_group() -> str:
    return os.environ.get("TRADINGCREW_KAFKA_WORKFLOW_GROUP", "workflow-runners").strip()


def num_partitions() -> int:
    try:
        return max(1, int(os.environ.get("TRADINGCREW_KAFKA_PARTITIONS", "24")))
    except (TypeError, ValueError):
        return 24


def console_url() -> str:
    """Best-effort Console URL for the UI/status payload (visualisation)."""
    return os.environ.get("TRADINGCREW_KAFKA_CONSOLE_URL", "http://localhost:18090").strip()


# ---------------------------------------------------------------------------
# Shared producer + lifecycle
# ---------------------------------------------------------------------------

_producer: Optional[Any] = None
_started = False


def _dumps(value: Dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")


async def ensure_topics() -> None:
    """Create the workflow topics if they don't exist (idempotent)."""
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers())
    await admin.start()
    try:
        want = [
            NewTopic(workflow_requests_topic(), num_partitions=num_partitions(), replication_factor=1),
            # Events is a low-volume status/observability stream — a few
            # partitions is plenty and keeps ordering easy to reason about.
            NewTopic(workflow_events_topic(), num_partitions=3, replication_factor=1),
        ]
        try:
            await admin.create_topics(want)
            logger.info("Kafka topics ensured: %s, %s (partitions=%d)",
                        workflow_requests_topic(), workflow_events_topic(), num_partitions())
        except TopicAlreadyExistsError:
            logger.info("Kafka topics already exist: %s, %s",
                        workflow_requests_topic(), workflow_events_topic())
    finally:
        await admin.close()


async def start_bus() -> None:
    """Start the shared producer and ensure topics exist.

    Raises if the broker is unreachable — callers (the FastAPI lifespan) log a
    clear, actionable error pointing at ``scripts/redpanda.sh start``.
    """
    global _producer, _started
    if _started:
        return
    from aiokafka import AIOKafkaProducer

    await ensure_topics()
    _producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers(),
        value_serializer=_dumps,
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        enable_idempotence=True,
        acks="all",
    )
    await _producer.start()
    _started = True
    logger.info("Kafka producer started (bootstrap=%s)", bootstrap_servers())


async def stop_bus() -> None:
    global _producer, _started
    if _producer is not None:
        try:
            await _producer.stop()
        except Exception:  # noqa: BLE001
            logger.exception("error stopping Kafka producer")
    _producer = None
    _started = False


def is_started() -> bool:
    return _started


def get_producer() -> Any:
    if _producer is None:
        raise RuntimeError(
            "Kafka bus not started. Ensure scripts/redpanda.sh is running and "
            "the FastAPI lifespan called start_bus()."
        )
    return _producer


async def publish(
    topic: str,
    value: Dict[str, Any],
    *,
    key: Optional[str] = None,
    partition: Optional[int] = None,
) -> None:
    """Publish one JSON message and wait for the broker ack.

    An explicit ``partition`` overrides the default partitioner — used to
    round-robin work evenly across partitions/consumers.
    """
    await get_producer().send_and_wait(topic, value=value, key=key, partition=partition)
