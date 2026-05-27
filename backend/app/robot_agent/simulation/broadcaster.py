"""Thread-safe SSE broadcaster with per-session registry.

:class:`EventBroadcaster` manages a set of ``asyncio.Queue`` subscribers that
FastAPI endpoints consume as SSE streams.  The :meth:`broadcast` method is
synchronous and thread-safe — it acquires a ``threading.Lock`` — and is called
from :class:`SimulationRunner`\'s background thread.  The :meth:`subscribe` and
:meth:`unsubscribe` methods are async and called from FastAPI coroutines.

Dead queues (clients that have disconnected) are detected via ``QueueFull``
exceptions and pruned after each broadcast cycle.

Use :func:`create_broadcaster` / :func:`get_broadcaster` / :func:`delete_broadcaster`
to manage one broadcaster per simulation session.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger("sse_broadcaster")


class EventBroadcaster:
    """Thread-safe SSE broadcaster.

    subscribe/unsubscribe are async (called from FastAPI coroutines).
    broadcast is sync (called from SimulationRunner background thread).
    A threading.Lock guards the _subscribers set for cross-thread access.
    Each Queue has maxsize=100 so QueueFull can actually occur for dead clients.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(queue)
        logger.debug(f"SSE client subscribed, total: {len(self._subscribers)}")
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)
        logger.debug(f"SSE client unsubscribed, total: {len(self._subscribers)}")

    def broadcast(self, event: dict[str, Any]) -> None:
        """Sync broadcast — safe to call from any thread.

        Iterates over a snapshot of subscribers. Dead queues (QueueFull)
        are removed after iteration to avoid mutating the set mid-loop.
        """
        dead: set[asyncio.Queue[dict[str, Any]]] = set()
        with self._lock:
            snapshot = list(self._subscribers)

        for queue in snapshot:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
            except Exception:
                dead.add(queue)

        if dead:
            with self._lock:
                self._subscribers -= dead
            logger.warning(f"Removed {len(dead)} dead SSE subscriber(s)")


# Per-session registry — keyed by sim_id
_broadcasters: dict[str, EventBroadcaster] = {}
_broadcaster_lock = threading.Lock()


def create_broadcaster(sim_id: str) -> EventBroadcaster:
    """Create and register a new :class:`EventBroadcaster` for *sim_id*."""
    broadcaster = EventBroadcaster()
    with _broadcaster_lock:
        _broadcasters[sim_id] = broadcaster
    return broadcaster


def get_broadcaster(sim_id: str) -> EventBroadcaster:
    """Return the :class:`EventBroadcaster` for *sim_id*.

    Raises:
        KeyError: if *sim_id* has no registered broadcaster.
    """
    with _broadcaster_lock:
        broadcaster = _broadcasters.get(sim_id)
    if broadcaster is None:
        raise KeyError(sim_id)
    return broadcaster


def delete_broadcaster(sim_id: str) -> None:
    """Remove the broadcaster for *sim_id* (no-op if absent)."""
    with _broadcaster_lock:
        _broadcasters.pop(sim_id, None)
