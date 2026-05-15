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
        self._lock = threading.Lock()  # threading.Lock, not asyncio.Lock

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


_broadcaster: EventBroadcaster | None = None
_broadcaster_lock = threading.Lock()


def get_broadcaster() -> EventBroadcaster:
    global _broadcaster
    with _broadcaster_lock:
        if _broadcaster is None:
            _broadcaster = EventBroadcaster()
    return _broadcaster