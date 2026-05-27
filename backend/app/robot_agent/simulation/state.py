"""Thread-safe simulation state machine with per-session registry.

:class:`SimulationState` holds the mutable runtime state for one simulation
session — status, current simulation time, tick index, accumulated alerts,
active incident id, speed factor, and HITL events.  All read/write methods
acquire an internal ``threading.Lock`` for safe access from the background
runner thread and FastAPI request handlers.

:func:`create_simulation_session` generates a UUID session key, stores a new
:class:`SimulationState` in a module-level dict, and returns the key.
:func:`get_simulation_state` looks up an existing session by key.
:func:`delete_simulation_session` removes a finished session from memory.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass
class SimulationState:
    status: Literal["idle", "running", "paused", "finished", "blocked"] = "idle"
    sim_time: datetime | None = None
    tick_index: int = 0
    total_ticks: int = 0
    alerts: list[dict[str, Any]] = field(default_factory=list)
    active_incident_id: str | None = None
    speed_factor: int = 3600
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    hitl_events: dict[str, threading.Event] = field(default_factory=dict)

    def set_status(self, status: Literal["idle", "running", "paused", "finished"]) -> None:
        with self._lock:
            self.status = status

    def get_status(self) -> Literal["idle", "running", "paused", "finished"]:
        with self._lock:
            return self.status

    def set_sim_time(self, sim_time: datetime) -> None:
        with self._lock:
            self.sim_time = sim_time

    def get_sim_time(self) -> datetime | None:
        with self._lock:
            return self.sim_time

    def set_tick_index(self, tick_index: int) -> None:
        with self._lock:
            self.tick_index = tick_index

    def get_tick_index(self) -> int:
        with self._lock:
            return self.tick_index

    def set_total_ticks(self, total_ticks: int) -> None:
        with self._lock:
            self.total_ticks = total_ticks

    def get_total_ticks(self) -> int:
        with self._lock:
            return self.total_ticks

    def add_alert(self, alert: dict[str, Any]) -> None:
        with self._lock:
            self.alerts.append(alert)

    def get_alerts(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.alerts)

    def set_active_incident(self, incident_id: str | None) -> None:
        with self._lock:
            self.active_incident_id = incident_id

    def get_active_incident(self) -> str | None:
        with self._lock:
            return self.active_incident_id

    def set_speed_factor(self, speed_factor: int) -> None:
        with self._lock:
            self.speed_factor = speed_factor

    def get_speed_factor(self) -> int:
        with self._lock:
            return self.speed_factor

    def reset(self) -> None:
        with self._lock:
            self.status = "idle"
            self.sim_time = None
            self.tick_index = 0
            self.total_ticks = 0
            self.alerts.clear()
            self.active_incident_id = None
            for event in self.hitl_events.values():
                event.set()
            self.hitl_events.clear()

    def set_hitl_event(self, queue_id: str, event: threading.Event) -> None:
        with self._lock:
            self.hitl_events[queue_id] = event

    def get_hitl_event(self, queue_id: str) -> threading.Event | None:
        with self._lock:
            return self.hitl_events.get(queue_id)

    def remove_hitl_event(self, queue_id: str) -> None:
        with self._lock:
            if queue_id in self.hitl_events:
                del self.hitl_events[queue_id]

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "sim_time": self.sim_time.isoformat() if self.sim_time else None,
                "tick_index": self.tick_index,
                "total_ticks": self.total_ticks,
                "alerts": list(self.alerts),
                "active_incident_id": self.active_incident_id,
                "speed_factor": self.speed_factor,
            }


# Per-session registry — keyed by sim_id (UUID hex string)
_sessions: dict[str, SimulationState] = {}
_registry_lock = threading.Lock()


def create_simulation_session() -> str:
    """Create a new simulation session and return its sim_id."""
    sim_id = uuid.uuid4().hex
    with _registry_lock:
        _sessions[sim_id] = SimulationState()
    return sim_id


def get_simulation_state(sim_id: str) -> SimulationState:
    """Return the :class:`SimulationState` for *sim_id*.

    Raises:
        KeyError: if *sim_id* is not a known session.
    """
    with _registry_lock:
        state = _sessions.get(sim_id)
    if state is None:
        raise KeyError(sim_id)
    return state


def delete_simulation_session(sim_id: str) -> None:
    """Remove session *sim_id* from the registry (no-op if absent)."""
    with _registry_lock:
        _sessions.pop(sim_id, None)


def find_state_with_hitl_event(queue_id: str) -> SimulationState | None:
    """Return the session state that owns a pending HITL event for *queue_id*.

    Used by :func:`~.hitl.resolve_hitl_entry_and_set_event` to locate the
    correct session without requiring the caller to know the *sim_id*.
    Returns ``None`` when no active session has that queue_id.
    """
    with _registry_lock:
        states = list(_sessions.values())
    for state in states:
        if state.get_hitl_event(queue_id) is not None:
            return state
    return None
