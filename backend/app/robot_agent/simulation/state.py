from __future__ import annotations

import threading
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


_simulation_state: SimulationState | None = None
_state_lock = threading.Lock()


def get_simulation_state() -> SimulationState:
    global _simulation_state
    with _state_lock:
        if _simulation_state is None:
            _simulation_state = SimulationState()
        return _simulation_state