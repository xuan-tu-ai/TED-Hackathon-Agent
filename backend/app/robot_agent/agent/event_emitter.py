from __future__ import annotations

from typing import Any, Callable


class AgentEventEmitter:
    def __init__(self):
        self.on_step: Callable[[dict[str, Any]], None] | None = None
        self.on_observation: Callable[[dict[str, Any]], None] | None = None
        self.on_plan: Callable[[dict[str, Any]], None] | None = None
        self.on_hitl_request: Callable[[dict[str, Any]], None] | None = None
        self.on_plan_task: Callable[[dict[str, Any]], None] | None = None

    def emit_step(self, payload: dict[str, Any]) -> None:
        if self.on_step is not None:
            self.on_step(payload)

    def emit_observation(self, payload: dict[str, Any]) -> None:
        if self.on_observation is not None:
            self.on_observation(payload)

    def emit_plan(self, payload: dict[str, Any]) -> None:
        if self.on_plan is not None:
            self.on_plan(payload)

    def emit_hitl_request(self, payload: dict[str, Any]) -> None:
        if self.on_hitl_request is not None:
            self.on_hitl_request(payload)

    def emit_plan_task(self, payload: dict[str, Any]) -> None:
        if self.on_plan_task is not None:
            self.on_plan_task(payload)
