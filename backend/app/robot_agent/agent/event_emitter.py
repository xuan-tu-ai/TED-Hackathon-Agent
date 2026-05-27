"""SSE event bus for the diagnostic agent lifecycle.

Provides a lightweight callback-based emitter that decouples agent
internals (plan steps, HITL requests, observations) from the transport
layer (SSE, WebSocket, etc.).  Extracted from DiagnosticAgent in Sprint 6.
"""

from __future__ import annotations

from typing import Any, Callable


class AgentEventEmitter:
    """Callback-based event bus for agent lifecycle events.

    Each event type is an optional callable slot.  External code (e.g. an
    SSE endpoint) sets the relevant slot to push events to connected clients.
    """

    def __init__(self):
        self.on_step: Callable[[dict[str, Any]], None] | None = None
        self.on_observation: Callable[[dict[str, Any]], None] | None = None
        self.on_plan: Callable[[dict[str, Any]], None] | None = None
        self.on_hitl_request: Callable[[dict[str, Any]], None] | None = None
        self.on_plan_task: Callable[[dict[str, Any]], None] | None = None

    def emit_step(self, payload: dict[str, Any]) -> None:
        """Emit a step-level event (e.g. plan task update)."""
        if self.on_step is not None:
            self.on_step(payload)

    def emit_observation(self, payload: dict[str, Any]) -> None:
        """Emit an observation event (e.g. sensor reading or query result)."""
        if self.on_observation is not None:
            self.on_observation(payload)

    def emit_plan(self, payload: dict[str, Any]) -> None:
        """Emit a plan-level event (e.g. plan created)."""
        if self.on_plan is not None:
            self.on_plan(payload)

    def emit_hitl_request(self, payload: dict[str, Any]) -> None:
        """Emit a HITL request event requiring human review."""
        if self.on_hitl_request is not None:
            self.on_hitl_request(payload)

    def emit_plan_task(self, payload: dict[str, Any]) -> None:
        """Emit a plan-task-level event (e.g. individual task status change)."""
        if self.on_plan_task is not None:
            self.on_plan_task(payload)
