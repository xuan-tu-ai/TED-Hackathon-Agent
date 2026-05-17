"""Plan lifecycle orchestrator for the diagnostic agent.

Manages creation, parsing, and step-wise execution of a diagnostic plan.
Extracted from DiagnosticAgent in Sprint 6 for separation of concerns.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .event_emitter import AgentEventEmitter
from . import skills as _skills_module

logger = logging.getLogger("plan_manager")


class PlanManager:
    """Orchestrates plan creation, task tracking, and lifecycle events.

    Parses diagnostic plans from a Markdown template, tracks execution
    progress via ``current_task_index``, and broadcasts state changes
    through the injected :class:`AgentEventEmitter`.
    """

    def __init__(self, emitter: AgentEventEmitter):
        self.emitter = emitter
        self.current_plan: dict[str, Any] | None = None
        self.current_task_index: int = 0

    def reset(self) -> None:
        """Clear the current plan and reset task index to zero."""
        self.current_plan = None
        self.current_task_index = 0

    def parse_from_template(self) -> bool:
        """Parse the ``create_plan`` Markdown template into a task list.

        Scans for checklist items of the form ``**Item N:** description``,
        infers a task type for each, and sets :attr:`current_plan`.

        Returns:
            True if at least one task was parsed, False otherwise.
        """
        template_path = _skills_module.SKILLS_BASE / "create_plan" / "_template.md"
        if not template_path.exists():
            logger.error("Plan template not found at %s", template_path)
            return False

        content = template_path.read_text(encoding="utf-8")
        tasks = []
        task_id = 1
        for line in content.splitlines():
            if "- [ ]" in line or "- [x]" in line:
                match = re.search(r"\*\*(Item \d+:)\*\*\s*(.+)", line)
                if match:
                    name = match.group(2).strip().rstrip(".")
                    tasks.append({
                        "id": task_id,
                        "name": name,
                        "type": self._infer_task_type(name),
                        "status": "pending",
                    })
                    task_id += 1

        if not tasks:
            logger.error("Plan template has no parseable tasks")
            return False

        self.current_plan = {"tasks": tasks}
        self._broadcast_plan_created()
        return True

    def _infer_task_type(self, name: str) -> str:
        """Map a task name to a type key (skill, diagnose, hitl, action, assess)."""
        name_lower = name.lower()
        if "load" in name_lower or "skill" in name_lower:
            return "skill"
        elif "diagnos" in name_lower or "check" in name_lower or "query" in name_lower:
            return "diagnose"
        elif "hitl" in name_lower or "approval" in name_lower:
            return "hitl"
        elif "dispatch" in name_lower or "send" in name_lower or "formulate" in name_lower or "action" in name_lower:
            return "action"
        elif "evaluate" in name_lower or "determine" in name_lower:
            return "assess"
        else:
            return "diagnose"

    def update_task(self, task_id: int | str, status: str) -> None:
        """Set the status of *task_id* and broadcast the change.

        Automatically advances :attr:`current_task_index` when the
        current task transitions to ``completed``.
        """
        if not self.current_plan:
            return

        str_id = str(task_id)
        for idx, task in enumerate(self.current_plan.get("tasks", [])):
            if str(task.get("id")) == str_id:
                task["status"] = status
                self._broadcast_plan_task_update(str_id, status)
                if status == "completed" and idx == self.current_task_index:
                    self.current_task_index += 1
                break

    def get_current_task(self) -> dict[str, Any] | None:
        """Return the task at the current index, or None if exhausted."""
        if not self.current_plan:
            return None
        tasks = self.current_plan.get("tasks", [])
        if self.current_task_index < len(tasks):
            return tasks[self.current_task_index]
        return None

    def advance(self) -> None:
        """Increment the current task index (move to next task)."""
        self.current_task_index += 1

    def is_done(self) -> bool:
        """Return True when every task in the plan is marked ``completed``."""
        if not self.current_plan:
            return False
        return all(
            task.get("status") == "completed"
            for task in self.current_plan.get("tasks", [])
        )

    def set_incident_id(self, incident_id: str | None) -> None:
        """Store the incident ID for inclusion in broadcast events."""
        self._incident_id = incident_id

    def _broadcast_plan_created(self) -> None:
        """Emit a ``plan_created`` event via the emitter."""
        if self.current_plan is not None:
            self.emitter.emit_plan({
                "type": "plan_created",
                "incident_id": getattr(self, '_incident_id', None),
                "tasks": self.current_plan["tasks"],
            })

    def _broadcast_plan_task_update(self, task_id: str, status: str) -> None:
        """Emit a ``plan_task_update`` step event via the emitter."""
        self.emitter.emit_step({
            "type": "plan_task_update",
            "incident_id": getattr(self, '_incident_id', None),
            "task_id": task_id,
            "status": status,
        })

    def _broadcast_plan_task_completed(self, task_id: str) -> None:
        """Shorthand to emit a *completed* status for *task_id*."""
        self._broadcast_plan_task_update(task_id, "completed")
