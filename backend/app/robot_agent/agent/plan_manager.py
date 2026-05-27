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

from .config import config
from .event_emitter import AgentEventEmitter

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
        template_path = config.skills_base / "create-plan" / "_template.md"
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

        When a task transitions to ``completed``, all earlier incomplete
        tasks (prerequisites) are auto-completed to enforce sequential
        ordering.  Advancement of :attr:`current_task_index` is handled
        by :meth:`get_current_task` (first-pending lookup) rather than
        auto-increment.
        """
        if not self.current_plan:
            return

        str_id = str(task_id)
        tasks = self.current_plan.get("tasks", [])
        target_idx = None
        for idx, task in enumerate(tasks):
            if str(task.get("id")) == str_id:
                target_idx = idx
                break

        if target_idx is None:
            return

        if status == "completed":
            # Auto-complete all prerequisite tasks before this one
            for i in range(target_idx):
                if tasks[i].get("status") != "completed":
                    tasks[i]["status"] = "completed"
                    self._broadcast_plan_task_update(str(tasks[i]["id"]), "completed")

        tasks[target_idx]["status"] = status
        self._broadcast_plan_task_update(str_id, status)

    def get_current_task(self) -> dict[str, Any] | None:
        """Return the first task whose status is not ``completed``, or None if all done.

        This replaces the old index-based lookup so that the agent can stay
        on the same task across multiple DSPy turns.
        """
        if not self.current_plan:
            return None
        for task in self.current_plan.get("tasks", []):
            if task.get("status") != "completed":
                return task
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
        """Emit a plan-task event via the emitter with correct type per status.

        - ``plan_task_started`` when status = ``running``
        - ``plan_task_completed`` when status = ``completed``
        - ``plan_task_update`` as a fallback for other statuses
        """
        event_type = {
            "running": "plan_task_started",
            "completed": "plan_task_completed",
        }.get(status, "plan_task_update")
        self.emitter.emit_plan_task({
            "type": event_type,
            "incident_id": getattr(self, '_incident_id', None),
            "task_id": task_id,
            "status": status,
        })

    def _broadcast_plan_task_completed(self, task_id: str) -> None:
        """Shorthand to emit a *completed* status for *task_id*."""
        self._broadcast_plan_task_update(task_id, "completed")
