"""Diagnostic trace recorder for agent sessions.

Persists a JSONL trace of every diagnostic step (action, reasoning,
skill loaded) under ``data/sessions/``.  Enables post-hoc analysis and
audit of agent decisions.  Extracted from DiagnosticAgent in Sprint 6.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SESSION_DIR = Path(__file__).resolve().parents[4] / "data" / "sessions"


class SessionTraceWriter:
    """Writes and reads per-incident diagnostic traces as JSONL files.

    Each incident produces one file at ``data/sessions/{incident_id}.jsonl``
    with one JSON object per line representing a single diagnostic step.
    """

    def __init__(self, incident_id: str):
        self.incident_id = incident_id
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        """Lazily resolved path to the trace file for this incident."""
        if self._path is None:
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            self._path = SESSION_DIR / f"{self.incident_id}.jsonl"
        return self._path

    def write_step(
        self,
        step_id: str,
        action: str,
        reasoning: str,
        skill_loaded: str | None = None,
    ) -> None:
        """Append a single diagnostic step to the trace file.

        Args:
            step_id: Unique step identifier.
            action: Action taken by the agent.
            reasoning: Free-text reasoning for the action.
            skill_loaded: Optional name of the skill invoked.
        """
        entry = {
            "step_id": step_id,
            "action": action,
            "skill_loaded": skill_loaded,
            "reasoning": reasoning,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_trace(self) -> list[dict[str, Any]]:
        """Read all recorded steps back from the trace file.

        Returns:
            List of deserialised step dicts in chronological order.
            Returns an empty list if the trace file does not exist.
        """
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return [json.loads(l) for l in lines]
