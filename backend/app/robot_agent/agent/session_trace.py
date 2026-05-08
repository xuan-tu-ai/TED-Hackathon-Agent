from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SESSION_DIR = Path("data/sessions")


class SessionTraceWriter:
    def __init__(self, incident_id: str):
        self.incident_id = incident_id
        self._path: Path | None = None

    @property
    def path(self) -> Path:
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
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return [json.loads(l) for l in lines]
