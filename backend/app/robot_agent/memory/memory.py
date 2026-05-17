"""Tiered memory system for the diagnostic agent.

Implements a four-tier memory contract:

- **Tier 0 (Context Compaction):**  Summarises and prunes long sessions
  when the turn count exceeds ``MAX_TURNS_BEFORE_COMPACTION``.
- **Tier 1 (Session Memory):**  Logs every agent reasoning turn to a
  JSONL file keyed by session ID.
- **Tier 2 (Project Rules):**  Reads persisted Markdown rule files from
  ``agent_knowledge/rules/``.
- **Tier 3 (Semantic Memory):**  Queries PM cards indexed in ChromaDB
  via vector similarity.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory")


MEMORY_DIR = Path("data") / "memory"
SESSION_DIR = MEMORY_DIR / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryEntry:
    memory_id: str
    robot_code: str
    agent_name: str
    incident_id: str
    severity: str
    summary: str
    confidence: float
    recorded_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionTurn:
    turn_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    skill_loaded: str | None = None
    action: str | None = None
    reasoning: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    context_snapshot: dict[str, Any] = field(default_factory=dict)


class Tier1SessionMemory:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        self.turns: list[SessionTurn] = []
        self.session_file = SESSION_DIR / f"{self.session_id}.jsonl"

    def add_turn(
        self,
        skill_loaded: str | None = None,
        action: str | None = None,
        reasoning: str | None = None,
        tool_calls: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> SessionTurn:
        turn = SessionTurn(
            turn_id=f"{self.session_id}-t{len(self.turns) + 1}",
            session_id=self.session_id,
            skill_loaded=skill_loaded,
            action=action,
            reasoning=reasoning,
            tool_calls=tool_calls or [],
            context_snapshot=context_snapshot or {},
        )
        self.turns.append(turn)
        return turn

    def save(self) -> None:
        if not self.session_file.parent.exists():
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, "w", encoding="utf-8") as f:
            for turn in self.turns:
                f.write(json.dumps(asdict(turn), ensure_ascii=False, default=str) + "\n")
        logger.info(f"Session {self.session_id} saved: {len(self.turns)} turns to {self.session_file}")

    def load(self, session_id: str) -> list[SessionTurn]:
        session_file = SESSION_DIR / f"{session_id}.jsonl"
        if not session_file.exists():
            return []
        turns = []
        with open(session_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    turns.append(SessionTurn(**data))
        return turns

    def get_turn_count(self) -> int:
        return len(self.turns)


class Tier0ContextCompaction:
    MAX_TURNS_BEFORE_COMPACTION = 20
    SUMMARY_TEMPLATE = "Previous turns summary: {summarized_content}"

    def __init__(self, session_memory: Tier1SessionMemory):
        self.session = session_memory

    def needs_compaction(self) -> bool:
        return self.session.get_turn_count() >= self.MAX_TURNS_BEFORE_COMPACTION

    def compact(self) -> str:
        if not self.needs_compaction():
            return ""

        recent_turns = self.session.turns[-self.MAX_TURNS_BEFORE_COMPACTION :]
        summarized_parts = []

        for turn in recent_turns:
            parts = []
            if turn.skill_loaded:
                parts.append(f"Loaded skill: {turn.skill_loaded}")
            if turn.action:
                parts.append(f"Action: {turn.action}")
            if turn.reasoning:
                parts.append(f"Reasoning: {turn.reasoning}")
            if turn.tool_calls:
                parts.append(f"Tools: {', '.join(turn.tool_calls)}")
            if parts:
                summarized_parts.append("; ".join(parts))

        summary = f"{len(recent_turns)} turns summarized. Last skills/actions: " + " | ".join(summarized_parts)

        self.session.turns = self.session.turns[: -self.MAX_TURNS_BEFORE_COMPACTION]

        compact_turn = SessionTurn(
            turn_id=f"{self.session.session_id}-t{len(self.session.turns) + 1}",
            session_id=self.session.session_id,
            skill_loaded=None,
            action="CONTEXT_COMPACTED",
            reasoning=summary,
            tool_calls=[],
            context_snapshot={},
        )
        self.session.turns.append(compact_turn)

        return summary

    def get_context_summary(self) -> str:
        if not self.needs_compaction():
            return ""
        return self.compact()


class Tier2ProjectRules:
    RULES_DIR = Path(__file__).resolve().parents[3] / "agent_knowledge" / "rules"
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_rule(cls, rule_name: str) -> str:
        rule_file = cls.RULES_DIR / f"{rule_name}.md"
        if not rule_file.exists():
            return ""
        return rule_file.read_text(encoding="utf-8")

    @classmethod
    def list_rules(cls) -> list[str]:
        if not cls.RULES_DIR.exists():
            return []
        return [f.stem for f in cls.RULES_DIR.glob("*.md")]


class Tier3SemanticMemory:
    def __init__(self):
        self._chromadb_available = self._check_chromadb()

    @staticmethod
    def _check_chromadb() -> bool:
        try:
            import chromadb
            return True
        except ImportError:
            return False

    def query_pm_cards(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self._chromadb_available:
            logger.warning("ChromaDB not available, returning empty results")
            return []

        try:
            from ..chroma.client import get_collection
            collection = get_collection("pm_cards")
            results = collection.query(query_texts=[query], n_results=limit)
            return results.get("documents", [[]])[0]
        except Exception as exc:
            logger.error(f"Tier3 query failed: {exc}")
            return [{"error": "Tier3 ChromaDB query failed", "detail": str(exc)}]

    def index_pm_card(self, card_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._chromadb_available:
            return

        try:
            from ..chroma.client import get_collection
            collection = get_collection("pm_cards")
            collection.upsert(
                ids=[card_id],
                documents=[text],
                metadatas=[metadata or {}],
            )
        except Exception as exc:
            logger.error(f"Tier3 indexing failed: {exc}")


def get_memory_dir() -> Path:
    return MEMORY_DIR


def get_session_dir() -> Path:
    return SESSION_DIR