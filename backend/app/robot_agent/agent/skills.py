from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar


SKILLS_BASE = Path(__file__).resolve().parents[3] / "agent_knowledge" / "skills"


class Skill:
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

    @classmethod
    def load(cls, skill_name: str) -> "Skill":
        skill_path = SKILLS_BASE / skill_name / "SKILL.md"
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill not found: {skill_name} at {skill_path}")
        content = skill_path.read_text(encoding="utf-8")
        return cls(name=skill_name, content=content)

    def __str__(self) -> str:
        return self.content


class LoadSkillTool:
    name: ClassVar[str] = "load_skill"
    description: ClassVar[str] = "Load a skill by name. Reads the SKILL.md file and returns its content for context."

    def __call__(self, skill_name: str) -> str:
        skill = Skill.load(skill_name)
        return skill.content
