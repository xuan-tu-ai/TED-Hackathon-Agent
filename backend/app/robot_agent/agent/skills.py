"""Skill loading system for the diagnostic agent.

Skills are Markdown files stored under ``agent_knowledge/skills/<name>/SKILL.md``.
Each skill encapsulates a diagnostic sub-routine (e.g. *routing*, *diagnose_surge*,
*create_plan*). The agent loads skills at runtime to guide its LM-powered reasoning.

Module-level constants
----------------------
SKILLS_BASE : Path
    Root directory containing skill sub-folders.
ALLOWED_SKILLS : frozenset[str]
    Names of skills that have a valid ``SKILL.md`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar


SKILLS_BASE = Path(__file__).resolve().parents[3] / "agent_knowledge" / "skills"


def _discover_allowed_skills() -> frozenset[str]:
    """Scan ``SKILLS_BASE`` for directories containing a ``SKILL.md`` file."""
    if not SKILLS_BASE.exists():
        return frozenset()
    return frozenset(
        d.name for d in SKILLS_BASE.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    )


ALLOWED_SKILLS: frozenset[str] = _discover_allowed_skills()


class Skill:
    """A single diagnostic skill backed by its ``SKILL.md`` file.

    Parameters
    ----------
    name : str
        The skill identifier (directory name).
    content : str
        Raw Markdown content of the skill.
    """

    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

    @classmethod
    def load(cls, skill_name: str) -> "Skill":
        """Load a skill by name from the filesystem.

        Raises
        ------
        FileNotFoundError
            If ``SKILLS_BASE / skill_name / SKILL.md`` does not exist.
        """
        skill_path = SKILLS_BASE / skill_name / "SKILL.md"
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill not found: {skill_name} at {skill_path}")
        content = skill_path.read_text(encoding="utf-8")
        return cls(name=skill_name, content=content)

    def __str__(self) -> str:
        return self.content


class LoadSkillTool:
    """DSPy-compatible tool that loads a skill by name for LLM context.

    When the requested skill is not in :data:`ALLOWED_SKILLS`, returns
    an error message listing the available skills.
    """

    name: ClassVar[str] = "load_skill"
    description: ClassVar[str] = "Load a skill by name. Reads the SKILL.md file and returns its content for context."

    def __call__(self, skill_name: str) -> str:
        if skill_name not in ALLOWED_SKILLS:
            available = sorted(ALLOWED_SKILLS)
            return (
                f"Skill '{skill_name}' not found. Available skills: {available}"
            )
        skill = Skill.load(skill_name)
        return skill.content
