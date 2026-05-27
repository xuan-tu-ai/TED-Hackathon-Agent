"""Skill loading system for the diagnostic agent.

Skills are Markdown files stored under ``agent_knowledge/skills/<name>/SKILL.md``.
Each skill encapsulates a diagnostic sub-routine (e.g. *diagnose*, *create-plan*,
*correlate-history*). The agent loads skills at runtime to guide its LM-powered reasoning.

Module-level constants
----------------------
SKILLS_BASE : Path
    Root directory containing skill sub-folders.
ALLOWED_SKILLS : frozenset[str]
    Names of skills that have a valid ``SKILL.md`` file.
"""

from __future__ import annotations

from typing import ClassVar

from .config import config


def _discover_allowed_skills() -> frozenset[str]:
    """Scan ``config.skills_base`` for directories containing a ``SKILL.md`` file."""
    skills_dir = config.skills_base
    if not skills_dir.exists():
        return frozenset()
    return frozenset(
        d.name for d in skills_dir.iterdir()
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

        Supports two formats:
        - ``<skill_name>``          → loads ``SKILLS_BASE/<skill_name>/SKILL.md``
        - ``<skill>/refs/<file>.md`` → loads ``SKILLS_BASE/<skill>/references/<file>.md``

        Raises
        ------
        FileNotFoundError
            If the skill file does not exist.
        """
        # Check for reference path format: diagnose/references/surge or diagnose/refs/surge
        if "/" in skill_name:
            parts = skill_name.split("/", 1)
            parent_skill = parts[0]
            ref_part = parts[1]
            # Normalise "refs/" to "references/"
            if ref_part.startswith("refs/"):
                ref_part = "references/" + ref_part[5:]
            skill_path = config.skills_base / parent_skill / ref_part
            if not skill_path.exists():
                # Try appending .md
                skill_path = config.skills_base / parent_skill / f"{ref_part}.md"
        else:
            skill_path = config.skills_base / skill_name / "SKILL.md"

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
        is_reference = "/" in skill_name
        if not is_reference and skill_name not in ALLOWED_SKILLS:
            available = sorted(ALLOWED_SKILLS)
            return (
                f"Skill '{skill_name}' not found. Available skills: {available}"
            )
        try:
            skill = Skill.load(skill_name)
            return skill.content
        except FileNotFoundError as e:
            available = sorted(ALLOWED_SKILLS)
            return (
                f"Skill '{skill_name}' not found. Available skills: {available} | {e}"
            )
