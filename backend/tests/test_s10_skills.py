from __future__ import annotations

import re
from pathlib import Path

import pytest


SKILLS_BASE = Path(__file__).resolve().parents[1] / "agent_knowledge" / "skills"


def _discover_skill_dirs() -> list[Path]:
    """Return all sub-directories under SKILLS_BASE that have a SKILL.md."""
    if not SKILLS_BASE.exists():
        return []
    return sorted(
        d for d in SKILLS_BASE.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    )


def _discover_all_dirs() -> list[Path]:
    """Return all sub-directories under SKILLS_BASE (including references)."""
    if not SKILLS_BASE.exists():
        return []
    return sorted(d for d in SKILLS_BASE.iterdir() if d.is_dir())


def _extract_yaml_name(content: str) -> str | None:
    """Extract the ``name`` field from YAML frontmatter (``---...---``)."""
    match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    frontmatter = match.group(1)
    name_match = re.search(r"^name:\s*(\S+)", frontmatter, re.MULTILINE)
    return name_match.group(1) if name_match else None


class TestSkillNaming:
    """Sprint 10 T11: Validate naming conventions."""

    def test_no_underscores_in_folder_names(self):
        """All skill folder names must use hyphens, not underscores."""
        bad = []
        for d in _discover_all_dirs():
            name = d.name
            if "_" in name:
                bad.append(name)
        assert not bad, (
            f"Found skill folder names with underscores: {bad}. "
            "Use hyphens (`-`) instead."
        )

    def test_yaml_names_use_hyphens(self):
        """All SKILL.md YAML ``name`` fields must use hyphens, not underscores."""
        bad = []
        for d in _discover_skill_dirs():
            content = (d / "SKILL.md").read_text(encoding="utf-8")
            yaml_name = _extract_yaml_name(content)
            if yaml_name and "_" in yaml_name:
                bad.append((d.name, yaml_name))
        assert not bad, (
            f"Found YAML names with underscores: {bad}. "
            "Use hyphens (`-`) instead."
        )

    def test_skill_folder_names_match_yaml_names(self):
        """Skill folder names should match their YAML ``name`` field."""
        mismatches = []
        for d in _discover_skill_dirs():
            content = (d / "SKILL.md").read_text(encoding="utf-8")
            yaml_name = _extract_yaml_name(content)
            if yaml_name and yaml_name != d.name:
                mismatches.append((d.name, yaml_name))
        assert not mismatches, (
            f"Folder name != YAML name: {mismatches}"
        )


class TestAllSkillsLoad:
    """Sprint 10 T11: Verify all skills can be loaded."""

    def test_all_skills_are_in_allowed_set(self):
        """All skill directories with SKILL.md appear in ALLOWED_SKILLS."""
        from app.robot_agent.agent.skills import ALLOWED_SKILLS

        expected = {d.name for d in _discover_skill_dirs()}
        assert ALLOWED_SKILLS == expected, (
            f"Mismatch. ALLOWED_SKILLS={sorted(ALLOWED_SKILLS)} "
            f"!= expected={sorted(expected)}"
        )

    def test_each_skill_loads_successfully(self):
        """Calling LoadSkillTool for each skill returns non-error content."""
        from app.robot_agent.agent.skills import LoadSkillTool

        tool = LoadSkillTool()
        for d in _discover_skill_dirs():
            result = tool(d.name)
            assert "not found" not in result.lower(), (
                f"Skill '{d.name}' failed to load: {result[:200]}"
            )
            assert len(result) > 10, (
                f"Skill '{d.name}' loaded but content is too short ({len(result)} chars)"
            )


class TestDiagnoseReferences:
    """Sprint 24: Verify sensor-based diagnose/references files load correctly."""

    SENSOR_NAMES = ["wire_feed_speed", "wire_temperature", "welding_current"]

    def test_reference_files_exist(self):
        """All 3 sensor reference files exist under diagnose/references/."""
        refs_dir = SKILLS_BASE / "diagnose" / "references"
        assert refs_dir.is_dir(), (
            f"diagnose/references directory not found at {refs_dir}"
        )
        for name in self.SENSOR_NAMES:
            ref_path = refs_dir / f"{name}.md"
            assert ref_path.exists(), (
                f"Reference file missing: {ref_path}"
            )

    def test_reference_files_have_content(self):
        """Each reference file has meaningful content (>50 chars)."""
        refs_dir = SKILLS_BASE / "diagnose" / "references"
        for name in self.SENSOR_NAMES:
            content = (refs_dir / f"{name}.md").read_text(encoding="utf-8")
            assert len(content) > 50, (
                f"Reference '{name}.md' content too short ({len(content)} chars)"
            )

    def test_references_load_via_skill_tool(self):
        """References load successfully via LoadSkillTool with path syntax."""
        from app.robot_agent.agent.skills import LoadSkillTool

        tool = LoadSkillTool()
        for name in self.SENSOR_NAMES:
            ref_path = f"diagnose/references/{name}"
            result = tool(ref_path)
            assert "not found" not in result.lower(), (
                f"Reference '{ref_path}' failed to load: {result[:200]}"
            )
            assert len(result) > 50, (
                f"Reference '{ref_path}' loaded but content too short ({len(result)} chars)"
            )

    def test_refs_prefix_alias_works(self):
        """Short alias 'diagnose/refs/wire_feed_speed' also loads correctly."""
        from app.robot_agent.agent.skills import LoadSkillTool

        tool = LoadSkillTool()
        result = tool("diagnose/refs/wire_feed_speed")
        assert "not found" not in result.lower()
        assert len(result) > 50


class TestNoOrphanSkills:
    """Sprint 10 T11: Verify no orphan or empty skill directories."""

    def test_no_empty_skill_dirs(self):
        """Every skill directory must have a non-empty SKILL.md."""
        empty = []
        for d in _discover_all_dirs():
            skill_md = d / "SKILL.md"
            if skill_md.exists():
                content = skill_md.read_text(encoding="utf-8")
                if len(content.strip()) == 0:
                    empty.append(d.name)
            else:
                empty.append(d.name)
        assert not empty, (
            f"Skill directories missing or empty SKILL.md: {empty}"
        )
