from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS_BASE = Path(__file__).resolve().parents[1] / "backend" / "agent_knowledge" / "skills"
IMPLEMENTED_TOOLS = {"load_skill", "query_pm_cards", "hitl_check", "update_task_status", "create_plan", "complete"}


def _list_skill_dirs() -> list[Path]:
    if not SKILLS_BASE.exists():
        return []
    return [d for d in SKILLS_BASE.iterdir() if d.is_dir()]


def _parse_frontmatter(path: Path) -> dict | None:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    raw = parts[1].strip()
    result = {}
    for line in raw.splitlines():
        m = re.match(r"(\w+):\s*(.*)", line)
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip("'\"")
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
            result[key] = val
    return result if result else None


class TestKnowledgeStructure:
    def test_all_skill_dirs_have_skill_md(self):
        for d in _list_skill_dirs():
            skill_md = d / "SKILL.md"
            assert skill_md.exists(), f"Missing SKILL.md in {d}"

    def test_skill_frontmatter_has_required_fields(self):
        for d in _list_skill_dirs():
            skill_md = d / "SKILL.md"
            fm = _parse_frontmatter(skill_md)
            assert fm is not None, f"No YAML frontmatter in {skill_md}"
            assert "name" in fm, f"Missing 'name' in {skill_md}"
            assert "description" in fm, f"Missing 'description' in {skill_md}"

    def test_no_phantom_tools_in_required_tools(self):
        for d in _list_skill_dirs():
            skill_md = d / "SKILL.md"
            fm = _parse_frontmatter(skill_md)
            if fm is None:
                continue
            required = fm.get("required_tools", [])
            if isinstance(required, list):
                for tool in required:
                    assert tool in IMPLEMENTED_TOOLS, (
                        f"Skill '{d.name}' lists '{tool}' in required_tools but it is not implemented. "
                        f"Implemented tools: {IMPLEMENTED_TOOLS}"
                    )


class TestLoadSkillTool:
    def test_unknown_name_returns_string(self):
        from app.robot_agent.agent.skills import LoadSkillTool

        tool = LoadSkillTool()
        result = tool("nonexistent_skill_xyz")
        assert isinstance(result, str)
        assert "not found" in result.lower() or "error" in result.lower()

    def test_known_name_returns_content(self):
        from app.robot_agent.agent.skills import LoadSkillTool

        tool = LoadSkillTool()
        result = tool("routing")
        assert isinstance(result, str)
        assert len(result) > 50


class TestDiagnosticAgentCleanup:
    def test_dead_stubs_removed(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent

        agent = DiagnosticAgent()
        assert not hasattr(agent, "explore")
        assert not hasattr(agent, "plan") or not callable(getattr(agent, "plan", None))

    def test_no_query_db_stub(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent

        agent = DiagnosticAgent()
        assert "query_db" not in dir(agent) or "query_db" not in agent.AVAILABLE_TOOLS

    def test_emitter_replaces_callbacks(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent

        agent = DiagnosticAgent()
        assert hasattr(agent, "emitter")
        assert hasattr(agent.emitter, "emit_step")
        assert not hasattr(agent, "on_step")
