from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_create_plan_skill_md_exists():
    skill_path = Path("backend/agent_knowledge/skills/create_plan/SKILL.md")
    assert skill_path.exists()


def test_create_plan_template_exists():
    template_path = Path("backend/agent_knowledge/skills/create_plan/_template.md")
    assert template_path.exists()


def test_create_plan_skill_has_steps():
    skill_path = Path("backend/agent_knowledge/skills/create_plan/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    assert "Step 1" in content
    assert "Step 2" in content
    assert "Step 3" in content
    assert "Step 4" in content


def test_create_plan_skill_has_strict_rules():
    skill_path = Path("backend/agent_knowledge/skills/create_plan/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    assert "Do NOT" in content
    assert "template" in content.lower()


def test_create_plan_skill_has_session_termination_rule():
    skill_path = Path("backend/agent_knowledge/skills/create_plan/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    assert "Session Termination Rule" in content or "session is complete" in content.lower()


def test_create_plan_template_has_checklist():
    template_path = Path("backend/agent_knowledge/skills/create_plan/_template.md")
    content = template_path.read_text(encoding="utf-8")
    assert "Item 1" in content
    assert "Item 2" in content
    assert "Item 3" in content
    assert "Item 4" in content
    assert "Item 5" in content
    assert "Item 6" in content


def test_create_plan_template_has_plan_execution_status():
    template_path = Path("backend/agent_knowledge/skills/create_plan/_template.md")
    content = template_path.read_text(encoding="utf-8")
    assert "Plan Execution Status" in content
    assert "PENDING" in content