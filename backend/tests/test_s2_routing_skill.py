from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_routing_skill_has_if_else_logic():
    skill_path = Path("backend/agent_knowledge/skills/routing/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    families = ["SURGE", "DROPOUT", "INSTABILITY", "DEVIATION", "TRANSITION"]
    for family in families:
        assert family in content
        assert f' abnormal_family == "{family}"' in content or f"abnormal_family == '{family}'" in content


def test_routing_skill_loads_diagnose_skills():
    skill_path = Path("backend/agent_knowledge/skills/routing/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    diagnose_skills = [
        "diagnose_surge",
        "diagnose_dropout",
        "diagnose_instability",
        "diagnose_deviation",
        "diagnose_transition",
    ]
    for skill in diagnose_skills:
        assert skill in content


def test_routing_skill_has_create_plan_reference():
    skill_path = Path("backend/agent_knowledge/skills/routing/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    assert "create_plan" in content