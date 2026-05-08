from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

DIAGNOSE_SKILLS = [
    "diagnose_surge",
    "diagnose_dropout",
    "diagnose_instability",
    "diagnose_deviation",
    "diagnose_transition",
]


AGENT_KNOWLEDGE = Path("backend/agent_knowledge")


@pytest.mark.parametrize("skill", DIAGNOSE_SKILLS)
def test_diagnose_skill_has_hitl_section(skill):
    skill_path = AGENT_KNOWLEDGE / "skills" / skill / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "## HITL Checkpoints" in content or "HITL Checkpoints" in content


@pytest.mark.parametrize("skill", DIAGNOSE_SKILLS)
def test_diagnose_skill_mentions_critical_actions(skill):
    skill_path = AGENT_KNOWLEDGE / "skills" / skill / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "dispatch_maintenance" in content or "CRITICAL" in content


@pytest.mark.parametrize("skill", DIAGNOSE_SKILLS)
def test_diagnose_skill_mentions_high_actions(skill):
    skill_path = AGENT_KNOWLEDGE / "skills" / skill / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "send_email" in content or "update_robot_config" in content or "HIGH" in content


@pytest.mark.parametrize("skill", DIAGNOSE_SKILLS)
def test_diagnose_skill_calls_hitl_check(skill):
    skill_path = AGENT_KNOWLEDGE / "skills" / skill / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "hitl_check" in content.lower()


@pytest.mark.parametrize("skill", DIAGNOSE_SKILLS)
def test_diagnose_skill_references_create_plan(skill):
    skill_path = AGENT_KNOWLEDGE / "skills" / skill / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "create_plan" in content


def test_routing_skill_has_hitl_checkpoints():
    skill_path = AGENT_KNOWLEDGE / "skills" / "routing" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "HITL Checkpoints" in content or "hitl_check" in content.lower()