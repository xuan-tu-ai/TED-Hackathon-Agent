from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.skills import Skill


SKILL_NAMES = [
    "diagnose",
    "correlate-history",
    "verify-defect",
    "check-fmea",
]


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_domain_skill_loads(skill_name):
    skill = Skill.load(skill_name)
    assert isinstance(skill.content, str)
    assert len(skill.content) > 0


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_domain_skill_has_yaml_frontmatter(skill_name):
    skill = Skill.load(skill_name)
    assert skill.content.startswith("---")
    assert f"name: {skill_name}" in skill.content


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_domain_skill_has_description(skill_name):
    skill = Skill.load(skill_name)
    assert "description:" in skill.content


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_domain_skill_has_required_tools(skill_name):
    skill = Skill.load(skill_name)
    assert "required_tools:" in skill.content


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_domain_skill_has_purpose_section(skill_name):
    skill = Skill.load(skill_name)
    assert "#" in skill.content
