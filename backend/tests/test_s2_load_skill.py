from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.skills import Skill, LoadSkillTool


def test_skill_load_returns_content():
    skill = Skill.load("routing")
    assert isinstance(skill.content, str)
    assert len(skill.content) > 0


def test_skill_load_unknown_raises():
    with pytest.raises(FileNotFoundError):
        Skill.load("nonexistent_skill")


def test_load_skill_tool_call():
    tool = LoadSkillTool()
    result = tool("routing")
    assert isinstance(result, str)
    assert len(result) > 0


def test_load_skill_tool_unknown():
    tool = LoadSkillTool()
    result = tool("unknown_skill")
    assert isinstance(result, str)
    assert "not found" in result.lower()