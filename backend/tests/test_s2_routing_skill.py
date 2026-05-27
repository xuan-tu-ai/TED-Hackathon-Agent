from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

BASE = Path(__file__).resolve().parent.parent


def test_routing_skill_has_if_else_logic():
    skill_path = BASE / "agent_knowledge" / "skills" / "diagnose" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")

    sensors = ["wire_feed_speed", "wire_temperature", "welding_current"]
    for sensor in sensors:
        assert sensor in content


def test_routing_skill_loads_diagnose_references():
    skill_path = BASE / "agent_knowledge" / "skills" / "diagnose" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")

    references = [
        "diagnose/references/wire_feed_speed",
        "diagnose/references/wire_temperature",
        "diagnose/references/welding_current",
    ]
    for ref in references:
        assert ref in content


def test_routing_skill_has_task_reference():
    skill_path = BASE / "agent_knowledge" / "skills" / "diagnose" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "update_task_status" in content
