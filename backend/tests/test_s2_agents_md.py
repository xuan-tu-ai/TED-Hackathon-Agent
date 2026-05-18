from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

AGENTS_PATH = Path(__file__).resolve().parent.parent / "agent_knowledge" / "AGENTS.md"
DIAGNOSE_PATH = Path(__file__).resolve().parent.parent / "agent_knowledge" / "skills" / "diagnose" / "SKILL.md"


def test_agents_md_exists():
    assert AGENTS_PATH.exists()


def test_agents_md_has_diagnose_instruction():
    content = AGENTS_PATH.read_text(encoding="utf-8")
    assert "load_skill" in content
    assert "diagnose" in content


def test_agents_md_has_signal_families():
    content = DIAGNOSE_PATH.read_text(encoding="utf-8")
    families = ["SURGE", "DROPOUT", "INSTABILITY", "DEVIATION", "TRANSITION"]
    for family in families:
        assert family in content


def test_agents_md_has_session_rules():
    content = AGENTS_PATH.read_text(encoding="utf-8")
    assert "Plan" in content
    assert "Execute Tasks" in content
    assert "Verify" in content
    assert "Correlate" in content
    assert "Check FMEA" in content
    assert "HITL" in content
    assert "Complete" in content
