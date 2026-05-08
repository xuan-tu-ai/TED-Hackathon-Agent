from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_agents_md_exists():
    agents_path = Path("backend/agent_knowledge/AGENTS.md")
    assert agents_path.exists()


def test_agents_md_has_routing_instruction():
    agents_path = Path("backend/agent_knowledge/AGENTS.md")
    content = agents_path.read_text(encoding="utf-8")
    assert "load_skill" in content
    assert "routing" in content


def test_agents_md_has_signal_families():
    agents_path = Path("backend/agent_knowledge/AGENTS.md")
    content = agents_path.read_text(encoding="utf-8")
    families = ["INSTABILITY", "DEVIATION", "TRANSITION", "DROPOUT", "SURGE"]
    for family in families:
        assert family in content


def test_agents_md_has_session_rules():
    agents_path = Path("backend/agent_knowledge/AGENTS.md")
    content = agents_path.read_text(encoding="utf-8")
    assert "Start" in content
    assert "Route" in content
    assert "Diagnose" in content
    assert "Plan" in content
    assert "Complete" in content