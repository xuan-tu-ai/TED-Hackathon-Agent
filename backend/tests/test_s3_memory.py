from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_tier1_instantiation():
    from app.robot_agent.memory.memory import Tier1SessionMemory

    mem = Tier1SessionMemory()
    assert mem is not None
    assert mem.session_id.startswith("sess-")


def test_tier1_add_turn():
    from app.robot_agent.memory.memory import Tier1SessionMemory

    mem = Tier1SessionMemory(session_id="test-sess-001")
    turn = mem.add_turn(skill_loaded="routing", action="load_skill", reasoning="Test reasoning")
    assert turn.turn_id == "test-sess-001-t1"
    assert turn.skill_loaded == "routing"
    assert mem.get_turn_count() == 1


def test_tier1_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr("app.robot_agent.memory.memory.SESSION_DIR", tmp_path / "sessions")

    from app.robot_agent.memory.memory import Tier1SessionMemory

    mem = Tier1SessionMemory(session_id="test-save-001")
    mem.add_turn(skill_loaded="routing", action="load_skill")
    mem.save()

    loaded = Tier1SessionMemory()
    turns = loaded.load("test-save-001")
    assert len(turns) == 1
    assert turns[0].skill_loaded == "routing"


def test_tier0_needs_compaction_false_under_limit():
    from app.robot_agent.memory.memory import Tier0ContextCompaction, Tier1SessionMemory

    mem = Tier1SessionMemory(session_id="compact-test-001")
    for i in range(5):
        mem.add_turn(skill_loaded=f"skill_{i}", action=f"action_{i}")

    compactor = Tier0ContextCompaction(mem)
    assert compactor.needs_compaction() is False


def test_tier0_needs_compaction_true_at_limit():
    from app.robot_agent.memory.memory import Tier0ContextCompaction, Tier1SessionMemory

    mem = Tier1SessionMemory(session_id="compact-test-002")
    for i in range(25):
        mem.add_turn(skill_loaded=f"skill_{i}", action=f"action_{i}")

    compactor = Tier0ContextCompaction(mem)
    assert compactor.needs_compaction() is True


def test_tier0_compact():
    from app.robot_agent.memory.memory import Tier0ContextCompaction, Tier1SessionMemory

    mem = Tier1SessionMemory(session_id="compact-test-003")
    for i in range(25):
        mem.add_turn(skill_loaded=f"skill_{i}", action=f"action_{i}")

    compactor = Tier0ContextCompaction(mem)
    summary = compactor.compact()

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert mem.get_turn_count() == 6  # 25 - 20 + 1 = 6 (20 removed + 1 summary turn + original 5 early turns)


def test_tier2_list_rules():
    from app.robot_agent.memory.memory import Tier2ProjectRules

    rules = Tier2ProjectRules.list_rules()
    assert isinstance(rules, list)


def test_tier3_instantiation():
    from app.robot_agent.memory.memory import Tier3SemanticMemory

    mem = Tier3SemanticMemory()
    assert mem is not None