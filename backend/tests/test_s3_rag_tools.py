from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_query_pm_cards_tool_instantiation():
    from app.robot_agent.agent.rag_tools import QueryPMCardsTool

    tool = QueryPMCardsTool()
    assert tool is not None
    assert tool.name == "query_pm_cards"


def test_index_pm_cards_tool_instantiation():
    from app.robot_agent.agent.rag_tools import IndexPMCardsTool

    tool = IndexPMCardsTool()
    assert tool is not None
    assert tool.name == "index_pm_card"


def test_query_pm_cards_fallback_search():
    from app.robot_agent.agent.rag_tools import QueryPMCardsTool

    tool = QueryPMCardsTool()
    results = tool("paint", limit=2)
    assert isinstance(results, list)
    assert len(results) <= 2


def test_query_pm_cards_returns_correct_fields():
    from app.robot_agent.agent.rag_tools import QueryPMCardsTool

    tool = QueryPMCardsTool()
    results = tool("paint", limit=1)
    if results:
        assert "id" in results[0] or "card_id" in results[0] or "cause" in results[0]


def test_get_rag_tools_returns_list():
    from app.robot_agent.agent.rag_tools import get_rag_tools

    tools = get_rag_tools()
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert tools[0].name == "query_pm_cards"
    assert tools[1].name == "index_pm_card"