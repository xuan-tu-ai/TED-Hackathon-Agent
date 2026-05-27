from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_chroma_dir_created():
    from app.robot_agent.chroma.client import get_client
    client = get_client()
    assert client is not None
    from app.robot_agent.agent.config import config
    assert config.chroma_db_dir.exists()


def test_chroma_get_client():
    from app.robot_agent.chroma.client import get_client
    client = get_client()
    assert client is not None


def test_chroma_get_collection():
    from app.robot_agent.chroma.client import get_collection
    collection = get_collection("pm_cards")
    assert collection is not None
    assert collection.name == "pm_cards"


def test_chroma_persistent_path():
    from app.robot_agent.agent.config import config
    assert str(config.chroma_db_dir).endswith("chroma_db")