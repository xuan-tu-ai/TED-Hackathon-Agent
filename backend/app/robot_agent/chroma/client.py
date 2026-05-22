"""ChromaDB persistent client wrapper.

Creates / reuses a local persistent ChromaDB instance stored under
``<project_root>/data/chroma_db/`` and exposes helpers to get or
create named collections (primarily ``pm_cards``).
"""

from __future__ import annotations

import chromadb


def _get_chroma_dir():
    """Lazy import to avoid circular dependency via agent.__init__."""
    from ..agent.config import config
    return config.chroma_db_dir


def get_client():
    chroma_dir = _get_chroma_dir()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(name: str = "pm_cards"):
    client = get_client()
    return client.get_or_create_collection(name=name)


def reset_client():
    client = get_client()
    client.reset()
    return client