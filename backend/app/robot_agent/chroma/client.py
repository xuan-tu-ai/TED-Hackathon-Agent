from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings


_CHROMA_DIR = Path(__file__).resolve().parents[4] / "data" / "chroma_db"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_client():
    return chromadb.PersistentClient(path=str(_CHROMA_DIR))


def get_collection(name: str = "pm_cards"):
    client = get_client()
    return client.get_or_create_collection(name=name)


def reset_client():
    client = get_client()
    client.reset()
    return client