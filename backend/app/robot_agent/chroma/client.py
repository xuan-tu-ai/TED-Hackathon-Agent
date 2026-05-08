from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings


CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = "pm_cards"):
    client = get_client()
    return client.get_or_create_collection(name=name)


def reset_client():
    client = get_client()
    client.reset()
    return client