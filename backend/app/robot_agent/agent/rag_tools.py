"""RAG tools for querying and indexing ChromaDB vector stores.

Provides DSPy-compatible tools that let the diagnostic agent search
historical Preventive Maintenance (PM) cards and index new records.

The underlying ChromaDB collection is managed via
:func:`~backend.app.robot_agent.chroma.client.get_collection`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..chroma.client import get_collection


class QueryPMCardsTool:
    """Search PM Cards via RAG to find similar past incidents.

    Queries the ``pm_cards`` ChromaDB collection and returns a list of
    matching documents with their metadata and IDs.

    Parameters
    ----------
    query : str
        Free-text natural language query.
    limit : int
        Maximum number of results (default 5).
    """

    name: ClassVar[str] = "query_pm_cards"
    description: ClassVar[str] = "Search PM Cards via RAG to find similar past incidents. Use this to correlate current incident with historical maintenance records."

    def __call__(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            collection = get_collection("pm_cards")
            try:
                count = collection.count()
                if count == 0:
                    return [{"error": "ChromaDB collection is empty", "detail": "pm_cards"}]
            except Exception as exc:
                return [{"error": "ChromaDB unavailable", "detail": str(exc)}]

            results = collection.query(query_texts=[query], n_results=limit)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]

            if not documents:
                return [{"error": "No results from ChromaDB", "detail": query}]

            cards = []
            for i, doc in enumerate(documents):
                cards.append({
                    "id": ids[i] if i < len(ids) else None,
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return cards
        except Exception as exc:
            return [{"error": "ChromaDB unavailable", "detail": str(exc)}]


class IndexPMCardsTool:
    """Index a PM Card into the ChromaDB vector store for RAG.

    Upserts a document into the ``pm_cards`` collection so it becomes
    discoverable by :class:`QueryPMCardsTool`.
    """

    name: ClassVar[str] = "index_pm_card"
    description: ClassVar[str] = "Index a PM Card into the ChromaDB vector store for RAG."

    def __call__(self, card_id: str, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            collection = get_collection("pm_cards")
            collection.upsert(
                ids=[card_id],
                documents=[text],
                metadatas=[metadata or {}],
            )
            return {"status": "indexed", "card_id": card_id}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


def get_rag_tools() -> list:
    """Return a list of all available RAG tool instances."""
    return [QueryPMCardsTool(), IndexPMCardsTool()]
