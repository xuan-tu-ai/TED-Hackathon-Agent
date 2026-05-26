"""RAG tools for querying and indexing PM Cards.

Provides DSPy-compatible tools that let the diagnostic agent search
historical Preventive Maintenance (PM) cards and index new records.

PM Cards are queried from Databricks Delta Lake when ``databricks_configured``
is True, with a fallback to the local ChromaDB vector store for backward
compatibility during development.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from sqlalchemy import select, text

from ..agent.config import config
from ..chroma.client import get_collection

logger = logging.getLogger(__name__)


def _query_databricks_pm_cards(query_text: str, limit: int = 5) -> list[dict[str, Any]]:
    """Query PM Cards directly from the Databricks ``pm_cards`` table.

    Returns a list of card dicts with ``card_id``, ``sensor``, ``robot_code``,
    ``cause``, ``symptom``, and ``action`` keys.
    """
    try:
        from ..db.databricks_session import create_databricks_session_factory

        sf = create_databricks_session_factory()
        session = sf()
        try:
            stmt = (
                select(
                    text("card_id"),
                    text("sensor"),
                    text("robot_code"),
                    text("cause"),
                    text("symptom"),
                    text("action"),
                )
                .select_from(text("pm_cards"))
                .where(
                    text(
                        "cause ILIKE :q OR symptom ILIKE :q OR action ILIKE :q"
                    )
                )
                .limit(limit)
            )
            rows = session.execute(stmt, {"q": f"%{query_text}%"}).fetchall()
            cards = []
            for row in rows:
                cards.append({
                    "id": row.card_id,
                    "content": f"Cause: {row.cause}\nSymptom: {row.symptom}\nAction: {row.action}",
                    "metadata": {
                        "card_id": row.card_id,
                        "sensor": row.sensor,
                        "robot_code": row.robot_code,
                        "cause": row.cause,
                        "symptom": row.symptom,
                        "action": row.action,
                        "source": "databricks",
                    },
                })
            return cards
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Databricks PMCard query failed, falling back: %s", exc)
        return []


class QueryPMCardsTool:
    """Search PM Cards to find similar past incidents.

    When Databricks is configured, queries the ``pm_cards`` Delta Lake table
    directly via SQL (ILIKE on cause/symptom/action). Falls back to ChromaDB
    vector search when Databricks is unavailable.

    Parameters
    ----------
    query : str
        Free-text natural language query.
    limit : int
        Maximum number of results (default 5).
    """

    name: ClassVar[str] = "query_pm_cards"
    description: ClassVar[str] = "Search PM Cards to find similar past incidents. Use this to correlate current incident with historical maintenance records."

    def __call__(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        # Prefer Databricks when configured
        if config.databricks_configured:
            cards = _query_databricks_pm_cards(query, limit=limit)
            if cards:
                return cards
            logger.info("Databricks query returned no results, falling back to ChromaDB")

        # Fallback: ChromaDB vector search
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
