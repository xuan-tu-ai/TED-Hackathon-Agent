from __future__ import annotations

from typing import Any, ClassVar

from ..chroma.client import get_collection
from ..db import create_session_factory
from ..db.models import PMCard


class QueryPMCardsTool:
    name: ClassVar[str] = "query_pm_cards"
    description: ClassVar[str] = "Search PM Cards via RAG to find similar past incidents. Use this to correlate current incident with historical maintenance records."

    def __call__(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            collection = get_collection("pm_cards")
            try:
                count = collection.count()
                if count == 0:
                    return self._fallback_search(query, limit)
            except Exception:
                return self._fallback_search(query, limit)

            results = collection.query(query_texts=[query], n_results=limit)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]

            if not documents:
                return self._fallback_search(query, limit)

            cards = []
            for i, doc in enumerate(documents):
                cards.append({
                    "id": ids[i] if i < len(ids) else None,
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return cards
        except Exception:
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        db_url = "sqlite:///data/demo.db"
        sf = create_session_factory(db_url)
        session = sf()

        query_lower = query.lower()
        cards = (
            session.query(PMCard)
            .filter(
                (PMCard.cause.ilike(f"%{query_lower}%")) |
                (PMCard.symptom.ilike(f"%{query_lower}%")) |
                (PMCard.action.ilike(f"%{query_lower}%")) |
                (PMCard.sensor.ilike(f"%{query_lower}%"))
            )
            .limit(limit)
            .all()
        )

        results = []
        for card in cards:
            results.append({
                "id": card.card_id,
                "sensor": card.sensor,
                "robot_code": card.robot_code,
                "cause": card.cause,
                "symptom": card.symptom,
                "action": card.action,
            })

        session.close()
        return results


class IndexPMCardsTool:
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
    return [QueryPMCardsTool(), IndexPMCardsTool()]
