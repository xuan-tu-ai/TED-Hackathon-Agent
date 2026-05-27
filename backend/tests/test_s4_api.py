from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set TEST_DATABASE_URL so conftest fixtures use the same path.
_test_db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite+pysqlite:///./data/demo.db"
os.environ["TEST_DATABASE_URL"] = _test_db_url
os.environ["DATABASE_URL"] = _test_db_url

from fastapi.testclient import TestClient
import pytest

from app.robot_agent.api.hitl_api import router
from app.robot_agent.main import app


client = TestClient(app)


class TestHitlApi:
    def test_list_pending_empty(self, clean_db):
        response = client.get("/api/hitl/pending")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["entries"] == []

    def test_list_pending_with_entries(self, clean_db):
        from app.robot_agent.agent.hitl import hitl_check
        hitl_check(
            action="dispatch_maintenance",
            incident_id="INC-API-001",
            robot_code="R-07",
            parameters={"robot_code": "R-07"},
            reasoning="API test",
        )
        response = client.get("/api/hitl/pending")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["entries"][0]["action"] == "dispatch_maintenance"
        assert data["entries"][0]["severity"] == "CRITICAL"

    def test_resolve_approved(self, clean_db):
        from app.robot_agent.agent.hitl import hitl_check
        result = hitl_check(
            action="send_email",
            incident_id="INC-RES-API-001",
            robot_code="R-07",
            parameters={"to": "supervisor@example.com"},
            reasoning="API test resolve",
        )
        queue_id = result.queue_id

        response = client.post("/api/hitl/resolve", json={
            "queue_id": queue_id,
            "status": "APPROVED",
            "resolved_by": "supervisor@example.com",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "APPROVED"

    def test_resolve_rejected(self, clean_db):
        from app.robot_agent.agent.hitl import hitl_check
        result = hitl_check(
            action="send_email",
            incident_id="INC-RES-API-002",
            robot_code="R-07",
            parameters={"to": "team@example.com"},
            reasoning="API test reject",
        )
        queue_id = result.queue_id

        response = client.post("/api/hitl/resolve", json={
            "queue_id": queue_id,
            "status": "REJECTED",
            "resolved_by": "supervisor@example.com",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "REJECTED"

    def test_resolve_not_found(self, clean_db):
        response = client.post("/api/hitl/resolve", json={
            "queue_id": "nonexistent",
            "status": "APPROVED",
            "resolved_by": "supervisor",
        })
        assert response.status_code == 404
