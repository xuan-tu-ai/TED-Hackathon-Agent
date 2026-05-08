from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DATABASE_URL"] = "sqlite:///data/demo.db"

from fastapi.testclient import TestClient
import pytest

from app.robot_agent.main import app


client = TestClient(app)


class TestIncidentsApi:
    def test_list_incidents_empty(self):
        response = client.get("/api/incidents/")
        assert response.status_code == 200
        data = response.json()
        assert "incidents" in data
        assert "count" in data
        assert isinstance(data["incidents"], list)

    def test_list_incidents_with_limit(self):
        response = client.get("/api/incidents/?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] <= 10

    def test_health_endpoint(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_trace_not_found_returns_404(self):
        response = client.get("/api/incidents/NONEXISTENT/trace")
        assert response.status_code == 404


class TestSessionTrace:
    def test_write_and_read_trace(self, clean_db, tmp_path):
        import uuid
        unique_id = f"INC-TRACE-{uuid.uuid4().hex[:8]}"
        from app.robot_agent.agent.session_trace import SessionTraceWriter
        writer = SessionTraceWriter(unique_id)
        writer.write_step(f"sess-{unique_id}-step-1", "load_skill(routing)", "Loading routing", skill_loaded="routing")
        writer.write_step(f"sess-{unique_id}-step-2", "load_skill(diagnose_surge)", "Loading diagnose_surge", skill_loaded="diagnose_surge")
        entries = writer.read_trace()
        assert len(entries) == 2
        assert entries[0]["skill_loaded"] == "routing"
        assert entries[1]["action"] == "load_skill(diagnose_surge)"
