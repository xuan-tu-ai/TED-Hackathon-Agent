from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


class TestGetDefaultDbUrl:
    def test_default_fallback(self):
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        from app.robot_agent.db.models import get_default_db_url
        url = get_default_db_url()
        assert url.startswith("sqlite+pysqlite:///")

    def test_respects_env_var(self):
        os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost/mydb"
        try:
            from app.robot_agent.db.models import get_default_db_url
            url = get_default_db_url()
            assert url == "postgresql+psycopg://user:pass@localhost/mydb"
        finally:
            os.environ.pop("DATABASE_URL", None)

    def test_normalizes_legacy_sqlite_scheme(self):
        os.environ["DATABASE_URL"] = "sqlite:///data/demo.db"
        try:
            from app.robot_agent.db.models import get_default_db_url
            url = get_default_db_url()
            assert url == "sqlite+pysqlite:///data/demo.db"
        finally:
            os.environ.pop("DATABASE_URL", None)


class TestDbUrlHelpers:
    def test_is_sqlite_true(self):
        from app.robot_agent.db.models import is_sqlite
        assert is_sqlite("sqlite+pysqlite:///data/demo.db")
        assert is_sqlite("sqlite:///data/demo.db")

    def test_is_sqlite_false(self):
        from app.robot_agent.db.models import is_sqlite
        assert not is_sqlite("postgresql+psycopg://localhost/db")

    def test_is_postgres_true(self):
        from app.robot_agent.db.models import is_postgres
        assert is_postgres("postgresql+psycopg://localhost/db")
        assert is_postgres("postgres://localhost/db")

    def test_is_postgres_false(self):
        from app.robot_agent.db.models import is_postgres
        assert not is_postgres("sqlite+pysqlite:///data/demo.db")
