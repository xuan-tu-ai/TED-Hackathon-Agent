from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


class TestGetDbEngine:
    def test_sqlite_engine_dialect(self):
        from app.robot_agent.db.models import get_db_engine
        engine = get_db_engine("sqlite+pysqlite:///./data/test.db")
        assert engine.dialect.name == "sqlite"

    def test_postgresql_engine_dialect(self):
        from app.robot_agent.db.models import get_db_engine
        engine = get_db_engine("postgresql+psycopg://user:pass@localhost/mydb")
        assert engine.dialect.name == "postgresql"

    def test_sqlite_check_same_thread(self):
        from app.robot_agent.db.models import get_db_engine
        engine = get_db_engine("sqlite+pysqlite:///./data/test.db")
        assert engine.dialect.name == "sqlite"

    def test_azure_warning_no_sslmode(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        from app.robot_agent.db.models import get_db_engine
        engine = get_db_engine(
            "postgresql+psycopg://user:pass@myhost.postgres.database.azure.com:5432/robot_agent"
        )
        assert engine.dialect.name == "postgresql"
        assert "sslmode" in caplog.text

    def test_azure_no_warning_with_sslmode(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        from app.robot_agent.db.models import get_db_engine
        engine = get_db_engine(
            "postgresql+psycopg://user:pass@myhost.postgres.database.azure.com:5432/robot_agent?sslmode=require"
        )
        assert engine.dialect.name == "postgresql"
        assert not any("sslmode" in msg for msg in caplog.text)

    def test_engine_cache_reuses(self):
        from app.robot_agent.db.models import get_db_engine, _engine_cache
        _engine_cache.clear()
        e1 = get_db_engine("sqlite+pysqlite:///./data/test.db")
        e2 = get_db_engine("sqlite+pysqlite:///./data/test.db")
        assert e1 is e2
