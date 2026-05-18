from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_psycopg_importable():
    import psycopg
    assert psycopg.__version__ is not None


def test_alembic_importable():
    import alembic
    assert alembic.__version__ is not None


def test_env_example_has_postgres_variants():
    env_path = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_path.read_text()
    assert "postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent" in content
    assert "sslmode=require" in content
    assert "sqlite+pysqlite:///./data/demo.db" in content
