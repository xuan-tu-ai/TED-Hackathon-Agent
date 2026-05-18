from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_alembic_ini_exists():
    ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    assert ini_path.exists()


def test_alembic_env_has_target_metadata():
    env_path = Path(__file__).resolve().parents[1] / "alembic" / "env.py"
    content = env_path.read_text()
    assert "target_metadata = Base.metadata" in content
    assert "from app.robot_agent.db.models import Base" in content


def test_alembic_env_reads_env_var():
    env_path = Path(__file__).resolve().parents[1] / "alembic" / "env.py"
    content = env_path.read_text()
    assert 'os.getenv("DATABASE_URL")' in content


def test_baseline_migration_generated():
    versions_dir = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    py_files = list(versions_dir.glob("*.py"))
    assert len(py_files) >= 1


def test_baseline_migration_has_8_tables():
    versions_dir = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    py_files = sorted(versions_dir.glob("*.py"))
    content = py_files[-1].read_text()
    for table in [
        "agent_memories",
        "alerts",
        "hitl_queue",
        "incidents",
        "maintenance_events",
        "perception_results",
        "pm_cards",
        "reports",
    ]:
        assert f"create_table('{table}'" in content


def test_alembic_upgrade_head_sqlite():
    import os
    import uuid
    from app.robot_agent.db.models import _engine_cache
    _engine_cache.clear()
    db_name = f"test_alembic_{uuid.uuid4().hex[:8]}.db"
    db_url = f"sqlite+pysqlite:///./data/{db_name}"
    os.environ["DATABASE_URL"] = db_url
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    from app.robot_agent.db.models import get_db_engine
    engine = get_db_engine(db_url)
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for table in [
        "agent_memories", "alerts", "hitl_queue", "incidents",
        "maintenance_events", "perception_results", "pm_cards", "reports",
    ]:
        assert table in tables, f"Missing table: {table}"

    command.downgrade(alembic_cfg, "base")
    engine.dispose()
    db_path = Path(__file__).resolve().parents[1] / "data" / db_name
    if db_path.exists():
        db_path.unlink()
