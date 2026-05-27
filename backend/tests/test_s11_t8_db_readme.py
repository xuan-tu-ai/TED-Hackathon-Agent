from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_db_readme_exists():
    readme = Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "db" / "README.md"
    assert readme.exists()


def test_db_readme_has_sqlite_section():
    readme = Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "db" / "README.md"
    content = readme.read_text()
    assert "SQLite" in content and "Postgres" in content


def test_db_readme_has_databas_url_variants():
    readme = Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "db" / "README.md"
    content = readme.read_text()
    assert "DATABASE_URL" in content
    assert "alembic upgrade head" in content
    assert "jsonb" in content
