"""Entry-point script for demo database seeding and SQL snapshot export.

:func:`run` creates a fresh database at ``data/demo.db`` (or the configured
``DATABASE_URL``), seeds it with 45 days of ``PerceptionResult`` data plus
PM Cards and maintenance events, then dumps the full database as an SQL
script to ``data/sql/seed_snapshot.sql`` (SQLite only).

Can be invoked directly: ``uv run python app/robot_agent/seed/run.py``.

Functions:
    seed_all: Seed perception results, PM cards, and maintenance events.
    seed_demo_data: Alias for ``seed_all`` with default 45-day window.
"""

from __future__ import annotations

from pathlib import Path

from ..db.models import Base, create_db_engine, create_session_factory, seed_maintenance_events
from .perception_results import seed_perception_results
from .pm_cards import seed_pm_cards


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = DATA_DIR / "sql"
DB_PATH = DATA_DIR / "demo.db"
SQL_PATH = SQL_DIR / "seed_snapshot.sql"


def seed_all(session, days: int = 30) -> dict[str, int]:
    perception_count = seed_perception_results(session, days=days)
    pm_cards_count = seed_pm_cards(session)
    maintenance_count = seed_maintenance_events(session)
    session.commit()
    return {
        "perception_results": perception_count,
        "pm_cards": pm_cards_count,
        "maintenance_events": maintenance_count,
    }


def seed_demo_data(session, days: int = 45) -> dict[str, int]:
    return seed_all(session, days=days)


def run() -> None:
    SQL_DIR.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite+pysqlite:///{DB_PATH.as_posix()}"
    engine = create_db_engine(db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    SF = create_session_factory(db_url)

    print("Seeding 45-day demo data…")
    with SF() as session:
        counts = seed_all(session, days=45)

    print("Seeded rows:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")

    print(f"\nExporting SQL snapshot to {SQL_PATH}…")
    try:
        raw_conn = engine.raw_connection()
        with open(SQL_PATH, "w", encoding="utf-8") as f:
            for line in raw_conn.iterdump():
                f.write(line + "\n")
        raw_conn.close()
    except AttributeError:
        print("  (SQL dump skipped — only available for SQLite)")

    size_kb = SQL_PATH.stat().st_size // 1024
    print(f"Done — {SQL_PATH.name} ({size_kb:,} KB)")


if __name__ == "__main__":
    run()
