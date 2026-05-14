from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine

from ..db.models import Base, seed_maintenance_events
from .perception_results import seed_perception_results
from .pm_cards import seed_pm_cards


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
SQL_DIR = DATA_DIR / "sql"
DB_PATH = DATA_DIR / "demo.db"
SQL_PATH = SQL_DIR / "seed_snapshot.sql"


def seed_all(session, days: int = 45) -> dict[str, int]:
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

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH}")

    db_url = f"sqlite:///{DB_PATH}"
    engine = create_engine(db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import sessionmaker
    SF = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    print("Seeding 45-day demo data…")
    with SF() as session:
        counts = seed_all(session, days=45)

    print("Seeded rows:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")

    print(f"\nExporting SQL snapshot to {SQL_PATH}…")
    conn = sqlite3.connect(str(DB_PATH))
    with open(SQL_PATH, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(line + "\n")
    conn.close()

    size_kb = SQL_PATH.stat().st_size // 1024
    print(f"Done — {SQL_PATH.name} ({size_kb:,} KB)")


if __name__ == "__main__":
    run()
