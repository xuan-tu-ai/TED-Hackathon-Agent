"""Quick connectivity test for the 3 Databricks OT tables.

Run:
    uv run python scripts/test_databricks_connection.py

Prints row counts and sample rows for each table, then tests PM Cards
Vector Search if the index is configured.
"""

import os
import sys
from pathlib import Path

# Make sure .env is loaded from the backend directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import text

def _engine():
    from app.robot_agent.db.databricks_session import get_databricks_engine
    return get_databricks_engine()

def _tables():
    from app.robot_agent.db.databricks_session import (
        DB_TABLE_PERCEPTION_RESULTS,
        DB_TABLE_MAINTENANCE_EVENTS,
        DB_TABLE_PM_CARDS,
    )
    return DB_TABLE_PERCEPTION_RESULTS, DB_TABLE_MAINTENANCE_EVENTS, DB_TABLE_PM_CARDS


def test_table(engine, table_fqn: str, sample_cols: str, label: str) -> bool:
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"  table: {table_fqn}")
    print(f"{'-'*60}")
    try:
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_fqn}")).scalar()
            print(f"  row count : {count}")
            rows = conn.execute(
                text(f"SELECT {sample_cols} FROM {table_fqn} LIMIT 3")
            ).fetchall()
            for r in rows:
                print(f"  sample    : {dict(r._mapping)}")
        print(f"  [OK] OK")
        return True
    except Exception as exc:
        print(f"  [FAIL] FAILED: {exc}")
        return False


def test_vector_search() -> bool:
    index_name = os.getenv("DATABRICKS_VS_INDEX_PM_CARDS")
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if not index_name:
        print("\n  [WARN] DATABRICKS_VS_INDEX_PM_CARDS not set — skipping VS test")
        return False

    print(f"\n{'-'*60}")
    print(f"  PM Cards — Vector Search")
    print(f"  index: {index_name}")
    print(f"{'-'*60}")
    try:
        from databricks.sdk import WorkspaceClient
        ws = WorkspaceClient(host=f"https://{host}", token=token)
        result = ws.vector_search_indexes.query_index(
            index_name=index_name,
            columns=["card_id", "sensor", "robot_code", "cause"],
            query_text="wire feed jam debris",
            num_results=3,
        )
        if result.result and result.result.data_array:
            cols = ["card_id", "sensor", "robot_code", "cause"]
            for row in result.result.data_array:
                print(f"  sample    : {dict(zip(cols, row))}")
            print(f"  [OK] OK ({len(result.result.data_array)} results)")
        else:
            print("  [WARN] No results returned")
        return True
    except Exception as exc:
        print(f"  [FAIL] FAILED: {exc}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Databricks connectivity test")
    print("=" * 60)
    print(f"  HOST    : {os.getenv('DATABRICKS_HOST')}")
    print(f"  CATALOG : {os.getenv('DATABRICKS_CATALOG')}")
    print(f"  SCHEMA  : {os.getenv('DATABRICKS_SCHEMA')}")

    try:
        engine = _engine()
        tbl_pr, tbl_me, tbl_pm = _tables()
    except RuntimeError as e:
        print(f"\n[FAIL] Cannot build engine: {e}")
        sys.exit(1)

    results = [
        test_table(engine, tbl_pr,
                   "robot_code, observed_at, abnormal_score, is_file_anomaly",
                   "perception_results"),
        test_table(engine, tbl_me,
                   "robot_code, event_ts, event_type",
                   "maintenance_events"),
        test_table(engine, tbl_pm,
                   "card_id, sensor, robot_code",
                   "pm_cards"),
        test_vector_search(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"  Result: {passed}/{len(results)} passed")
    print("=" * 60)
    sys.exit(0 if passed == len(results) else 1)
