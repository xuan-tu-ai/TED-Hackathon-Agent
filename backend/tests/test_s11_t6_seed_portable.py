from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_no_raw_sqlite3_import():
    import ast
    run_path = Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "seed" / "run.py"
    tree = ast.parse(run_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "sqlite3", "run.py must not import sqlite3 directly"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "sqlite3", "run.py must not import sqlite3 directly"


def test_create_session_factory_used():
    content = Path(
        Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "seed" / "run.py"
    ).read_text()
    assert "create_session_factory" in content


def test_create_db_engine_used():
    content = Path(
        Path(__file__).resolve().parents[1] / "app" / "robot_agent" / "seed" / "run.py"
    ).read_text()
    assert "create_db_engine" in content


def test_seed_module_importable():
    from app.robot_agent.seed import seed_all, seed_demo_data
    assert callable(seed_all)
    assert callable(seed_demo_data)
