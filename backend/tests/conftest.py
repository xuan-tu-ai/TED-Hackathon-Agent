"""Pytest fixtures for the robot_agent test suite.

Provides session-scoped database connection helpers (``db_url``,
``session_factory``, ``db_session``) and a function-scoped ``clean_db``
fixture that creates a fresh database schema for each test.
"""

import pytest

from app.robot_agent.seed import create_session_factory


@pytest.fixture(scope="session")
def db_url():
    """Resolve the demo.db path relative to the project root (two levels up)."""
    from pathlib import Path
    db_path = Path(__file__).resolve().parents[2] / "data" / "demo.db"
    return f"sqlite+pysqlite:///{db_path.as_posix()}"


@pytest.fixture(scope="session")
def session_factory(db_url):
    """Return a SQLAlchemy sessionmaker bound to the demo database."""
    return create_session_factory(db_url)


@pytest.fixture
def db_session(session_factory):
    """Provide a transactional database session (yielded, then closed)."""
    session = session_factory()
    yield session
    session.close()


@pytest.fixture(scope="function")
def clean_db(db_url):
    """Re-create all tables before and after each test for isolation."""
    import os
    os.environ["DATABASE_URL"] = db_url
    from app.robot_agent.db.models import Base, create_db_engine
    engine = create_db_engine(db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()