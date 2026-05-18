"""Pytest fixtures for the robot_agent test suite.

Provides session-scoped database connection helpers (``db_url``,
``session_factory``) and a function-scoped ``clean_db`` fixture
that creates a fresh database schema for each test.

All fixtures respect the ``TEST_DATABASE_URL`` environment variable
(default: ``sqlite+pysqlite:///./data/demo.db``).
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def db_url():
    """Return database URL from TEST_DATABASE_URL env var or default."""
    env_val = os.getenv("TEST_DATABASE_URL")
    if env_val:
        return env_val
    return f"sqlite+pysqlite:///{Path(__file__).resolve().parents[2] / 'data' / 'demo.db'}"


@pytest.fixture(scope="session")
def db_engine(db_url):
    """Return a SQLAlchemy engine bound to the test database URL,
    with all tables created once per session."""
    from app.robot_agent.db.models import Base, create_db_engine, _engine_cache

    _engine_cache.clear()
    engine = create_db_engine(db_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def session_factory(db_url):
    """Return a SQLAlchemy sessionmaker bound to the test database URL."""
    from app.robot_agent.db.models import create_session_factory

    return create_session_factory(db_url)


@pytest.fixture
def db_session(db_engine):
    """Provide a transactional database session (yielded, then closed)."""
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def clean_db(db_url):
    """Re-create all tables before and after each test for isolation."""
    from app.robot_agent.db.models import Base, create_db_engine, _engine_cache

    _engine_cache.clear()
    engine = create_db_engine(db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
    _engine_cache.clear()
