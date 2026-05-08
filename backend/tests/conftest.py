import pytest

from app.robot_agent.seed import create_session_factory


@pytest.fixture(scope="session")
def db_url():
    return "sqlite+pysqlite:///C:/Users/admin/Desktop/Workspace/TED-Hackathon-Agent/data/demo.db"


@pytest.fixture(scope="session")
def session_factory(db_url):
    return create_session_factory(db_url)


@pytest.fixture
def db_session(session_factory):
    session = session_factory()
    yield session
    session.close()


@pytest.fixture(scope="function")
def clean_db(db_url):
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