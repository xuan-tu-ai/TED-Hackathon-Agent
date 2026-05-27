from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_json_type_defined():
    from app.robot_agent.db.models import JSONType
    assert JSONType is not None


def test_jsontype_has_postgresql_variant():
    from app.robot_agent.db.models import JSONType
    from sqlalchemy.dialects import postgresql as pg
    # Compile JSONType against the Postgres dialect via the public TypeEngine API.
    compiled = JSONType.compile(dialect=pg.dialect())
    assert "JSONB" in str(compiled).upper()


def test_all_json_columns_render_sqlite():
    from app.robot_agent.db.models import (
        PerceptionResult, Incident, Alert, Report, HitlQueue,
        Base, get_db_engine,
    )
    engine = get_db_engine("sqlite+pysqlite:///./data/test_t4_cols.db")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    assert engine.dialect.name == "sqlite"
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_json_round_trip_sqlite():
    from app.robot_agent.db.models import (
        Base, Incident, get_db_engine,
    )
    engine = get_db_engine("sqlite+pysqlite:///./data/test_t4.db")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()

    inc = Incident(
        incident_id="INC-T4-TEST",
        event_id="EVT-001",
        packet_json={"key": "value", "nested": {"a": 1}},
        decision_json={"action": "investigate"},
        trace_json=[{"step": 1}],
    )
    session.add(inc)
    session.commit()

    fetched = session.query(Incident).filter_by(incident_id="INC-T4-TEST").first()
    assert fetched is not None
    assert fetched.packet_json["key"] == "value"
    assert fetched.packet_json["nested"]["a"] == 1
    assert fetched.decision_json["action"] == "investigate"

    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()
