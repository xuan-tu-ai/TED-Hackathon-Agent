"""SQLAlchemy ORM models for the paint-shop diagnostic system.

Defines the relational schema used by the robot_agent subsystem,
covering perception results, maintenance history, preventive-maintenance
(PM) cards, incidents, alerts, diagnostic reports, agent memory entries,
and the human-in-the-loop (HITL) queue.

Database URL is driven by the ``DATABASE_URL`` environment variable;
defaults to a SQLite database at ``<project_root>/data/demo.db``.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..agent.config import config


logger_db = logging.getLogger(__name__)


def get_default_db_url() -> str:
    return _normalize_sqlite_scheme(config.database_url)


def _normalize_sqlite_scheme(url: str) -> str:
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+pysqlite:///", 1)
    return url


def is_sqlite(url: str) -> bool:
    return "sqlite" in url


def is_postgres(url: str) -> bool:
    return "postgresql" in url or "postgres" in url


def _is_azure_host(url_str: str) -> bool:
    return ".postgres.database.azure.com" in url_str


JSONType = JSON().with_variant(JSONB, "postgresql")

_engine_cache: dict[str, Engine] = {}


def get_db_engine(db_url: str | None = None) -> Engine:
    key = db_url or get_default_db_url()
    if key not in _engine_cache:
        _engine_cache[key] = _build_engine(key)
        logger_db.debug("Created DB engine: %s", key)
    return _engine_cache[key]


def _build_engine(url: str) -> Engine:
    if is_sqlite(url):
        return create_engine(url, connect_args={"check_same_thread": False})
    if is_postgres(url):
        connect_args: dict[str, Any] = {}
        if _is_azure_host(url) and "sslmode" not in url:
            logger_db.warning("Azure Postgres URL detected without sslmode=require — connection may fail")
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args=connect_args,
        )
    return create_engine(url)


class Base(DeclarativeBase):
    pass


class PerceptionResult(Base):
    __tablename__ = "perception_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    robot_code: Mapped[str] = mapped_column(String(16), index=True)
    program_number: Mapped[int] = mapped_column(Integer)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    abnormal_type: Mapped[str] = mapped_column(String(64), index=True)
    abnormal_score: Mapped[float] = mapped_column(Float)
    severity_hint: Mapped[str] = mapped_column(String(16))
    is_file_anomaly: Mapped[bool] = mapped_column(Boolean, index=True)
    per_sensor_scores: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class MaintenanceEvent(Base):
    __tablename__ = "maintenance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    robot_code: Mapped[str] = mapped_column(String(16), index=True)
    event_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    notes: Mapped[str] = mapped_column(Text)


class PMCard(Base):
    __tablename__ = "pm_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    card_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    sensor: Mapped[str] = mapped_column(String(64), index=True)
    robot_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    cause: Mapped[str] = mapped_column(Text)
    symptom: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    packet_json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    decision_json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    trace_json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    channels_json: Mapped[list[str]] = mapped_column(JSONType)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    requires_ack: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    final_severity: Mapped[str] = mapped_column(String(16), index=True)
    summary: Mapped[str] = mapped_column(Text)
    root_cause_candidates_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONType)
    recommended_action: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True,
        default=lambda: f"mem-{uuid.uuid4().hex[:8]}",
    )
    robot_code: Mapped[str] = mapped_column(String(16), index=True)
    agent_name: Mapped[str] = mapped_column(String(64))
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    summary: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class HitlQueue(Base):
    __tablename__ = "hitl_queue"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex[:8])
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    robot_code: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(Text)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONType)
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    reasoning: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)


def create_db_engine(db_url: str | None = None):
    return get_db_engine(db_url)


def create_session_factory(db_url: str | None = None):
    engine = get_db_engine(db_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine=None) -> None:
    db_engine = engine or get_db_engine()
    Base.metadata.create_all(db_engine)


def seed_maintenance_events(session: Session) -> int:
    records = [
        MaintenanceEvent(
            robot_code="R-07",
            event_ts=datetime(2026, 4, 10, 9, 0, tzinfo=UTC),
            event_type="filter_check",
            notes="Đã kiểm tra lọc inline, phát hiện cặn sơn tăng.",
        ),
        MaintenanceEvent(
            robot_code="R-07",
            event_ts=datetime(2026, 4, 14, 14, 30, tzinfo=UTC),
            event_type="pressure_calibration",
            notes="Hiệu chuẩn áp suất đầu phun sau cảnh báo giảm áp.",
        ),
        MaintenanceEvent(
            robot_code="R-13",
            event_ts=datetime(2026, 4, 11, 10, 15, tzinfo=UTC),
            event_type="bearing_inspection",
            notes="Kiểm tra bạc đạn motor khí nén do dao động RPM theo chu kỳ.",
        ),
        MaintenanceEvent(
            robot_code="R-31",
            event_ts=datetime(2026, 4, 13, 8, 45, tzinfo=UTC),
            event_type="ground_cleaning",
            notes="Vệ sinh điểm tiếp địa khu vực spray zone.",
        ),
        MaintenanceEvent(
            robot_code="R-44",
            event_ts=datetime(2026, 4, 9, 16, 0, tzinfo=UTC),
            event_type="solenoid_tuning",
            notes="Điều chỉnh van điện-khí do hysteresis tăng.",
        ),
    ]
    session.add_all(records)
    return len(records)
