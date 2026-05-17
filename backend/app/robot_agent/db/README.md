# robot_agent / DB

SQLAlchemy ORM models and session helpers for the paint-shop diagnostic system.

## Tables

| Model              | Table                 | Purpose                                                  |
|--------------------|-----------------------|----------------------------------------------------------|
| `PerceptionResult` | `perception_results`  | Raw anomaly detections per robot / sensor                |
| `MaintenanceEvent` | `maintenance_events`  | Historical maintenance actions logged by operators       |
| `PMCard`           | `pm_cards`            | Preventive-maintenance reference cards (cause → action)  |
| `Incident`         | `incidents`           | Diagnosed incidents with packet + decision payloads      |
| `Alert`            | `alerts`              | Notifications dispatched through configured channels     |
| `Report`           | `reports`             | Final diagnostic reports with root-cause candidates      |
| `AgentMemory`      | `agent_memories`      | Cross-session agent knowledge persisted to SQL           |
| `HitlQueue`        | `hitl_queue`          | Pending human-in-the-loop resolution requests            |

## Utilities

- `create_db_engine(url)` – build or retrieve a cached SQLAlchemy engine
- `create_session_factory(url)` – build a sessionmaker bound to an engine
- `init_db(engine)` – run `Base.metadata.create_all`
- `seed_maintenance_events(session)` – insert sample maintenance records

The default database is a SQLite file at `<project_root>/data/demo.db`.
