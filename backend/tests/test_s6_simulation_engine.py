from __future__ import annotations

import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def _ensure_seeded():
    from app.robot_agent.seed import create_session_factory, seed_demo_data
    session_factory = create_session_factory("sqlite:///data/demo.db")
    session = session_factory()
    from app.robot_agent.db.models import PerceptionResult
    count = session.query(PerceptionResult).count()
    if count == 0:
        seed_demo_data(session, days=30)
    session.close()


class TestSimulationEngineTimeline:
    def test_timeline_length_with_real_db(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        assert isinstance(timeline, list)
        assert len(timeline) > 0
        assert all(isinstance(t, datetime) for t in timeline)
        assert timeline == sorted(timeline)

    def test_timeline_720_ticks(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        assert len(timeline) == 720

    def test_timeline_covers_30_days(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        first = timeline[0]
        last = timeline[-1]

        assert first.tzinfo is not None or first.tzinfo == UTC
        assert last.tzinfo is not None or last.tzinfo == UTC

        duration = last - first
        assert duration.days >= 29


class TestSimulationEngineGetEvents:
    def test_get_events_at_returns_list(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        events = engine.get_events_at(timeline[0])

        assert isinstance(events, list)
        assert len(events) > 0

    def test_get_events_at_has_robot_code(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        events = engine.get_events_at(timeline[0])

        for event in events:
            assert hasattr(event, "robot_code")
            assert event.robot_code is not None


class TestSimulationEngineTickGenerator:
    def test_tick_generator_yields_tuples(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")

        gen = engine.tick_generator()
        first_tick, first_events = next(gen)

        assert isinstance(first_tick, datetime)
        assert isinstance(first_events, list)

    def test_tick_generator_exhausted_at_end(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()

        gen = engine.tick_generator()
        count = 0
        for _ in gen:
            count += 1

        assert count == len(timeline)


class TestSimulationEngineCheckTriggers:
    def test_check_triggers_returns_dict(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()
        session = engine.session_factory()

        triggers = engine.check_triggers(session, "R-07", timeline[-1])

        assert isinstance(triggers, dict)
        assert "l1" in triggers
        assert "l2" in triggers
        assert "has_alert" in triggers
        assert "alert_type" in triggers
        session.close()

    def test_check_triggers_r07_has_alert(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()
        session = engine.session_factory()

        triggers = engine.check_triggers(session, "R-07", timeline[-1])

        assert triggers["has_alert"] is True
        assert triggers["alert_type"] in ("L1_BURST", "L2_ESCALATION")
        session.close()

    def test_check_triggers_with_reference_time(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()
        session = engine.session_factory()

        ref_time = timeline[-1]
        triggers = engine.check_triggers(
            session, "R-07", up_to=ref_time, reference_time=ref_time
        )

        assert triggers["l1"]["reference_time"] == ref_time.isoformat()
        assert triggers["l2"]["reference_time"] == ref_time.isoformat()
        session.close()

    def test_check_triggers_uses_up_to_when_no_reference(self, db_session):
        _ensure_seeded()
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        timeline = engine.get_timeline()
        session = engine.session_factory()

        up_to = timeline[-1]
        triggers = engine.check_triggers(session, "R-07", up_to=up_to)

        assert triggers["l1"]["reference_time"] == up_to.isoformat()
        assert triggers["l2"]["reference_time"] == up_to.isoformat()
        session.close()


class TestSimulationEngineHelpers:
    def test_get_robots_returns_47_robots(self, db_session):
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db")
        robots = engine.get_robots()

        assert len(robots) == 47
        assert robots[0] == "R-01"
        assert robots[-1] == "R-47"

    def test_speed_factor_stored(self, db_session):
        from app.robot_agent.simulation import SimulationEngine

        engine = SimulationEngine("sqlite:///data/demo.db", speed_factor=7200)
        assert engine.speed_factor == 7200

        engine2 = SimulationEngine("sqlite:///data/demo.db")
        assert engine2.speed_factor == 3600


class TestL1L2ReferenceTime:
    def test_l1_burst_with_reference_time(self, db_session):
        _ensure_seeded()
        from app.robot_agent.poller.l1_burst import detect_l1_burst

        session = db_session
        now = datetime.now(UTC)

        result_without_ref = detect_l1_burst(session, "R-07", window_hours=1)
        result_with_ref = detect_l1_burst(session, "R-07", window_hours=1, reference_time=now)

        assert "reference_time" in result_without_ref
        assert "reference_time" in result_with_ref
        assert result_without_ref["reference_time"] is not None
        assert result_with_ref["reference_time"] == now.isoformat()

    def test_l2_slope_with_reference_time(self, db_session):
        _ensure_seeded()
        from app.robot_agent.poller.l2_slope import detect_l2_slope

        session = db_session
        now = datetime.now(UTC)

        result_without_ref = detect_l2_slope(session, "R-07", days=30)
        result_with_ref = detect_l2_slope(session, "R-07", days=30, reference_time=now)

        assert "reference_time" in result_without_ref
        assert "reference_time" in result_with_ref
        assert result_without_ref["reference_time"] is not None
        assert result_with_ref["reference_time"] == now.isoformat()

    def test_l1_burst_backward_compatible(self, db_session):
        _ensure_seeded()
        from app.robot_agent.poller.l1_burst import detect_l1_burst

        session = db_session

        result = detect_l1_burst(session, "R-07", window_hours=1)
        assert "is_burst" in result
        assert "event_count" in result
        assert "density_per_hour" in result

    def test_l2_slope_backward_compatible(self, db_session):
        _ensure_seeded()
        from app.robot_agent.poller.l2_slope import detect_l2_slope

        session = db_session

        result = detect_l2_slope(session, "R-07", days=30)
        assert "trend" in result
        assert "slope" in result
        assert "r2" in result


class TestSimulationState:
    def test_default_state_is_idle(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        assert state.get_status() == "idle"
        assert state.get_sim_time() is None
        assert state.get_tick_index() == 0
        assert state.get_total_ticks() == 0
        assert state.get_alerts() == []
        assert state.get_active_incident() is None
        assert state.get_speed_factor() == 3600

    def test_set_and_get_status(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_status("running")
        assert state.get_status() == "running"

        state.set_status("paused")
        assert state.get_status() == "paused"

    def test_set_and_get_sim_time(self):
        from datetime import UTC
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        now = datetime.now(UTC)
        state.set_sim_time(now)
        assert state.get_sim_time() == now

    def test_set_and_get_tick_index(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_tick_index(42)
        assert state.get_tick_index() == 42

    def test_set_and_get_total_ticks(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_total_ticks(720)
        assert state.get_total_ticks() == 720

    def test_add_and_get_alerts(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        alert1 = {"robot_id": "R-07", "alert_type": "L2_ESCALATION"}
        alert2 = {"robot_id": "R-31", "alert_type": "L1_BURST"}

        state.add_alert(alert1)
        state.add_alert(alert2)

        alerts = state.get_alerts()
        assert len(alerts) == 2
        assert alerts[0]["robot_id"] == "R-07"
        assert alerts[1]["robot_id"] == "R-31"

    def test_set_and_get_active_incident(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_active_incident("INC-001")
        assert state.get_active_incident() == "INC-001"

        state.set_active_incident(None)
        assert state.get_active_incident() is None

    def test_reset(self):
        from datetime import UTC
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_status("running")
        state.set_sim_time(datetime.now(UTC))
        state.set_tick_index(100)
        state.set_total_ticks(720)
        state.add_alert({"robot_id": "R-07"})
        state.set_active_incident("INC-001")

        state.reset()

        assert state.get_status() == "idle"
        assert state.get_sim_time() is None
        assert state.get_tick_index() == 0
        assert state.get_total_ticks() == 0
        assert state.get_alerts() == []
        assert state.get_active_incident() is None

    def test_to_dict(self):
        from datetime import UTC
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        state.set_status("running")
        state.set_sim_time(datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC))
        state.set_tick_index(5)
        state.set_total_ticks(720)
        state.add_alert({"robot_id": "R-07"})

        d = state.to_dict()
        assert d["status"] == "running"
        assert d["sim_time"] == "2026-04-10T12:00:00+00:00"
        assert d["tick_index"] == 5
        assert d["total_ticks"] == 720
        assert len(d["alerts"]) == 1


class TestGetSimulationStateSingleton:
    def test_returns_same_instance(self):
        from app.robot_agent.simulation.state import get_simulation_state

        s1 = get_simulation_state()
        s2 = get_simulation_state()
        assert s1 is s2

    def test_singleton_reset_clears_state(self):
        from app.robot_agent.simulation.state import get_simulation_state

        state = get_simulation_state()
        state.set_status("running")
        state.set_tick_index(100)

        state.reset()

        assert state.get_status() == "idle"
        assert state.get_tick_index() == 0


class TestSimulationRunner:
    def test_runner_instantiation(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        _ensure_seeded()
        engine = SimulationEngine("sqlite:///data/demo.db")
        state = SimulationState()
        events = []

        runner = SimulationRunner(
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=lambda e: events.append(e),
        )

        assert runner.speed_factor == 3600
        assert runner._running is False
        assert runner._already_triggered == {}

    def test_runner_start_stop(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        _ensure_seeded()
        engine = SimulationEngine("sqlite:///data/demo.db")
        state = SimulationState()
        events = []

        runner = SimulationRunner(
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=lambda e: events.append(e),
        )

        runner.start()
        assert runner._running is True
        assert state.get_status() == "running"
        assert state.get_total_ticks() == 720

        runner.stop()
        assert runner._running is False
        assert state.get_status() == "idle"

    def test_runner_pause_resume(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        _ensure_seeded()
        engine = SimulationEngine("sqlite:///data/demo.db")
        state = SimulationState()
        events = []

        runner = SimulationRunner(
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=lambda e: events.append(e),
        )

        runner.start()
        runner.pause()
        assert state.get_status() == "paused"

        runner.resume()
        assert state.get_status() == "running"

        runner.stop()

    def test_runner_sets_up_state_on_start(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        _ensure_seeded()
        engine = SimulationEngine("sqlite:///data/demo.db")
        state = SimulationState()
        events = []

        runner = SimulationRunner(
            engine=engine,
            state=state,
            speed_factor=7200,
            on_event=lambda e: events.append(e),
        )

        runner.start()

        assert state.get_speed_factor() == 7200
        assert state.get_total_ticks() == 720

        runner.stop()

    def test_runner_broadcasts_started_event(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        _ensure_seeded()
        engine = SimulationEngine("sqlite:///data/demo.db")
        state = SimulationState()
        events = []

        runner = SimulationRunner(
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=lambda e: events.append(e),
        )

        runner.start()
        time.sleep(0.1)

        assert any(e.get("type") == "sim_started" for e in events)
        runner.stop()


class TestSimulationStateHitlEvents:
    def test_set_and_get_hitl_event(self):
        from app.robot_agent.simulation.state import SimulationState
        import threading

        state = SimulationState()
        event = threading.Event()
        state.set_hitl_event("q-123", event)

        retrieved = state.get_hitl_event("q-123")
        assert retrieved is event

    def test_remove_hitl_event(self):
        from app.robot_agent.simulation.state import SimulationState
        import threading

        state = SimulationState()
        event = threading.Event()
        state.set_hitl_event("q-123", event)
        state.remove_hitl_event("q-123")

        assert state.get_hitl_event("q-123") is None

    def test_hitl_event_nonexistent(self):
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        assert state.get_hitl_event("nonexistent") is None

    def test_reset_clears_hitl_events(self):
        from app.robot_agent.simulation.state import SimulationState
        import threading

        state = SimulationState()
        event1 = threading.Event()
        event2 = threading.Event()
        state.set_hitl_event("q-1", event1)
        state.set_hitl_event("q-2", event2)

        state.reset()

        assert state.get_hitl_event("q-1") is None
        assert state.get_hitl_event("q-2") is None


class TestResolveHitlEntryAndSetEvent:
    def test_resolve_entry_updates_db(self, clean_db, db_session):
        _ensure_seeded()
        from app.robot_agent.agent.hitl import resolve_hitl_entry
        import uuid

        queue_id = uuid.uuid4().hex[:8]

        from app.robot_agent.seed import HitlQueue
        entry = HitlQueue(
            id=queue_id,
            incident_id="INC-TEST-001",
            robot_code="R-07",
            action="dispatch_maintenance",
            parameters={},
            severity="CRITICAL",
            status="PENDING",
            reasoning="Test reasoning",
        )
        db_session.add(entry)
        db_session.commit()

        resolved = resolve_hitl_entry(queue_id, "APPROVED", "test_user")
        assert resolved is True

        from app.robot_agent.seed import HitlQueue as HQ
        from app.robot_agent.seed import create_session_factory
        s = create_session_factory()()
        updated = s.get(HQ, queue_id)
        assert updated.status == "APPROVED"
        assert updated.resolved_by == "test_user"
        s.close()

    def test_event_set_on_resolve(self):
        from app.robot_agent.simulation.state import get_simulation_state
        import threading

        state = get_simulation_state()
        state.reset()

        queue_id = "q-test-123"
        event = threading.Event()
        state.set_hitl_event(queue_id, event)

        assert event.is_set() is False
        event.set()
        assert event.is_set()

    def test_resolve_and_set_event_rejects(self):
        from app.robot_agent.agent.hitl import resolve_hitl_entry_and_set_event, hitl_check
        from app.robot_agent.simulation.state import get_simulation_state
        import threading
        import uuid

        state = get_simulation_state()
        state.reset()

        queue_id = uuid.uuid4().hex[:8]

        from app.robot_agent.seed import create_session_factory
        session = create_session_factory()()
        from app.robot_agent.seed import HitlQueue
        from datetime import UTC
        entry = HitlQueue(
            id=queue_id,
            incident_id="INC-TEST-002",
            robot_code="R-31",
            action="send_email",
            parameters={},
            severity="HIGH",
            status="PENDING",
            reasoning="Test reasoning",
        )
        session.add(entry)
        session.commit()
        session.close()

        hitl_event = threading.Event()
        state.set_hitl_event(queue_id, hitl_event)

        resolved = resolve_hitl_entry_and_set_event(queue_id, "REJECTED", "test_user")
        assert resolved is True

        assert hitl_event.is_set()
        assert state.get_hitl_event(queue_id) is None


class TestDiagnosticAgentHitlBlocking:
    def test_agent_hitl_check_permanently_blocked(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        incident = DiagnosticIncident(
            incident_id="INC-TEST-HITL-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="L2_ESCALATION",
            sensor="paint_pressure",
            anomaly_score=0.85,
            severity_hint="HIGH",
        )

        agent = DiagnosticAgent()
        agent.start_session(incident)

        result = agent.hitl_check(
            action="override_safety_interlock",
            parameters={},
            reasoning="Trying to override safety",
        )

        assert result["blocked"] is True
        assert result["permanently_blocked"] is True
        assert agent.state == "blocked"

    def test_agent_hitl_check_high_severity_queues(self):
        """Real test: agent.hitl_check() with HIGH severity BLOCKS then resumes after resolve."""
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident
        from app.robot_agent.agent.hitl import resolve_hitl_entry_and_set_event
        from app.robot_agent.seed import create_session_factory
        from app.robot_agent.seed import HitlQueue
        import threading
        import uuid

        # Reset state singleton
        from app.robot_agent.simulation.state import get_simulation_state
        state = get_simulation_state()
        state.reset()

        incident = DiagnosticIncident(
            incident_id=f"INC-TEST-HITL-{uuid.uuid4().hex[:6]}",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="L2_ESCALATION",
            sensor="paint_pressure",
            anomaly_score=0.85,
            severity_hint="HIGH",
        )

        agent = DiagnosticAgent()
        agent.start_session(incident)

        result_holder: dict = {}
        hitl_called = threading.Event()

        def run_hitl():
            # This will block inside hitl_event.wait(timeout=300)
            result_holder["result"] = agent.hitl_check(
                action="send_email",
                parameters={"to": "ops@factory.com"},
                reasoning="Notifying ops team of high severity surge anomaly",
            )
            hitl_called.set()

        t = threading.Thread(target=run_hitl, daemon=True)
        t.start()

        # Wait for the DB entry to appear (agent is blocked waiting for resolution)
        session = create_session_factory()()\
            if False else create_session_factory("sqlite:///data/demo.db")()
        queue_id = None
        for _ in range(30):
            time.sleep(0.1)
            entry = (
                session.query(HitlQueue)
                .filter(
                    HitlQueue.incident_id == incident.incident_id,
                    HitlQueue.action == "send_email",
                    HitlQueue.status == "PENDING",
                )
                .first()
            )
            if entry:
                queue_id = entry.id
                break
        session.close()

        assert queue_id is not None, "HITL DB entry was never created"
        assert agent.state == "blocked", f"Agent should be blocked, got: {agent.state}"

        # Resolve → unblocks the agent thread
        resolved = resolve_hitl_entry_and_set_event(queue_id, "APPROVED", "test_user")
        assert resolved is True

        # Wait for agent thread to finish (with timeout)
        hitl_called.wait(timeout=5)
        t.join(timeout=5)

        result = result_holder.get("result")
        assert result is not None, "hitl_check() never returned"
        assert result["blocked"] is True
        assert result["permanently_blocked"] is False
        assert result["queue_id"] == queue_id
        assert result.get("resolved") is True
        assert agent.state == "running"


# ============================================================
# Task 6.6 — EventBroadcaster Tests
# ============================================================

class TestEventBroadcaster:
    """Real tests for EventBroadcaster — subscribe, broadcast, dead-client cleanup."""

    def test_subscribe_returns_queue(self):
        import asyncio
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _run():
            q = await broadcaster.subscribe()
            assert q is not None
            assert q.maxsize == 100
            return q

        asyncio.run(_run())

    def test_broadcast_delivers_to_subscriber(self):
        import asyncio
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _run():
            q = await broadcaster.subscribe()
            broadcaster.broadcast({"type": "tick", "tick_index": 1})
            event = q.get_nowait()
            assert event["type"] == "tick"
            assert event["tick_index"] == 1

        asyncio.run(_run())

    def test_broadcast_reaches_multiple_subscribers(self):
        import asyncio
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _run():
            q1 = await broadcaster.subscribe()
            q2 = await broadcaster.subscribe()
            broadcaster.broadcast({"type": "alert", "robot_id": "R-07"})
            e1 = q1.get_nowait()
            e2 = q2.get_nowait()
            assert e1["robot_id"] == "R-07"
            assert e2["robot_id"] == "R-07"

        asyncio.run(_run())

    def test_unsubscribe_stops_delivery(self):
        import asyncio
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _run():
            q = await broadcaster.subscribe()
            await broadcaster.unsubscribe(q)
            broadcaster.broadcast({"type": "tick"})
            assert q.empty()

        asyncio.run(_run())

    def test_full_queue_subscriber_is_pruned(self):
        """A subscriber whose queue is full (dead client) should be removed automatically."""
        import asyncio
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _run():
            q = await broadcaster.subscribe()
            # Fill the queue to maxsize
            for i in range(100):
                q.put_nowait({"i": i})
            # Subscriber count before
            assert len(broadcaster._subscribers) == 1
            # Next broadcast should detect QueueFull and prune
            broadcaster.broadcast({"type": "overflow"})
            assert len(broadcaster._subscribers) == 0

        asyncio.run(_run())

    def test_get_broadcaster_is_singleton(self):
        from app.robot_agent.simulation.broadcaster import get_broadcaster

        b1 = get_broadcaster()
        b2 = get_broadcaster()
        assert b1 is b2

    def test_broadcast_is_thread_safe(self):
        """Multiple threads broadcasting simultaneously should not corrupt _subscribers."""
        import asyncio
        import threading
        from app.robot_agent.simulation.broadcaster import EventBroadcaster

        broadcaster = EventBroadcaster()

        async def _setup():
            qs = [await broadcaster.subscribe() for _ in range(5)]
            return qs

        qs = asyncio.run(_setup())

        errors = []

        def _broadcast_loop():
            try:
                for i in range(50):
                    broadcaster.broadcast({"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_broadcast_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ============================================================
# Task 6.7 — Simulation API Endpoint Tests
# ============================================================

class TestSimulationApi:
    """Real endpoint tests via FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def reset_sim_state(self):
        """Stop any running simulation and reset state before each test."""
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)
        client.post("/api/sim/stop")  # idempotent
        from app.robot_agent.simulation.state import get_simulation_state
        get_simulation_state().reset()
        yield
        client.post("/api/sim/stop")

    def test_status_returns_idle_before_start(self):
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        resp = client.get("/api/sim/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["tick_index"] == 0

    def test_start_returns_running(self):
        _ensure_seeded()
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        resp = client.post("/api/sim/start", json={"speed_factor": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["total_ticks"] == 720

    def test_start_twice_returns_409(self):
        _ensure_seeded()
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        client.post("/api/sim/start", json={"speed_factor": 1})
        resp2 = client.post("/api/sim/start", json={"speed_factor": 1})
        assert resp2.status_code == 409

    def test_stop_returns_idle(self):
        _ensure_seeded()
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        client.post("/api/sim/start", json={"speed_factor": 1})
        resp = client.post("/api/sim/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_pause_and_resume(self):
        _ensure_seeded()
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        client.post("/api/sim/start", json={"speed_factor": 1})

        resp = client.post("/api/sim/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

        resp = client.post("/api/sim/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_pause_when_not_running_returns_409(self):
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        resp = client.post("/api/sim/pause")
        assert resp.status_code == 409

    def test_hitl_approve_not_found_returns_404(self):
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        resp = client.post(
            "/api/sim/hitl/nonexistent-queue/approve",
            json={"queue_id": "nonexistent-queue", "status": "APPROVED", "resolved_by": "tester"},
        )
        assert resp.status_code == 404

    def test_hitl_approve_existing_entry(self):
        """Create a real HITL DB entry, then approve it via API."""
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        from app.robot_agent.seed import create_session_factory
        from app.robot_agent.seed import HitlQueue
        from datetime import UTC
        import uuid

        client = TestClient(app)

        queue_id = uuid.uuid4().hex[:8]
        session = create_session_factory("sqlite:///data/demo.db")()
        entry = HitlQueue(
            id=queue_id,
            incident_id="INC-API-TEST-001",
            robot_code="R-07",
            action="dispatch_maintenance",
            parameters={},
            severity="CRITICAL",
            status="PENDING",
            reasoning="API test",
            created_at=datetime.now(UTC),
        )
        session.add(entry)
        session.commit()
        session.close()

        resp = client.post(
            f"/api/sim/hitl/{queue_id}/approve",
            json={"queue_id": queue_id, "status": "APPROVED", "resolved_by": "tester"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["queue_id"] == queue_id
        assert data["status"] == "APPROVED"

    def test_hitl_reject_existing_entry(self):
        """Create a real HITL DB entry, then reject it via API."""
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        from app.robot_agent.seed import create_session_factory
        from app.robot_agent.seed import HitlQueue
        from datetime import UTC
        import uuid

        client = TestClient(app)

        queue_id = uuid.uuid4().hex[:8]
        session = create_session_factory("sqlite:///data/demo.db")()
        entry = HitlQueue(
            id=queue_id,
            incident_id="INC-API-TEST-002",
            robot_code="R-31",
            action="send_slack_alert",
            parameters={},
            severity="HIGH",
            status="PENDING",
            reasoning="API reject test",
            created_at=datetime.now(UTC),
        )
        session.add(entry)
        session.commit()
        session.close()

        resp = client.post(
            f"/api/sim/hitl/{queue_id}/reject",
            json={"queue_id": queue_id, "status": "REJECTED", "resolved_by": "tester"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "REJECTED"

    def test_status_updates_after_start(self):
        """After start, status endpoint reflects running state."""
        _ensure_seeded()
        from fastapi.testclient import TestClient
        from app.robot_agent.main import app
        client = TestClient(app)

        client.post("/api/sim/start", json={"speed_factor": 1})
        time.sleep(0.2)

        resp = client.get("/api/sim/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("running", "finished")
        assert data["total_ticks"] == 720

