"""Simulation runner — background-thread tick loop with incident dispatch.

:class:`SimulationRunner` drives the replay timeline obtained from
:class:`SimulationEngine`, evaluates L1/L2 triggers per robot per tick,
creates :class:`DiagnosticIncident` objects for alerts, performs triage,
and dispatches incidents to the multi-agent pipeline via a thread-pool
executor.

Supports pause / resume, warp-to-first-alert, jump-to-tick, debounce
(deduplication of repeated alerts per robot within a configurable window),
and alert-storm detection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable

from ..agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident
from ..agent.session_trace import SessionTraceWriter
from ..agent.triage_agent import triage_incident, TriageDecision
from ..agent.systemic_agent import SystemicIncident
from ..inference.v9_scores import (
    dominant_sensor,
    normalize_mae,
    parse_v9_scores,
    score_to_severity,
    sensor_to_family,
)
from ..seed import create_session_factory
from ..db.models import Incident

from .engine import SimulationEngine
from .state import SimulationState

logger = logging.getLogger("simulation_runner")

ALERT_STORM_THRESHOLD = 5
DEBOUNCE_WINDOW_TICKS = 24


class SimulationRunner:
    def __init__(
        self,
        sim_id: str,
        engine: SimulationEngine,
        state: SimulationState,
        speed_factor: int = 3600,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.sim_id = sim_id
        self.engine = engine
        self.state = state
        self.speed_factor = speed_factor
        self.on_event = on_event or (lambda e: None)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._debounce_timestamps: dict[str, int] = {}
        self._running = False
        self._pending_systemic_alerts: list[SystemicIncident] = []
        self._pending_incidents: list[DiagnosticIncident] = []
        self._active_incident_count = 0
        self._warp_mode = False
        self._target_tick: int | None = None

        self._incident_queue: queue.Queue[DiagnosticIncident] = queue.Queue()
        self._agent_executor = ThreadPoolExecutor(max_workers=5)
        self._dispatcher_thread: threading.Thread | None = None
        self._dispatcher_stop_event = threading.Event()

    def _broadcast(self, event: dict[str, Any]) -> None:
        self.on_event(event)

    def _handle_alert_storm(self, tick_time: datetime) -> None:
        if len(self.state.get_alerts()) >= ALERT_STORM_THRESHOLD:
            recent_alerts = [
                a for a in self.state.get_alerts()
                if abs((tick_time - datetime.fromisoformat(a["sim_time"])).total_seconds()) < 3600
            ]
            if len(recent_alerts) >= ALERT_STORM_THRESHOLD:
                logger.warning(f"ALERT STORM detected: {len(recent_alerts)} alerts in recent window")
                self._broadcast({
                    "type": "alert_storm_detected",
                    "alert_count": len(recent_alerts),
                    "sim_time": tick_time.isoformat(),
                })

    def _create_incident(self, robot_id: str, triggers: dict[str, Any], tick_time: datetime | None = None) -> DiagnosticIncident:
        incident_id = f"INC-SIM-{uuid.uuid4().hex[:8]}"
        l1 = triggers.get("l1", {})
        l2 = triggers.get("l2", {})
        alert_type = triggers.get("alert_type", "L2_ESCALATION")

        # Fetch latest anomalous event from Silver at tick_time to extract V9 scores
        v9_scores = {}
        if tick_time is not None:
            latest_event = self.engine.get_latest_anomaly_at(robot_id, tick_time)
            if latest_event is not None:
                raw = getattr(latest_event, "per_sensor_scores", None)
                if raw is not None:
                    v9_scores = parse_v9_scores(raw)

        # Derive dominant sensor and family from V9 scores
        sensor, sensor_score = dominant_sensor(v9_scores)
        abnormal_family = sensor_to_family(sensor, v9_scores)

        # Normalize MAE against sensor threshold → 0-1 score → severity
        if v9_scores:
            from ..agent.config import config as _cfg
            sensor_threshold = _cfg.sensor_thresholds.get(sensor, 1.0)
            score = normalize_mae(float(sensor_score), sensor_threshold)
            severity = score_to_severity(score)
        else:
            # No V9 scores: derive from L1 density or L2 slope
            if alert_type == "L1_BURST":
                from ..agent.config import config as _cfg
                density = l1.get("density_per_hour", 0.0)
                threshold = _cfg.l1_density_threshold
                score = round(min(1.0, density / (2.0 * threshold)), 2)
                severity = score_to_severity(score)
            else:
                slope = l2.get("slope", 0.0)
                score = round(min(1.0, abs(slope) / 0.15), 2)
                severity = score_to_severity(score)

        inc = DiagnosticIncident(
            incident_id=incident_id,
            robot_code=robot_id,
            abnormal_family=abnormal_family,
            abnormal_subtype=alert_type,
            sensor=sensor,
            anomaly_score=round(score, 2),
            severity_hint=severity,
            per_sensor_scores=v9_scores,
        )
        return inc

    def _save_incident_to_db(self, inc: DiagnosticIncident, triggers: dict[str, Any]) -> None:
        session = create_session_factory()()
        try:
            l1_data = triggers.get("l1", {})
            l2_data = triggers.get("l2", {})

            l1_serializable = {
                "is_burst": l1_data.get("is_burst", False),
                "event_count": l1_data.get("event_count", 0),
                "density_per_hour": l1_data.get("density_per_hour", 0.0),
                "threshold": l1_data.get("threshold", 0.0),
                "window_hours": l1_data.get("window_hours", 0),
                "reference_time": l1_data.get("reference_time"),
            }

            l2_serializable = {
                "trend": l2_data.get("trend", "unknown"),
                "slope": l2_data.get("slope", 0.0),
                "r2": l2_data.get("r2", 0.0),
                "days_analyzed": l2_data.get("days_analyzed", 0),
                "min_slope_threshold": l2_data.get("min_slope_threshold", 0.0),
                "reference_time": l2_data.get("reference_time"),
            }

            db_inc = Incident(
                incident_id=inc.incident_id,
                event_id=f"evt-{inc.robot_code}-sim",
                packet_json={
                    "robot_code": inc.robot_code,
                    "abnormal_family": inc.abnormal_family,
                    "sensor": inc.sensor,
                    "severity_hint": inc.severity_hint,
                    "score": inc.anomaly_score,
                    "alert_type": triggers.get("alert_type"),
                    "per_sensor_scores": inc.per_sensor_scores,
                    "l1": l1_serializable,
                    "l2": l2_serializable,
                },
                decision_json={},
                trace_json={},
            )
            session.add(db_inc)
            session.commit()
            logger.info(f"Incident {inc.incident_id} saved to DB")
        except Exception as e:
            logger.error(f"Failed to save incident to DB: {e}")
            session.rollback()
        finally:
            session.close()

    def _dispatcher_loop(self) -> None:
        while not self._dispatcher_stop_event.is_set():
            try:
                inc = self._incident_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            try:
                self._submit_to_executor(inc)
            except Exception as e:
                logger.exception("Dispatcher failed for %s: %s", inc.incident_id, e)

    def _submit_incident(self, inc: DiagnosticIncident) -> None:
        self._incident_queue.put_nowait(inc)

    def _submit_to_executor(self, inc: DiagnosticIncident) -> None:
        if self._active_incident_count < 2:
            self._active_incident_count += 1
            self._broadcast({
                "type": "incident_queued",
                "incident_id": inc.incident_id,
                "robot_id": inc.robot_code,
                "status": "PROCESSING",
                "queue_position": None,
            })
            self._agent_executor.submit(self._launch_agent, inc)
        else:
            self._pending_incidents.append(inc)
            queue_pos = len(self._pending_incidents)
            self._broadcast({
                "type": "incident_queued",
                "incident_id": inc.incident_id,
                "robot_id": inc.robot_code,
                "status": "PENDING",
                "queue_position": queue_pos,
            })
            logger.info(f"Incident {inc.incident_id} queued (position {queue_pos})")

    def _launch_agent(self, inc: DiagnosticIncident) -> None:
        logger.info("Agent _launch_agent invoked for %s", inc.incident_id)
        logger.info("Agent STARTED for incident %s (robot %s)", inc.incident_id, inc.robot_code)
        self.state.set_active_incident(inc.incident_id)

        try:
            agent = DiagnosticAgent()
            agent.hitl_gate.set_state_getter(lambda: self.state)
            agent.emitter.on_hitl_request = self._broadcast
            agent.emitter.on_step = self._broadcast
            agent.emitter.on_observation = self._broadcast
            agent.emitter.on_plan = self._broadcast
            agent.emitter.on_plan_task = self._broadcast

            result = agent.run_incident(inc)

            self._broadcast({
                "type": "agent_done",
                "incident_id": inc.incident_id,
                "robot_id": inc.robot_code,
                "status": result.get("status"),
            })
        except Exception as e:
            logger.error(f"Agent error for {inc.incident_id}: {e}")
            self._broadcast({
                "type": "agent_done",
                "incident_id": inc.incident_id,
                "robot_id": inc.robot_code,
                "status": "failed",
                "error": str(e),
            })
        finally:
            self._active_incident_count = max(0, self._active_incident_count - 1)
            if self.state.get_active_incident() == inc.incident_id:
                self.state.set_active_incident(None)
            self._process_pending_incidents()

    def _process_pending_incidents(self) -> None:
        while self._pending_incidents and self._active_incident_count < 2:
            next_inc = self._pending_incidents.pop(0)
            self._active_incident_count += 1
            self._broadcast({
                "type": "incident_queued",
                "incident_id": next_inc.incident_id,
                "robot_id": next_inc.robot_code,
                "status": "PROCESSING",
                "queue_position": None,
            })
            self._agent_executor.submit(self._launch_agent, next_inc)

    def start(self) -> None:
        if self._running:
            logger.warning("Runner already running")
            return

        self._stop_event.clear()
        self._dispatcher_stop_event.clear()
        self._running = True
        self._debounce_timestamps.clear()
        self.state.reset()
        self.state.set_speed_factor(self.speed_factor)

        timeline = self.engine.get_timeline()
        self.state.set_total_ticks(len(timeline))
        self.state.set_status("running")

        self._dispatcher_thread = threading.Thread(target=self._dispatcher_loop, name="incident_dispatcher", daemon=True)
        self._dispatcher_thread.start()

        self._thread = threading.Thread(target=self._run_loop, name="simulation_runner", daemon=True)
        self._thread.start()
        logger.info(f"Simulation runner started with {len(timeline)} ticks")

    def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping simulation runner")
        self._stop_event.set()
        self._dispatcher_stop_event.set()
        self._pause_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5)
            self._dispatcher_thread = None

        self._agent_executor.shutdown(wait=False, cancel_futures=True)
        self.state.set_status("idle")
        self.state.reset()
        logger.info("Simulation runner stopped")

    def pause(self) -> None:
        if not self._running:
            return
        self._pause_event.clear()
        self.state.set_status("paused")
        self._broadcast({"type": "sim_paused"})

    def resume(self) -> None:
        if not self._running:
            return
        self._pause_event.set()
        self.state.set_status("running")
        self._broadcast({"type": "sim_resumed"})

    def warp_to_alert(self) -> None:
        if not self._running:
            return
        self._warp_mode = True
        self.resume() # Ensure we are not paused
        logger.info("Simulation warping to first alert...")

    def jump_to_tick(self, tick_index: int) -> None:
        if not self._running:
            return
        self._target_tick = tick_index
        self.resume()
        logger.info(f"Simulation jumping to tick {tick_index}")

    def _run_loop(self) -> None:
        timeline = self.engine.get_timeline()
        real_seconds_per_tick = 1.0  # 1 tick = 1 simulated hour = 1 real second at 1x speed
        if self.speed_factor > 1:
            real_seconds_per_tick = 1.0 / self.speed_factor
        # speed_factor semantics: 1x = 1h/1s, 2x = 1h/0.5s, etc.

        self._broadcast({
            "type": "sim_started",
            "total_ticks": len(timeline),
            "speed_factor": self.speed_factor,
        })

        tick_index = 0
        while tick_index < len(timeline):
            if self._stop_event.is_set():
                break

            self._pause_event.wait()

            # Handle jump
            if self._target_tick is not None:
                tick_index = self._target_tick
                self._target_tick = None
                if tick_index >= len(timeline):
                    break
                logger.debug(f"Runner jumped to tick {tick_index}")

            tick_time = timeline[tick_index]
            self.state.set_tick_index(tick_index)
            self.state.set_sim_time(tick_time)

            session = self.engine.session_factory()
            try:
                events = self.engine.get_events_at(tick_time)
                events_count = len(events)
            finally:
                session.close()

            logger.debug(f"Tick {tick_index} processed")

            self._broadcast({
                "type": "tick",
                "tick_index": tick_index,
                "sim_time": tick_time.isoformat(),
                "events_count": events_count,
            })

            # Broadcast per-robot file-anomaly count for this tick
            for robot_id in self.engine.get_robots():
                if self._stop_event.is_set():
                    break
                error_count = self.engine.get_error_count_at(robot_id, tick_time)
                if error_count > 0:
                    self._broadcast({
                        "type": "error_count",
                        "robot_id": robot_id,
                        "sim_time": tick_time.isoformat(),
                        "tick_index": tick_index,
                        "count": error_count,
                    })

            for robot_id in self.engine.get_robots():
                if self._stop_event.is_set():
                    break

                session = self.engine.session_factory()
                try:
                    triggers = self.engine.check_triggers(
                        session, robot_id, up_to=tick_time
                    )
                finally:
                    session.close()

                if triggers.get("has_alert"):
                    last_triggered_tick = self._debounce_timestamps.get(robot_id)
                    if last_triggered_tick is not None and tick_index - last_triggered_tick < DEBOUNCE_WINDOW_TICKS:
                        self._broadcast({
                            "type": "alert_suppressed",
                            "robot_id": robot_id,
                            "reason": "debounce",
                            "sim_time": tick_time.isoformat(),
                            "ticks_remaining": DEBOUNCE_WINDOW_TICKS - (tick_index - last_triggered_tick),
                        })
                        continue

                    self._debounce_timestamps[robot_id] = tick_index
                    inc = self._create_incident(robot_id, triggers, tick_time=tick_time)
                    self._save_incident_to_db(inc, triggers)

                    triage_result = triage_incident(inc)

                    self.state.add_alert({
                        "robot_id": robot_id,
                        "incident_id": inc.incident_id,
                        "alert_type": triggers.get("alert_type"),
                        "sim_time": tick_time.isoformat(),
                    })

                    self._broadcast({
                        "type": "alert",
                        "robot_id": robot_id,
                        "incident_id": inc.incident_id,
                        "alert_type": triggers.get("alert_type"),
                        "sim_time": tick_time.isoformat(),
                    })

                    if triage_result.decision == TriageDecision.AUTO_RESOLVE:
                        logger.info(f"Triage auto-resolved: {inc.incident_id}")
                        self._broadcast({
                            "type": "auto_resolved",
                            "incident_id": inc.incident_id,
                            "robot_id": robot_id,
                            "reason": triage_result.reason,
                            "message": triage_result.auto_resolve_message,
                        })
                    elif triage_result.decision == TriageDecision.CRITICAL_ESCALATE:
                        logger.warning(f"Triage CRITICAL ESCALATE: {inc.incident_id}")
                        self._broadcast({
                            "type": "critical_escalate",
                            "incident_id": inc.incident_id,
                            "robot_id": robot_id,
                            "reason": triage_result.reason,
                        })
                        self._submit_incident(inc)
                    else:
                        self._submit_incident(inc)

                    # Stop warp mode if we found an alert
                    if self._warp_mode:
                        self._warp_mode = False
                        logger.info(f"Warp stopped at tick {tick_index} (Alert triggered)")

            self._handle_alert_storm(tick_time)

            if not self._warp_mode:
                time.sleep(real_seconds_per_tick)

            tick_index += 1

        self.state.set_status("finished")
        self._broadcast({
            "type": "sim_finished",
            "total_ticks": len(timeline),
        })
        self._running = False
        logger.info("Simulation runner finished")
        self._running = False
        logger.info("Simulation runner finished")