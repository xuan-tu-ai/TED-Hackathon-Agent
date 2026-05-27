# Simulation Sub-package

Time-series replay engine that drives historical `PerceptionResult` data through the multi-agent diagnostic pipeline.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ SimulationEngine │────▶│ SimulationRunner │────▶│ EventBroadcaster │
│  (data access)   │     │   (tick loop)    │     │   (SSE stream)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
   ┌──────────┐          ┌──────────────┐
   │   DB     │          │ SimulationState │
   │ (SQLite) │          │  (state machine)│
   └──────────┘          └──────────────┘
```

## Simulation Lifecycle

```
IDLE ──start()──▶ RUNNING ──pause()──▶ PAUSED
                   │                     │
                   │  resume()◀──────────┘
                   │
                   │  stop()
                   ▼
                 FINISHED ──▶ IDLE (reset)
```

Additional controls:
- **warp-to-alert**: Fast-forwards through ticks until the first alert fires.
- **jump-to-tick**: Instantly advances to a specific tick index.

## State Machine (`SimulationState`)

| Field | Type | Description |
|---|---|---|
| `status` | `Literal["idle","running","paused","finished","blocked"]` | Current lifecycle phase |
| `sim_time` | `datetime \| None` | Current simulated time |
| `tick_index` | `int` | Current tick (0-indexed) |
| `total_ticks` | `int` | Total ticks in the timeline |
| `alerts` | `list[dict]` | Accumulated alerts during run |
| `active_incident_id` | `str \| None` | Incident currently being processed |
| `speed_factor` | `int` | Simulation speed multiplier (1 = 1 h per real second) |
| `hitl_events` | `dict[str, threading.Event]` | Human-in-the-loop blocking events |

All read/write methods are thread-safe via an internal `threading.Lock`.

## SSE Events (`EventBroadcaster`)

Events broadcast to SSE subscribers:

| Event Type | Payload Description |
|---|---|
| `sim_started` | `total_ticks`, `speed_factor` |
| `tick` | `tick_index`, `sim_time`, `events_count` |
| `alert` | `robot_id`, `incident_id`, `alert_type`, `sim_time` |
| `alert_suppressed` | `robot_id`, `reason`, `sim_time`, `ticks_remaining` |
| `alert_storm_detected` | `alert_count`, `sim_time` |
| `incident_queued` | `incident_id`, `robot_id`, `status`, `queue_position` |
| `auto_resolved` | `incident_id`, `robot_id`, `reason`, `message` |
| `critical_escalate` | `incident_id`, `robot_id`, `reason` |
| `agent_done` | `incident_id`, `robot_id`, `status`, `error` |
| `sim_paused` / `sim_resumed` | (none) |
| `sim_finished` | `total_ticks` |

## Trigger Debounce

Repeated alerts for the same robot are suppressed for `DEBOUNCE_WINDOW_TICKS` (24) ticks to prevent flooding. Alert-storm detection fires when ≥ `ALERT_STORM_THRESHOLD` (5) alerts occur within a 1-hour window.

## Incident Dispatch

1. Alert fires → `DiagnosticIncident` created and saved to DB.
2. `triage_incident()` decides: `AUTO_RESOLVE`, `CRITICAL_ESCALATE`, or normal dispatch.
3. Normal incidents are queued via an `asyncio.Queue` and dispatched to a `ThreadPoolExecutor` (max 2 concurrent).
4. Pending incidents are processed as slots free up.

## Key Constants

| Constant | Default | Purpose |
|---|---|---|
| `ALL_ROBOTS` | `R-01` … `R-03` | All welding robots |
| `ALERT_STORM_THRESHOLD` | 5 | Alerts per hour to trigger storm detection |
| `DEBOUNCE_WINDOW_TICKS` | 24 | Ticks to wait before re-triggering same robot |
