# Poller Sub-package

Cognitive polling loop for real-time L1 (burst) and L2 (slope) anomaly detection across all welding robots.

## Architecture

```
┌──────────────┐     ┌──────────────────┐
│ CognitivePoller │────▶ Alert Handler (callback)
│  (poll loop)   │     └──────────────────┘
└──────┬───────┘
       │
       ├──▶ detect_l1_burst(session, robot_id, ...)
       │       └── returns {"is_burst", "event_count", "density_per_hour", ...}
       │
       └──▶ detect_l2_slope(session, robot_id, ...)
               └── returns {"trend", "slope", "r2", "daily_averages", ...}
```

## Cognitive Polling (`CognitivePoller`)

The poller iterates over all robots at a configurable `poll_interval` (default 10 s) and evaluates both L1 and L2 conditions.

```python
poller = CognitivePoller(
    db_url="sqlite:///data/demo.db",
    robots=None,                           # None = all robots
    poll_interval=10,                      # seconds
    l1_threshold=8.0,                      # density threshold
    l2_min_slope=0.10,                     # min slope for escalation
)
poller.set_alert_handler(my_alert_callback)
poller.start()   # blocking loop
```

## L1 Burst Detection (`l1_burst.py`)

Detects short-term anomaly bursts by measuring event density.

| Parameter | Default | Description |
|---|---|---|
| `window_hours` | 1 | Time window for density calculation |
| `density_threshold` | 2.5 | Events/hour to declare a burst |

**Multi-window summary** (`get_l1_summary`): evaluates 1 h, 6 h, and 24 h windows with a threshold of 8.0.

**Return value**:
```python
{
    "is_burst": bool,
    "event_count": int,
    "density_per_hour": float,
    "threshold": float,
    "window_hours": int,
    "reference_time": str,       # ISO 8601
    "recent_events": [...],      # up to 10 PerceptionResult objects
}
```

## L2 Slope Detection (`l2_slope.py`)

Detects long-term progressive degradation via linear regression on daily-average `abnormal_score`.

| Parameter | Default | Description |
|---|---|---|
| `days` | 30 | Look-back window in days |
| `min_slope` | 0.015 | Minimum slope to classify as escalating/improving |

**Multi-window summary** (`get_l2_summary`): evaluates 7 d, 14 d, and 30 d windows.

**Return value**:
```python
{
    "trend": "escalating" | "stable" | "improving" | "insufficient_data",
    "slope": float,             # change in score per day
    "r2": float,                # fit quality (0–1)
    "daily_averages": [...],    # list of (date, avg_score) tuples
    "days_analyzed": int,
    "min_slope_threshold": float,
    "reference_time": str,      # ISO 8601
}
```

## Threshold Configuration

| Parameter | Default | Affects |
|---|---|---|
| `l1_threshold` | 8.0 | `poll_once()` uses `get_l1_summary` which applies threshold across windows |
| `l2_min_slope` | 0.10 | `poll_once()` uses `get_l2_summary` which checks trend per window |
| `window_hours` | 1 (single), 1/6/24 (summary) | L1 analysis window |
| `density_threshold` | 2.5 (single), 8.0 (summary) | L1 burst trigger |
| `days` | 30 (single), 7/14/30 (summary) | L2 regression window |
| `min_slope` | 0.015 (single), varies (summary) | L2 escalation trigger |

## Usage as CLI

```bash
uv run python -c "from app.robot_agent.poller.poller import run_poller; run_poller()"
```
