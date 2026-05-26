# Seed Sub-package

Database seeding scripts for populating the demo database with synthetic `PerceptionResult` and `PMCard` data across welding robots.

## Seed Scripts

| Module | Strategy | Anomaly Pattern | Rows |
|---|---|---|---|
| `perception_results.py` | **Deterministic** | Exactly one spike: R-07 `paint_pressure` at day 5 hour 0 | 47 robots × *days* × 24 h |
| `perception_results_legacy.py` | **Probabilistic** (numpy) | R-03 L1 burst, R-04 L2 escalation, R-05 L1 burst | 47 robots × *days* × 60 cycles |
| `pm_cards.py` | **Static list** | 8 PM cards for paint-pressure anomalies (R-07 + 1 shared) | 8 |
| `run.py` | **Orchestrator** | Calls `seed_all` + SQL snapshot export | — |

## Data Distributions

### Deterministic Seed (`perception_results.py`)

```
47 robots × 30 days × 24 h = 33 840 rows

Normal data (everything except day 5, hour 0, R-07):
  • shape = 0.05, win = 0.04, error_steps = []
  • abnormal_score = 0.10, severity = "low", is_file_anomaly = False

Anomaly (R-07, day 5, hour 0):
  • paint_pressure: shape = 0.88, win = 0.78, error_steps = [100..149]
  • abnormal_score = 1.00, severity = "critical", is_file_anomaly = True
```

### Probabilistic Seed (`perception_results_legacy.py`)

```
47 robots × 45 days × 60 cycles = 126 900 rows

R-03: paint_flow_rate L1 burst        (~65 % probability per cycle)
R-04: high_pressure_current L2 slope   (gradual: 0.30 → 0.93 over 45 days)
R-05: electro_pneumatic_output L1 burst (~55 % probability per cycle)
```

### PM Cards (`pm_cards.py`)

8 static cards, all relating to `paint_pressure` — 7 specific to R-07 and 1 shared-manifold card (robot_code = `None`).

## Usage

```bash
# Seed the demo database and export SQL snapshot
uv run python app/robot_agent/seed/run.py

# Or programmatically:
from app.robot_agent.seed import seed_all, seed_demo_data
counts = seed_all(session, days=30)
```

## Output

- SQLite database at `data/demo.db`
- SQL snapshot at `data/sql/seed_snapshot.sql`

The snapshot is tracked in version control so CI and other environments can load a reproducible state.
