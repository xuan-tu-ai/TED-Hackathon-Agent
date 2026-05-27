# robot_agent / API

FastAPI routing layer for the robot diagnostic agent.

## Endpoints

### Simulation (`simulation_api.py`)

| Method | Path                 | Description                                  |
|--------|----------------------|----------------------------------------------|
| POST   | `/api/sim/start`     | Initialise and start the discrete-event sim  |
| POST   | `/api/sim/pause`     | Pause a running simulation                   |
| POST   | `/api/sim/resume`    | Resume a paused simulation                   |
| POST   | `/api/sim/stop`      | Stop and clean up the simulation             |
| GET    | `/api/sim/status`    | Return current simulation state              |
| GET    | `/api/sim/stream`    | SSE stream of simulation events              |

### Incidents (`incidents_api.py`)

| Method | Path                              | Description                                     |
|--------|-----------------------------------|-------------------------------------------------|
| GET    | `/api/incidents/`                 | List incidents (filterable by robot/severity)   |
| POST   | `/api/incidents/trigger_demo`     | Inject a random demo incident & run agent       |
| GET    | `/api/incidents/{id}/trace`       | Fetch agent reasoning trace for an incident     |

### HITL (`hitl_api.py`)

| Method | Path                                  | Description                       |
|--------|---------------------------------------|-----------------------------------|
| GET    | `/api/hitl/pending`                   | List entries awaiting human review|
| POST   | `/api/hitl/resolve`                   | Approve or reject a HITL entry    |
| POST   | `/api/sim/hitl/{id}/approve`          | Approve (simulation-integrated)   |
| POST   | `/api/sim/hitl/{id}/reject`           | Reject (simulation-integrated)    |
