# Frontend Dashboard — WeldBot AI

React dashboard for the WeldBot AI diagnostic system. Provides real-time monitoring of welding robot anomalies, agent diagnostic traces, and Human-in-the-Loop (HITL) approval workflow.

## Structure

```
frontend/
├── src/
│   ├── components/    # React components (agent chat, HITL modals, robot status)
│   ├── hooks/         # Custom React hooks (SSE streaming, API calls)
│   ├── stores/        # Zustand state stores (incidents, HITL, simulation)
│   └── main.tsx       # Application entry point
├── public/            # Static assets
├── package.json       # Node dependencies
├── Dockerfile         # Docker image (served via nginx)
├── nginx.conf         # nginx configuration for production
└── vite.config.js     # Vite configuration
```

## Key Features

- **Real-time SSE streaming** — Agent steps, observations, and HITL requests streamed live
- **HITL approval modals** — Approve/reject HIGH/CRITICAL actions inline
- **Incident timeline** — View diagnostic traces with reasoning steps
- **Robot status dashboard** — 4-robot overview with anomaly indicators
- **Simulation controls** — Start/stop/step simulation engine from the UI

## Running

### Development

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173 (proxies API to localhost:8000)
```

### Production

```bash
npm run build
npm run preview
# → http://localhost:4173
```

### Docker

```bash
docker compose up --build   # from project root
# → http://localhost:3000 (served via nginx, proxies to backend container)
```

## Tech Stack

- **React 19** — UI framework
- **TypeScript** — Type safety
- **Vite** — Build tool
- **Zustand** — State management
- **SSE** — Server-Sent Events for real-time agent streaming

## Environment

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend API base URL |
| `VITE_WS_URL` | `ws://localhost:8000` | WebSocket URL |

## API Integration

The frontend connects to the backend via:

| Endpoint | Purpose |
|---|---|
| `GET /api/incidents` | List incidents |
| `GET /api/robots` | Robot status |
| `GET /api/hitl/pending` | Pending HITL approvals |
| `POST /api/hitl/{id}/approve` | Approve HITL action |
| `POST /api/hitl/{id}/reject` | Reject HITL action |
| `GET /api/simulation/events` | SSE stream |
| `POST /api/simulation/start` | Start simulation |
| `POST /api/simulation/stop` | Stop simulation |
