# BLUEPRINT-03: KIẾN TRÚC AGENT ĐÃ TRIỂN KHAI (AS-BUILT — Sprint 10)

> **Trạng thái:** Tài liệu thực tế (không phải thiết kế lý thuyết).  
> Mô tả đúng những gì chạy trong `backend/` tại thời điểm Sprint 10.  
> Đọc cùng [Blueprint-01](./Blueprint-01.md) (lý thuyết Databricks) và [Blueprint-02](./Blueprint-02.md) (Q&A thiết kế CLI) để thấy khoảng cách design→implementation.

---

## 1. TỔNG QUAN HỆ THỐNG

Hệ thống hiện tại là một **simulation-first diagnostic platform** — không kết nối OT thật, thay vào đó phát lại lịch sử 30 ngày từ bảng `perception_results` để demo luồng agent end-to-end.

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
│   ChatPanel · PlanTracker · CollapsibleStep · SimulationStore   │
└────────────────────────┬─────────────────────────────────────────┘
                         │ SSE (EventSource)  │ REST (fetch)
┌────────────────────────┴─────────────────────────────────────────┐
│                        FASTAPI SERVER                            │
│  /api/sim/*   /api/hitl/*   /api/incidents/*   /api/health      │
└───┬───────────────────┬──────────────────────────────────────────┘
    │                   │
    │ async             │ sync (thread-safe bridge)
    ▼                   ▼
EventBroadcaster   SimulationRunner  ◄──── SimulationEngine
(asyncio.Queue)    (background thread)      (SQLite query)
    ▲                   │
    │ broadcast()       │ ThreadPoolExecutor (max 5)
    └───────────────────┤
                        ▼
                  DiagnosticAgent
                  (DSPy + skill files)
                        │
               ┌────────┴────────┐
               │                 │
          TriageAgent      SystemicAgent
          (rule-based)     (stub, future)
```

**Công nghệ cốt lõi:** Python 3.13 · FastAPI · SQLAlchemy · DSPy-AI · ChromaDB · SQLite

---

## 2. TẦNG AGENT

### 2.1 TriageAgent — Cổng Phân Loại

File: `backend/agent/triage_agent.py`

Agent nhẹ nhất, **không dùng LLM**, chạy hoàn toàn bằng rule-based logic. Nhận `DiagnosticIncident` từ runner và ra quyết định định tuyến.

```
Input: DiagnosticIncident
         │
         ├── severity_hint == "CRITICAL"?  →  CRITICAL_ESCALATE
         │
         ├── family ∈ {SURGE, DROPOUT, INSTABILITY}?  →  QUEUE_FOR_DIAGNOSTIC
         │
         ├── anomaly_score >= 0.5?  →  QUEUE_FOR_DIAGNOSTIC
         │
         └── else  →  AUTO_RESOLVE
```

Output: `TriageResult(decision, incident_id, robot_code, reason, auto_resolve_message)`

**Tại sao tách TriageAgent riêng?** Ngăn LLM bị wake up bởi nhiễu L2 thấp — tiết kiệm token và giữ queue sạch.

---

### 2.2 DiagnosticAgent — Lõi Suy Luận

File: `backend/agent/diagnostic_agent.py`

Agent chính dùng **DSPy ChainOfThought** để lập luận qua hai pha bắt buộc.

#### Cấu trúc dữ liệu

```python
@dataclass
class DiagnosticIncident:
    incident_id: str
    robot_code: str          # "R-07"
    abnormal_family: str     # "SURGE" | "DROPOUT" | ...
    abnormal_subtype: str    # "spike" | "flatline" | ...
    sensor: str              # "motor_current_a"
    anomaly_score: float     # 0.0 – 1.0
    severity_hint: str       # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"

@dataclass
class AgentStep:
    step_id: str
    action: str              # tên tool hoặc "think"
    tool_name: str | None
    tool_input: dict | None
    reasoning: str
    observation: str | None
    timestamp: str
```

#### Vòng đời một Incident

```
run_incident(incident)
    │
    ├─ [PHASE 1: PLAN] max_turns=5
    │      Bước đầu tiên BẮT BUỘC: load_skill("create_plan")
    │      → current_plan: { tasks: [{id, name, type, status}] }
    │      → broadcast: plan_created event
    │
    └─ [PHASE 2: EXECUTE] max_turns=20
           Lần lượt từng task trong plan:
           └─ ChainOfThought → {action, tool_name, tool_input, reasoning}
                  │
                  ├─ load_skill(name)       → đọc SKILL.md vào context
                  ├─ query_pm_cards(q)      → ChromaDB RAG
                  ├─ verify_surface_defect(robot_id) → mock vision API
                  ├─ query_db(q)            → mock DB query
                  ├─ hitl_check(action, params, reasoning) → BLOCKS thread
                  └─ update_task_status(id, status)
```

#### Progressive Skill Loading

Đây là pattern quan trọng nhất của DiagnosticAgent:

```
Default context: chỉ AGENTS.md (system prompt nhỏ gọn)
         │
         ▼ load_skill("routing")
context += agent_knowledge/skills/routing/SKILL.md
         │
         ▼ load_skill("create_plan")
context += agent_knowledge/skills/create_plan/SKILL.md
         │
         ▼ load_skill("diagnose_surge")   (nếu cần)
context += agent_knowledge/skills/diagnose_surge/SKILL.md
```

**Lý do:** Ngăn LLM hallucinate bằng cách chỉ nạp kiến thức đúng lúc. Mỗi skill là một SKILL.md trong `agent_knowledge/skills/<name>/`.

#### Callback System (Event Broadcasting)

DiagnosticAgent không tự broadcast. Nó gọi callback được inject từ ngoài:

| Callback | Khi nào | Payload |
|---|---|---|
| `on_step(step)` | Mỗi lần agent chọn action | AgentStep (chưa có observation) |
| `on_observation(step_id, obs)` | Sau khi tool trả về | string observation |
| `on_plan(tasks)` | Sau khi tạo plan xong | list of task dicts |
| `on_plan_task(task_id, status)` | Mỗi khi task thay đổi trạng thái | task_id, new_status |
| `on_hitl_request(payload)` | Khi hitl_check cần người duyệt | HitlRequest dict |

---

### 2.3 SystemicAgent — Stub Tương Lai

File: `backend/agent/systemic_agent.py`

Phát hiện storm nhiều robot (≥5 alert trong 1 giờ). **Chưa implement DSPy logic**, hiện chỉ là dataclass và interface. SimulationRunner phát `alert_storm_detected` event nhưng không spawn SystemicAgent.

---

## 3. TẦNG HITL (HUMAN-IN-THE-LOOP)

File: `backend/agent/hitl.py`

### Flow Chặn-Chờ-Tiếp

```
DiagnosticAgent.hitl_check(action, parameters, reasoning)
         │
         ├─ action ∈ BLOCKED_ACTIONS?  →  PERMANENTLY_BLOCKED (raise exception)
         │
         ├─ ACTION_SEVERITY[action] ∈ {"LOW", "MEDIUM"}?  →  ALLOWED (continue immediately)
         │
         └─ severity ∈ {"HIGH", "CRITICAL"}:
                  │
                  1. INSERT HitlQueue(status=PENDING)
                  2. Create threading.Event
                  3. Store: state.hitl_events[queue_id] = event
                  4. broadcast: hitl_request event → frontend hiển thị
                  5. event.wait(timeout=300)  ← AGENT THREAD BLOCKED HERE
                  │
                  ┌────────────────────────────────┐
                  │  User clicks Approve/Reject     │
                  │  POST /api/sim/hitl/{id}/approve│
                  └────────────┬───────────────────┘
                               │
                  6. resolve_hitl_entry_and_set_event()
                     → UPDATE HitlQueue status
                     → event.set()  ← UNBLOCKS agent thread
                  │
                  7. Agent tiếp tục với kết quả APPROVED/REJECTED
```

### Danh Sách Actions

**BLOCKED_ACTIONS** (từ chối vĩnh viễn — safety interlock):
`override_safety_interlock`, `disable_emergency_stop`, `update_firmware`, `modify_safety_threshold`, `disable_painter_mode`, `force_robot_home`

**HIGH severity** (cần người duyệt):
`send_email`, `send_teams_message`, `send_slack_alert`, `create_jira_ticket`, `update_robot_config`, `adjust_paint_pressure`, `trigger_clean_cycle`, `approve_plan`

**CRITICAL severity** (cần người duyệt ngay):
`dispatch_maintenance`, `request_emergency_stop`

---

## 4. TẦNG SIMULATION

### 4.1 SimulationEngine — Phát Lại Lịch Sử

File: `backend/simulation/engine.py`

Không tạo dữ liệu giả — đọc thực từ `perception_results` (SQLite) và phát lại từng giờ.

```python
timeline = get_timeline()
# → list of hourly buckets: [datetime, ...]
# → 720 entries (30 ngày)

events_at(sim_time)
# → SELECT * FROM perception_results
#   WHERE observed_at IN [sim_time, sim_time+1h)
```

**L1 Detection (Burst):**
```
window = 24h, density_threshold = 15 events/hour
is_burst = (count(anomalies in window) / 24) >= 15
```

**L2 Detection (Slope):**
```
window = 30 ngày
group_by(date) → avg(abnormal_score) per day
linear_regression(x=day_index, y=avg_score)
is_escalating = slope >= 0.005
```

Ngưỡng L1/L2 được tune sau Sprint 9 để giảm alert noise.

---

### 4.2 SimulationRunner — Bộ Điều Phối

File: `backend/simulation/runner.py`

Chạy trên 3 thread + 1 executor pool:

| Thread | Vai trò |
|---|---|
| `_run_loop()` | Tick qua timeline, gọi engine, triage |
| `_dispatcher_loop()` | Lấy incident từ `queue.Queue`, dispatch vào executor |
| ThreadPoolExecutor (max 5) | Chạy DiagnosticAgent song song |

**Debounce (chống alert lặp):**
```
DEBOUNCE_WINDOW_TICKS = 24
Mỗi robot có cooldown 24 tick (= 24 giờ sim time)
→ alert bị suppress nếu robot đã alert trong cửa sổ đó
```

**Concurrency control:**
```
active_incident_count: int (max 2)
pending_incidents: list

Khi count < 2 → submit to executor ngay
Khi count >= 2 → đẩy vào pending
Khi agent xong → process_pending_incidents()
```

**Warp Mode** (skip đến alert đầu tiên):
```
_warp_mode = True → không sleep giữa các tick
Dừng warp khi first alert detected
```

---

### 4.3 SimulationState — Trạng Thái Toàn Cục

File: `backend/simulation/state.py`

Singleton, thread-safe với `threading.Lock`. Là source of truth cho toàn bộ FastAPI.

```python
status: "idle" | "running" | "paused" | "finished" | "blocked"
sim_time: datetime
tick_index: int
total_ticks: int
speed_factor: int
active_incident_id: str | None
hitl_events: dict[queue_id → threading.Event]
```

---

## 5. TẦNG SỰ KIỆN (EVENT SYSTEM)

### EventBroadcaster

File: `backend/simulation/broadcaster.py`

Cầu nối giữa sync thread (agent) và async event loop (FastAPI SSE):

```
sync thread:
  broadcaster.broadcast(event)   ← gọi từ bất kỳ thread nào
       │  threading.Lock
       ▼
  for queue in subscribers:
      asyncio.run_coroutine_threadsafe(queue.put(event), loop)

async event loop (FastAPI):
  GET /api/sim/stream
       │
       subscriber = await broadcaster.subscribe()  → asyncio.Queue
       │
       async for event in subscriber:
           yield f"data: {json.dumps(event)}\n\n"
```

### Catalog Sự Kiện

| event_type | Nguồn | Ý nghĩa |
|---|---|---|
| `tick` | runner | Mỗi tick sim |
| `alert` | runner | Robot vừa trigger L1/L2 |
| `alert_suppressed` | runner | Alert bị debounce |
| `alert_storm_detected` | runner | ≥5 alert trong 1 giờ |
| `incident_queued` | runner | Incident vào hàng chờ agent |
| `auto_resolved` | runner | TriageAgent → AUTO_RESOLVE |
| `critical_escalate` | runner | TriageAgent → CRITICAL |
| `agent_step` | agent | Một bước suy luận (chưa có obs) |
| `agent_step_update` | agent | Observation trả về |
| `plan_created` | agent | Plan xong, list tasks |
| `plan_task_update` | agent | Task đổi trạng thái |
| `hitl_request` | agent | Cần người duyệt |
| `hitl_timeout` | agent | 300s không có phản hồi |
| `agent_done` | runner | Agent xong incident |
| `sim_started/paused/resumed/finished` | runner | Lifecycle sim |

---

## 6. TẦNG API

### Simulation API — `backend/api/simulation_api.py`

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/sim/start` | Bắt đầu sim, body: `{speed_factor}` |
| POST | `/api/sim/pause` | Tạm dừng |
| POST | `/api/sim/resume` | Tiếp tục |
| POST | `/api/sim/skip_to_alert` | Warp đến alert đầu tiên |
| POST | `/api/sim/stop` | Dừng và reset |
| GET | `/api/sim/status` | SimulationState.to_dict() |
| GET | `/api/sim/stream` | SSE stream (EventSource) |
| POST | `/api/sim/hitl/{queue_id}/approve` | Duyệt HITL |
| POST | `/api/sim/hitl/{queue_id}/reject` | Từ chối HITL |

### HITL API — `backend/api/hitl_api.py`

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/hitl/pending` | Danh sách entries đang PENDING |
| POST | `/api/hitl/resolve` | Approve/Reject theo queue_id |

### Incidents API — `backend/api/incidents_api.py`

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/incidents/` | List incidents (filter: robot_code, severity) |
| GET | `/api/incidents/{id}/trace` | JSONL trace của agent session |
| POST | `/api/incidents/trigger_demo` | Trigger random demo incident |

---

## 7. TẦNG BỘ NHỚ VÀ TRI THỨC

### Memory Hierarchy (đã triển khai)

```
TIER 0 — In-Context RAM (mỗi DSPy call)
  ChainOfThought predictor, incident + history steps + loaded skills
  Xoá khi session kết thúc.

TIER 1 — Session Trace (JSONL)
  data/sessions/<incident_id>.jsonl
  Mỗi entry = một AgentStep. Persist ngay sau mỗi step.

TIER 2 — Project Rules (Markdown, đọc khi cần)
  agent_knowledge/rules/hitl-rules.md
  agent_knowledge/rules/severity-policy.md
  Inject vào system prompt lúc khởi tạo agent.

TIER 3 — Semantic Memory (ChromaDB)
  Collection: pm_cards
  Embed PM Cards (card_id, sensor, cause, symptom, action)
  Tool: query_pm_cards(query) → cosine similarity
```

### DB Schema (SQLite, chính)

| Table | Vai trò |
|---|---|
| `perception_results` | Input từ model (robot_code, observed_at, abnormal_type, score) |
| `incidents` | Incident tạo bởi runner |
| `hitl_queue` | Queue phê duyệt HITL |
| `pm_cards` | Thẻ bảo trì (dùng cho RAG) |
| `alerts` | Lịch sử alert gửi đi |
| `maintenance_events` | Lịch sử bảo trì robot |
| `reports` | Báo cáo chẩn đoán cuối |
| `agent_memories` | Ghi chú học từ session (future) |

---

## 8. PATTERN KIẾN TRÚC ĐÁng CHÚ Ý

### Pattern 1: Async/Sync Boundary

FastAPI chạy async; agent chạy sync trong thread. Boundary được xử lý hoàn toàn bởi `EventBroadcaster` — không có shared mutable state ngoài `SimulationState` (đã lock).

```
Sync World                      Async World
────────────────                ──────────────────────
SimulationRunner                FastAPI event loop
DiagnosticAgent         bridge  asyncio.Queue (per subscriber)
threading.Event    ────────►    SSE StreamingResponse
threading.Lock                  REST endpoints
```

### Pattern 2: Planning-First Workflow

Không để LLM tự do chọn tool ngay lập tức. Phase PLAN bắt buộc trước, tạo `current_plan` có thứ tự. Phase EXECUTE chỉ đi theo plan. Kết quả:

- Auditability: người dùng thấy plan trước khi agent hành động
- Controllability: UI có PlanTracker hiển thị progress real-time
- Anti-hallucination: plan constraints hạn chế tool choice

### Pattern 3: Surgical Context Loading

```
Agent biết gì ngay từ đầu?    →  Chỉ AGENTS.md (system rules)
Agent biết thêm gì sau đó?    →  load_skill() khi cần
Agent không bao giờ biết?      →  Skill của family khác
```

Ngăn "context poisoning" giữa các loại incident (SURGE vs DROPOUT có diagnostic approach khác nhau).

### Pattern 4: Thread-Safe HITL Gate

Agent chạy trong ThreadPoolExecutor thread. HITL cần đợi UI. Giải pháp: `threading.Event` + timeout 300s. Nếu timeout → `hitl_timeout` event + agent tự quyết định (tiếp tục với mức rủi ro thấp hơn hoặc abort).

---

## 9. KHOẢNG CÁCH DESIGN vs IMPLEMENTATION

| Blueprint-01 (lý thuyết) | Blueprint-03 (thực tế) | Lý do |
|---|---|---|
| Databricks Unity Catalog | SQLite + SQLAlchemy | Demo/prototype scope |
| Spark Stateful Streaming | SimulationRunner thread | Không cần Kafka ở scale này |
| Mosaic AI Model Serving | DSPy + env LM config | Provider-agnostic |
| 6-layer Kernel | 3-tier (Triage/Diagnostic/Systemic) | Simplified for demo |
| Vector Search (Databricks) | ChromaDB local | Cost và deployment |
| Multi-cloud federation | Single FastAPI process | MVP scope |
| MES/ERP integration | Mocked tool calls | Out of scope Sprint 10 |

---

## 10. CẤU TRÚC THƯ MỤC BACKEND

```
backend/
├── agent/
│   ├── diagnostic_agent.py    ← Core LLM reasoning
│   ├── triage_agent.py        ← Rule-based routing
│   ├── systemic_agent.py      ← Stub (future)
│   ├── hitl.py                ← HITL gate + DB
│   ├── rag_tools.py           ← ChromaDB query
│   ├── skills.py              ← Markdown skill loader
│   ├── session_trace.py       ← JSONL writer
│   └── llm_config.py          ← DSPy LM setup
├── simulation/
│   ├── engine.py              ← L1/L2 detection
│   ├── runner.py              ← Tick loop + dispatcher
│   ├── state.py               ← Global state singleton
│   └── broadcaster.py         ← SSE bridge (sync→async)
├── api/
│   ├── simulation_api.py      ← /api/sim/*
│   ├── hitl_api.py            ← /api/hitl/*
│   └── incidents_api.py       ← /api/incidents/*
├── poller/
│   ├── l1_burst.py            ← Density detection
│   └── l2_slope.py            ← Trend detection
├── seed/
│   ├── models.py              ← SQLAlchemy ORM
│   ├── run.py                 ← Seed script
│   └── perception_results.py  ← Seed data generator
└── main.py                    ← FastAPI app entrypoint

agent_knowledge/
├── AGENTS.md                  ← System prompt base
├── skills/
│   ├── create_plan/SKILL.md
│   ├── routing/SKILL.md
│   └── ...
└── rules/
    ├── hitl-rules.md
    └── severity-policy.md
```

---

*Cập nhật lần cuối: Sprint 10 — 2026-05-05*  
*Tác giả: DI-trietlm*