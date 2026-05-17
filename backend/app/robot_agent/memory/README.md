# robot_agent / Memory

Four-tier memory system for the diagnostic agent.

## Tiers

| Tier | Class                    | Backend     | Purpose                                  |
|------|--------------------------|-------------|------------------------------------------|
| T0   | `Tier0ContextCompaction` | In-memory   | Summarise & prune long sessions          |
| T1   | `Tier1SessionMemory`     | JSONL file  | Per-session turn-level reasoning log     |
| T2   | `Tier2ProjectRules`      | Markdown    | Read-only project rule files             |
| T3   | `Tier3SemanticMemory`    | ChromaDB    | Vector-similarity PM-card look-up        |

## Data Classes

- **`MemoryEntry`** – A persisted cross-session memory record (robot, incident, summary, confidence).
- **`SessionTurn`** – A single agent reasoning step (action, skill, tool calls, context snapshot).
