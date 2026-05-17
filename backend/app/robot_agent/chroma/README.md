# robot_agent / Chroma

ChromaDB vector-store wrapper for semantic PM-card retrieval.

## Functions

| Function         | Description                                       |
|------------------|---------------------------------------------------|
| `get_client()`   | Return a persistent ChromaDB client               |
| `get_collection(name)` | Get or create a named collection (default: `pm_cards`) |
| `reset_client()` | Wipe all collections and start fresh              |

Data is stored under `<project_root>/data/chroma_db/`.
