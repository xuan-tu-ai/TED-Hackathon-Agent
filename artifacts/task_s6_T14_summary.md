# Task Summary: T14 — Fix Silent Exception Handlers

**Sprint:** Sprint 6
**Task:** T14

## Summary of Work
Fixed three silent exception handlers:
1. `diagnostic_agent.py` — bare `except: pass` in `_parse_llm_output` → `except Exception as exc: logger.debug(...)`
2. `diagnostic_agent.py` — hitl_check JSON parse → added `logger.warning(...)` and `"parse_error"` to params
3. `memory.py` — ChromaDB failure returning `[]` → returns `[{"error": "Tier3 ChromaDB query failed", "detail": str(exc)}]`

## Files Modified
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Fixed 2 silent handlers
- `backend/app/robot_agent/memory/memory.py` — Fixed 1 silent handler

## Verification
Tests in `TestSilentExceptionRegression` all pass.
