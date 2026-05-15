# Task Summary: T11 — Remove Silent Fallback from rag_tools.py

**Sprint:** Sprint 6
**Task:** T11

## Summary of Work
Removed the silent `_fallback_search` method from `QueryPMCardsTool`. Replaced all `except Exception: return self._fallback_search(...)` paths with explicit error dicts containing `"error"` and `"detail"`.

## Files Modified
- `backend/app/robot_agent/agent/rag_tools.py` — Removed `_fallback_search`; replaced fallback paths with `[{"error": "...", "detail": str(exc)}]`; removed unused imports

## Verification
Agent now receives legible error observations instead of silently degraded SQL results.
