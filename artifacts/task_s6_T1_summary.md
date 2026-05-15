# Task Summary: T1 — Deduplicate LLM Parsing

**Sprint:** Sprint 6
**Task:** T1

## Summary of Work
Extracted the duplicated action-parsing block from `_run_plan_phase` and `_run_execute_phase` into a private method `_parse_llm_output(result) -> tuple[str, str, str]`.

## Files Modified
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Added `_parse_llm_output` method; replaced both parsing blocks with calls to it

## Verification
All existing tests pass — no behavior change.
