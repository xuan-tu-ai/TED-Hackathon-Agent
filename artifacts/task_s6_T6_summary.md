# Task Summary: T6 — Create HitlGate

**Sprint:** Sprint 6
**Task:** T6

## Summary of Work
Created `HitlGate` class that owns the HITL threading/blocking logic. Public `check()` method handles BLOCKED, PERMANENTLY_BLOCKED, and ALLOWED outcomes.

## Files Created
- `backend/app/robot_agent/agent/hitl_gate.py` — `HitlGate` class

## Verification
Tests in `TestHitlGate` and `TestHitlGatePermanentlyBlocked` all pass.
