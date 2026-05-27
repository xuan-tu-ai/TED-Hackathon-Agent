# Task Summary: Sprint 11 — T11

**Sprint:** Sprint 11
**Task:** T11 — Migrate hardcoded URLs in tests — Group B (S6–S8)

## Summary of Work
- **test_s6_simulation_engine.py**: 23 occurrences of `"sqlite:///data/demo.db"` → `db_url` fixture across 31 test methods
- **test_s7_l1_rebalance.py**: 1 occurrence → `db_url` fixture
- **test_s7_priority_queue.py**: 4 occurrences → `db_url` fixture
- test_s7_e2e.py and test_s8_l1_balance.py had no hardcoded URLs needing migration.

## Files Modified
- [backend/tests/test_s6_simulation_engine.py](backend/tests/test_s6_simulation_engine.py)
- [backend/tests/test_s7_l1_rebalance.py](backend/tests/test_s7_l1_rebalance.py)
- [backend/tests/test_s7_priority_queue.py](backend/tests/test_s7_priority_queue.py)

## Testing
- All Group B tests with migrated URLs pass on SQLite (pre-existing failures in other tests unchanged)
