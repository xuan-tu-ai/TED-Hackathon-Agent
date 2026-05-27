# Task Summary: T5 — Welding Sample Generator

**Sprint:** Sprint 14
**Task:** T5 — Build the normal sample generator (`generator.py`)

## Summary of Work
Implemented `WeldingSampleGenerator` that composes profile selection → phase envelope → LTI current dynamics → thermal dynamics → colored noise into a complete welding cycle. Supports random or fixed robot_id (0-3) and program_id (0-15). Produces both clean (physics only) and noisy (physics + noise) samples. All 4 robots produce valid data, with Robot 3 having +5°C temperature baseline.

## Files Modified
- `scripts/data_simulator/generator.py` — Core generator class
- `tests/test_data_simulator/test_generator.py` — 13 unit tests

## Testing
- **Test File:** `tests/test_data_simulator/test_generator.py`
- **Status:** 13/13 passed
- **Execution Command:** `uv run python -m pytest tests/test_data_simulator/test_generator.py -v`

## Additional Notes
- Sensor ranges verified: feed_speed 1000-5000 mm/min, current 50-170A, temp 25-130°C.
- Samples vary in length (100-600+ steps) depending on program profile.
- Phase boundaries (idle_end, rampup_end, active_end, winddown_end) are preserved after noise addition.
