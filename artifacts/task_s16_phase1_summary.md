# Task Summary: Sprint 16 — Phase 1 (T1–T8)

**Sprint:** Sprint 16
**Task:** T1–T8: Noise Model Refinement

## Summary of Work
Replaced i.i.d. white noise with AR(1) autoregressive noise (momentum) across all 3 sensor channels. Added local-window variance estimation to replace global `signal_amplitude`. Added U-pattern regime-switching noise for future setpoint-controlled sensors. Added config params to `NoiseConfig` dataclass. Wrote 12 new tests covering AR(1) autocorrelation, local std behavior, and U-pattern plateau holding.

## Files Modified
- `backend/data_simulator/physics/noise.py` — Added `_ar1_noise()`, `_local_std()`, refactored `add_current_noise()`, `add_wire_feed_noise()`, `add_temperature_noise()`; added `add_u_pattern_noise()`; added config params to `NoiseConfig`

## Testing
- **Test File:** `backend/tests/test_data_simulator/test_noise.py`
- **Status:** 30/30 passed
- **New tests:** `TestAR1Noise` (4), `TestLocalStd` (4), `TestUPatternNoise` (3), updated `TestCurrentNoise` (2)
- **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_noise.py -v`

## Additional Notes
- AR(1) phi=0.85 by default — strong momentum, gradual transitions
- Local std uses `scipy.ndimage.uniform_filter1d` for O(n) rolling computation
- U-pattern is opt-in (`u_pattern_enabled=False` by default)
- Full simulator suite (138 tests) passes
