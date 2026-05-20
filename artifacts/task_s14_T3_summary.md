# Task Summary: T3 — Colored Noise Generators

**Sprint:** Sprint 14
**Task:** T3 — Implement colored noise generators (`physics/noise.py`)

## Summary of Work
Implemented three sensor-specific noise models: S0 (wire_feed_speed) — mechanical vibration via sinusoidal + Gaussian noise; S1 (welding_current) — arc white noise at 5% of signal amplitude (~7.5A std for 150A signal); S2 (temperature) — thermal random walk with leakage (0.999× per step) for smooth drift + rare glitch spikes (~1 in 10,000 steps, ±5°C). Also added `arc_strike_fluctuation` — a normal micro-event at arc ignition.

## Files Modified
- `scripts/data_simulator/physics/noise.py` — Full noise generator implementation (3 noise types + arc strike)
- `tests/test_data_simulator/test_noise.py` — 18 unit tests

## Testing
- **Test File:** `tests/test_data_simulator/test_noise.py`
- **Status:** 18/18 passed
- **Execution Command:** `uv run python -m pytest tests/test_data_simulator/test_noise.py -v`

## Additional Notes
- Noise is reproducible given fixed seed.
- S0 vibration frequency (0.05 cycles/step) aliases a ~20Hz mechanical vibration at 4Hz sampling.
- S2 random walk uses 0.999 leakage to prevent unbounded drift over long sequences.
