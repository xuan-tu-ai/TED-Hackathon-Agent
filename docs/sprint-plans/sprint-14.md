# Sprint 14 Plan

## Sprint Goal
Build a **physics-based welding robot data simulator** that generates realistic 30-day multivariate time-series datasets (train: 100% normal, test: 50% normal + 50% abnormal) matching the output format of `data/mock_welding_robot_data.csv`, with LTI dynamics, colored noise per sensor, and 4 distinct anomaly injection types.

## Motivation
- The existing `data/mock_welding_robot_data.csv` (~200MB, 1.8M rows) is flat mock data with only 1 robot (code=6), 1 program (116), and no labeled anomalies.
- The **GLC-Anomaly-MaskFormer** model needs diverse, physics-realistic data with:
  - Multiple robots (0–3) with different thermal baselines
  - Multiple programs (0–15) with varying cycle lengths, amplitudes, idle durations
  - LTI dynamics: current overshoot on arc ignition, exponential thermal response, thermal cooling curves
  - Sensor-specific colored noise: mechanical vibration (S0), arc white noise (S1), thermal random walk (S2)
  - 4 labeled anomaly types for test set evaluation
- Output must match the legacy CSV schema: `robot_code`, `start_time`, `program_number`, `infer_recorded_date`, `serial_number`, `phase`, `wire_feed_speed_mm_min`, `wire_temperature_c`, `welding_current_a`, `timestamp`

## Design Strategy
- **`scripts/data_simulator/`** — new directory under project root (not inside `backend/`), since this is a data generation tool, not application code.
- Modular architecture: **Physics Engine** → **Noise Generator** → **Program Profile** → **Sample Builder** → **Anomaly Injector** → **CSV Writer**
- Each module is independently testable and configurable via dataclass configs.
- Physics models use `scipy.signal.lti` / `scipy.signal.lsim` for LTI dynamics (current overshoot) and first-order ODE for thermal model.
- All randomness seeded for reproducibility.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Physics Engine (T1–T3) — ✅ Complete

- [x] **T1: Implement LTI current dynamics (`physics/current_dynamics.py`)**
  - Second-order underdamped LTI system via `scipy.signal.TransferFunction` / `lsim` to model the arc ignition overshoot (~12% overshoot, 1 step delay).
  - Input: `wire_feed_speed` time-series (reference signal). Output: `welding_current` time-series.
  - Tunable params: `natural_freq`, `damping_ratio`, `delay_steps`, `dc_gain`.
  - Unit test: 10/10 passed.

- [x] **T2: Implement thermal dynamics (`physics/thermal_dynamics.py`)**
  - First-order exponential charge/discharge model.
  - Different time constants for heating vs cooling (cooling is ~3× slower).
  - Robot-specific baseline offset: robot 3 has +5°C higher equilibrium.
  - Unit test: 11/11 passed.

- [x] **T3: Implement colored noise generators (`physics/noise.py`)**
  - S0: Mechanical vibration — sinusoidal + Gaussian.
  - S1: Arc white noise — Gaussian at 5% of signal amplitude.
  - S2: Thermal random walk with leakage + rare glitch spikes.
  - Normal arc strike fluctuation micro-event.
  - Unit test: 18/18 passed.

---

### Phase 2: Program Profiles & Normal Sample Generation (T4–T6) — ✅ Complete

- [x] **T4: Implement program profile system (`profiles.py`)**
  - 16 profiles (IDs 0–15) with varying phase durations and feed speeds.
  - `PhaseRanges`, `ProfileConfig`, `SampledProfile` dataclasses.
  - `build_phase_envelope()` generates piecewise-linear reference signal.
  - Unit test: 19/19 passed.

- [x] **T5: Build the normal sample generator (`generator.py`)**
  - `WeldingSampleGenerator` composing profile → envelope → physics → noise.
  - Supports random/fixed robot_id (0-3) and program_id (0-15).
  - Clean (physics only) and noisy (physics + noise) output.
  - Unit test: 13/13 passed.

- [x] **T6: Build the 30-day batch generator (`batch_generator.py`)**
  - `BatchGenerator` scheduling cycles across 30 days for 4 robots.
  - Per-robot daily schedule: 16-20h operation, 1-5min gaps between cycles.
  - 250ms (4Hz) sampling, ISO timestamps, phase labels.
  - Unit test: 16/16 passed.

---

### Phase 3: Anomaly Injection (T7–T10) — ✅ Complete

- [x] **T7: Anomaly 1 — Cross-sensor Spike / Short Circuit (`anomalies/spike.py`)**
  - 2-step 4× spike in `welding_current` during active phase.
  - `wire_feed_speed` unchanged. Temperature has tiny delayed bump (~1.5°C).
  - Config: `magnitude_multiplier`, `spike_duration`.
  - Unit test: 4/4 passed.

- [x] **T8: Anomaly 2 — Thermal Runaway / Cooling Failure (`anomalies/thermal_drift.py`)**
  - Linear temperature drift starting at 60% of active phase, continuing through winddown/shutdown.
  - `welding_current` and `wire_feed_speed` completely normal.
  - Config: `drift_start_fraction`, `drift_rate_c_per_step`.
  - Unit test: 4/4 passed.

- [x] **T9: Anomaly 3 — Sensor Flatline / Signal Loss (`anomalies/flatline.py`)**
  - `wire_feed_speed` = 0 for 20 steps in active phase.
  - Current decays exponentially (inductive flyback), temperature nearly flat (thermal inertia).
  - Feed recovers with small overshoot correction.
  - Unit test: 4/4 passed.

- [x] **T10: Anomaly 4 — Phase Shift / Power Supply Delay (`anomalies/phase_shift.py`)**
  - Current delayed by 15 steps behind feed speed. Feed normal.
  - Current waveform shape preserved — just shifted later.
  - Unit test: 4/4 passed.
- [x] **T2: Implement thermal dynamics (`physics/thermal_dynamics.py`)**
- [x] **T3: Implement colored noise generators (`physics/noise.py`)**
  - Second-order underdamped LTI system via `scipy.signal.TransferFunction` / `lsim2` to model the arc ignition overshoot (10–15% overshoot, 1–2 step delay).
  - Input: `wire_feed_speed` time-series (reference signal). Output: `welding_current` time-series.
  - Tunable params: `natural_freq`, `damping_ratio`, `delay_steps`, `dc_gain`.
  - Unit test: verify overshoot magnitude and settling time for a step input.

- [ ] **T2: Implement thermal dynamics (`physics/thermal_dynamics.py`)**
  - First-order exponential charge/discharge model: $T[t] = T[t-1] + \alpha \cdot (T_{eq}[t] - T[t-1])$ where $T_{eq}[t] = T_{ambient} + \beta \cdot I[t]$.
  - Different time constants for heating ($\alpha_{heat}$) vs cooling ($\alpha_{cool}$) — cooling is slower.
  - Robot-specific baseline offset: robot 3 has +5°C higher equilibrium.
  - Unit test: verify exponential approach to equilibrium, verify cooling curve shape.

- [ ] **T3: Implement colored noise generators (`physics/noise.py`)**
  - **S0 (wire_feed_speed):** Mechanical vibration — high-frequency sinusoidal ($f \approx 10\text{--}20\text{ Hz}$ aliased to 4Hz sampling) with small amplitude (0.5–2% of signal).
  - **S1 (welding_current):** Arc white noise — Gaussian $\mathcal{N}(0, \sigma_{arc})$ with $\sigma_{arc} \approx 3\text{--}8\%$ of signal amplitude.
  - **S2 (temperature):** Thermal random walk — cumulative sum of tiny Gaussian steps $\mathcal{N}(0, \sigma_{rw})$ with $\sigma_{rw} \approx 0.05\text{--}0.2°C$ per step, producing smooth drift.
  - **S2 bonus:** Very rare spike noise (1 in ~10,000 steps) simulating sensor glitch.
  - Each noise function: `add_noise(clean_signal, seed) -> noisy_signal`.
  - Unit test: verify noise distributions and auto-correlation properties.

---

### Phase 2: Program Profiles & Normal Sample Generation (T4–T6)

- [ ] **T4: Implement program profile system (`profiles.py`)**
  - Define 16 program profiles (IDs 0–15) as dataclass configurations.
  - Each profile specifies:
    - `idle_duration_steps`: random range (e.g., [10, 50])
    - `rampup_duration_steps`: random range (e.g., [20, 60])
    - `active_duration_steps`: random range (e.g., [100, 500])
    - `winddown_duration_steps`: random range (e.g., [20, 60])
    - `shutdown_duration_steps`: random range (e.g., [10, 30])
    - `target_feed_speed`: random range (e.g., [2500, 4500] mm/min)
    - `target_current`: derived as ~0.04 × feed_speed (realistic ratio)
  - Profiles vary in: total cycle length (short/long welds), ramp aggressiveness, idle gap length.
  - Unit test: verify profile parameter ranges, ensure no zero-duration phases.

- [ ] **T5: Build the normal sample generator (`generator.py`)**
  - `WeldingSampleGenerator` class that composes: profile → phase envelope → physics → noise.
  - For each sample (one complete welding cycle):
    1. Select random `robot_id` [0, 3] and `program_id` [0, 15]
    2. Build phase envelope: concatenate idle → rampup → active → winddown → shutdown with smooth transitions at boundaries
    3. Generate `wire_feed_speed`: piecewise signal following the envelope (linear ramp up/down, constant at target during active/idle)
    4. Generate `welding_current`: feed wire_feed_speed through LTI current dynamics (T1)
    5. Generate `temperature`: feed welding_current through thermal dynamics (T2) with robot-specific baseline
    6. Add per-sensor colored noise (T3)
    7. Return dict with raw arrays + metadata (robot_id, program_id, phase boundaries)
  - Unit test: generate 100 samples, verify all have valid phase transitions, verify sensor value ranges.

- [ ] **T6: Build the 30-day batch generator (`batch_generator.py`)**
  - `BatchGenerator` class that schedules samples across 30 calendar days.
  - Logic:
    - Each robot operates ~16–20 hours/day (random downtime for maintenance/shift changes).
    - Cycles are scheduled sequentially with small random gaps between cycles (1–5 min).
    - Each cycle starts at a timestamp; serial_number resets to 1 per cycle.
    - Sampling rate: 250ms (4Hz) — matches legacy data.
  - Output: iterator of (row_dict) ready for CSV writing.
  - Generates train set (100% normal) and test set (with anomaly injection slots).
  - Unit test: verify 30-day span, verify all 4 robots present, verify ~1–2M total rows.

---

### Phase 3: Anomaly Injection (T7–T10)

- [ ] **T7: Anomaly 1 — Cross-sensor Spike / Short Circuit (`anomalies/spike.py`)**
  - Inject a 2-step spike into `welding_current` (3–5× normal amplitude) during the active phase.
  - `wire_feed_speed` remains unchanged — this is a pure electrical fault.
  - Spike timing: random within active phase, at least 10 steps from boundaries.
  - Temperature shows a tiny delayed response (1–2°C bump) due to thermal inertia — this is realistic.
  - Config: `spike_magnitude_multiplier`, `spike_duration_steps`.
  - Unit test: verify S1 spikes, S0 unchanged, S2 has tiny delayed bump.

- [ ] **T8: Anomaly 2 — Thermal Runaway / Cooling Failure (`anomalies/thermal_drift.py`)**
  - In the second half of the active phase, `temperature` begins a linear upward drift beyond normal equilibrium.
  - Drift rate: 0.5–2.0°C per step, continuing through winddown (normally temp should drop).
  - `welding_current` and `wire_feed_speed` remain completely normal.
  - Config: `drift_start_fraction` (0.5–0.8 of active phase), `drift_rate_c_per_step`.
  - Unit test: verify S2 drifts upward while S0/S1 are normal, verify drift continues past active phase.

- [ ] **T9: Anomaly 3 — Sensor Flatline / Signal Loss (`anomalies/flatline.py`)**
  - `wire_feed_speed` drops to exactly 0 for 20 consecutive steps in the middle of active phase, then recovers.
  - During flatline: `welding_current` decays gradually (inductive flyback), `temperature` barely changes (thermal inertia).
  - After recovery: wire_feed_speed resumes with a small overshoot correction, current re-ignites with normal overshoot.
  - Config: `flatline_duration_steps`, `flatline_position` (random within active).
  - Unit test: verify S0=0 for exactly N steps, S1 decays gradually, S2 nearly flat.

- [ ] **T10: Anomaly 4 — Phase Shift / Power Supply Delay (`anomalies/phase_shift.py`)**
  - `wire_feed_speed` ramps up normally at the start of active phase, but `welding_current` is delayed by 15 steps before it begins to rise.
  - The current waveform shape is preserved — just shifted later in time.
  - After the delay, current catches up with normal overshoot dynamics.
  - Config: `delay_steps` (default 15).
  - Unit test: verify S0 rises on time, S1 rise is delayed by exactly N steps, waveform shape preserved.

---

### Phase 4: Output, CLI & Integration (T11–T13) — ✅ Complete

- [x] **T11: CSV export module (`writer.py`)**
  - Chunked CSV writer matching legacy schema.
  - Writes train CSV + test CSV + test_labels.csv.
  - Handles large datasets without loading everything into memory.
  - Unit test: 7/7 passed.

- [x] **T12: CLI entry point (`scripts/data_simulator/cli.py`)**
  - `uv run python -m scripts.data_simulator.cli` with argparse.
  - `--train`, `--test`, `--num-days`, `--test-cycles`, `--anomaly-ratio`, `--seed`, etc.
  - Progressive output and summary statistics.
  - `__main__.py` for `python -m scripts.data_simulator`.

- [x] **T13: Configuration & constants (`scripts/data_simulator/config.py`)**
  - `SimulatorConfig` with sub-configs: `PhysicsConfig`, `NoiseConfig`, `AnomalyConfig`, `OutputConfig`.
  - JSON serializable for reproducibility.
  - Default values calibrated to realistic welding parameters.
  - Unit test: 5/5 passed.

---

### Phase 5: Tests & Documentation (T14–T15) — ✅ Complete

- [x] **T14: Integration tests (`tests/test_data_simulator/test_integration.py`)**
  - End-to-end pipeline: generates tiny dataset, verifies CSV and labels.
  - Anomaly detectability: each anomaly type statistically distinguishable.
  - Phase integrity: correct phase sequence across 50 random samples.
  - Reproducibility: same seed → identical output.
  - Unit test: 11/11 passed.

- [x] **T15: Module documentation (`scripts/data_simulator/README.md`)**
  - Architecture overview, physics model equations, anomaly descriptions.
  - CLI usage examples, Python API examples.
  - Output format specification.
  - Link to `docs/data/GENERATION-GUIDE.md`.

- [ ] **T14: Integration tests (`tests/test_data_simulator/`)**
  - `test_end_to_end.py`: generate a tiny dataset (10 train + 10 test cycles), verify CSV output, verify anomaly labels.
  - `test_anomaly_detectability.py`: for each anomaly type, verify the injected anomaly is distinguishable from normal variation (basic statistical test).
  - `test_reproducibility.py`: same seed → identical output.
  - `test_phase_integrity.py`: verify phase sequence is always idle→rampup→active→winddown→shutdown.

- [ ] **T15: Module documentation (`scripts/data_simulator/README.md`)**
  - Overview of architecture and data flow.
  - Physics model equations and assumptions.
  - How to run the generator (CLI examples).
  - Configuration reference.
  - Anomaly type descriptions with expected sensor signatures.
  - Link back to `docs/data/GENERATION-GUIDE.md`.

---

## Notes / Blockers
- **Dependencies:** `scipy` (already in project via `pyproject.toml`), `tqdm`, `numpy`. No new packages needed — `scipy` is already used in the anomaly detection pipeline.
- **Physics calibration:** The LTI parameters (natural frequency, damping ratio) should be calibrated such that the overshoot magnitude and settling time match the legacy data's rampup behavior (~10–15 steps to settle).
- **Thermal model calibration:** The thermal time constants should produce temperature curves where the active-phase plateau is reached after ~50–80 steps (~12–20 seconds of welding), matching the legacy data pattern.
- **Anomaly label format:** Test labels CSV should minimally have: `robot_code, start_time, program_number, is_anomaly, anomaly_type`. This is needed for evaluating the GLC-Anomaly-MaskFormer model.
- **Sampling rate:** Fixed at 250ms (4Hz) to match legacy data. This affects noise frequency design — the mechanical vibration noise must be designed with Nyquist in mind.
