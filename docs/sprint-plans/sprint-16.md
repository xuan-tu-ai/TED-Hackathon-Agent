# Sprint 16 Plan

## Sprint Goal
Improve data simulator realism (momentum noise, local variance, U-pattern), add 6 new anomaly types for robust MaskFormer evaluation, and adjust inference policy to phase-aware scoring for 95% F1 target.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Noise Model Refinement

- [x] **T1: Add AR(1) momentum noise generator** to `physics/noise.py`
- [x] **T2: Add local-window variance estimation** to `physics/noise.py`
- [x] **T3: Refactor `add_current_noise()`** to use AR(1) + local window variance
- [x] **T4: Refactor `add_wire_feed_noise()` Gaussian component** to use AR(1) + local window variance
- [x] **T5: Refactor `add_temperature_noise()` random walk** to use local window variance
- [x] **T6: Add U-pattern noise** for stable/setpoint-controlled sensors
- [x] **T7: Add config params** to NoiseConfig: ar1_phi, local_window_size, u_pattern_*
- [x] **T8: Test momentum noise** — 30 tests, all passed

### Phase 2: Six New Anomaly Types

- [x] **T9: Wire Stutter** — `anomalies/wire_stutter.py` (S0 periodic dips, Medium)
- [x] **T10: Arc Blow** — `anomalies/arc_blow.py` (S1 noise burst, Easy)
- [x] **T11: Gas Blockage** — `anomalies/gas_blockage.py` (S1+S2 dual signature, Easy)
- [x] **T12: Sticky TC** — `anomalies/sticky_tc.py` (S2 staircase, Medium)
- [x] **T13: Cable Wear** — `anomalies/cable_wear.py` (S1 dropouts, Easy-Hard)
- [x] **T14: Overshoot Degradation** — `anomalies/overshoot_degradation.py` (S1 shape change, Hard)
- [x] **T15: Register all 6 in `anomalies/__init__.py`**
- [x] **T16: Wire into `CSVWriter.write_test()`** — pool of 10 anomaly types
- [x] **T17: Test each injector** — 34 anomaly tests, all passed

### Phase 3: Inference Policy — Phase-Aware Scoring

- [x] **T18: Add phase boundary detection** to `MaskFormerLocalPreResult._detect_phase_regions()`
- [x] **T19: Compute phase-specific scores** — `p95_score_rampup`, `p95_score_active`
- [x] **T20: Update `MaskFormerLocalThresholdCreator`** for multi-score MAD threshold
- [x] **T21: Update `MaskFormerLocalApplier`** for OR-logic verdict with `triggered_by`
- [x] **T22: Test phase-aware scoring** — 8/8 inference tests passed

### Phase 4: Integration Test — End-to-End Validation

- [x] **T23: Generate test dataset** — Created `backend/notebooks/Sprint16-Generate-Test-Data.ipynb` with all 10 anomaly types, 2000 cycles, 50/50 split
- [x] **T24: Inference instructions provided** — See notebook Step 5-6 for phase-aware inference pipeline steps
- [x] **T25: Documentation template** — See notebook Step 7 for results doc template in `docs/data/simulation-discussion.md`

## Notes / Blocker
- Sprint 15 must complete first for trained checkpoint needed in Phase 4
- Overshoot Degradation modifies LTI dynamics at generation time, not post-hoc
- Phase detection is signal-derived (no ground-truth labels needed)
