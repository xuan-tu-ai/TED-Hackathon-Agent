# Sprint 18 Plan

## Sprint Goal
Regenerate welding robot data with convergent normal signals, simplified anomaly types (10 → 5 based on real-world reliability), and verified real-world timing parameters. Address the core insight: "Anomaly is not an absolute concept" — some anomalies are only detectable via historical baseline comparison or cross-sensor context.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Research & Timing Verification

- [x] **T1: Research real MIG/MAG welding cycle parameters** — ✅ 4Hz / 240-step defaults verified realistic. Wire feed speed ranges confirmed. Documented in `docs/data/welding-program-research.md`.
- [x] **T2: Update config if research warrants changes** — ✅ No changes needed. Current defaults confirmed realistic.

### Phase 2: Normal Data Convergence

- [x] **T3: Deterministic sample generation per (robot_id, program_id)** — added `deterministic` param to `generate_sample()`, `_make_deterministic_seed()` helper
- [x] **T4: Tighten noise parameters for training data** — added `NoiseConfig.for_training()` factory with ~50% reduced variance
- [x] **T5: Update BatchGenerator for deterministic training** — `training_mode` flag + deterministic `generate_train()`

### Phase 3: Anomaly Simplification (10 → 5)

- [x] **T6: Deprecate 5 anomaly types from `__init__.py`** — kept `spike`, `thermal_drift`, `flatline`, `cable_wear`, `arc_blow`
- [x] **T7: Update `CSVWriter.write_test()` to 5-type pool** — removed `overshoot_degradation` special case, updated `anomaly_types` list
- [x] **T8: Archive removed anomaly files** — moved deprecated modules to `_archived/`, updated test imports

### Phase 4: Regeneration & Validation

- [x] **T9: Regenerate training dataset** — convergent normal data (30 days, 4 robots, reduced noise + deterministic)
- [x] **T10: Regenerate test dataset** — 2000 cycles, 50/50 split, 5 anomaly types → 715,450 rows
- [x] **T11: Quality validation checks** — 140/140 simulator tests passing. Generated datasets validated.

### Phase 5: Documentation

- [x] **T12: Update GENERATION-GUIDE.md** — reflects 5 anomaly types + convergence section + archived anomalies list
- [x] **T13: Update simulation discussion doc + research findings** — `docs/data/welding-program-research.md` created, `GENERATION-GUIDE.md` updated
- [ ] **T10: Regenerate test dataset** — 2000 cycles, 50/50 split, exactly 5 anomaly types
- [ ] **T11: Quality validation** — convergence <2% deviation, no NaN/inf, all simulator tests pass

### Phase 5: Documentation

- [ ] **T12: Update `GENERATION-GUIDE.md`** — reflect 5 anomaly types, add convergence notes, update Section 4
- [ ] **T13: Update `simulation-discussion.md`** — document detectability classification rationale

## Decisions

- **5 kept anomalies:** `spike` (cross-sensor, recon-error detectable), `thermal_drift` (historical-baseline needed), `flatline` (local-RevIN test), `cable_wear` (prolonged degradation, recon-error), `arc_blow` (cross-sensor context needed). Covers all 3 detectability classes.
- **5 removed:** `phase_shift`, `wire_stutter`, `gas_blockage`, `sticky_tc`, `overshoot_degradation` — archived to `_archived/`.
- **Convergence:** Deterministic base signal per (robot, program) + ~50% reduced training noise. Test noise unchanged for generalization evaluation.
- **Research:** Lightweight — confirm 4Hz and 30s–3min ranges match industrial MIG/MAG welding.
