# Task Summary: Tasks 16-18 — Anomaly Scorer + Localization

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Tasks 16-18 — Implement MADThreshold, AnomalyScorer, PatchMasker + PerturbationLoop

## Summary of Work
Implemented `models/scorer.py` with three components:

**MADThreshold** (Task 16): Robust anomaly threshold using Median Absolute Deviation. Formula: `threshold = median + 3 * 1.4826 * MAD`. Robust to up to 50% contamination. Methods: `fit(scores)`, `predict(scores)`, `get_p_value(score)`.

**AnomalyScorer** (Task 17): Wraps CNF to compute anomaly scores via negative log-likelihood. Higher score = more anomalous. `score(z_pooled)` returns [B] scores. `forward(z_pooled)` returns scores + z_noise for logging.

**PatchMasker + PerturbationLoop** (Task 18): Localization via Log-Likelihood Perturbation. PatchMasker produces 84 masked versions of a file (6 sensors × 14 patches). PerturbationLoop tests which (sensor, patch) region causes biggest log-likelihood improvement when masked — that region is the root cause. 84 forward passes per flagged file.

## Files Modified
- [backend/app/anomaly_detection/models/scorer.py](backend/app/anomaly_detection/models/scorer.py) — New file with MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop
- [backend/tests/test_scorer.py](backend/tests/test_scorer.py) — 11 unit tests

## Testing
- **Test File:** [backend/tests/test_scorer.py](backend/tests/test_scorer.py)
- **Status:** Passed (11 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_scorer.py -v`
- **Coverage:**
  - MADThreshold: fit sets threshold, predict above threshold, robust to outliers, requires fit
  - AnomalyScorer: score shape [B], score finite values
  - PatchMasker: single patch masked correctly, 84 all_masks, all masks different
  - PerturbationLoop: localize returns [6,14] heatmap, localize_batch returns list of heatmaps

## Additional Notes
- Bug fix: `CNF.forward()` now flattens any leading dimensions of z to handle calls from PerturbationLoop that pass batch-dim tensors in unexpected ways
- Bug fix: PerturbationLoop.localize() now passes `z_pooled` (encoder output) to scorer instead of raw_file directly
- Sprint plan: Tasks 16, 17, 18 marked [x] done