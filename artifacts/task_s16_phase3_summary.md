# Task Summary: Sprint 16 — Phase 3 (T18–T22)

**Sprint:** Sprint 16
**Task:** T18–T22: Phase-Aware Scoring for MaskFormer Inference

## Summary of Work
Added signal-derived phase boundary detection to the MaskFormer inference pipeline. The `_detect_phase_regions()` function identifies rampup, active, and winddown regions from the S0 feed-speed gradient — no ground-truth labels needed. The `score_batch()` method now computes three scores per file: `p95_score` (global), `p95_score_rampup`, and `p95_score_active`. The `MaskFormerLocalThresholdCreator` supports multi-score MAD calibration. The `MaskFormerLocalApplier` uses OR-logic: a file is anomalous if ANY score exceeds its threshold, with a `triggered_by` field identifying which score(s) triggered.

## Files Modified
- `backend/app/anomaly_detection/inference/maskformer/local.py` — Added `_detect_phase_regions()`, updated `score_batch()` (3 scores), `MaskFormerLocalThresholdCreator` (multi-key thresholds), `MaskFormerLocalApplier` (multi-score verdict + triggered_by)
- `backend/tests/test_local_inference.py` — Added 3 new tests: `test_phase_scores_detected`, `test_multi_score_threshold_creator`, `test_multi_score_applier_or_logic`; updated existing tests for new API keys

## Testing
- **Test File:** `backend/tests/test_local_inference.py::TestMaskFormerLocalPipeline`
- **Status:** 8/8 passed
- **Execution Command:** `uv run pytest tests/test_local_inference.py::TestMaskFormerLocalPipeline -v`

## Additional Notes
- Phase detection is robust: on random/noisy data, it gracefully returns `(0, 0)` for undetected phases
- Backward-compatible: `MaskFormerLocalThresholdCreator.threshold` property returns the first score's threshold
- The `MaskFormerLocalApplier` falls back to global threshold if no `score_thresholds` are provided
