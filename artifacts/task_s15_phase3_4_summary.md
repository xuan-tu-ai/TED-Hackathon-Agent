# Task Summary: Phase 3 + Phase 4 — CNN Decoder & Phase-A Warmup

**Sprint:** Sprint 15 — MaskFormer Training Fixes
**Tasks:** T7, T8, T9, T10, T11, T12

## Summary of Work
Implemented two intertwined phases: the V6-style CNN decoder (Phase 3) and Phase-A-only warmup (Phase 4). These changes together ensure the encoder+decoder can learn reconstruction before the adversarial minimax game competes for shared parameters.

### Phase 3 — CNN Decoder (T7, T8, T9)

**T7 — Implement `CNNDecoder` in `reconstruction_decoder.py`**
- Refactored the file into two classes: `PerTokenDecoder` (original `Linear→GELU→Linear`) and `CNNDecoder` (adds V6-style depthwise-separable Conv1d temporal refinement).
- `CNNDecoder` pipeline: per-token decode → reshape `[B, N, PS]` → `[B, C, P*PS]` → `Conv1d(C→4C, k=3, groups=C) → SiLU → Conv1d(4C→C, k=1)` → reshape back.
- `ReconstructionDecoder = PerTokenDecoder` kept as backward-compatible alias.

**T8 — Wire CNN decoder into model**
- Added `use_cnn_decoder: bool = False` parameter to `GLCAnomalyMaskFormer.__init__()`
- When True, creates `CNNDecoder(num_sensors, num_patches, d_model, patch_size)`
- Forward pass (`_forward_impl`) unchanged — decoder has same `[B, N, D] → [B, N, PS]` signature

**T9 — Tests for CNN decoder**
- `test_cnn_decoder_output_shape`: shape `[B, N, PS]` matches PerTokenDecoder
- `test_cnn_decoder_no_nan`: no NaN/Inf
- `test_cnn_decoder_differs_from_per_token`: CNN temporal refinement changes output

### Phase 4 — Phase-A Warmup (T10, T11, T12)

**T10 — Add warmup params to Runner**
- Added `warmup_epochs: int = 0` to both `MaskFormerTrainer` and `MaskFormerTrainerRunner`

**T11 — Implement warmup mode in `train_step()`**
- `train_step(x, epoch)` now accepts an `epoch` parameter (0-based)
- If `epoch < warmup_epochs`: Phase B is skipped entirely, `lambda_assoc=0` during Phase A
- Returns `assoc_loss_max=0.0` as placeholder during warmup
- After warmup: full minimax resumes with normal `lambda_assoc`

**T12 — Warmup tests**
- `test_warmup_skips_phase_b`: sigma unchanged during warmup (Phase B skipped)
- `test_warmup_then_phase_b_resumes`: sigma changes after warmup ends (Phase B resumes)

## Files Modified
- [backend/app/anomaly_detection/layers/reconstruction_decoder.py](backend/app/anomaly_detection/layers/reconstruction_decoder.py) — `PerTokenDecoder` + `CNNDecoder` classes
- [backend/app/anomaly_detection/models/glc_anomaly_maskformer.py](backend/app/anomaly_detection/models/glc_anomaly_maskformer.py) — `use_cnn_decoder` parameter
- [backend/app/anomaly_detection/training/maskformer_trainer.py](backend/app/anomaly_detection/training/maskformer_trainer.py) — `warmup_epochs`, `epoch` param in `train_step`, Phase B gating
- [backend/tests/test_maskformer_trainer.py](backend/tests/test_maskformer_trainer.py) — 6 new tests (3 warmup + 3 CNN decoder)

## Testing
- **Test Files:** `tests/test_maskformer_trainer.py`, `tests/test_anomaly_attention.py`
- **Status:** All 30 tests passed
- **Execution Command:** `cd backend && uv run pytest tests/test_maskformer_trainer.py tests/test_anomaly_attention.py -v`

## Additional Notes
- The warmup is designed to be configured via `MaskFormerTrainerRunner(model, warmup_epochs=5)` — no notebook code changes needed.
- `train_step` now expects an `epoch` parameter. The runner's `train()` method passes it automatically.
- The CNN decoder is backward-compatible: existing code using `ReconstructionDecoder` imports still works (it's an alias for `PerTokenDecoder`).
- Gate B (after Phase 4) will be the first full test with JS + CNN + warmup — run `warmup_epochs=5, epochs=30, lambda=0.02`.
