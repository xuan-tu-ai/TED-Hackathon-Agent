# Task Summary: Phase 5 + Phase 6 — Grad-Norm Logging & λ Ramp

**Sprint:** Sprint 15 — MaskFormer Training Fixes
**Tasks:** T13, T14, T15, T16, T17

## Summary of Work
Implemented Phase 5 (gradient-norm ratio logging) and Phase 6 (λ ramp schedule). These two phases work together to let the data determine the optimal λ value rather than guessing.

### Phase 5 — Gradient-Norm Logging (T13, T14)

**T13 — Add grad-norm computation to `train_step()`**
- Added `log_grad_norms: bool = False` parameter to `MaskFormerTrainer.__init__()`
- Pre-identified Q/K/V parameters via `self._qkv_params` (filter by "W_q", "W_k", "W_v" in param names)
- Added `_compute_qkv_grad_norm()` helper that computes L2 norm of gradients on Q/K/V params
- During training: runs two partial backward passes (recon-only → record norm → zero_grad, then adversarial-only → record norm → zero_grad) before the full `loss_a.backward()`
- Ratio `||∇_QKV assoc|| / ||∇_QKV recon||` stored as `grad_ratio` in step output

**T14 — Log grad_ratio in history**
- Added `"grad_ratio"` key to the history dict in `MaskFormerTrainerRunner.train()`
- Grad ratio is averaged per-epoch and stored in history

### Phase 6 — λ Ramp (T15, T16, T17)

**T15 — Add ramp params to Runner**
- `lambda_end: float` — target λ after ramp completes (default: same as `lambda_assoc`, no ramp)
- `lambda_ramp_epochs: int` — number of epochs over which to linearly ramp (default: 0, no ramp)

**T16 — Implement ramp logic in `train()`**
- Before each epoch: `current_lambda = linear_interpolate(lambda_assoc, lambda_end, ramp_progress)`
- `self.trainer.lambda_assoc = current_lambda` — trainer uses this in Phase A and Phase B
- Ramp starts after `warmup_epochs` complete (so warmup + ramp are pipelined naturally)
- Current λ printed in epoch summary

**T17 — Tests**
- `test_ramp_linear_interpolation`: λ reaches `lambda_end` after `ramp_epochs`
- `test_ramp_clamps_at_end`: λ stays at `lambda_end` (doesn't exceed)
- `test_ramp_starts_after_warmup`: ramp begins after warmup epochs, not during
- `test_grad_ratio_in_output`: `grad_ratio` is present and non-negative when logging enabled
- `test_grad_ratio_during_warmup`: `grad_ratio` is 0 during warmup (no adversarial term)

## Files Modified
- [backend/app/anomaly_detection/training/maskformer_trainer.py](backend/app/anomaly_detection/training/maskformer_trainer.py) — Grad-norm logging + λ ramp logic + epoch printout with λ and grad_ratio
- [backend/tests/test_maskformer_trainer.py](backend/tests/test_maskformer_trainer.py) — 5 new tests (2 grad-norm + 3 λ ramp)

## Testing
- **Test Files:** `tests/test_maskformer_trainer.py`, `tests/test_anomaly_attention.py`
- **Status:** 22 passed (test_maskformer_trainer.py) + 13 passed (test_anomaly_attention.py) = 35 total
- **Execution Command:** `cd backend && uv run pytest tests/test_maskformer_trainer.py tests/test_anomaly_attention.py -v`

## Additional Notes
- Grad-norm logging uses 2 extra backward passes + 2 zero_grad calls per step when enabled. This is intentional — it's a diagnostic tool, not meant to be always-on. Set `log_grad_norms=False` (default) for production training.
- The λ ramp is linear: `λ = λ_start + progress × (λ_end − λ_start)` where `progress = clamp((epoch − warmup_epochs) / ramp_epochs, 0, 1)`.
- Default configuration: `warmup_epochs=5, lambda_end=2.0, lambda_ramp_epochs=10` will warm up for 5 epochs, then ramp λ from 0.02 → 2.0 over the next 10 epochs.
