# Task Summary: Phase 2 — JS Divergence

**Sprint:** Sprint 15 — MaskFormer Training Fixes
**Tasks:** T4, T5, T6

## Summary of Work
Replaced symmetric KL divergence with Jensen-Shannon divergence in the Anomaly-Attention block. JS is bounded in [0, log(2)] ≈ [0, 0.69] for any pair of normalized distributions, eliminating the eps-floor pathology (`log(1e-8)` ≈ −18 per element) that inflated discrepancy to O(28) and made the λ hyperparameter impossible to reason about.

### T4 — Replace `_symmetric_kl` with `_jensen_shannon`
- **File:** `backend/app/anomaly_detection/layers/anomaly_attention.py`
- Renamed method from `_symmetric_kl` to `_jensen_shannon`
- Replaced `KL(p||q) + KL(q||p)` with `0.5 * (KL(p||m) + KL(q||m))` where `m = 0.5*(p+q)`
- Updated call site in `forward_with_detach_mode` (line 167: `self._symmetric_kl(…)` → `self._jensen_shannon(…)`)
- Updated module-level docstring (line 10) and class docstring (line 32) from "symmetric KL" to "Jensen-Shannon"

### T5 — Update `AssociationDiscrepancyLoss` docstrings
- **File:** `backend/app/anomaly_detection/criterions/association_discrepancy.py`
- Updated module docstring and class docstring to reference JS divergence
- Added note explaining why JS fixes the eps-floor pathology
- No functional code changes — `forward()` still calls `.mean()` on the incoming discrepancy tensor

### T6 — Update tests
- **File:** `backend/tests/test_anomaly_attention.py`
- Updated `test_discrepancy_non_negative` assertion comment from "Symmetric KL must be non-negative" to "JS divergence must be non-negative"
- All existing tests pass unchanged (JS is still non-negative, no NaN/Inf, gradient routing tests unaffected)

## Files Modified
- [backend/app/anomaly_detection/layers/anomaly_attention.py](backend/app/anomaly_detection/layers/anomaly_attention.py) — `_symmetric_kl` → `_jensen_shannon` with JS formula; docstrings updated
- [backend/app/anomaly_detection/criterions/association_discrepancy.py](backend/app/anomaly_detection/criterions/association_discrepancy.py) — Docstrings reference JS divergence
- [backend/tests/test_anomaly_attention.py](backend/tests/test_anomaly_attention.py) — Test comment updated

## Testing
- **Test Files:** `tests/test_anomaly_attention.py`, `tests/test_maskformer_trainer.py`
- **Status:** All 25 tests passed
- **Execution Command:** `cd backend && uv run pytest tests/test_anomaly_attention.py tests/test_maskformer_trainer.py -v`

## Additional Notes
- JS divergence is strictly bounded: `0 ≤ JS(p, q) ≤ log(2)` for any valid probability distributions p, q. This means `assoc_loss` values will drop from O(28) to O(0.1–0.5) after this change, making λ=3 (per the official paper) directly meaningful.
- No checkpoint compatibility issue — all existing checkpoints used symmetric KL values which are now mathematically different. A full retrain is required (see Gate A in sprint plan).
