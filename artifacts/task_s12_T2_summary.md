# Task Summary: T2 — Numerical Stability in _symmetric_kl

**Sprint:** Sprint 12
**Task:** T2

## Summary of Work
Changed the KL divergence computation in `AnomalyAttentionBlock._symmetric_kl` from `torch.log((p + eps) / (q + eps))` to `torch.log(p + eps) - torch.log(q + eps)`. This matches the official `my_kl_loss` from `thuml/Anomaly-Transformer` and avoids numerical overflow/underflow when `(p+eps)/(q+eps)` produces extreme ratios.

## Files Modified
- `backend/app/anomaly_detection/layers/anomaly_attention.py` — Lines 122-123: Split `log(a/b)` into `log(a) - log(b)`.

## Testing
- **Test File:** `backend/tests/test_anomaly_attention.py`
- **Status:** Passed (13/13 tests, including 2 new: `test_kl_equivalent_to_old_formulation` and `test_kl_stable_with_extreme_values`)
- **Execution Command:** `uv run pytest tests/test_anomaly_attention.py -v`

## Additional Notes
The new formulation produces bitwise-identical results for normal inputs while being more stable for near-zero probabilities. The `test_kl_equivalent_to_old_formulation` test verifies numerical equivalence.
