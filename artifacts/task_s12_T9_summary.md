# Task Summary: T9 — Fix Minimax Phase B sign

**Sprint:** Sprint 12
**Task:** T9

## Summary of Work
Fixed the critical Phase B sign bug in `MaskFormerTrainer.train_step`. Changed `loss_b = -self.lambda_assoc * assoc_dict_max["loss"]` to `loss_b = self.lambda_assoc * assoc_dict_max["loss"]`. The original code had the wrong sign — Phase B should maximize (positive) association discrepancy so sigma learns to make the Gaussian prior contrast with the learned series association. The negative sign caused both phases to minimize discrepancy, leading to mode collapse and sigma divergence.

## Files Modified
- `backend/app/anomaly_detection/training/maskformer_trainer.py` — Line 144: Changed `-self.lambda_assoc` to `self.lambda_assoc`.

## Testing
- **Test File:** `backend/tests/test_maskformer_trainer.py`
- **Status:** Passed (9/9 trainer tests, including new `test_phase_b_loss_is_positive`)
- **Execution Command:** `uv run pytest tests/test_maskformer_trainer.py -v`

## Additional Notes
This fix aligns the implementation with the official `thuml/Anomaly-Transformer` solver.py and the model's own docstrings. After the fix, `assoc_loss_max` is always positive, correctly driving sigma to increase discrepancy during the maximize phase.
