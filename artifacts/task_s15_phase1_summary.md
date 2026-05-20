# Task Summary: Phase 1 — Observability

**Sprint:** Sprint 15 — MaskFormer Training Fixes
**Tasks:** T1, T2, T3

## Summary of Work
Implemented Phase 1 (Observability) for the MaskFormer training pipeline. All changes are notebook-only — no model or trainer code was modified. The training curves visualization was restructured to stop treating total loss as a health metric, and a new AUROC/AUPRC evaluation section was added using synthetic anomalies.

### T1 — Add `recon_loss` subplot as primary training curve
- Replaced the old 2×3 grid (which included total loss) with a 2×2 grid showing: recon_loss (with trend annotation + improvement %), assoc_min vs assoc_max overlaid (with gap fill), sigma_mean, and val_loss (or composite train_loss as fallback).
- Added a printed "Training Health Report" summary box with target indicators.

### T2 — Add per-epoch AUROC/AUPRC evaluation
- Added a new Section 7 with a markdown explanation (AUROC/AUPRC definitions) and a code cell that:
  - Creates 100 synthetic anomaly windows by injecting Gaussian noise spikes into normal validation windows.
  - Evaluates the trained model via `MaskFormerEvaluator`.
  - Prints AUROC, AUPRC, MAD threshold, and sample counts with qualitative assessment.

### T3 — Stop treating total loss as health signal
- Updated the training cell's final printout to list `recon_loss` first with "(MUST trend down)" annotation and a footer: "Total loss is a composite metric — use recon_loss + AUROC for health."
- Removed total loss from the primary visualization grid (demoted to a grayed-out fallback in the val_loss subplot).
- Added a new markdown cell (Section 5.5) explaining which metrics to track and why.

## Files Modified
- [backend/notebooks/Training-MaskFormer.ipynb](backend/notebooks/Training-MaskFormer.ipynb) — Restructured training visualization (cells 15-18), added AUROC evaluation (cells 19-20), updated training cell printout (cell 15)

## Testing
Phase 1 is notebook-only with no new Python code. The changes are visual/logging only and do not affect model training. Validation:
- `uv run pytest tests/test_maskformer_trainer.py -v` — all 12 tests still pass (notebook changes do not touch trainer code)

## Additional Notes
- Synthetic anomalies for AUROC are generated ad-hoc (Gaussian noise spikes). These are not physics-realistic; they only test whether the model can distinguish a visibly corrupted signal from a clean one. Physics-based anomalies from Sprint 14's data simulator should replace this once available.
- The evaluation runs on the final model only (not per-epoch). True per-epoch AUROC tracking would require saving checkpoints or adding evaluation to the training loop, which is planned for Phase 5 (grad-norm logging integration).
