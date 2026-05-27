# Task Summary: Task 21 — Training Loop (Stage1Trainer, Stage2Trainer, Evaluator)

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Task 21 — Training loop (Stage1Trainer, Stage2Trainer, Evaluator)

## Summary of Work
Implemented `train.py` with three trainer classes and one evaluator class, providing the complete two-stage training pipeline for GLC-Flow V7.

### Classes Implemented

**Stage1Trainer** — Encoder + Dual-Head training (CNF frozen)
- Computes `GlobalRevIN + PerPatchRevIN → Patchify → Dual Encoders → Transformer → Dual Pooling`
- Loss = MSE(global_head(z_cls), file_stats) + MSE(local_head(z_patches), patch_stats) + λ_orth × orthogonality_loss(z_pooled)
- Calls `_freeze_cnf()` to freeze CNF params before training

**Stage2Trainer** — CNF/Flow Matching training (encoder frozen)
- Uses encoder to get z_pooled, then samples z0~N(0,I), z_t=(1-t)z0+tz1
- Predicts vector field v(z_t, t), targets (z1 - z0)
- Calls `_freeze_encoder()` to freeze all non-CNF params

**Evaluator** — Model evaluation
- `compute_scores(x)`: runs model.forward → CNF.log_likelihood → returns anomaly scores
- `evaluate(normal_files, anomaly_files)`: computes AUROC and AUPRC using MAD threshold
- `_compute_auroc`: pairwise concordance formula (O(n²), correct for any score distribution)
- `_compute_auprc`: precision-recall curve area

**GLCFlowTrainer** — High-level orchestrator
- `train_stage1()`: Stage 1 encoder training with history tracking
- `train_stage2()`: Stage 2 CNF training with history tracking
- `evaluate()`: End-to-end evaluation on normal/anomaly sets

## Files Modified
- [backend/app/anomaly_detection/train.py](backend/app/anomaly_detection/train.py) — New file with Stage1Trainer, Stage2Trainer, Evaluator, GLCFlowTrainer, EvalResult
- [backend/tests/test_train.py](backend/tests/test_train.py) — 17 unit tests

## Bugs Fixed During Implementation

### Bug 1: Stage2 gradient flow issue
**Problem:** `z1 = out["z_pooled"].detach()` was the correct fix, but the initial implementation used a `with torch.no_grad()` block which could cause issues with gradient tracking in some configurations.

**Fix:** Changed to `z1 = out["z_pooled"].detach()` without the `no_grad` context (which is still correct since z1 is explicitly detached).

### Bug 2: Stage2Trainer._freeze_encoder didn't re-enable CNF gradients
**Problem:** After Stage1Trainer freezes the entire model (including CNF), Stage2Trainer's `_freeze_encoder` was correctly unfreezing the encoder but CNF params could end up with `requires_grad=False` if they were already frozen.

**Fix:** Added explicit `param.requires_grad = True` for all CNF parameters in `Stage2Trainer._freeze_encoder`:
```python
def _freeze_encoder(self) -> None:
    for name, param in self.model.named_parameters():
        if "cnf" not in name:
            param.requires_grad = False
    for param in self.model.cnf.parameters():
        param.requires_grad = True
```

## Testing
- **Test File:** [backend/tests/test_train.py](backend/tests/test_train.py)
- **Status:** All 17 tests PASS
- **Execution:** `cd backend && uv run pytest tests/test_train.py -v`
- **Coverage:**
  - Stage1Trainer: weights update, CNF frozen, loss positive, orth loss scalar
  - Stage2Trainer: CNF updates, encoder frozen, loss positive
  - Evaluator: score shape, finite scores, EvalResult return, threshold fitting
  - GLCFlowTrainer: initialization, config, stage1/stage2 history, evaluate, full pipeline

## Additional Notes
- `EvalResult` is a `@dataclass` for structured evaluation results (AUROC, AUPRC, threshold, counts)
- AUROC computed via pairwise concordance (fraction of normal-anomaly pairs correctly ordered)
- All 130 tests in backend/tests/ pass after this task
- Sprint plan Task 21 marked [x] done