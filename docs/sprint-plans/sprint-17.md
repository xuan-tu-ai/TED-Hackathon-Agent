# Sprint 17 Plan

## Sprint Goal
Refactor MaskFormer-based anomaly detection into GLCReconV8 full reconstruction with dual-stream CNN encoders and a standard Transformer.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Remove MaskFormer Artifacts

- [x] **T1: Delete masking layer file** `backend/app/anomaly_detection/layers/masking.py`
- [x] **T2: Delete association discrepancy loss** `backend/app/anomaly_detection/criterions/association_discrepancy.py`
- [x] **T3: Delete masked reconstruction loss** `backend/app/anomaly_detection/criterions/masked_reconstruction.py`
- [x] **T4: Delete MaskFormer model** `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py`
- [x] **T5: Remove AnomalyAttention classes** from `backend/app/anomaly_detection/layers/attention.py`
- [x] **T6: Update layer exports** in `backend/app/anomaly_detection/layers/__init__.py` (remove masking + anomaly attention, add CNN encoders)
- [x] **T7: Update criterion exports** in `backend/app/anomaly_detection/criterions/__init__.py`
- [x] **T8: Update model exports** in `backend/app/anomaly_detection/models/__init__.py`

### Phase 2: Add Dual-Stream CNN Encoders

- [x] **T9: Add PatchEncoderCNNGlobal + PatchEncoderCNNLocal** to `backend/app/anomaly_detection/layers/encoders.py`

### Phase 3: Add GLCReconV8 Model and Registration

- [x] **T10: Add GLCReconV8 model** in `backend/app/anomaly_detection/models/glc_recon_v8.py`
- [x] **T11: Register GLCReconV8** in `backend/app/anomaly_detection/models/factory.py` (registry + enums)

### Phase 4: Training + Smoke Test

- [x] **T12: Add trainer** `backend/app/anomaly_detection/training/glc_recon_v8_trainer.py`
- [x] **T13: Smoke test forward pass** B=2, C=6, T=240 -> loss scalar

### Phase 5: ContextEmbedding + Improved Trainer

- [x] **T14: Add ContextEmbedding to GLCReconV8** in `backend/app/anomaly_detection/models/glc_recon_v8.py`
  - Added `num_robots`, `num_programs`, `use_context_embeddings` to `GLCReconV8Config`
  - `ContextEmbedding(merge_mode="sum")` summed into all patch tokens after positional embedding
  - `_tokenize()`, `_forward_impl()`, `compute_loss()`, `compute_anomaly_score()`, `compute_heatmap()`, `detect_anomaly()` all accept `robot_idx`/`program_idx`
  - Graceful degradation: context skipped when indices not provided

- [x] **T15: Improve GLCReconV8Trainer** in `backend/app/anomaly_detection/training/glc_recon_v8_trainer.py`
  - Linear warmup + CosineAnnealing LR schedule (`warmup_epochs`, `min_lr_ratio`)
  - Best checkpoint tracking by lowest `val_loss` (`best_state_dict`, `restore_best()`)
  - Full `train(train_loader, val_loader, epochs, lr, save_best)` method
  - `GLCReconV8History` dataclass with `loss`, `val_loss`, `lr` lists
  - `GLCReconV8Evaluator` class: AUROC/AUPRC with synthetic anomaly injection (spike, oscillation, flatline)
  - `EvalResult` dataclass with `auroc`, `auprc`, `threshold`, `normal_scores`, `anomaly_scores`

- [x] **T16: Update inference pipeline for context** in `backend/app/anomaly_detection/inference/recon_v8/local.py`
  - `Reconf8LocalPreResult` accepts `robot_codebook` and `program_codebook` dicts
  - `_get_context_tensors()` resolves context indices per-record from codebooks
  - `score_batch()` passes `robot_idx`/`program_idx` to `compute_heatmap()` (expanded to batch size)

### Phase 6: Training + Inference Notebooks

- [x] **T17: Create Training-GLCReconV8.ipynb** `backend/notebooks/Training-GLCReconV8.ipynb`
  - Sections: data loading with codebooks, Dataset/DataLoader, smoke test, training with warmup+cosine, loss curves, AUROC evaluation, reconstruction sanity check, checkpoint save
  - Saves `training_extra.json` with history, codebooks, best epoch, AUROC, MAD threshold

- [x] **T18: Create Inference-GLCReconV8.ipynb** `backend/notebooks/Inference-GLCReconV8.ipynb`
  - Sections: load checkpoint + extras, load train/val/test CSV, build macro-window records, score both sets with codebooks, fit multi-score MAD threshold, apply OR-logic verdict, per-type metrics (precision/recall/F1/ROC-AUC), score distribution visualization, per-sensor heatmap comparison, summary report

## Notes / Blockers
- Ensure no remaining MaskFormer-specific exports or registry entries remain after removal.
- Validation should use the same forward path as training (no masking path).
- Context codebooks (`robot_codebook`, `program_codebook`) must be saved to `training_extra.json` at training time and loaded at inference time for consistent index mapping.
