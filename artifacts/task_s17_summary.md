# Sprint 17 Summary: MaskFormer → GLCReconV8 Refactor

**Sprint:** Sprint 17
**Goal:** Refactor MaskFormer-based anomaly detection into GLCReconV8 full reconstruction with dual-stream CNN encoders and a standard Transformer.

## Summary of Work

Successfully migrated the anomaly detection codebase from the MaskFormer paradigm (masked reconstruction + anomaly attention + association discrepancy) to a simpler, more effective full-reconstruction paradigm (GLCReconV8). All MaskFormer-specific files, classes, exports, and tests were removed. The new architecture uses dual-stream CNN patch encoders (global + local), a standard Transformer encoder, and per-token MSE reconstruction loss. A complete inference pipeline (PreResult → ThresholdCreator → Applier) was built for the new model, along with an updated inference notebook and comprehensive test suite.

## Files Modified

### Deleted (MaskFormer artifacts)
- `backend/app/anomaly_detection/layers/masking.py` — MaskToken, SensorBlockMasking
- `backend/app/anomaly_detection/criterions/association_discrepancy.py` — AssociationDiscrepancyLoss
- `backend/app/anomaly_detection/criterions/masked_reconstruction.py` — MaskedReconstructionLoss
- `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py` — GLCAnomalyMaskFormer model
- `backend/app/anomaly_detection/training/maskformer_trainer.py` — MaskFormerTrainer
- `backend/app/anomaly_detection/inference/maskformer/` — Entire maskformer inference module
- `backend/tests/test_glc_anomaly_maskformer.py`, `test_maskformer_trainer.py`, `test_maskformer_criterions.py`, `test_mask_token.py`, `test_sensor_block_masking.py`, `test_anomaly_attention.py`

### Modified
- `backend/app/anomaly_detection/layers/attention.py` — Removed AnomalyAttentionBlock and AnomalyAttentionTransformer classes; kept CLSToken, TransformerEncoder, DualPooling
- `backend/app/anomaly_detection/layers/encoders.py` — Added PatchEncoderCNNGlobal and PatchEncoderCNNLocal classes
- `backend/app/anomaly_detection/layers/__init__.py` — Removed masking/anomaly-attention exports; added CNN encoder exports
- `backend/app/anomaly_detection/criterions/__init__.py` — Removed MaskedReconstructionLoss, AssociationDiscrepancyLoss
- `backend/app/anomaly_detection/models/__init__.py` — Replaced GLCAnomalyMaskFormer with GLCReconV8; removed MaskFormerModels
- `backend/app/anomaly_detection/models/factory.py` — Removed MaskFormerModels enum; added GLCReconV8 to registry and ReconstructionModels
- `backend/app/anomaly_detection/training/__init__.py` — Replaced MaskFormerTrainer with GLCReconV8Trainer
- `backend/app/anomaly_detection/inference/__init__.py` — Replaced maskformer imports with recon_v8 imports
- `backend/tests/test_local_inference.py` — Replaced MaskFormer tests with ReconV8 tests

### Created
- `backend/app/anomaly_detection/models/glc_recon_v8.py` — GLCReconV8 model (full reconstruction with dual-stream CNN encoders, standard Transformer, CNNDecoder)
- `backend/app/anomaly_detection/training/glc_recon_v8_trainer.py` — Simple training loop (single forward/backward, no minimax)
- `backend/app/anomaly_detection/inference/recon_v8/__init__.py` — ReconV8 inference module exports
- `backend/app/anomaly_detection/inference/recon_v8/local.py` — Reconf8LocalPreResult, Reconf8LocalThresholdCreator, Reconf8LocalApplier
- `backend/tests/test_glc_recon_v8.py` — 24 tests covering factory, forward pass, loss, anomaly scoring, heatmap, checkpoint
- `backend/notebooks/Inference-GLCReconV8.ipynb` — Updated inference notebook for V8

## Testing

**Test Files:**
- `tests/test_glc_recon_v8.py` — 24 tests, all pass
- `tests/test_local_inference.py::TestReconf8LocalPipeline` — 8 inference tests, all pass

**Execution:**
```bash
uv run pytest tests/test_glc_recon_v8.py -v       # 24 passed
uv run pytest tests/test_local_inference.py -v     # 15 passed (all 3 methodologies)
```

## Architecture Highlights

### GLCReconV8 Architecture
```
Input [B, C, T]
  ├── Global stream: GlobalRevIN → Patchify → PatchEncoderCNNGlobal → [B, C*P, D/2]
  └── Local stream:  Patchify → PerPatchRevIN → PatchEncoderCNNLocal → [B, C*P, D/2]
                     Concat → [B, C*P, D] + PosEmbed + [CLS]
                     → TransformerEncoder → CNNDecoder → recon [B, C*P, patch_size]
```

- **Loss:** MSE over all tokens (no masking, no minimax, no association discrepancy)
- **Encoding:** 1D Conv1d(1→16→32) per patch replaces Linear projection
- **Target:** GlobalRevIN-normalized patches (consistent across overlapping windows)
- **Inference:** Per-patch reconstruction MSE → heatmap [B, C, P] → phase-aware P95 scoring
