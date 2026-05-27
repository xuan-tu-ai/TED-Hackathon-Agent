# Task Summary: GLCMaeV8 — Asymmetric MAE for Sensor Anomaly Detection

## Status: COMPLETED

## What was implemented

Refactored GLCReconV8 → GLCMaeV8 following the MAE paradigm (He et al., 2022)
adapted for multivariate time-series sensor data.

### Files created / modified

| File | Action | Description |
|---|---|---|
| `layers/masking.py` | Created | `MaskToken` (learnable [MASK] param), `random_masking()` (shuffle/restore) |
| `layers/reconstruction.py` | Modified | Added `LightDecoder` (kept); added `use_tanh` param to `CNNDecoder` |
| `layers/__init__.py` | Modified | Export `MaskToken`, `random_masking`, `LightDecoder` |
| `models/glc_mae_v8.py` | Created | `GLCMaeV8` + `GLCMaeV8Config` |
| `models/factory.py` | Modified | Added `GLC_MAE_V8` to `SupportedAnomalyDetectionModels`, `ReconstructionModels`, `_STR_TO_MODEL_CLS` |
| `models/__init__.py` | Modified | Export `GLCMaeV8`, `GLCMaeV8Config` |
| `training/glc_mae_v8_trainer.py` | Created | `GLCMaeV8Trainer`, `GLCMaeV8Evaluator`, `EvalResult` |
| `tests/test_glc_mae_v8.py` | Created | 16 smoke tests (all passing) |

### Architecture (C=3, seq_len=240, patch_size=32, patch_stride=16)

```
num_patches = (240-32)//16 + 1 = 14
num_tokens  = 3 × 14 = 42
n_visible   = int(42 × 0.55) = 23   (mask_ratio=0.45)
n_masked    = 42 - 23 = 19
```

**Training forward (asymmetric):**
```
raw [B, 3, 240]
→ dual-stream CNN tokenize (global + local RevIN) → tokens [B, 42, D]
→ add PositionalEmbedding [42, D]
→ random_masking(45%) → visible_tokens [B, 23, D] + ids_shuffle + mask
→ TransformerEncoder (only 23 visible tokens) → latent [B, 23, D]
→ _unshuffle: scatter latent + (mask_token + pos_emb) at masked positions → [B, 42, D]
→ CNNDecoder(use_tanh=False) → recon [B, 42, patch_size]
Loss: MSE on 19 masked positions only
```

**Inference forward (full, no masking):**
```
raw [B, 3, 240]
→ dual-stream CNN tokenize → tokens [B, 42, D] → TransformerEncoder → CNNDecoder(use_tanh=False)
→ per-token MSE → heatmap [B, 3, 14]
```

### Key design decisions

- **No train/inference gap**: encoder never sees [MASK] during training or inference.
- **CNNDecoder(use_tanh=False)**: For anomaly detection the reconstruction error IS the anomaly score → stronger decoder lowers MSE(normal) → wider gap with MSE(anomaly) → better AUROC. Masking prevents encoder shortcut independently of decoder strength. GlobalRevIN targets are unbounded so Tanh is off.
- **compute_loss always uses masked forward**: val_loss computed with same masking as train → comparable scale.
- **compute_heatmap/compute_anomaly_score use full inference forward**: deterministic scores for threshold calibration.
- **ContextEmbedding preserved**: robot_idx + program_idx still supported via codebooks.

## Test results

```
16 passed in 5.93s
```

All tests pass:
- `test_mask_token_shape` — MaskToken expands correctly
- `test_random_masking_shapes` — visible/mask tensors correct shape + ratio
- `test_light_decoder_no_tanh` — output not bounded to [-1,1]
- `test_training_forward_shapes` — recon/target/mask correct
- `test_inference_forward_no_mask` — no mask key in inference output
- `test_compute_loss_scalar` — scalar loss, >0, no NaN
- `test_compute_heatmap_shape` — [B, C, P] = [2, 3, 14]
- `test_compute_anomaly_score_shape` — [B] = [2], non-negative
- `test_detect_anomaly` — correct keys + heatmap shape
- `test_config_round_trip` — to_dict / from_dict roundtrip
- `test_backward_pass` — mask_token.grad is not None
- Factory + enum registration tests (3)
