# Task Summary: Tasks 19-20 — GLC-Flow V7 Model Assembly + Factory

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Tasks 19-20 — GLC-Flow V7 main model class + factory functions

## Summary of Work

Assembled `glc_flow_v7.py` and `factory.py` — the complete GLC-Flow V7 model from all previously implemented components. The full pipeline is now runnable end-to-end with correct shape propagation.

### Model Architecture (GLCFlowV7.forward)

```
Input: x['input'] [B, 6, 240]
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
GlobalRevIN                                  PerPatchRevIN
(normalize per-file)                        (patchify first, normalize per-patch)
  │                                              │
  ▼                                              ▼
Patchify + FlattenPatches                   Patchify + FlattenPatches
→ [B, 84, 32]                               → [B, 84, 32]
  │                                              │
  ▼                                              ▼
PatchEncoderGlobal                           PatchEncoderLocal
(Linear 32→64 + LN + GELU)                 (Linear 32→64 + LN + GELU, independent)
→ [B, 84, 64]                               → [B, 84, 64]
  │                                              │
  └──────────────────┬───────────────────────────┘
                   ▼
     Concat → [B, 84, 128] + PositionalEmbedding + CLS token
     → TransformerEncoder → [B, 85, 128]
     → DualPooling: z_pooled = concat([z_cls, z_max_local]) → [B, 256]
  │
  ├─→ GlobalHead(z_cls) → file_stats_pred [B, 36]
  └─→ LocalHead(z_patches) → patch_stats_pred [B, 12]

  CNF(z_pooled) for anomaly scoring (d_model=256 for CNF since z_pooled is [B, 2D])
```

### Output Dict Keys
- `z_pooled: [B, 256]` — dual-pooled representation (CLS + max local) for CNF
- `z_cls: [B, 128]` — CLS token (global drift)
- `z_patches: [B, 84, 128]` — per-patch tokens
- `z_max_local: [B, 128]` — max-pooled local (spike detection)
- `file_stats_pred: [B, 36]` — auxiliary file-level stats (μ, σ, slope, peak, RMS, range × 6 sensors)
- `patch_stats_pred: [B, 12]` — auxiliary patch-level stats (max/min per sensor)
- `transformer_output: [B, 85, 128]` — full transformer sequence with CLS

## Files Modified
- [backend/app/anomaly_detection/models/glc_flow_v7.py](backend/app/anomaly_detection/models/glc_flow_v7.py) — Full model assembly
- [backend/app/anomaly_detection/models/factory.py](backend/app/anomaly_detection/models/factory.py) — Factory + utility functions
- [backend/app/anomaly_detection/layers/patchify.py](backend/app/anomaly_detection/layers/patchify.py) — **Fix: removed Linear from FlattenPatches** (projection now done by PatchEncoderGlobal/Local per stream)
- [backend/tests/test_patchify.py](backend/tests/test_patchify.py) — Updated tests for FlattenPatches (reshape-only, no projection)
- [backend/tests/test_glc_flow_v7.py](backend/tests/test_glc_flow_v7.py) — 13 integration tests

## Bugs Fixed During Assembly

### Bug 1: FlattenPatches double projection (critical)
**Problem:** FlattenPatches had `Linear(32, d_half)` AND PatchEncoderGlobal/Local also project `Linear(32, d_half)`. This caused `tokens_g` to go from `[B, 84, 32]` → `[B, 84, 64]` via FlattenPatches, then `[B, 84, 64]` → `[B, 84, 64]` via PatchEncoderGlobal (identity mapping with wrong dim). The second linear expected input=32 but got 64.

**Fix:** Removed Linear from FlattenPatches — it now just reshapes `[B, C, P, L] → [B, C*P, L]`. Each stream's PatchEncoder does the 32→64 projection independently, enabling disentangled learning as designed.

**Files:** `layers/patchify.py`, `models/glc_flow_v7.py`

### Bug 2: CNF d_model mismatch (critical)
**Problem:** `z_pooled` is `[B, 2D] = [B, 256]` after dual pooling, but CNF was initialized with `d_model=config.d_model=128`. VectorFieldMLP expected input `[B, 128+32]=[B, 160]` but received `[B, 256+32]=[B, 288]`.

**Fix:** Changed CNF initialization to `CNF(d_model=config.d_model * 2, ...)` so it correctly handles `[B, 2D]` input.

**File:** `models/glc_flow_v7.py`

## Testing
- **Test File:** [backend/tests/test_glc_flow_v7.py](backend/tests/test_glc_flow_v7.py)
- **Status:** All 13 tests PASS
- **Execution:** `cd backend && uv run pytest tests/test_glc_flow_v7.py -v`
- **Coverage:**
  - Factory: create default/config/summary/param count
  - Forward: dict keys, shapes (z_pooled=[2,256], z_cls=[2,128], z_patches=[2,84,128], etc.), single sample, batch sizes
  - detect_anomaly: returns correct dict keys and scalar scores
  - stage1_loss: returns all loss components
  - Model summary: 790,064 total parameters

## Additional Notes
- FlattenPatches no longer does projection — this is intentional per spec (each stream's encoder does independent 32→64 projection)
- CNF uses d_model=256 (2×d_model) because z_pooled = concat([z_cls, z_max_local]) = [B, 2D]
- All 113 tests in backend/tests/ pass (including newly updated test_patchify.py)
- Sprint plan Tasks 19, 20 marked [x] done