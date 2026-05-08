# Sprint 1 Plan

## Sprint Goal
Implement GLC-Flow V7 anomaly detection model — Dual-Stream Encoder (Global/Local RevIN), Transformer with CLS, CNF-based anomaly scoring, and localization pipeline. Sprint 1 covers model implementation only (data generation is Sprint 2).

## Directory Structure
```
backend/app/anomaly_detection/
├── layers/              # Standalone building blocks (RevIN, Patchify, TimeEmbedding, etc.)
│   ├── __init__.py
│   ├── revin.py          # GlobalRevIN + PerPatchRevIN
│   ├── patchify.py       # Patchify layer
│   └── time_embedding.py # SinusoidalTimeEmbedding
├── criterions/          # Loss functions (FlowMatching, DualHead, Orthogonality)
│   ├── __init__.py
│   ├── flow_matching.py
│   ├── dual_head.py
│   └── orthogonality.py
├── models/              # Final model = assembly of layers
│   ├── __init__.py
│   ├── factory.py       # Model creation / registry
│   └── glc_flow_v7.py   # Main model class
├── utils.py             # Config, helpers
└── train.py             # Training script (Stage 1 + Stage 2)
```

## Dependencies
- Sprint 0 (project structure)
- Sprint 2 (mock data — Sprint 1 uses stub/synthetic data for shape testing)

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 0: Config & Constants
- [x] Task 1: Create `anomaly_detection/utils.py` with GLCFlowConfig dataclass

### Phase 1: Layers (Preprocessing)
- [x] Task 2: Create `anomaly_detection/layers/revin.py` — GlobalRevIN + PerPatchRevIN with inverse()
- [x] Task 3: Create `anomaly_detection/layers/patchify.py` — Patchify: unfold(32, 16) → [B, 6, 14, 32]
- [x] Task 4: Create `anomaly_detection/layers/time_embedding.py` — SinusoidalTimeEmbedding

### Phase 2: Dual-Stream Patch Encoders
- [x] Task 5: Implement PatchEncoderGlobal: Linear(32, D/2) + LayerNorm + GELU
- [x] Task 6: Implement PatchEncoderLocal: Linear(32, D/2) + LayerNorm + GELU (independent weights)

### Phase 3: Transformer Encoder + CLS
- [x] Task 7: Implement CLSToken: learnable parameter [1, 1, D] expanded to batch
- [x] Task 8: Implement TransformerEncoder: 4 layers, 4 heads, FFN=256, Dropout=0.1

### Phase 4: Dual Pooling
- [x] Task 9: Implement dual_pooling: z_cls + z_max_local concat → [B, 2D]

### Phase 5: Criterions (Loss Functions)
- [x] Task 10: Create `anomaly_detection/criterions/flow_matching.py` — FlowMatchingLoss
- [x] Task 11: Create `anomaly_detection/criterions/dual_head.py` — DualHeadLoss
- [x] Task 12: Create `anomaly_detection/criterions/orthogonality.py` — OrthogonalityLoss

### Phase 6: CNF (Flow Matching Network)
- [x] Task 13: Implement VectorFieldMLP: 3-layer MLP [D+32→256→256→256→D]
- [x] Task 14: Implement forward_ode with torchdiffeq.odeint for inference
- [x] Task 15: Implement log_likelihood computation (Hutchinson log-det)

### Phase 7: Anomaly Scorer & Localization
- [x] Task 16: Implement AnomalyScorer + MADThreshold
- [x] Task 17: Implement MADThreshold: robust threshold fitting
- [x] Task 18: Implement PatchMasker + PerturbationLoop for localization

### Phase 8: Main Model Assembly
- [x] Task 19: Create `anomaly_detection/models/glc_flow_v7.py` — GLCFlowV7
- [x] Task 20: Create `anomaly_detection/models/factory.py` — factory function

### Phase 9: Training Loop
- [x] Task 21: Create `anomaly_detection/train.py` — Stage1Trainer + Stage2Trainer + Evaluator

### Phase 10: Tests
- [x] Task 22: Create unit tests for RevIN
- [x] Task 23: Create unit tests for CNF
- [x] Task 24: Create integration test for detect_anomaly() pipeline

## Dependency Graph
```
Phase 0 (Config)
    ↓
Phase 1 (Layers: revin, patchify, time_emb)
    ↓
Phase 2 (Encoders) ─→ Phase 3 (Transformer) ─→ Phase 4 (Dual Pooling)
                                                   ↓
Phase 5 (Criterions) ← ─────────────── ─ ─ ─ ─ ─ ┘
                                                   ↓
                                           Phase 6 (CNF)
                                                   ↓
Phase 7 (Scorer + Localization)
                                                   ↓
Phase 8 (Model Assembly)
                                                   ↓
Phase 9 (Training Loop)
                                                   ↓
Phase 10 (Tests)
```

**Note:** Phase 1-4 can be tested end-to-end for shape correctness before Phase 5-6 is ready. Phase 6 (CNF) can be developed in parallel with Phase 5 since its input is just [B, 2D] vector.

## Notes / Blockers
- **torchdiffeq**: Must `uv add torchdiffeq` before Task 13
- **Data stub**: Sprint 1 uses synthetic stub data (random tensors) for shape testing; Sprint 2 implements real mock data generator
- **Forward dict I/O**: Per skill template, all model forward() accept dict, return dict
- **CNF input**: z_pooled = [B, 2D], time_emb = [32], so MLP input dim = 2D + 32

## Estimated Time
~15.5 hours (per DL_MODEL.md Section 10)

## Status: COMPLETED ✓

All 24 tasks completed, 153 tests passing. Implementation aligns with DL_MODEL.md spec after draft comparison. Key changes from draft alignment: SinusoidalTimeEmbedding frequency formula, GlobalHead/LocalHead output shapes [B,C,6]/[B,C,N,2], compute_patch_stats stride=16, PerPatchRevIN min-max inverse, VectorFieldMLP (no zero-init).