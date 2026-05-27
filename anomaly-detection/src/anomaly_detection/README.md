# Anomaly Detection Module

Multi-paradigm deep-learning anomaly detection for welding robots.  Six
sensors per robot (air motor rotation speed, electro-pneumatic output, high
pressure current/voltage, paint pressure, paint flow rate) are monitored at
100 Hz and scored by models ranging from simple auto-encoders to flow-matching
and masked-prediction transformers.

---

## Architecture Overview

```
backend/app/anomaly_detection/
├── __init__.py          # Package root, development workflow docs
├── constants.py         # Sensor names, stat suffixes, window sizes
├── schemas.py           # Config dataclasses (GLCFlowV7Config, RevINStat, …)
├── utils.py             # Standalone helpers (seeding, param counting, device)
│
├── layers/              # Reusable nn.Module building blocks
│   ├── preprocessing.py
│   ├── revin.py               # GlobalRevIN, PerPatchRevIN
│   ├── patchify.py            # Patchify, FlattenPatches
│   ├── encoders.py            # PatchEncoderGlobal, PatchEncoderLocal
│   ├── transformer.py         # TransformerEncoder, CLSToken, DualPooling
│   ├── time_embedding.py      # SinusoidalTimeEmbedding
│   ├── backbone.py            # Shared Transformer backbone
│   ├── bottleneck.py          # VQ / RVQ bottleneck
│   ├── cnf.py                 # VectorFieldMLP, CNF (continuous normalising flow)
│   ├── anomaly_attention.py   # AnomalyAttentionBlock / Transformer
│   ├── context_embedding.py   # RobotEmbedding, ProgramEmbedding
│   ├── mask_token.py          # Learnable mask token
│   ├── sensor_block_masking.py
│   ├── reconstruction_decoder.py
│   ├── feature_extraction.py
│   └── scorer.py              # MADThreshold, AnomalyScorer, PatchMasker
│
├── models/              # Concrete architectures, all extending BaseModel
│   ├── base.py                # BaseAnomalyDetectionModel (abstract)
│   ├── factory.py             # create_model(), model enums
│   ├── v1_vae.py              # Convolutional VAE
│   ├── v5_vq.py               # Vector-quantised VAE
│   ├── v5_vq_file.py          # VQ + file-level codebook
│   ├── v6_vq.py               # VQ extension
│   ├── v6_rvq.py              # Residual VQ
│   ├── glc_flow_v7.py         # Continuous Normalising Flow
│   └── glc_anomaly_maskformer.py  # Masked-prediction dual-head
│
├── criterions/          # Loss functions
│   ├── base.py                # BaseAnomalyDetectionLoss (abstract)
│   ├── flow_matching.py       # FlowMatchingLoss (CNF likelihood)
│   ├── dual_head.py           # DualHeadLoss, GlobalHead, LocalHead
│   ├── orthogonality.py       # Orthogonality regularisation
│   ├── masked_reconstruction.py
│   └── association_discrepancy.py
│
├── training/            # Trainers and datasets
│   ├── datasets/
│   │   ├── raw.py             # RawWindowDataset (window-level)
│   │   └── stat.py            # StatWindowDataset (window + file stats)
│   ├── single_stage_trainer.py   # Trainer for V1–V6
│   ├── flow_matching_trainer.py  # Two-stage trainer for GLC-Flow V7
│   └── maskformer_trainer.py     # Minimax trainer for MaskFormer
│
└── inference/           # Post-training inference pipelines
    ├── reconstruction/   # V1–V6: recon error + percentile threshold
    ├── flow_matching/    # GLCFlowV7: NLL + MAD threshold
    └── maskformer/       # MaskFormer: dual heatmap + percentile threshold
```

### Data Flow (Training → Inference)

```
Raw 100 Hz sensor files
        │
        ▼
┌──────────────────┐
│  Dataset Loader  │  RawWindowDataset / StatWindowDataset
│  (training/)     │  → windows of shape [B, 6, WINDOW_SIZE]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model (models/) │  Forward pass: batch → dict{out, patches_pred, …}
│  + Criterion     │  Criterion computes dict{loss, global_loss, …}
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Trainer         │  Backprop, checkpoint via save_manual_checkpoint()
│  (training/)     │
└────────┬─────────┘
         │  (frozen weights)
         ▼
┌──────────────────┐
│  Inference       │  1. PreResult  → raw anomaly scores / heatmaps
│  (inference/)    │  2. Threshold  → fit threshold from normal files
│                  │  3. Applier    → produce binary verdicts
└──────────────────┘
```

Each **inference paradigm** implements the same three-step pipeline:

| Paradigm | PreResult | Threshold | Applier |
|---|---|---|---|
| Reconstruction (V1–V6) | MSE per window + shape error | Percentile | Window + file verdict |
| Flow Matching (GLC-Flow V7) | CNF negative log-likelihood | MAD (median abs deviation) | Macro-window verdict |
| MaskFormer | Dual heatmap (recon × assoc. disc.) | Percentile | Macro-window verdict |

---

## Three-Paradigm Evolution

The module has evolved through four generations, each addressing a limitation
of the previous one:

```
AE (V1) → VQ-VAE (V5–V6) → Flow Matching (V7) → MaskFormer (V7)
```

| Generation | Models | Core Idea | Limitation Addressed |
|---|---|---|---|
| **Auto-Encoder** | V1 VAE | Convolutional encoder-decoder; reconstruction error = anomaly score | Baseline — no explicit normal-abnormal separation |
| **VQ-VAE** | V5 VQ, V5 VQ File, V6 VQ, V6 RVQ | Discrete codebook bottleneck forces compact normal representations; out-of-distribution inputs produce high recon error | Poor separation in continuous latent space |
| **Flow Matching** | GLC-Flow V7 | Continuous Normalising Flow learns a probability path from noise to data; NLL = anomaly score; dual-stream encoder (global + local) | VQ codebook collapse; need for probabilistic scoring |
| **MaskFormer** | GLC-Anomaly-MaskFormer | Masked patch prediction + association discrepancy dual-head; minimax training alternates between mask reconstruction and attention divergence | Single-head saturation; need for explainable heatmaps |

The **factory** (`models/factory.py`) groups models by paradigm via enums:

```python
ReconstructionModels  # V1–V6: recon error
FlowMatchingModels    # GLC-Flow V7: CNF NLL
MaskFormerModels      # MaskFormer: dual heatmap
```

---

## `BaseAnomalyDetectionModel` Contract

Every model inherits from `BaseAnomalyDetectionModel` (`models/base.py`), which
provides a uniform external interface:

```python
class BaseAnomalyDetectionModel(nn.Module):
    PARADIGM: ClassVar[str] = "window"        # "window" | "macro"

    @property
    def name(self) -> str: ...                 # e.g. "GLCFlowV7"

    @property
    def required_input_args(self) -> set: ...  # {"input", ...}

    def forward(self, batch: dict) -> dict: ...   # checks + delegates
    def compute_anomaly_score(self, batch): ...   # abstract
    def save_manual_checkpoint(self, epoch, val_loss, folder): ...
    @classmethod
    def load_from_checkpoint(cls, checkpoint, ...): ...
    @classmethod
    def load_from_metadata(cls, metadata): ...     # used by factory
```

Key design points:

- **`forward`** validates the batch via `check_input()`, then calls
  `_forward_args(**filtered)` — subclasses only override `_forward_args`.
- **`required_input_args`** declares what keys the model expects (e.g.
  `{"input"}` or `{"input", "robot_idx", "program_idx"}`).
- **Serialisation** is handled through `get_metadata()` (constructor-arg
  dict) + `state_dict()`, enabling round-trip save/load.
- The **factory** (`create_model()`) is the *only* way to instantiate
  models — callers must never call constructors directly.

---

## How to Develop

### Adding a New Model

1. **Create the model file** in `models/` (e.g. `models/v8_foo.py`).
2. **Inherit `BaseAnomalyDetectionModel`** — implement `name`,
   `required_input_args`, `_forward_args`, `compute_anomaly_score`,
   `get_metadata`.
3. **Register** in `models/__init__.py` (import) and `models/factory.py`
   (`_STR_TO_MODEL_CLS` enum + inference-group enum).
4. **Add a config dataclass** in `schemas.py` if the model has new
   hyperparameters.
5. **Create (or reuse) a criterion** in `criterions/`.
6. **Create (or reuse) a trainer** in `training/` — or extend an existing
   one.
7. **Add inference logic** in `inference/` under the appropriate paradigm
   directory (pre_result, threshold, applier).

### Adding a New Layer

1. **Create the layer file** in `layers/` (e.g. `layers/my_module.py`).
2. **Export** in `layers/__init__.py` and `__all__`.
3. Layers are `nn.Module` subclasses with no reference to training logic
   (no optimizers, no loss computation).
4. Use existing patterns: `GlobalRevIN` for normalisation, `Patchify` for
   patching, `TransformerEncoder` for attention.

### Adding a New Criterion

1. **Create the criterion file** in `criterions/` (e.g.
   `criterions/my_loss.py`).
2. **Inherit `BaseAnomalyDetectionLoss`** — implement `forward(model_output) → dict{loss, ...}` and `get_metadata()`.
3. **Export** in `criterions/__init__.py` and `__all__`.
4. The `forward` method receives the model's output dict and must return a
   dict with at least the key `"loss"`. Additional keys (e.g.
   `"orth_loss"`, `"global_loss"`) are logged automatically by trainers.

---

## File-by-File Reference

| File | Role | Exports |
|---|---|---|
| `__init__.py` | Package root, dev workflow docs | — |
| `constants.py` | Centralised magic values | `SENSORS`, `STATS_COLS`, `WINDOW_SIZE`, … |
| `schemas.py` | Config dataclasses | `GLCFlowV7Config`, `RevINStat`, `PatchStats` |
| `utils.py` | Standalone helpers | `set_random_seed`, `count_parameters`, `to_device`, `detach_tensor` |
| `models/base.py` | Abstract base + protocol | `BaseAnomalyDetectionModel`, `ModelConfig`, `filter_model_kwargs` |
| `models/factory.py` | Central model registry & instantiation | `create_model`, `SupportedAnomalyDetectionModels`, `ReconstructionModels`, `FlowMatchingModels`, `MaskFormerModels`, `count_parameters`, `get_model_summary` |
| `models/v1_vae.py` | Convolutional VAE (V1) | `V1VAEAnomalyDetectionModel` |
| `models/v5_vq.py` | Vector-quantised VAE (V5) | `V5VQAnomalyDetectionModel` |
| `models/v5_vq_file.py` | VQ + file-level codebook (V5) | `V5VQFileAnomalyDetectionModel` |
| `models/v6_vq.py` | VQ extension (V6) | `V6VQAnomalyDetectionModel` |
| `models/v6_rvq.py` | Residual VQ (V6) | `V6RVQAnomalyDetectionModel` |
| `models/glc_flow_v7.py` | CNF flow-matching (V7) | `GLCFlowV7` |
| `models/glc_anomaly_maskformer.py` | Masked prediction dual-head (V7) | `GLCAnomalyMaskFormer` |
| `criterions/base.py` | Abstract loss base | `BaseAnomalyDetectionLoss` |
| `criterions/flow_matching.py` | CNF NLL loss | `FlowMatchingLoss` |
| `criterions/dual_head.py` | Global + local head loss | `DualHeadLoss`, `GlobalHead`, `LocalHead` |
| `criterions/orthogonality.py` | Orthogonality regularisation | `OrthogonalityLoss`, `orthogonality_loss` |
| `criterions/masked_reconstruction.py` | Masked patch MSE loss | `MaskedReconstructionLoss` |
| `criterions/association_discrepancy.py` | Attention divergence loss | `AssociationDiscrepancyLoss` |
| `layers/__init__.py` | Layer module index | All layer exports |
| `layers/revin.py` | Reversible instance normalisation | `GlobalRevIN`, `PerPatchRevIN` |
| `layers/patchify.py` | Patching / unpatching | `Patchify`, `FlattenPatches` |
| `layers/encoders.py` | Patch-level encoders | `PatchEncoderGlobal`, `PatchEncoderLocal`, `PositionalEmbedding` |
| `layers/transformer.py` | Transformer backbone | `TransformerEncoder`, `CLSToken`, `DualPooling` |
| `layers/time_embedding.py` | Sinusoidal timestep encoding | `SinusoidalTimeEmbedding`, `get_time_embedding` |
| `layers/cnf.py` | Continuous normalising flow | `VectorFieldMLP`, `CNF` |
| `layers/backbone.py` | Shared transformer backbone | `Backbone` |
| `layers/bottleneck.py` | VQ / RVQ discretisation | `VQBottleneck`, `RVQBottleneck` |
| `layers/anomaly_attention.py` | MaskFormer attention blocks | `AnomalyAttentionBlock`, `AnomalyAttentionTransformer` |
| `layers/context_embedding.py` | Robot / program ID embeddings | `RobotEmbedding`, `ProgramEmbedding`, `ContextEmbedding` |
| `layers/mask_token.py` | Learnable mask token | `MaskToken` |
| `layers/sensor_block_masking.py` | Block-wise sensor masking | `SensorBlockMasking` |
| `layers/reconstruction_decoder.py` | Patch reconstruction head | `ReconstructionDecoder` |
| `layers/feature_extraction.py` | Generic feature extractor | `FeatureExtraction` |
| `layers/scorer.py` | Anomaly scoring utilities | `MADThreshold`, `AnomalyScorer`, `PatchMasker`, `PerturbationLoop` |
| `training/datasets/raw.py` | Raw-window dataset | `RawWindowDataset`, `RawWindowMosaicDataset` |
| `training/datasets/stat.py` | Stat-context dataset | `StatWindowDataset`, `StatWindowMosaicDataset`, `write_sdf_to_mds` |
| `training/single_stage_trainer.py` | Trainer for V1–V6 | `SingleStageTrainer` |
| `training/flow_matching_trainer.py` | Two-stage trainer for GLC-Flow V7 | `FlowMatchingTrainer`, `Stage1Trainer`, `Stage2Trainer`, `Evaluator`, `EvalResult` |
| `training/maskformer_trainer.py` | Minimax trainer for MaskFormer | `MaskFormerTrainer`, `MaskFormerEvaluator`, `MaskFormerTrainerRunner` |
| `inference/__init__.py` | Inference pipeline index | All three paradigm pipelines |
| `inference/reconstruction/` | Recon-based inference (V1–V6) | `ReconPreResult`, `ReconThresholdCreator`, `ReconThresholdApplier` |
| `inference/flow_matching/` | CNF NLL inference (V7) | `FlowMatchingPreResult`, `FlowMatchingThresholdCreator`, `FlowMatchingApplier` |
| `inference/maskformer/` | Dual-heatmap inference (V7) | `MaskFormerPreResult`, `MaskFormerThresholdCreator`, `MaskFormerApplier` |
