# Sprint 7 Plan

## Sprint Goal
Implement **GLC-Anomaly-MaskFormer V7** — a sibling architecture to GLC-Flow V7 that replaces the CNF likelihood head with a **Masked Predictive + Association Discrepancy** dual-criterion detector — fully integrated into the existing [backend/app/anomaly_detection/](backend/app/anomaly_detection/) module following the same `BaseAnomalyDetectionModel` / `BaseAnomalyDetectionLoss` patterns as V7. Deliver model, single-stage minimax trainer, dual-scoring inference (`recon × association` heatmap + MAD threshold), and unit tests.

## Reference Design
- [docs/new-anomaly-detection/DL_MODEL_GLC_ANOMALY_MASKFORMER_v7.md](docs/new-anomaly-detection/DL_MODEL_GLC_ANOMALY_MASKFORMER_v7.md)

## Dependencies
- Sprint 1 (GLC-Flow V7 layers — RevIN, Patchify, Encoders, Orthogonality loss, MAD threshold — all reused).
- Sprint 3 (V6 pattern / `BaseAnomalyDetectionModel` + `ModelConfig` Protocol — followed strictly).

## Architecture Mapping to Existing Codebase

| Design Doc Concern | Existing Reusable Asset | New Asset Needed |
|---|---|---|
| Dual RevIN normalization | [revin.py](backend/app/anomaly_detection/layers/revin.py): `GlobalRevIN`, `PerPatchRevIN` | — |
| Patchify (240→14×32) | [patchify.py](backend/app/anomaly_detection/layers/patchify.py): `Patchify`, `FlattenPatches` | — |
| Dual-stream patch projection | [encoders.py](backend/app/anomaly_detection/layers/encoders.py): `PatchEncoderGlobal`, `PatchEncoderLocal` | — |
| Positional embedding (84 tokens) | [encoders.py](backend/app/anomaly_detection/layers/encoders.py): `PositionalEmbedding` | — |
| Robot / Program embeddings | — | `ContextEmbedding` (new) |
| Learnable `[MASK]` token | — | `MaskToken` (new) |
| Sensor-block masking | — | `SensorBlockMasking` (new) |
| Anomaly-Attention block (A_series, A_prior, KL discrepancy) | — | `AnomalyAttentionBlock` (new — replaces stock `TransformerEncoder`) |
| Stacked anomaly-attention transformer | — | `AnomalyAttentionTransformer` (new) |
| Masked-patch reconstruction decoder | — | `ReconstructionDecoder` (new) |
| Reconstruction loss | — | `MaskedReconstructionLoss` (new) |
| Association discrepancy loss (minimax) | — | `AssociationDiscrepancyLoss` (new) |
| Orthogonality regularization | [orthogonality.py](backend/app/anomaly_detection/criterions/orthogonality.py): `OrthogonalityLoss` | — |
| MAD threshold | [scorer.py](backend/app/anomaly_detection/layers/scorer.py): `MADThreshold` | — |
| Top-level model | follows [glc_flow_v7.py](backend/app/anomaly_detection/models/glc_flow_v7.py) pattern | `GLCAnomalyMaskFormer` + `GLCAnomalyMaskFormerConfig` |
| Trainer (single-stage minimax) | follows [single_stage_trainer.py](backend/app/anomaly_detection/training/single_stage_trainer.py) pattern | `MaskFormerTrainer` (new — minimax phase logic) |
| Dual scoring inference | follows `detect_anomaly()` method on [glc_flow_v7.py](backend/app/anomaly_detection/models/glc_flow_v7.py) | model-level `compute_combined_heatmap()` + `detect_anomaly()` |
| Factory registration | [models/factory.py](backend/app/anomaly_detection/models/factory.py): `_STR_TO_MODEL_CLS`, `SupportedAnomalyDetectionModels` | string key `glc_anomaly_maskformer` |

## Affected Files
```
backend/app/anomaly_detection/
  layers/
    mask_token.py                  ← NEW: MaskToken
    sensor_block_masking.py        ← NEW: SensorBlockMasking
    context_embedding.py           ← NEW: RobotEmbedding, ProgramEmbedding, ContextEmbedding (composite)
    anomaly_attention.py           ← NEW: AnomalyAttentionBlock, AnomalyAttentionTransformer
    reconstruction_decoder.py      ← NEW: ReconstructionDecoder
    __init__.py                    ← EDIT: export new layers
  criterions/
    masked_reconstruction.py       ← NEW: MaskedReconstructionLoss
    association_discrepancy.py     ← NEW: AssociationDiscrepancyLoss (minimax via stop-grad)
    __init__.py                    ← EDIT: export new losses
  models/
    glc_anomaly_maskformer.py      ← NEW: GLCAnomalyMaskFormer + GLCAnomalyMaskFormerConfig
    factory.py                     ← EDIT: register glc_anomaly_maskformer
    __init__.py                    ← EDIT: export GLCAnomalyMaskFormer
  training/
    maskformer_trainer.py          ← NEW: MaskFormerTrainer + MaskFormerEvaluator (dual scoring + MAD)
    __init__.py                    ← EDIT: export trainer
  inference/
    maskformer/
      __init__.py                  ← NEW
      heatmap_scorer.py            ← NEW: helper for batch-level dual heatmap computation (optional, mirrors v7/preresult_inferencer.py if Spark integration needed; otherwise model methods suffice)
backend/tests/
  test_mask_token.py               ← NEW: MaskToken shape + learnability
  test_sensor_block_masking.py     ← NEW: time-column masking correctness
  test_anomaly_attention.py        ← NEW: AnomalyAttentionBlock shapes + KL discrepancy properties
  test_reconstruction_decoder.py   ← NEW: decoder output shape
  test_maskformer_criterions.py    ← NEW: MaskedReconstructionLoss + AssociationDiscrepancyLoss
  test_glc_anomaly_maskformer.py   ← NEW: model forward, save/load, factory wiring
  test_maskformer_trainer.py       ← NEW: single training step runs, minimax sign correct
```

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: New Layers (5 tasks)

- [x] **T1: Create [layers/mask_token.py](backend/app/anomaly_detection/layers/mask_token.py)** — `MaskToken(nn.Module)`. Single learnable parameter `[1, 1, d_model]` initialized via `nn.init.normal_(std=0.02)`. Forward: given a boolean mask `[B, N]` and tokens `[B, N, D]`, return `[B, N, D]` with masked positions replaced by the learned MASK embedding. Match the `CLSToken` convention in [transformer.py](backend/app/anomaly_detection/layers/transformer.py).

- [x] **T2: Create [layers/sensor_block_masking.py](backend/app/anomaly_detection/layers/sensor_block_masking.py)** — `SensorBlockMasking(nn.Module)`. Constructor takes `num_sensors=6`, `num_patches=14`, `mask_ratio=0.3`. Forward: takes tokens `[B, 84, D]` and produces `(mask_indices: BoolTensor[B, 84], masked_time_ids: LongTensor[B, n_masked])`. Sample which time positions to hide once per batch element, then expand across all `num_sensors`. Used in **training only** — at inference, callers either skip masking (for discrepancy path) or pass an explicit single-token mask (for per-token recon path). Pure module — no `[MASK]` replacement; that happens in the model via `MaskToken`.

- [x] **T3: Create [layers/context_embedding.py](backend/app/anomaly_detection/layers/context_embedding.py)** — three modules:
  - `RobotEmbedding(num_robots, d_model)` — `nn.Embedding`, returns `[B, D]` from `robot_idx [B]`.
  - `ProgramEmbedding(num_programs, d_model)` — same pattern.
  - `ContextEmbedding(num_robots, num_programs, d_model)` — wraps both; forward takes `(robot_idx, program_idx)` and returns `[B, 1, D]` (broadcast-ready). Allow `robot_idx=None`/`program_idx=None` to return a zero tensor for backward-compat with `RawWindowDataset` (which does not yet provide context IDs).

- [x] **T4: Create [layers/anomaly_attention.py](backend/app/anomaly_detection/layers/anomaly_attention.py)** — TWO classes following the design doc §6.6:
  - `AnomalyAttentionBlock(d_model, n_heads, n_tokens, ffn_dim, dropout, initial_sigma=5.0)` — multi-head Q/K/V (separate `nn.Linear`s), learnable per-head `sigma` parameter `nn.Parameter(torch.ones(n_heads) * initial_sigma)`. Forward `(x: [B,N,D]) → dict(out=[B,N,D], discrepancy=[B,N], A_series=[B,H,N,N], A_prior=[B,H,N,N])`. Pre-norm residual ordering matching V7 (`norm_first=True`). KL: `kl(P||Q) + kl(Q||P)` with `eps=1e-8`; mean over heads → `[B, N]`. **Critical detail (minimax stop-grad):** expose `forward_with_detach_mode(x, detach: Literal["series","prior","none"]="none")` so the trainer can switch which side carries gradients in each phase. Specifically: `detach="series"` → `A_series_d = A_series.detach()` is used to build discrepancy (sigma gets the gradient, "maximize" phase); `detach="prior"` → `A_prior_d = A_prior.detach()` (Q/K get the gradient, "minimize" phase). The standard `forward` should call with `detach="none"` for inference. Follow `define-deeplearning-model` skill — typed args, dict output.
  - `AnomalyAttentionTransformer(num_layers, num_heads, d_model, ffn_dim, dropout, n_tokens, initial_sigma)` — stacks `AnomalyAttentionBlock` modules. Forward `(x) → dict(out=[B,N,D], discrepancies=[B,L,N] stacked-per-layer)`. Average across layers later is the trainer's responsibility.

- [x] **T5: Create [layers/reconstruction_decoder.py](backend/app/anomaly_detection/layers/reconstruction_decoder.py)** — `ReconstructionDecoder(d_model, patch_size)`. Per design doc §8.1: `nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch_size))`. Forward `(tokens: [B,N,D]) → patches_pred: [B,N,patch_size]`. Targets are the **raw, non-RevIN'd** patches (or alternatively the per-patch RevIN'd patches — pick one and document it; recommended: predict per-patch RevIN'd targets so the decoder isn't fighting baseline shifts).

### Phase 2: New Criterions (2 tasks)

- [x] **T6: Create [criterions/masked_reconstruction.py](backend/app/anomaly_detection/criterions/masked_reconstruction.py)** — `MaskedReconstructionLoss(BaseAnomalyDetectionLoss)`. Forward `(patches_pred: [B,N,L], patches_target: [B,N,L], mask: [B,N]) → dict(loss=scalar)`. Compute MSE **only on masked positions**: `((pred - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * L + eps)`. `get_metadata()` returns `{type: "MaskedReconstructionLoss"}`.

- [x] **T7: Create [criterions/association_discrepancy.py](backend/app/anomaly_detection/criterions/association_discrepancy.py)** — `AssociationDiscrepancyLoss(BaseAnomalyDetectionLoss)`. Forward `(discrepancies: [B,L,N]) → dict(loss=scalar)`. Compute `loss = discrepancies.mean()` averaged over batch, layers, and tokens. **This module does NOT implement the minimax — it only computes the discrepancy magnitude.** The sign and detach mechanics live in the trainer (T10): the trainer adds `-lambda_assoc * L_assoc` (minimize phase) and `+L_assoc` (maximize phase) with the appropriate `detach_mode` switched on the transformer (T4). Document the contract clearly in the docstring.

### Phase 3: Model (2 tasks)

- [x] **T8: Create [models/glc_anomaly_maskformer.py](backend/app/anomaly_detection/models/glc_anomaly_maskformer.py)** — two classes:
  - `GLCAnomalyMaskFormerConfig` (`@dataclass`) with fields: `num_sensors=6, seq_len=240, patch_size=32, patch_stride=16, d_model=128, transformer_layers=4, transformer_heads=4, ffn_dim=512, dropout=0.1, mask_ratio=0.3, initial_sigma=5.0, lambda_assoc=2.0, lambda_orth=0.05, num_robots=47, num_programs=16, use_context_embeddings=True`. Include `d_half` and `num_patches` as `field(init=False)`. Implement `to_dict()` / `from_dict()` mirroring `GLCFlowV7Config`.
  - `GLCAnomalyMaskFormer(BaseAnomalyDetectionModel, nn.Module)` with `PARADIGM="macro"`, `CONFIG_CLASS=GLCAnomalyMaskFormerConfig`, `name="GLCAnomalyMaskFormer"`. Required input args: `{"input"}` when `use_context_embeddings=False`, else `{"input", "robot_idx", "program_idx"}`. Forward pipeline:
    1. Global stream: `GlobalRevIN → Patchify → FlattenPatches → PatchEncoderGlobal` → `[B, 84, D/2]`.
    2. Local stream: `Patchify → PerPatchRevIN → FlattenPatches → PatchEncoderLocal` → `[B, 84, D/2]`.
    3. Concat → `[B, 84, D]`, add `PositionalEmbedding`, add `ContextEmbedding(robot_idx, program_idx)`.
    4. **Training mode only** — sample mask via `SensorBlockMasking`, replace masked tokens via `MaskToken`. Save original tokens (before masking) for reconstruction target.
    5. `AnomalyAttentionTransformer` → `out [B,84,D]`, `discrepancies [B,L,84]`.
    6. `ReconstructionDecoder(out) → patches_pred [B,84,L]`.
    Forward output dict keys: `out, discrepancies, patches_pred, patches_target, mask, z_global, z_local` (last two for orthogonality loss — `z_global = tokens_pre_transformer[:, :, :d_half]`, `z_local = tokens_pre_transformer[:, :, d_half:]`, flatten over tokens before passing to `OrthogonalityLoss`).
    
    Plus these public methods:
    - `compute_anomaly_score(batch) -> Tensor[B]` — convenience: returns `combined_heatmap.amax(dim=(-1,-2))` (file-level score).
    - `compute_combined_heatmap(batch, fast=True) -> Tensor[B,6,14]` — runs association pass + reconstruction pass (84 forward passes per file if `fast=False`, ~10 random-mask passes if `fast=True` per design doc §8.3) and returns the **multiplicative** combined heatmap normalized to `[0,1]`.
    - `detect_anomaly(raw_file, threshold=None, robot_idx=None, program_idx=None) -> dict` mirroring V7's signature; returns `{anomaly_score, is_anomaly, heatmap [6,14], assoc_heatmap, recon_heatmap}`.
    - `get_metadata() -> dict` listing all config fields (for `BaseAnomalyDetectionModel.save_manual_checkpoint`).
    
    Adhere to `define-deeplearning-model` skill: typed `__init__`, dict-only `forward`. Follow the structure of [glc_flow_v7.py](backend/app/anomaly_detection/models/glc_flow_v7.py).

- [x] **T9: Register model in factory** — edit [models/factory.py](backend/app/anomaly_detection/models/factory.py): add `"glc_anomaly_maskformer": GLCAnomalyMaskFormer` to `_STR_TO_MODEL_CLS` and `GLC_ANOMALY_MASKFORMER = GLCAnomalyMaskFormer` to the `SupportedAnomalyDetectionModels` enum. Edit [models/__init__.py](backend/app/anomaly_detection/models/__init__.py) to export `GLCAnomalyMaskFormer`. Edit [layers/__init__.py](backend/app/anomaly_detection/layers/__init__.py) and [criterions/__init__.py](backend/app/anomaly_detection/criterions/__init__.py) to export the new layers and losses.

### Phase 4: Training (2 tasks)

- [x] **T10: Create [training/maskformer_trainer.py](backend/app/anomaly_detection/training/maskformer_trainer.py)** — three classes:
  - `MaskFormerTrainer(model, optimizer, lambda_assoc, lambda_orth)` with `train_step(x: dict) -> dict[str, Tensor]`. Implements the **two-phase minimax** in a single step per design doc §7.4:
    - **Phase A (minimize)** — forward through transformer with `detach_mode="prior"` (Q/K get gradients; sigma's contribution to discrepancy is detached). Loss: `L_recon - lambda_assoc * L_assoc_min + lambda_orth * L_orth`. Backward, optimizer.step() on Q/K/V/decoder/encoder/embeddings — NOT sigma.
    - **Phase B (maximize)** — forward through transformer with `detach_mode="series"` (sigma gets gradients). Loss: `+ lambda_assoc * L_assoc_max` (note opposite sign — increases discrepancy by *decreasing* sigma). Backward, optimizer.step() on sigma only.
    - **Practical implementation:** maintain two parameter groups in a single Adam optimizer (`group_main` and `group_sigma`) and gate `param.requires_grad` per phase, or use two separate optimizers — pick whichever is cleaner; document the choice. Returns dict with `loss, recon_loss, assoc_loss_min, assoc_loss_max, orth_loss, sigma_mean` for logging.
  - `MaskFormerEvaluator(model, mad: MADThreshold = None)` — `compute_scores(batch) -> Tensor[B]` (file-level combined score via `compute_anomaly_score`), `evaluate(normal_files, anomaly_files) -> EvalResult` (AUROC + AUPRC + threshold). Reuses the `EvalResult` dataclass and `_compute_auroc` / `_compute_auprc` helpers from [flow_matching_trainer.py](backend/app/anomaly_detection/training/flow_matching_trainer.py) — extract them to a shared `training/eval_utils.py` only if doing so is trivial; otherwise duplicate the small functions to avoid touching V7 code.
  - `MaskFormerTrainerRunner(model_or_config, device=None)` — top-level orchestrator parallel to `FlowMatchingTrainer` in [flow_matching_trainer.py](backend/app/anomaly_detection/training/flow_matching_trainer.py). Methods: `train(train_loader, val_loader=None, epochs=50, lr=1e-4)`, `evaluate(normal_files, anomaly_files)`. Single-stage training loop (no Stage 1 / Stage 2 split like V7 — MaskFormer trains end-to-end).

- [x] **T11: Export new trainer** — edit [training/__init__.py](backend/app/anomaly_detection/training/__init__.py) to add: `from app.anomaly_detection.training.maskformer_trainer import MaskFormerTrainer, MaskFormerEvaluator, MaskFormerTrainerRunner`.

### Phase 5: Inference (optional, deferrable)

- [x] **T12: Optional Spark-style inferencer** — create [inference/maskformer/heatmap_scorer.py](backend/app/anomaly_detection/inference/maskformer/heatmap_scorer.py) following the pattern of [inference/v7/preresult_inferencer.py](backend/app/anomaly_detection/inference/v7/preresult_inferencer.py): takes broadcasted metadata + state_dict, exposes a Pandas UDF that batches files and produces `{robot_code, program_number, start_time, p95_score, total_steps}`. **Skip this task if no Spark integration is needed for the hackathon demo** — the model-level `detect_anomaly()` (T8) is sufficient for notebook demos. Mark as `[~]` deferred if T8 is verified end-to-end before this task starts.

### Phase 6: Tests (5 tasks)

- [x] **T13: Create [tests/test_mask_token.py](backend/tests/test_mask_token.py) + [tests/test_sensor_block_masking.py](backend/tests/test_sensor_block_masking.py)** — combined task (both files are tiny):
  - MaskToken: shape correctness (`[B, N, D]` output); only positions where mask is True change; parameter is leaf and learnable (`requires_grad=True`).
  - SensorBlockMasking: with `mask_ratio=0.3`, exactly `floor(14 * 0.3) = 4` time positions are masked; masking is per-batch-element (different batches have different masked time IDs); EVERY masked time position has all 6 sensor tokens hidden (no partial-column masks).

- [x] **T14: Create [tests/test_anomaly_attention.py](backend/tests/test_anomaly_attention.py)** — `AnomalyAttentionBlock`:
  - Output shapes: `out=[B,N,D]`, `discrepancy=[B,N]`, `A_series=[B,H,N,N]`, `A_prior=[B,H,N,N]`.
  - `A_prior` rows sum to 1 (within `1e-5`).
  - `A_series` rows sum to 1 (softmax property).
  - Discrepancy is non-negative (symmetric KL ≥ 0).
  - With `detach_mode="prior"`, sigma's gradient is zero after backward; with `detach_mode="series"`, Q/K/V's gradient is zero.
  - Stacked `AnomalyAttentionTransformer` returns `discrepancies` of shape `[B, num_layers, N]`.

- [x] **T15: Create [tests/test_reconstruction_decoder.py](backend/tests/test_reconstruction_decoder.py) + [tests/test_maskformer_criterions.py](backend/tests/test_maskformer_criterions.py)** — combined task:
  - `ReconstructionDecoder`: output shape `[B, N, patch_size]`.
  - `MaskedReconstructionLoss`: loss is zero when pred==target on masked positions; loss is non-zero otherwise; gradient flows through to `patches_pred`.
  - `AssociationDiscrepancyLoss`: simple averaging correctness on a fixed `[B, L, N]` input.

- [x] **T16: Create [tests/test_glc_anomaly_maskformer.py](backend/tests/test_glc_anomaly_maskformer.py)** — model integration tests:
  - Forward output dict has expected keys: `out, discrepancies, patches_pred, patches_target, mask, z_global, z_local`.
  - Shapes for `B=2, num_sensors=6, seq_len=240`: `out=(2,84,128)`, `discrepancies=(2,4,84)`, `patches_pred=(2,84,32)`, `mask=(2,84)`.
  - `model.eval()` mode disables masking — `mask` is all-False, `patches_pred` covers all 84 tokens.
  - `compute_anomaly_score(batch).shape == (2,)`.
  - `compute_combined_heatmap(batch).shape == (2, 6, 14)` and values in `[0, 1]`.
  - `detect_anomaly(raw_file)` returns dict with keys `anomaly_score, is_anomaly, heatmap, assoc_heatmap, recon_heatmap`.
  - Factory: `create_model("glc_anomaly_maskformer")` returns a working `GLCAnomalyMaskFormer`; `create_model("glc_anomaly_maskformer", {"d_model": 64, "transformer_layers": 2})` honors overrides.
  - Save/load round-trip: `model.save_manual_checkpoint(...)` then `GLCAnomalyMaskFormer.load_from_checkpoint(...)` recovers the same `state_dict`.
  - `use_context_embeddings=False`: model accepts batch without `robot_idx`/`program_idx`; `required_input_args == {"input"}`.

- [x] **T17: Create [tests/test_maskformer_trainer.py](backend/tests/test_maskformer_trainer.py)** — trainer behavior:
  - `MaskFormerTrainer.train_step(x)` runs without errors on a `[2, 6, 240]` random tensor; returns dict with all expected keys (`loss, recon_loss, assoc_loss_min, assoc_loss_max, orth_loss, sigma_mean`).
  - **Minimax sign check:** after one step, `sigma.detach().mean()` shifts in the direction predicted by the maximize phase (regression test against the most likely sign bug).
  - `MaskFormerEvaluator.evaluate(normal_files, anomaly_files)` returns an `EvalResult` with `auroc ∈ [0, 1]`.
  - End-to-end: `MaskFormerTrainerRunner` initialized with `GLCAnomalyMaskFormerConfig(d_model=32, transformer_layers=1)` runs 1 epoch on 4 random files and completes without NaN losses.

### Phase 7: Verification

- [x] **T18: Run full test suite** — `uv run pytest backend/tests/test_mask_token.py backend/tests/test_sensor_block_masking.py backend/tests/test_anomaly_attention.py backend/tests/test_reconstruction_decoder.py backend/tests/test_maskformer_criterions.py backend/tests/test_glc_anomaly_maskformer.py backend/tests/test_maskformer_trainer.py -v` — all new tests pass. Then run `uv run pytest backend/tests/test_glc_flow_v7.py backend/tests/test_factory.py backend/tests/test_revin.py backend/tests/test_encoders.py backend/tests/test_patchify.py backend/tests/test_transformer.py backend/tests/test_criterions.py -v` — verify V7 tests still pass (regression check that registering new model + editing `__init__.py` did not break V7).

## Dependency Graph
```
T1 (MaskToken) ───────────────┐
T2 (SensorBlockMasking) ──────┤
T3 (ContextEmbedding) ────────┤
T4 (AnomalyAttention*) ───────┤
T5 (ReconstructionDecoder) ───┤
                              ▼
                   T6 (MaskedReconstructionLoss)
                   T7 (AssociationDiscrepancyLoss)
                              │
                              ▼
                  T8 (GLCAnomalyMaskFormer model)
                              │
                              ▼
              T9 (Factory + __init__ exports)
                              │
                              ▼
              T10 (MaskFormerTrainer + Evaluator)
                              │
                              ▼
                  T11 (training/__init__ export)
                              │
              ┌───────────────┴─────────────────┐
              ▼                                 ▼
       T12 (Spark inferencer       T13–T17 (unit tests, 
            — OPTIONAL)                  can run in parallel)
                              │
                              ▼
                          T18 (full verification)
```

## Notes / Blockers

- **Hackathon scope reminder.** Per design doc §12 "Implementation Plan (10 Days)", the critical-path risk is the minimax (T10). If minimax is unstable, the documented fallback is "drop the discrepancy maximization phase and use only reconstruction loss with association discrepancy computed at inference only." If T10 takes longer than 1.5 days, switch to the fallback: keep `AssociationDiscrepancyLoss` as **inference-only**, train with `L_recon + lambda_orth * L_orth` alone, and document the simplification in the model docstring.

- **Dataset compatibility.** Existing [RawWindowDataset](backend/app/anomaly_detection/training/datasets/raw.py) yields `{"input": tensor}` only — no `robot_idx`/`program_idx`. T8's `use_context_embeddings=False` path supports this directly. If real OT data with metadata becomes available later, a `RawWindowWithContextDataset` can be added without touching the model. **Do not edit the existing dataset in Sprint 7** — out of scope.

- **Per-patch RevIN target for reconstruction.** The design doc is silent on whether the decoder predicts raw patches or per-patch RevIN'd patches. Recommendation: **predict per-patch RevIN'd patches** (the same tokens that flow into the local stream). Rationale: the decoder learns to invert the encoder, not to invert RevIN. Document this choice in T5's docstring and in T8's `patches_target` construction.

- **Sigma parameter naming.** The minimax phase's parameter selection (T10) is the single most error-prone part of this sprint. The cleanest mechanism is to name sigma parameters distinctly (e.g., `block.sigma`) and gate via a name-based filter: `[p for n, p in model.named_parameters() if "sigma" in n]` for the max-phase optimizer. Use exactly one such filter in `_freeze_for_phase()` and write the regression test in T17.

- **`detach_mode` API surface.** The `forward_with_detach_mode` method on `AnomalyAttentionBlock` (T4) is the contract between layer and trainer. Test it directly in T14 — a subtle gradient bug here propagates silently into the training loop.

- **Inference cost.** Per design doc §8.1, exact recon scoring is 84 forward passes per file (~5–10 seconds on CPU). The `fast=True` mode in `compute_combined_heatmap` uses ~10 random-mask passes per file. Default `compute_combined_heatmap(fast=True)` for batch eval; allow `fast=False` for forensic single-file analysis.

- **Define-DL-model skill compliance** (T4, T5, T8): `__init__` arguments must be explicitly typed (no `Any`); `forward` input AND output must be `dict` of tensors. Audit all new layers against [`_pytorch_model.py`](.claude/skills/define-deeplearning-model/references/_pytorch_model.py).

- **Operational rules** (per [CLAUDE.md](CLAUDE.md)): use `uv add` for any new dependency (none expected — pure PyTorch); relative imports only at the same level inside subpackages; `uv run pytest` for verification.

- **Out of scope for Sprint 7** (do NOT touch):
  - Existing V7 model, V7 trainer, V6 models — pure additive sprint.
  - Real OT data integration (M9 backlog).
  - Production Spark deployment hardening (M10 backlog) — T12 is optional and Databricks-shaped but does not require a real Spark run.
  - Multi-agent system integration (`DiagnosticAgent` consuming MaskFormer heatmaps is a separate downstream sprint).

- **Commit cadence.** Per [CLAUDE.md](CLAUDE.md): propose a commit only when logic + test exist for a milestone. Natural commit boundaries: end of Phase 1 (T1–T5 + tests T13–T15), end of Phase 3 (T8–T9 + test T16), end of Phase 4 (T10–T11 + test T17), end of Phase 7 (T18 verification).
