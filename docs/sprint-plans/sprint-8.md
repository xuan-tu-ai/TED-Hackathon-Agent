# Sprint 8 Plan

## Sprint Goal
Document the entire codebase — every class, function, and file across `anomaly_detection/`, `robot_agent/`, and `agent_knowledge/` — with **Google Style docstrings** and **module-level README.md** files that explain structure, development patterns, and design rationale.

## Documentation Standards

### Google Style Docstrings
- Every public class: one-line summary, blank line, detailed paragraph explaining purpose/design choices, then `Args:` / `Attributes:` / `Returns:` / `Raises:` sections as applicable.
- Every public function/method: one-line imperative summary, blank line, extended description of why it exists and how it fits, then `Args:` / `Returns:` / `Raises:`.
- Every module (`__init__.py` or standalone): module-level docstring describing the module's role in the architecture.
- Constants/variables at module top-level: inline `#:` comment or docstring explaining purpose, units, allowed values.

**Requirement:** Docstrings must be *meaningful* — not "The forward function does forward pass" but rather *what semantic computation it performs, what shape assumptions it makes, and how it fits the broader pipeline*.

### Module README.md
Every sub-package directory gets a `README.md` covering:
- **Purpose:** What this module does in the larger system.
- **Structure:** List of files and brief description of each.
- **How to develop/extend:** Patterns to follow, which files to touch for a new feature.
- **Key classes/functions:** Highlight the main entry points.
- **Example usage:** Minimal code snippet if applicable.

## Affected Scope

```
backend/app/
├── anomaly_detection/                     ← 48 Python files + 6 READMEs (root + models + layers + criterions + training + inference)
│   ├── models/                            base, factory, v1, v5×2, v6×2, v7_flow, v7_maskformer, __init__
│   ├── layers/                            revin, patchify, preprocessing, backbone, bottleneck, encoders, transformer,
│   │                                      time_embedding, cnf, scorer, context_embedding, mask_token,
│   │                                      sensor_block_masking, anomaly_attention, reconstruction_decoder,
│   │                                      feature_extraction, __init__
│   ├── criterions/                        base, flow_matching, dual_head, orthogonality, masked_reconstruction,
│   │                                      association_discrepancy, __init__
│   ├── training/                          flow_matching_trainer, single_stage_trainer, maskformer_trainer,
│   │                                      datasets/{raw, stat, __init__}, __init__
│   └── inference/                         root + flow_matching/{applier,pre_result,threshold}, maskformer/{applier,pre_result,threshold},
│                                          reconstruction/{applier,pre_result,threshold,utils}, __init__s
├── robot_agent/                           ← 40 Python files + 7 READMEs (root + agent + api + simulation + poller + db + seed)
│   ├── agent/                             diagnostic_agent, systemic_agent, skills, rag_tools, llm_config,
│   │                                      hitl, hitl_gate, event_emitter, plan_manager, session_trace, __init__
│   ├── api/                               simulation_api, incidents_api, hitl_api, __init__
│   ├── simulation/                        engine, state, runner, broadcaster, __init__
│   ├── poller/                            poller, l1_burst, l2_slope, __init__
│   ├── memory/                            memory, __init__
│   ├── db/                                models, __init__
│   ├── chroma/                            client, __init__
│   └── seed/                              run, pm_cards, perception_results, perception_results_legacy, __init__
└── root files:                            conftest.py, __init__.py

backend/agent_knowledge/                   ← README for root + each skill directory
```

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Anomaly Detection — Models (3 tasks)

- [x] **T1: Document [models/base.py](backend/app/anomaly_detection/models/base.py) + [models/factory.py](backend/app/anomaly_detection/models/factory.py) + [models/\_\_init\_\_.py](backend/app/anomaly_detection/models/__init__.py) + create [models/README.md](backend/app/anomaly_detection/models/README.md)**
  - `base.py`: Google Style docstrings for `BaseAnomalyDetectionModel` (abstract interface, `PARADIGM`, `CONFIG_CLASS`, `name`, `required_input_args`, `save_manual_checkpoint`, `load_from_checkpoint`, `compute_anomaly_score`, `detect_anomaly`, `get_metadata`, `to_config_dict`). Explain the abstract contract and when/why to override each method.
  - `factory.py`: docstrings for `ModelConfig` Protocol, `SupportedAnomalyDetectionModels` enum, `create_model()`. Document the factory pattern and how to register a new model.
  - `__init__.py`: module docstring listing all exported model classes and their paradigms.
  - `README.md`: explain the model hierarchy (what `BaseAnomalyDetectionModel` enforces), how each variant differs (VAE vs VQ vs Flow vs MaskFormer), how to add a new model.

- [x] **T2: Document [models/v1_vae.py](backend/app/anomaly_detection/models/v1_vae.py) + [models/v5_vq.py](backend/app/anomaly_detection/models/v5_vq.py) + [models/v5_vq_file.py](backend/app/anomaly_detection/models/v5_vq_file.py)**
  - Google Style docstrings for all classes (`V1VAE` + `V1VAEConfig`, `V5VQ` + `V5VQConfig`, `V5VQFile` + `V5VQFileConfig`). Explain each model's architecture paradigm (pure AE, VQ-VAE, per-file VQ), the input/output shapes, and how they differ from adjacent variants. Document `compute_anomaly_score` and `detect_anomaly` behavior for each.

- [x] **T3: Document [models/v6_vq.py](backend/app/anomaly_detection/models/v6_vq.py) + [models/v6_rvq.py](backend/app/anomaly_detection/models/v6_rvq.py) + [models/glc_flow_v7.py](backend/app/anomaly_detection/models/glc_flow_v7.py) + [models/glc_anomaly_maskformer.py](backend/app/anomaly_detection/models/glc_anomaly_maskformer.py)**
  - V6 models: docstrings for `V6VQ` + `V6RVQ` and their configs. Explain the residual VQ extension over V5.
  - `glc_flow_v7.py`: full Google Style docstrings for `GLCFlowV7Config` (every field documented), `GLCFlowV7` (dual-stream encoder → transformer → CNF pipeline documented step-by-step in forward docstring). `compute_anomaly_score`, `compute_combined_heatmap`, `detect_anomaly`, `get_metadata`, `save_manual_checkpoint`, `load_from_checkpoint`. Explain the CNF flow-matching paradigm and how the two-stage training interacts with the model.
  - `glc_anomaly_maskformer.py`: full docstrings for `GLCAnomalyMaskFormerConfig`, `GLCAnomalyMaskFormer`. Document the masked predictive + association discrepancy dual-criterion paradigm, the `forward_with_detach_mode` contract, and minimax shape assumptions.

### Phase 2: Anomaly Detection — Layers (3 tasks)

- [x] **T4: Document [layers/revin.py](backend/app/anomaly_detection/layers/revin.py) + [layers/patchify.py](backend/app/anomaly_detection/layers/patchify.py) + [layers/preprocessing.py](backend/app/anomaly_detection/layers/preprocessing.py) + [layers/backbone.py](backend/app/anomaly_detection/layers/backbone.py) + [layers/bottleneck.py](backend/app/anomaly_detection/layers/bottleneck.py)**
  - `revin.py`: docstrings for `GlobalRevIN`, `PerPatchRevIN`. Document the instance normalization semantics, the affine parameter shapes, and the difference between global vs per-patch normalization.
  - `patchify.py`: docstrings for `Patchify`, `FlattenPatches`. Explain the sliding window / non-overlapping patch extraction, output shape `[B, N, P]`.
  - `preprocessing.py`: docstrings for any data preprocessing layers.
  - `backbone.py`: docstrings explaining shared backbone logic.
  - `bottleneck.py`: docstrings explaining dimensionality reduction / expansion logic.

- [x] **T5: Document [layers/encoders.py](backend/app/anomaly_detection/layers/encoders.py) + [layers/transformer.py](backend/app/anomaly_detection/layers/transformer.py) + [layers/time_embedding.py](backend/app/anomaly_detection/layers/time_embedding.py) + [layers/cnf.py](backend/app/anomaly_detection/layers/cnf.py) + [layers/scorer.py](backend/app/anomaly_detection/layers/scorer.py)**
  - `encoders.py`: docstrings for `PatchEncoderGlobal`, `PatchEncoderLocal`, `PositionalEmbedding`. Explain dual-stream encoding design rationale.
  - `transformer.py`: docstrings for `TransformerEncoderLayer`, `CLSToken`, `DualPooling`. Document CLS token usage and pooling strategy.
  - `time_embedding.py`: docstrings explaining sinusoidal / learned time encoding for the CNF.
  - `cnf.py`: docstrings for the Continuous Normalizing Flow ODE, how probability paths are learned, sampling vs inference paths.
  - `scorer.py`: docstrings for `MADThreshold`, `AnomalyScorer`, `PatchMasker`, `PerturbationLoop`. Explain the MAD theory and each scorer's role in anomaly localization.

- [x] **T6: Document [layers/context_embedding.py](backend/app/anomaly_detection/layers/context_embedding.py) + [layers/mask_token.py](backend/app/anomaly_detection/layers/mask_token.py) + [layers/sensor_block_masking.py](backend/app/anomaly_detection/layers/sensor_block_masking.py) + [layers/anomaly_attention.py](backend/app/anomaly_detection/layers/anomaly_attention.py) + [layers/reconstruction_decoder.py](backend/app/anomaly_detection/layers/reconstruction_decoder.py) + [layers/feature_extraction.py](backend/app/anomaly_detection/layers/feature_extraction.py) + [layers/\_\_init\_\_.py](backend/app/anomaly_detection/layers/__init__.py) + create [layers/README.md](backend/app/anomaly_detection/layers/README.md)**
  - All Sprint 7 layers: document the `detach_mode` contract in `AnomalyAttentionBlock`, the mask-sampling semantics in `SensorBlockMasking`, the context embedding fallback to zero-tensor, the reconstruction decoder design choice (predict per-patch RevIN'd targets). For each, explain motivation and shape contracts.
  - `feature_extraction.py`: document feature extraction utilities.
  - `__init__.py`: module docstring listing all exported layers.
  - `README.md`: explain the layer taxonomy (normalization → patchification → encoding → transformer → scoring/detection), the dual-stream pattern, how to add a new layer.

### Phase 3: Anomaly Detection — Criterions (1 task)

- [x] **T7: Document [criterions/](backend/app/anomaly_detection/criterions/) (all 7 files) + create [criterions/README.md](backend/app/anomaly_detection/criterions/README.md)**
- [x] **T8: Document [training/](backend/app/anomaly_detection/training/) (all 6 files) + create [training/README.md](backend/app/anomaly_detection/training/README.md) + [training/datasets/README.md](backend/app/anomaly_detection/training/datasets/README.md)**
- [x] **T9: Document [inference/](backend/app/anomaly_detection/inference/) (all 12 files) + create READMEs for `inference/`, `inference/flow_matching/`, `inference/maskformer/`, `inference/reconstruction/`**
- [x] **T10: Document [anomaly_detection/\_\_init\_\_.py](backend/app/anomaly_detection/__init__.py) + [constants.py](backend/app/anomaly_detection/constants.py) + [schemas.py](backend/app/anomaly_detection/schemas.py) + [utils.py](backend/app/anomaly_detection/utils.py) + create [anomaly_detection/README.md](backend/app/anomaly_detection/README.md)**
  - `constants.py`: document every constant (`SENSOR_NAMES`, `STATS_COLUMNS`, window sizes) with units and purpose. Use inline `#:` comments for each.
  - `schemas.py`: Google Style docstrings for `GLCFlowV7Config`, `RevINStat`, `PatchStats`. Document every field.
  - `utils.py`: document all utility functions with usage examples.
  - `__init__.py`: module docstring explaining the top-level module structure.
  - `README.md`: **the most important README** — explain the entire anomaly detection module at a high level:
    - Architecture: models → layers → criterions → training → inference data flow
    - The three-paradigm evolution: AE (V1) → VQ-VAE (V5/V6) → Flow Matching (V7) → MaskFormer (V7)
    - How the `BaseAnomalyDetectionModel` contract unifies all variants
    - How to develop: adding a new model, adding a new layer, adding a new criterion
    - File-by-file reference table linking to sprint docs for deep dives

### Phase 7: Robot Agent — Agent (2 tasks)

- [x] **T11: Document [agent/](backend/app/robot_agent/agent/) — core files: [diagnostic_agent.py](backend/app/robot_agent/agent/diagnostic_agent.py) + [systemic_agent.py](backend/app/robot_agent/agent/systemic_agent.py) + [skills.py](backend/app/robot_agent/agent/skills.py) + [rag_tools.py](backend/app/robot_agent/agent/rag_tools.py) + [llm_config.py](backend/app/robot_agent/agent/llm_config.py) + [\_\_init\_\_.py](backend/app/robot_agent/agent/__init__.py) + create [agent/README.md](backend/app/robot_agent/agent/README.md)**
- [x] **T12: Document [agent/](backend/app/robot_agent/agent/) — HITL & helpers: [hitl.py](backend/app/robot_agent/agent/hitl.py) + [hitl_gate.py](backend/app/robot_agent/agent/hitl_gate.py) + [event_emitter.py](backend/app/robot_agent/agent/event_emitter.py) + [plan_manager.py](backend/app/robot_agent/agent/plan_manager.py) + [session_trace.py](backend/app/robot_agent/agent/session_trace.py)**
- [x] **T13: Document [api/](backend/app/robot_agent/api/) + [db/](backend/app/robot_agent/db/) + [chroma/](backend/app/robot_agent/chroma/) + [memory/](backend/app/robot_agent/memory/) + create READMEs for each**
- [x] **T14: Document [simulation/](backend/app/robot_agent/simulation/) + [poller/](backend/app/robot_agent/poller/) + [seed/](backend/app/robot_agent/seed/) + create READMEs for each**
  - `simulation/engine.py`, `state.py`, `runner.py`, `broadcaster.py`: document the simulation lifecycle, state machine, event broadcasting over SSE.
  - `poller/poller.py`, `l1_burst.py`, `l2_slope.py`: document the cognitive polling architecture — what L1 (burst) and L2 (slope) detect, how thresholds are configured, how results feed into the diagnostic system.
  - `seed/run.py`, `pm_cards.py`, `perception_results.py`, `perception_results_legacy.py`: document database seeding scripts — what data they generate, how much, the random distribution parameters.
  - `README.md`s: explain the simulation and polling architecture, how to configure new detection algorithms, the seed data schema.

### Phase 10: Robot Agent — Root Module (1 task)

- [x] **T15: Document [robot_agent/main.py](backend/app/robot_agent/main.py) + [robot_agent/\_\_init\_\_.py](backend/app/robot_agent/__init__.py) + [backend/conftest.py](backend/backend/conftest.py) + create [robot_agent/README.md](backend/app/robot_agent/README.md)**

- [x] **T16: Document [agent_knowledge/](backend/agent_knowledge/) + verify docstring compliance**
  - Add/improve docstrings to skill directories' `SKILL.md` files where applicable.
  - Run Google Style docstring compliance check: every public class/func in `anomaly_detection/` and `robot_agent/` must have a docstring with the correct sections. Fix any violations found.
  - Verify all 18+ README files exist and are substantive (not stubs).
  - Use `uv run pytest backend/tests/ -v` — confirm all existing tests still pass after docstring-only changes.

## Notes / Blockers

- **Docstring-only sprint.** No logic changes. No refactoring. If any docstring task reveals a code bug, file a GitHub issue — do NOT fix the bug in this sprint.
- **Google Style reference.** Use numpy-style Google docstrings (Args/Returns/Raises as sections, types in parentheses). Example:
  ```python
  def compute_anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
      """Compute per-file anomaly score from the combined heatmap.

      Takes the patch-level heatmap produced by `compute_combined_heatmap`
      and reduces it to a scalar score via spatial max-pooling over the
      heatmap's sensor × time dimensions.

      Args:
          batch: Input tensor of shape [B, 6, 240] containing
              normalized sensor readings across 6 sensors and 240 time steps.

      Returns:
          Tensor of shape [B] with scalar anomaly scores in ℝ⁺.
      """
  ```
- **README location rules.** README.md goes in the same directory as the `__init__.py` it documents. E.g., `layers/README.md` documents the `anomaly_detection/layers/` package.
- **Large-file strategy.** Some files (e.g., `diagnostic_agent.py`) exceed 700 lines. For those, group related method docstrings together under their section, and ensure the class-level docstring covers the overall architecture so individual method docstrings can be shorter.
- **Commit cadence.** Proposed commits at natural boundaries:
  1. End of Phase 2 (T1–T6: anomaly_detection models + layers — 2 commits: models block, then layers block)
  2. End of Phase 5 (T7–T10: criterions + training + inference + root)
  3. End of Phase 8 (T11–T13: agent core + HITL helpers + data layer)
  4. End of Phase 10 (T14–T15: simulation/poller/seed + robot_agent root)
  5. End of Phase 11 (T16: agent_knowledge + verification)
