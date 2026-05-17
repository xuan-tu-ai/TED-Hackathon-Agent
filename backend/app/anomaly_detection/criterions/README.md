# Loss Functions (`criterions/`)

## Purpose

This module provides all loss functions used for training the anomaly detection
models. Each loss inherits from `BaseAnomalyDetectionLoss` and implements a
`forward()` method that returns a `dict` with at least a `"loss"` key.

## Taxonomy

Losses are organised by **training paradigm** and **stage**:

| Criterion | Stage | Paradigm | Role |
|---|---|---|---|
| `DualHeadLoss` | 1 | Two-stage | Encoder supervision via global + local decoder heads |
| `OrthogonalityLoss` | 1 | Two-stage | Auxiliary regularisation (disentangle subspaces) |
| `FlowMatchingLoss` | 2 | Two-stage | CNF likelihood via conditional flow matching |
| `MaskedReconstructionLoss` | — | Single-stage (MaskFormer) | Masked-patch MSE for anomaly attention |
| `AssociationDiscrepancyLoss` | — | Single-stage (MaskFormer) | Minimax association discrepancy |

### Two-Stage Pipeline (GLC-Flow-V7)

```
Stage 1 — Encoder Pre-training
  L = L_global(DualHeadLoss) + L_local(DualHeadLoss) + lambda_orth * L_orth(OrthogonalityLoss)

Stage 2 — CNF Training
  L = L_fm(FlowMatchingLoss)       [encoder frozen]
```

### Single-Stage Pipeline (MaskFormer)

```
Single Stage
  L = L_mask(MaskedReconstructionLoss) + lambda_assoc * L_assoc(AssociationDiscrepancyLoss)
```

The `AssociationDiscrepancyLoss` is used with a **minimax** sign schedule:
the trainer alternately applies `+lambda_assoc` (maximise, update prior) and
`-lambda_assoc` (minimise, update series).

## How Losses Are Composed

- **Stage 1** (`glc_flow_v7.py:stage1_loss`): `DualHeadLoss` produces
  `global_loss` + `local_loss`.  `OrthogonalityLoss` adds a regularisation
  term.  All three are summed into a single scalar.
- **Stage 2** (`flow_matching_trainer.py:Stage2Trainer`): `FlowMatchingLoss`
  is the sole objective.  The encoder is frozen; only the CNF head receives
  gradients.
- **Single-stage** (`maskformer_trainer.py:MaskFormerTrainer`):
  `MaskedReconstructionLoss` + `AssociationDiscrepancyLoss` are summed with a
  configurable `lambda_assoc` weight.  The sign of the association term
  alternates per minimax phase.

## How to Add a New Criterion

1. Create a new file `criterions/<name>.py`.
2. Subclass `BaseAnomalyDetectionLoss` and implement `forward()` and
   `get_metadata()`.
3. Add a module-level docstring in Google Style explaining:
   - The mathematical formulation.
   - Which training paradigm/stage it belongs to.
   - Any interaction with the trainer (e.g. sign-flipping, stop-gradient).
4. Export the class in `__init__.py` and add it to `__all__`.
5. If the loss has a functional (non-class) variant, export it separately.
