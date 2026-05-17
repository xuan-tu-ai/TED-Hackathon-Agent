# Anomaly Detection — Layers

## Purpose

This package contains all reusable neural network components (layers) that
are assembled by the model architectures in `models/`.  Layers are designed
to be independent, composable, and testable in isolation.

## Layer Taxonomy

```
Normalisation
├── revin.py          GlobalRevIN (per-file z-score) + PerPatchRevIN (per-patch min-max)
├── patchify.py       Patchify + FlattenPatches — time-series → patch tokens
├── preprocessing.py  Legacy RevIN, LogScaler, GlobalStatsStandardizer (V1–V6)

Encoding
├── encoders.py       PatchEncoderGlobal + PatchEncoderLocal + PositionalEmbedding (V7)
├── feature_extraction.py  ShapeEncoder, StatProjector, RelativePositionEncoder (V1–V6)
├── backbone.py       MLPMixerBackbone (V1–V6)
├── bottleneck.py     VAE head, VQ head, Residual VQ head (V1–V6)
├── transformer.py    Standard TransformerEncoder + CLSToken + DualPooling (V7)

Detection
├── cnf.py            Continuous Normalising Flow (V7 — density estimation)
├── scorer.py         MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop (V7)
├── anomaly_attention.py   AnomalyAttentionBlock + AnomalyAttentionTransformer (MaskFormer V7)
├── context_embedding.py   RobotEmbedding + ProgramEmbedding + ContextEmbedding (MaskFormer V7)
├── mask_token.py          Learnable [MASK] token (MaskFormer V7)
├── sensor_block_masking.py  Time-column masking (MaskFormer V7)
├── reconstruction_decoder.py  Linear → GELU → Linear patch decoder (MaskFormer V7)
├── time_embedding.py    Sinusoidal time encoding for CNF
```

## How to Develop

### Adding a New Layer

1. Create a new `.py` file in this directory.
2. Implement your `nn.Module` class following the conventions:
   - Typed `__init__` arguments.
   - `forward(input: Tensor) -> Tensor | dict` with shape-annotated docstring.
3. Export from `__init__.py` (both import line and `__all__`).
4. Create a unit test in `backend/tests/` that validates output shapes.

### Layer Design Principles

- **No coupling between layers** — layers should not import other layers from
  this package (the only exception is `cnf.py` importing `time_embedding.py`).
- **Shape annotations** — every `forward` docstring must document input and
  output shapes.
- **No training logic** — layers are stateless w.r.t. training; the minimax
  training logic is handled by the trainer, not the anomaly attention layer
  (though the layer exposes `forward_with_detach_mode` as a contract).
