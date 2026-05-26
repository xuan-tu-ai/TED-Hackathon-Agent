# Anomaly Detection Models

## Purpose

This package contains all concrete neural-network architectures for the
anomaly detection subsystem.  Every model inherits from
[`BaseAnomalyDetectionModel`](base.py) and is instantiated through a
central [factory](factory.py), never directly.

## Architecture Overview

```
BaseAnomalyDetectionModel  (abstract — base.py)
 ├── V1VAEAnomalyDetectionModel      (v1_vae.py)         — AE paradigm
 ├── V5VQAnomalyDetectionModel       (v5_vq.py)           — VQ-VAE
 ├── V5VQFileAnomalyDetectionModel   (v5_vq_file.py)      — per-file VQ
 ├── V6VQAnomalyDetectionModel       (v6_vq.py)           — VQ (V6)
 ├── V6RVQAnomalyDetectionModel      (v6_rvq.py)          — Residual VQ
 ├── GLCFlowV7                       (glc_flow_v7.py)     — Flow Matching
 └── GLCAnomalyMaskFormer            (glc_anomaly_maskformer.py) — MaskFormer
```

### Paradigms

| Paradigm | Models | Inference Strategy | Window Type |
|----------|--------|--------------------|-------------|
| **AE / VQ-VAE** | V1, V5, V6 | Reconstruction error → percentile threshold | Window (≤64) |
| **Flow Matching** | GLC-Flow V7 | NLL under CNF → MAD threshold | Macro (240) |
| **MaskFormer** | GLC-Anomaly-MaskFormer V7 | Recon × Association discrepancy heatmap → percentile threshold | Macro (240) |

## Model Contract

Every model must satisfy the `BaseAnomalyDetectionModel` interface:

```python
model = create_model("glc_flow_v7", {"d_model": 128})
output = model.forward({"input": tensor})       # → dict of tensors
scores = model.compute_anomaly_score({"input": tensor})  # → Tensor[B]
result = model.detect_anomaly(raw_file)          # → dict with heatmap
```

## How to Add a New Model

1. Create a new file (e.g. `v8_my_arch.py`).
2. Define a config dataclass implementing the `ModelConfig` protocol
   (``to_dict`` / ``from_dict``).
3. Implement your model by subclassing `BaseAnomalyDetectionModel` and
   providing all abstract members.
4. Register in `factory.py`:
   - Add the import.
   - Add to `_STR_TO_MODEL_CLS` dict.
   - Add to `SupportedAnomalyDetectionModels` enum.
   - Add to the appropriate inference-group enum (`ReconstructionModels`,
     `FlowMatchingModels`, `MaskFormerModels`) or create a new one.
5. Export from `__init__.py`.

## File Reference

| File | Contents |
|------|----------|
| `base.py` | `BaseAnomalyDetectionModel` ABC + `ModelConfig` protocol + `filter_model_kwargs` |
| `factory.py` | `create_model()`, enums, `count_parameters()`, `get_model_summary()` |
| `v1_vae.py` | V1 convolutional auto-encoder (window paradigm) |
| `v5_vq.py` | V5 vector-quantised VAE |
| `v5_vq_file.py` | V5 VQ with per-file codebook |
| `v6_vq.py` | V6 VQ evolution |
| `v6_rvq.py` | V6 residual VQ (multiple codebooks) |
| `glc_flow_v7.py` | GLC-Flow V7 — dual-stream encoder + transformer + CNF |
| `glc_anomaly_maskformer.py` | GLC-Anomaly-MaskFormer V7 — masked recon + association discrepancy |
