# Inference Pipeline

The inference package implements a three-step pipeline shared by all anomaly-detection methodologies:

```
PreResult → ThresholdCreator → Applier
```

## Architecture

Each methodology family (``flow_matching``, ``maskformer``, ``reconstruction``) provides three classes:

| Step | Role | Databricks Pattern |
|------|------|--------------------|
| **PreResult** | Compute raw anomaly scores / heatmaps from a trained model. No thresholds applied. | Broadcast model weights + metadata → returns a Pandas UDF |
| **ThresholdCreator** | Fit threshold(s) from pre-results on normal (non-anomalous) files. | Fits in the driver (torch/numpy), outputs a serialisable dict |
| **Applier** | Apply the learned thresholds to pre-results to produce binary verdicts. | Broadcast threshold config → returns a Pandas UDF |

### PreResult (score computation)

- Receives a Spark DataFrame of macro-windows (sorted by ``window_start_serial``).
- Runs batched model inference on GPU.
- Accumulates per-timestep scores across overlapping windows using a configurable weighting scheme.
- Returns a **single-row** Spark DataFrame per file with:
  - A p{quantile} file-level anomaly score (or shape + win scores for reconstruction).
  - Metadata (``robot_code``, ``program_number``, ``start_time``, serial bounds, total steps).
  - Per-sensor heatmaps (MaskFormer) or score maps + raw signals (reconstruction).

### ThresholdCreator (threshold fitting)

- Consumes collected pre-results from normal (healthy) files.
- Fits a threshold (MAD-based, percentile-based, or per-sensor) that separates normal from anomalous.
- Supports ``to_dict()`` / ``from_dict()`` serialisation for broadcasting.

### Applier (verdict)

- Receives a broadcasted threshold dict + a pre-result DataFrame.
- Applies the threshold per file (and optionally per sensor).
- Returns verdict columns (``is_file_anomaly``, per-sensor anomaly flags, error steps).

## How to Add a New Methodology

1. **Create a new sub-package** under ``inference/`` (e.g. ``inference/<method>/``).
2. Implement the three classes:
   - ``<Method>PreResult`` — extends the PreResult pattern (model + metadata broadcast → inference UDF).
   - ``<Method>ThresholdCreator`` — implements ``fit()``, ``to_dict()``, ``from_dict()``.
   - ``<Method>Applier`` — implements ``make_apply_fn()`` (threshold broadcast → apply UDF).
3. **Export** from the sub-package ``__init__.py`` and add to ``inference/__init__.py`` ``__all__``.
4. Wire into the orchestration layer (see ``app/anomaly_detection/orchestrator/``).

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| ``quantile`` | Score quantile used as the file-level summary (e.g. 0.95 → ``p95_score``). |
| ``k_skip`` | Number of leading/trailing windows excluded from the valid region to avoid boundary effects. |
