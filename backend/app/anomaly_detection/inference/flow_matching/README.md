# Flow-Matching Inference (GLCFlowV7)

CNF-based (Continuous Normalising Flow) anomaly detection using NLL scores.

## Classes

### FlowMatchingPreResult

Computes per-file NLL (negative log-likelihood) anomaly scores.

- Model: Flow-matching CNF (``model.cnf.compute_log_likelihood``).
- Accumulation: Uniform weighting across overlapping macro-windows.
- Score: p{quantile} of per-timestep NLL values in the valid region.
- Edge case: fewer than ``2 * k_skip`` windows → score set to ``100.0``, ``total_steps = 0``.

### FlowMatchingThresholdCreator

MAD-based threshold robust to ~50% contamination:

```
threshold = median(scores) + multiplier * MAD(scores)
```

- ``multiplier`` parameter controls sensitivity (default 3.0).
- Fits from a PyTorch Tensor or a Spark-collected pre-result DataFrame.

### FlowMatchingApplier

Applies the global threshold to pre-results:

- ``is_file_anomaly = 1`` if p{quantile} score > global_threshold.
- ``total_steps == 0`` → always anomalous (score = 100.0).

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| ``robot_code`` | String | Robot identifier |
| ``program_number`` | Integer | Program identifier |
| ``start_time`` | String | File start timestamp |
| ``is_file_anomaly`` | Integer | Binary verdict (0/1) |
| ``p{q}_score`` | Float | File-level NLL anomaly score |
| ``is_anomaly_per_sensor`` | Integer | Same as file verdict for this method |
