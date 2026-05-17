# Reconstruction Inference (v1–v6 VQ/RVQ)

Dual-criterion anomaly detection using reconstruction error (shape) and window-statistic error (win).

## Classes

### ReconPreResult

Computes per-timestep shape (reconstruction MSE) and window (z-score) error maps.

- Model: VQ / RVQ reconstruction models (v1–v6).
- Inputs: ``raw_window``, ``stats``, ``rel_pos``, ``robot_idx``, ``program_idx``.
- Accumulation: Gaussian-weighted overlap (``scipy.signal.windows.gaussian``, std = W/4).
- Two criteria:
  - **Shape**: ``(recon_norm - x_norm)²`` — reconstruction fidelity per timestep.
  - **Win**: ``(pred_win_stats_z - true_win_stats_z)²`` — window statistic deviation.
- Edge case: fewer than ``2 * k_skip`` windows → scores set to ``100.0``, ``total_steps = 0``.

### ReconThresholdCreator

Per-sensor percentile thresholds for both shape and win scores.

- Fits on collected pre-results from normal files.
- Produces ``{sensor_name: {"shape": float, "win": float}}`` dict.

### ReconThresholdApplier

Applies per-sensor shape/win thresholds to produce:

- ``err_steps_{s}`` — list of anomalous timestep serials per sensor.
- ``trigger_source_{s}`` — ``"shape"`` / ``"win"`` / ``"both"`` / ``"none"`` per sensor.
- ``is_file_anomaly`` — True if any sensor has anomalous steps.
- ``is_shape_triggered`` / ``is_win_triggered`` — file-level trigger flags.
- Raw signal arrays (``actual_raw_{s}``, ``recon_raw_{s}``) for visualisation.

### Utility Functions

- ``analyze_trigger_sources(spark, result_table)`` — prints trigger-source breakdown from a Spark table.
- ``visualize_feature_level_report_vq(spark, result_row, gold_table_name)`` — plots actual vs reconstructed signals with detected fault overlays.

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| ``robot_code`` | String | Robot identifier |
| ``program_number`` | Integer | Program identifier |
| ``start_time`` | String | File start timestamp |
| ``is_file_anomaly`` | Boolean | Binary file-level verdict |
| ``is_shape_triggered`` | Boolean | File flagged by shape criterion |
| ``is_win_triggered`` | Boolean | File flagged by win criterion |
| ``err_steps_{s}`` | Array[Integer] | Anomalous timestep serial numbers |
| ``trigger_source_{s}`` | String | shape / win / both / none |
| ``p{q}_shape_score`` | Float | File-level shape anomaly score |
| ``p{q}_win_score`` | Float | File-level window anomaly score |
| ``p{q}_shape_{s}`` | Float | Per-sensor shape score |
| ``p{q}_win_{s}`` | Float | Per-sensor window score |
| ``actual_raw_{s}`` | Array[Float] | Raw sensor signal |
| ``recon_raw_{s}`` | Array[Float] | Reconstructed sensor signal |
