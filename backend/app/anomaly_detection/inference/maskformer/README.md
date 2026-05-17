# MaskFormer Inference (GLCAnomalyMaskFormer)

Dual-criterion anomaly detection using per-sensor heatmaps from a MaskFormer model.

## Classes

### MaskFormerPreResult

Computes per-sensor anomaly heatmaps (combined mask + reconstruction error).

- Model: ``MaskFormer.compute_combined_heatmap`` with optional fast mode.
- Upsampling: Patch-level heatmaps are upsampled to timestep-level via ``_heatmap_to_timestep_scores`` (max-overlap strategy).
- Accumulation: Uniform weighting across overlapping macro-windows.
- Edge case: fewer than ``2 * k_skip`` windows → score set to ``100.0``, empty heatmaps.

### MaskFormerThresholdCreator

Percentile-based threshold on file-level heatmap scores.

- ``threshold = quantile(normal_scores, self.quantile)``.
- Fits from a PyTorch Tensor or a list of dicts from ``MaskFormerPreResult.score_batch``.

### MaskFormerApplier

Applies global + per-sensor heatmap thresholds:

- ``is_file_anomaly`` based on p{quantile} score vs global threshold.
- ``is_anomaly_{s}`` based on max of valid-region heatmap vs per-sensor threshold.
- Returns per-sensor heatmap arrays for downstream analysis.

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| ``robot_code`` | String | Robot identifier |
| ``program_number`` | Integer | Program identifier |
| ``start_time`` | String | File start timestamp |
| ``is_file_anomaly`` | Boolean | Binary file-level verdict |
| ``p{q}_score`` | Float | File-level anomaly score |
| ``is_anomaly_{s}`` | Boolean | Per-sensor binary verdict |
| ``heatmap_{s}`` | Array[Float] | Per-sensor timestep heatmap |
