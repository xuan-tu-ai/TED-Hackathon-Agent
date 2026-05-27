"""Reconstruction methodology (v1–v6 VQ/RVQ): shape + window error, per-sensor thresholds.

Exports (Spark)
---------------
ReconPreResult
    Compute per-sensor shape/win score maps via Spark Pandas UDF.
ReconThresholdApplier
    Apply per-sensor thresholds via Spark Pandas UDF.

Exports (Local)
---------------
ReconLocalPreResult
    Compute per-sensor shape/win score maps (no Spark, returns list[dict]).
ReconLocalThresholdCreator
    Per-sensor shape + win threshold from normal-file pre-results.
ReconLocalApplier
    Apply per-sensor thresholds (no Spark, returns list[dict]).

Utilities (Spark-free)
----------------------
analyze_trigger_sources
    Summarise trigger-source breakdown (requires Spark).
visualize_feature_level_report_vq
    Plot actual vs reconstructed signals with fault overlays.
_row_to_dict
    Normalise a Spark Row / pd.Series / dict to a plain dict.
_sanitize_start_time
    Sanitise a timestamp string for filesystem-safe filenames.
"""
from src.anomaly_detection.inference.reconstruction.spark import (
    ReconPreResult,
    ReconThresholdApplier,
)
from src.anomaly_detection.inference.reconstruction.local import (
    ReconLocalPreResult,
    ReconLocalThresholdCreator,
    ReconLocalApplier,
)
from src.anomaly_detection.inference.reconstruction.utils import (
    analyze_trigger_sources,
    visualize_feature_level_report_vq,
    _row_to_dict,
    _sanitize_start_time,
)
