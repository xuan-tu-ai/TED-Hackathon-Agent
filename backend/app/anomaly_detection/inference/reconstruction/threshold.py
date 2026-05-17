"""Per-sensor shape + win threshold for reconstruction scores.

Fits a second-level percentile over per-sensor scores already computed by
ReconPreResult. Produces a dict ``{sensor: {"shape": float, "win": float}}``
used downstream by ReconThresholdApplier.
"""
from typing import Optional, List

import numpy as np
import pandas as pd

import app.anomaly_detection.constants as constants


class ReconThresholdCreator:
    """Create per-sensor shape/win thresholds from ReconPreResult output on normal files.

    Fits a second-level quantile over the per-sensor scores already computed by
    ReconPreResult, matching the same fit-from-pre-results pattern used by
    FlowMatchingThresholdCreator and MaskFormerThresholdCreator.

    Workflow:
        1. Run ReconPreResult (Spark) on normal files → collect pre-result DataFrame
        2. ReconThresholdCreator().fit(normal_pre_results) → thresholds dict
        3. Broadcast thresholds dict → ReconThresholdApplier
    """

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.thresholds: Optional[dict] = None  # {sensor: {"shape": float, "win": float}}

    def fit(
        self,
        normal_pre_results: pd.DataFrame,
        sensors: Optional[List[str]] = None,
    ) -> "ReconThresholdCreator":
        """Fit per-sensor thresholds from a collected ReconPreResult DataFrame.

        Args:
            normal_pre_results: Spark-collected pre-result rows for normal files.
                                 Must contain p{q}_shape_{s} and p{q}_win_{s} columns.
            sensors: Subset of sensor names to fit. Defaults to all constants.SENSORS.
        """
        pq = f"p{self.quantile * 100:.0f}"
        sensors = sensors if sensors is not None else list(constants.SENSORS)
        q = self.quantile * 100

        self.thresholds = {}
        for s in sensors:
            shape_col = f"{pq}_shape_{s}"
            win_col = f"{pq}_win_{s}"
            self.thresholds[s] = {
                "shape": float(np.percentile(normal_pre_results[shape_col].dropna().values, q)),
                "win": float(np.percentile(normal_pre_results[win_col].dropna().values, q)),
            }
        return self

    def to_dict(self) -> dict:
        """Serialize per-sensor thresholds to a JSON-compatible dict."""
        return {"thresholds": self.thresholds, "quantile": self.quantile}

    @classmethod
    def from_dict(cls, d: dict) -> "ReconThresholdCreator":
        """Deserialize per-sensor thresholds from a dict.

        Args:
            d: Dict with ``thresholds`` and ``quantile`` keys.

        Returns:
            Reconstituted instance.
        """
        obj = cls(quantile=d.get("quantile", 0.95))
        obj.thresholds = d.get("thresholds")
        return obj
