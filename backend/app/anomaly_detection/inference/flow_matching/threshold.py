from typing import Optional

import torch
import pandas as pd


class FlowMatchingThresholdCreator:
    """Create a MAD-based global threshold from normal file NLL scores.

    Uses Median Absolute Deviation, which is robust to ~50% contamination.
    Threshold = median(scores) + multiplier * MAD(scores)
    """

    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
        self.threshold: Optional[float] = None

    def fit(self, normal_scores: torch.Tensor) -> "FlowMatchingThresholdCreator":
        median = normal_scores.median()
        mad = (normal_scores - median).abs().median()
        self.threshold = (median + self.multiplier * mad).item()
        return self

    def fit_from_pre_results(
        self, pre_result_df: pd.DataFrame, score_col: str = "p95_score"
    ) -> "FlowMatchingThresholdCreator":
        """Fit from a Spark-collected pre-result DataFrame."""
        scores = torch.tensor(
            pre_result_df[score_col].dropna().values, dtype=torch.float32
        )
        return self.fit(scores)

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "multiplier": self.multiplier}

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMatchingThresholdCreator":
        obj = cls(multiplier=d.get("multiplier", 3.0))
        obj.threshold = d.get("threshold")
        return obj
