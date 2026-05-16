from typing import Optional, List

import torch


class MaskFormerThresholdCreator:
    """Create a percentile threshold from normal file heatmap scores.

    Fits on file-level anomaly scores (max over heatmap) from MaskFormerPreResult.score_batch.
    """

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.threshold: Optional[float] = None

    def fit(self, normal_scores: torch.Tensor) -> "MaskFormerThresholdCreator":
        self.threshold = torch.quantile(normal_scores, self.quantile).item()
        return self

    def fit_from_pre_results(
        self, pre_result_list: List[dict]
    ) -> "MaskFormerThresholdCreator":
        """Fit from a list of dicts returned by MaskFormerPreResult.score_batch."""
        all_scores = torch.cat([r["file_scores"] for r in pre_result_list])
        return self.fit(all_scores)

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "quantile": self.quantile}

    @classmethod
    def from_dict(cls, d: dict) -> "MaskFormerThresholdCreator":
        obj = cls(quantile=d.get("quantile", 0.95))
        obj.threshold = d.get("threshold")
        return obj
