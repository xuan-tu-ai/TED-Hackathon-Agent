"""Percentile-based threshold for MaskFormer heatmap scores.

Fits on file-level anomaly scores (max over valid heatmap region) from
MaskFormerPreResult using ``torch.quantile``. Single global threshold
applied downstream by MaskFormerApplier.
"""
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
        """Fit percentile threshold from normal-file heatmap scores.

        Args:
            normal_scores: Tensor of file-level scores from
                MaskFormerPreResult (max over valid heatmap region).

        Returns:
            Self for chaining.
        """
        self.threshold = torch.quantile(normal_scores, self.quantile).item()
        return self

    def fit_from_pre_results(
        self, pre_result_list: List[dict]
    ) -> "MaskFormerThresholdCreator":
        """Fit from a list of dicts returned by MaskFormerPreResult.score_batch."""
        all_scores = torch.cat([r["file_scores"] for r in pre_result_list])
        return self.fit(all_scores)

    def to_dict(self) -> dict:
        """Serialize threshold config to a JSON-compatible dict."""
        return {"threshold": self.threshold, "quantile": self.quantile}

    @classmethod
    def from_dict(cls, d: dict) -> "MaskFormerThresholdCreator":
        """Deserialize threshold config from a dict.

        Args:
            d: Dict with ``threshold`` and ``quantile`` keys.

        Returns:
            Reconstituted instance.
        """
        obj = cls(quantile=d.get("quantile", 0.95))
        obj.threshold = d.get("threshold")
        return obj
