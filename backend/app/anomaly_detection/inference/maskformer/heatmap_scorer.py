"""Batch-level dual heatmap scorer for GLCAnomalyMaskFormer.

Provides a pure-PyTorch batch scorer (no Spark dependency) that mirrors
the pattern of inference/v7/preresult_inferencer.py for local and notebook use.
A Spark/Databricks Pandas-UDF wrapper can be added as a future extension.
"""

from typing import Optional

import torch
import numpy as np

from app.anomaly_detection.models.glc_anomaly_maskformer import GLCAnomalyMaskFormer


class MaskFormerHeatmapScorer:
    """Batch scorer for GLCAnomalyMaskFormer that returns per-file heatmaps.

    Loads model from metadata + state_dict (mirrors V7PreResultInferencer API
    but without the Spark Broadcast wrapper — use directly from notebooks or
    inference scripts).

    Args:
        model:    Pre-loaded GLCAnomalyMaskFormer instance (eval mode).
        device:   Inference device.
        fast:     Use approximate reconstruction scoring (default True).
    """

    def __init__(
        self,
        model: GLCAnomalyMaskFormer,
        device: Optional[torch.device] = None,
        fast: bool = True,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fast = fast
        self.model.to(self.device).eval()

    @classmethod
    def from_metadata_and_weights(
        cls,
        metadata: dict,
        state_dict: dict,
        device: Optional[torch.device] = None,
        fast: bool = True,
    ) -> "MaskFormerHeatmapScorer":
        """Instantiate scorer from serialized metadata + state_dict."""
        model = GLCAnomalyMaskFormer.load_from_metadata(metadata)
        model.load_state_dict(state_dict)
        return cls(model=model, device=device, fast=fast)

    def score_batch(self, raw: torch.Tensor) -> dict:
        """Score a batch of files.

        Args:
            raw: [B, num_sensors, seq_len] raw sensor files

        Returns:
            dict with:
                file_scores:      [B] file-level combined anomaly score
                combined_heatmap: [B, num_sensors, num_patches]
        """
        raw = raw.to(self.device)
        batch = {"input": raw}
        with torch.no_grad():
            heatmaps = self.model.compute_combined_heatmap(batch, fast=self.fast)
            file_scores = heatmaps.amax(dim=(-1, -2))
        return {
            "file_scores": file_scores.cpu(),
            "combined_heatmap": heatmaps.cpu(),
        }
