"""Anomaly scoring, thresholding, and localisation utilities.

Provides the inference-time components for the GLC-Flow V7 model:

*   :class:`MADThreshold` — robust threshold estimation using Median Absolute
    Deviation (insensitive to up to 50% contamination).
*   :class:`AnomalyScorer` — wraps a CNF to produce negative log-likelihood
    scores (higher = more anomalous).
*   :class:`PatchMasker` — zeroes individual (sensor, patch) regions for
    perturbation-based localisation.
*   :class:`PerturbationLoop` — runs 84 forward passes per file to identify
    the root-cause (sensor, patch) of an anomaly.
"""

import torch
import torch.nn as nn
from typing import Optional


class MADThreshold:
    """Robust anomaly threshold using Median Absolute Deviation (MAD).

    Unlike mean+3σ which is sensitive to outliers, MAD is robust to contamination
    up to ~50% of the data. This is critical for our setting where validation data
    may contain up to 2% unlabeled anomalies.

    threshold = median + k * MAD

    where k=3.0 and the constant 1.4826 makes MAD equivalent to σ for Gaussian:

        threshold = median + 3 * 1.4826 * MAD

    This gives the standard "3-sigma" behavior but robustly estimated.

    Args:
        k: Number of MAD units for threshold (default 3.0).
    """

    def __init__(self, k: float = 3.0) -> None:
        self.k: float = k
        self.median: Optional[torch.Tensor] = None
        self.mad: Optional[torch.Tensor] = None
        self.threshold: Optional[torch.Tensor] = None

    def fit(self, scores: torch.Tensor) -> None:
        """Compute threshold from validation anomaly scores.

        Args:
            scores: Tensor of anomaly scores [N] from validation set
        """
        self.median = torch.median(scores)
        abs_dev = torch.abs(scores - self.median)
        self.mad = torch.median(abs_dev)
        self.threshold = self.median + self.k * 1.4826 * self.mad

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        """Return boolean mask of anomalies.

        Args:
            scores: [N] anomaly scores

        Returns:
            is_anomaly: [N] boolean tensor
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict()")
        return scores > self.threshold

    def get_p_value(self, score: torch.Tensor) -> torch.Tensor:
        """Return approximate p-value (fraction of scores below this score).

        Not a true p-value (uses empirical distribution), but provides
        a calibrated measure of extremity.

        Args:
            score: Single score or [N] tensor

        Returns:
            p-value tensor
        """
        if self.threshold is None or self.median is None:
            raise RuntimeError("Must call fit() before get_p_value()")
        z = (score - self.median) / (1.4826 * self.mad + 1e-8)
        from scipy.stats import norm
        return norm.cdf(z)

    def extra_repr(self) -> str:
        return f"k={self.k}"


class AnomalyScorer(nn.Module):
    """Anomaly scoring combining LogLikelihood + MAD threshold.

    Pipeline:
        1. Compute log-likelihood of z_pooled under learned CNF distribution
        2. Use negative log-likelihood as anomaly score (higher = more anomalous)
        3. Compare against MAD threshold to flag anomalies

    Args:
        cnf: Trained CNF model
    """

    def __init__(self, cnf: nn.Module) -> None:
        super().__init__()
        self.cnf: nn.Module = cnf

    def score(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores (negative log-likelihood).

        Args:
            z_pooled: [B, 2D] pooled representation from encoder

        Returns:
            anomaly_scores: [B] — higher = more anomalous
        """
        ll, _ = self.cnf.compute_log_likelihood(z_pooled)
        return -ll  # higher log-likelihood loss = more anomalous

    def forward(self, z_pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and return anomaly metrics.

        Args:
            z_pooled: [B, 2D] pooled representation

        Returns:
            scores: [B] anomaly scores
            z_noise: [B, 2D] noise representation for logging
        """
        ll, z_noise = self.cnf.compute_log_likelihood(z_pooled)
        scores = -ll
        return scores, z_noise


class PatchMasker:
    """Utility for masking individual (sensor, patch) regions during localization.

    Used by PerturbationLoop to test which patch, when zeroed-out, causes the
    biggest improvement in log-likelihood (i.e., which patch is the root cause).

    Args:
        num_sensors: Number of sensor channels (default 6)
        num_patches: Number of patches per sensor (default 14)
        patch_stride: Stride between patches (default 16)
    """

    def __init__(
        self,
        num_sensors: int = 6,
        num_patches: int = 14,
        patch_stride: int = 16,
    ) -> None:
        self.num_sensors: int = num_sensors
        self.num_patches: int = num_patches
        self.patch_stride: int = patch_stride

    def mask(self, raw_file: torch.Tensor, sensor: int, patch: int) -> torch.Tensor:
        """Zero out a specific (sensor, patch) region in the raw file.

        The raw file has shape [B, num_sensors, seq_len]. After patchify,
        each sensor has num_patches patches of patch_size=32 steps.

        Args:
            raw_file: [B, num_sensors, seq_len]
            sensor: Sensor index [0, num_sensors)
            patch: Patch index [0, num_patches)

        Returns:
            masked_file: Same shape as raw_file with patch zeroed
        """
        B, C, T = raw_file.shape
        masked = raw_file.clone()
        start = patch * self.patch_stride
        end = start + 32
        masked[:, sensor, start:end] = 0.0
        return masked

    def all_masks(self, raw_file: torch.Tensor) -> list[tuple[int, int, torch.Tensor]]:
        """Generate all 84 masked versions of a file (one per sensor×patch).

        Args:
            raw_file: [B, num_sensors, seq_len]

        Returns:
            List of (sensor, patch, masked_file) tuples
        """
        results = []
        for s in range(self.num_sensors):
            for p in range(self.num_patches):
                masked = self.mask(raw_file, s, p)
                results.append((s, p, masked))
        return results


class PerturbationLoop:
    """Localization via Log-Likelihood Perturbation.

    For a flagged anomaly file, tests which (sensor, patch) region is the root
    cause by masking each region and measuring improvement in log-likelihood.

    The patch whose masking causes the BIGGEST increase in log-likelihood is
    the most anomalous (causal root).

    84 forward passes for each flagged file (6 sensors × 14 patches).

    Args:
        encoder: Trained encoder model (produces z_pooled from raw file)
        cnf: Trained CNF model
        scorer: AnomalyScorer instance
        patch_masker: PatchMasker instance
    """

    def __init__(
        self,
        encoder: nn.Module,
        cnf: nn.Module,
        scorer: AnomalyScorer,
        patch_masker: PatchMasker,
    ) -> None:
        self.encoder: nn.Module = encoder
        self.cnf: nn.Module = cnf
        self.scorer: AnomalyScorer = scorer
        self.patch_masker: PatchMasker = patch_masker

    def localize(self, raw_file: torch.Tensor) -> torch.Tensor:
        """Compute perturbation heatmap for a single file.

        Args:
            raw_file: [B, num_sensors, seq_len] or [num_sensors, seq_len]

        Returns:
            heatmap: [num_sensors, num_patches] tensor of log-likelihood delta
                    heatmap[s, p] = improvement when patch (s,p) is masked
                    Higher = more anomalous
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            z_pooled = self.encoder(raw_file)
            original_ll, _ = self.scorer(z_pooled)
            if original_ll.ndim == 0:
                original_ll = original_ll.unsqueeze(0)

        heatmap = torch.zeros(
            self.patch_masker.num_sensors,
            self.patch_masker.num_patches,
            device=raw_file.device,
            dtype=raw_file.dtype,
        )

        masks = self.patch_masker.all_masks(raw_file)
        for sensor, patch, masked_file in masks:
            z_masked = self.encoder(masked_file)
            masked_ll, _ = self.scorer(z_masked)
            if masked_ll.ndim == 0:
                masked_ll = masked_ll.unsqueeze(0)
            delta = masked_ll - original_ll
            heatmap[sensor, patch] = delta

        return heatmap

    def localize_batch(self, files: torch.Tensor) -> list[torch.Tensor]:
        """Localize a batch of files sequentially.

        Args:
            files: [B, num_sensors, seq_len]

        Returns:
            List of B heatmaps, each [num_sensors, num_patches]
        """
        return [self.localize(files[i]) for i in range(files.shape[0])]