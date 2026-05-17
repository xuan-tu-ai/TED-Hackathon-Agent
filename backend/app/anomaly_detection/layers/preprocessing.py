"""Pre-processing layers for V1–V6 window-models (legacy).

Used by the older (pre-V7) model variants that operate on short windows
(seq_len ≤ 64).  Provides three normalisation modules:

*   :class:`RevIN` — per-window z-score normalisation with denormalisation.
*   :class:`LogScaler` — signed log-transform for statistical features.
*   :class:`GlobalStatsStandardizer` — online z-score via ``BatchNorm1d``.

.. note::
   V7 models (GLC-Flow, MaskFormer) use the modern RevIN layers from
   :mod:`app.anomaly_detection.layers.revin` instead of this module.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Legacy per-window RevIN for short sequences (V1–V6).

    Computes z-score normalisation over the **time dimension** of each window
    independently, per sensor channel.  Statistics are detached so gradients
    do not flow through them.

    Args:
        num_sensors: Number of sensor channels.
        eps: Small constant to prevent division by zero (default 1e-5).
    """

    def __init__(self, num_sensors: int, eps: float = 1e-5):
        super().__init__()
        self.num_sensors = num_sensors
        self.eps = eps

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalise input over the time dimension.

        Args:
            x: Input tensor ``[B, C, W]``.

        Returns:
            Tuple of ``(x_norm [B, C, W], (mean [B, C, 1], std [B, C, 1]))``.
        """
        mean = torch.mean(x, dim=-1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x_norm = (x - mean) / stdev
        return x_norm, (mean, stdev)

    def denormalize(
        self,
        x_norm: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse the normalisation.

        Args:
            x_norm: Normalised tensor ``[B, C, W]``.
            stats: ``(mean, stdev)`` tuple from the forward pass.

        Returns:
            Denormalised tensor ``[B, C, W]``.
        """
        mean, stdev = stats
        return x_norm * stdev + mean


class LogScaler(nn.Module):
    """Signed log-transform for statistical features.

    Maps ``[-∞, ∞] → [-∞, ∞]`` via ``sign(x) * log(1 + |x|)``.
    Reduces skew in heavy-tailed statistical distributions.

    Input shape: ``[B, C, K]``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply signed log1p.

        Args:
            x: Input tensor.

        Returns:
            Log-transformed tensor of the same shape.
        """
        return torch.sign(x) * torch.log1p(torch.abs(x))


class GlobalStatsStandardizer(nn.Module):
    """Online z-score standardisation for statistical features.

    Wraps ``nn.BatchNorm1d`` with ``affine=False`` to accumulate running
    mean and variance across the training set.  This is applied **after**
    :class:`LogScaler`.

    Args:
        num_sensors: Number of sensor channels.
    """

    def __init__(self, num_sensors: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.standarizer = nn.BatchNorm1d(
            num_features=num_sensors, affine=False, track_running_stats=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise input using running statistics.

        Args:
            x: Input ``[B, C, K]`` — log-scaled statistical features.

        Returns:
            Z-scored tensor of the same shape.
        """
        return self.standarizer(x)