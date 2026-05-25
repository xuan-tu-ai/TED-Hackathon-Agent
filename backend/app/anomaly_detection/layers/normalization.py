"""Normalisation layers — RevIN variants, log-scaling, and standardisation.

This module consolidates all normalisation layers previously split across
``revin.py`` and ``preprocessing.py``:

**V7 modern layers** (macro-window models):
*   :class:`GlobalRevIN` — per-file z-score normalisation over the full
    sequence length (240 steps).  Catches baseline drift without amplifying
    small deviations on flat lines.
*   :class:`PerPatchRevIN` — per-patch min-max normalisation to ``[-1, 1]``.
    Preserves the amplitude of local spikes relative to the patch's own range.
**V1–V6 legacy layers** (short-window models):
*   :class:`RevIN` — per-window z-score normalisation with denormalisation
    (legacy, used by pre-V7 models).
*   :class:`LogScaler` — signed log-transform for statistical features.
*   :class:`GlobalStatsStandardizer` — online z-score via ``BatchNorm1d``.
"""

import torch
import torch.nn as nn


class GlobalRevIN(nn.Module):
    """Global RevIN — per-file normalization.

    Computes μ and σ over the ENTIRE sequence (all 240 steps) for each sensor
    channel independently. This catches drift/baseline shift without amplifying
    small deviations on flat lines.

    Forward:
        x_norm[c, t] = (x_raw[c, t] - μ_c) / (σ_c + ε)
        where μ_c, σ_c are computed over all t ∈ [0, 240).

    Inverse:
        x_raw[c, t] = x_norm[c, t] * σ_c + μ_c

    Args:
        num_sensors: Number of sensor channels (default 6).
        eps: Small constant to prevent division by zero (default 1e-5).
    """

    def __init__(self, num_sensors: int = 6, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalize input tensor per sensor over full sequence length.

        Args:
            x: Input tensor [B, num_sensors, seq_len] e.g. [B, 6, 240]

        Returns:
            x_norm: Normalized tensor same shape as input
            stats: Tuple of (mean, std) tensors each [B, num_sensors, 1]
        """
        mu = x.mean(dim=-1, keepdim=True)      # [B, C, 1]
        sigma = x.std(dim=-1, keepdim=True)     # [B, C, 1]
        sigma = torch.where(sigma < self.eps, torch.ones_like(sigma), sigma)
        x_norm = (x - mu) / sigma
        return x_norm, (mu.detach(), sigma.detach())

    def inverse(self, x_norm: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Denormalize using stored statistics.

        Args:
            x_norm: Normalized tensor [B, num_sensors, seq_len]
            stats: Tuple of (mean, std) tensors each [B, num_sensors, 1] from forward pass

        Returns:
            x_raw: Reconstructed raw tensor
        """
        mu, sigma = stats
        return x_norm * sigma + mu


class PerPatchRevIN(nn.Module):
    """Per-Patch RevIN — normalization applied independently to each patch.

    Each patch (32 steps) is normalized using its own min-max scaling to
    [-1, 1] range, which preserves spike amplitude relative to the patch's
    own range rather than diluting it by global statistics.

    This ensures that a local spike of 1-2 steps is not normalized away
    by the surrounding normal steps in the same patch.

    Patch layout after unfold: [B, num_sensors, num_patches, patch_size]
    e.g. [B, 6, 14, 32] for seq_len=240, patch_size=32, patch_stride=16.

    Args:
        num_sensors: Number of sensor channels.
        num_patches: Number of patches per sensor.
        patch_size: Size of each patch (default 32).
        eps: Small constant to prevent division by zero (default 1e-6).
    """

    def __init__(
        self,
        num_sensors: int = 6,
        num_patches: int = 14,
        patch_size: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalize input tensor per (sensor, patch) pair using min-max scaling.

        Args:
            x: Input tensor [B, num_sensors, num_patches, patch_size]

        Returns:
            x_norm: Normalized tensor same shape, scaled to [-1, 1] per patch
            stats: Tuple of (p_min, p_max) tensors each [B, num_sensors, num_patches, 1]
        """
        p_min = x.amin(dim=-1, keepdim=True)   # [B, C, P, 1]
        p_max = x.amax(dim=-1, keepdim=True)   # [B, C, P, 1]
        denom = (p_max - p_min) + self.eps
        x_norm = 2.0 * (x - p_min) / denom - 1.0
        return x_norm, (p_min.detach(), p_max.detach())

    def inverse(
        self, x_norm: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Denormalize using stored per-patch statistics (min-max based).

        Args:
            x_norm: Normalized tensor [B, num_sensors, num_patches, patch_size]
            stats: Tuple of (p_min, p_max) tensors each [B, num_sensors, num_patches, 1]

        Returns:
            x_raw: Reconstructed raw tensor
        """
        p_min, p_max = stats
        denom = (p_max - p_min) + self.eps
        return (x_norm + 1.0) * denom / 2.0 + p_min


# ---------------------------------------------------------------------------
# Legacy V1–V6 layers  (moved from preprocessing.py)
# ---------------------------------------------------------------------------


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


class GlobalSeqNorm(nn.Module):
    """Global sequence normalisation for raw sensor signals [B, C, T].

    Wraps ``nn.BatchNorm1d`` with ``affine=False`` to accumulate running
    mean and variance across the training set **per sensor channel**.
    Unlike :class:`GlobalStatsStandardizer` (designed for statistical
    features [B, C, K]), this class is explicitly named for use at the
    sequence level [B, C, T].

    Running statistics are saved in ``model.state_dict()`` automatically
    (``glob_seq_norm.bn.running_mean``, ``glob_seq_norm.bn.running_var``).
    No explicit "fit" call is needed — stats accumulate during training.

    Args:
        num_sensors: Number of sensor channels C.
    """

    def __init__(self, num_sensors: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.bn = nn.BatchNorm1d(
            num_features=num_sensors, affine=False, track_running_stats=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input [B, C, T] using running statistics per channel.

        Args:
            x: Input ``[B, C, T]``.

        Returns:
            Normalized tensor ``[B, C, T]``.
        """
        return self.bn(x)