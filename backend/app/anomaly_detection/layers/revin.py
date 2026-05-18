"""Reversible Instance Normalisation (RevIN) layers.

Provides two normalisation strategies used by V7 models:

*   :class:`GlobalRevIN` — per-file z-score normalisation over the full
    sequence length (240 steps).  Catches baseline drift without amplifying
    small deviations on flat lines.
*   :class:`PerPatchRevIN` — per-patch min-max normalisation to ``[-1, 1]``.
    Preserves the amplitude of local spikes relative to the patch's own range.

Helper functions :func:`compute_file_stats` and :func:`compute_patch_stats`
produce auxiliary targets for the V7 stage-1 training heads.
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


def compute_file_stats(x: torch.Tensor) -> torch.Tensor:
    """Compute file-level statistics for auxiliary head target.

    Computes: mean, std, min, max, kurtosis proxy, slope
    for each sensor channel over the full sequence.

    Args:
        x: Input tensor [B, num_sensors, seq_len]

    Returns:
        stats: [B, num_sensors, 6] — per-sensor file-level stats
    """
    B, C, T = x.shape

    mu    = x.mean(dim=-1)                                          # [B, C]
    sigma = x.std(dim=-1)                                          # [B, C]
    mn    = x.amin(dim=-1)                                        # [B, C]
    mx    = x.amax(dim=-1)                                        # [B, C]

    x_c   = x - mu.unsqueeze(-1)
    kurt  = (x_c ** 4).mean(dim=-1) / (sigma ** 4 + 1e-6)        # [B, C]

    t     = torch.linspace(-1, 1, T, device=x.device).unsqueeze(0).unsqueeze(0)
    slope = (x_c * t).mean(dim=-1) / (t ** 2).mean()              # [B, C]

    return torch.stack([mu, sigma, mn, mx, kurt, slope], dim=-1)   # [B, C, 6]


def compute_patch_stats(x: torch.Tensor, num_patches: int, patch_size: int, patch_stride: int) -> torch.Tensor:
    """Compute patch-level statistics for local auxiliary head target.

    For each (sensor, patch), computes max and min values.

    Args:
        x: Input tensor [B, num_sensors, seq_len]
        num_patches: Number of patches
        patch_size: Size of each patch
        patch_stride: Stride between patches

    Returns:
        stats: [B, num_sensors, num_patches, 2] — per-(sensor, patch) (max, min)
    """
    B, C, T = x.shape

    patches = x.unfold(-1, patch_size, patch_stride)  # [B, C, num_patches, patch_size]
    patch_max = patches.amax(dim=-1)  # [B, C, num_patches]
    patch_min = patches.amin(dim=-1)  # [B, C, num_patches]

    return torch.stack([patch_max, patch_min], dim=-1)  # [B, C, num_patches, 2]