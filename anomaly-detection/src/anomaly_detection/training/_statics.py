"""Standalone helper functions for training-time auxiliary target computation.

These were moved out of ``layers/normalization.py`` because they are functions,
not ``nn.Module`` subclasses — they belong in the training pipeline, not the
layer module.
"""

import torch


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
