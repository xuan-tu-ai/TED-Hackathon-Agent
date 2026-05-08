import torch
import torch.nn as nn


class Patchify(nn.Module):
    """Convert continuous time-series into patch tokens via unfolding.

    Transforms [B, num_sensors, seq_len] → [B, num_sensors, num_patches, patch_size]
    via `tensor.unfold(dim=-1, size=patch_size, step=patch_stride)`.

    For seq_len=240, patch_size=32, patch_stride=16:
        num_patches = (240 - 32) // 16 + 1 = 14

    Input shape:  [B, C, T]   e.g. [B, 6, 240]
    Output shape: [B, C, P, L] e.g. [B, 6, 14, 32]

    The unfolded tensor can then be reshaped for encoding:
        [B, C, P, L] → [B, C*P, L] → Linear(L, D/2) → [B, C*P, D/2]

    Args:
        patch_size: Size of each patch (default 32 steps).
        patch_stride: Step between consecutive patches (default 16).
    """

    def __init__(self, patch_size: int = 32, patch_stride: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unfold input into non-overlapping or overlapping patches.

        Args:
            x: Input tensor [B, num_sensors, seq_len]

        Returns:
            patches: [B, num_sensors, num_patches, patch_size]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got shape {x.shape}")

        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return patches  # [B, C, num_patches, patch_size]

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, patch_stride={self.patch_stride}"


class FlattenPatches(nn.Module):
    """Flatten (sensor, patch) dimensions into a single token dimension.

    Transforms [B, C, P, L] → [B, C*P, L]
    Projection to D_half is done by each stream's PatchEncoder (independent
    32→D/2 projection per stream, enabling disentangled global/local learning).

    Args:
        patch_size: Patch size L (default 32) — kept for extra_repr only.
    """

    def __init__(self, patch_size: int = 32) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Reshape patches to token sequence.

        Args:
            patches: [B, C, P, L] e.g. [B, 6, 14, 32]

        Returns:
            tokens: [B, C*P, L] e.g. [B, 84, 32]
        """
        B, C, P, L = patches.shape
        tokens = patches.reshape(B, C * P, L)  # [B, C*P, L]
        return tokens

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}"


def unfold_to_patches(x: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    """Functional version of Patchify.forward().

    Args:
        x: [B, C, T]
        patch_size: size of each patch
        patch_stride: step between patches

    Returns:
        patches: [B, C, P, patch_size]
    """
    return x.unfold(dimension=-1, size=patch_size, step=patch_stride)


def flatten_to_tokens(patches: torch.Tensor, d_half: int) -> torch.Tensor:
    """Functional version of FlattenPatches.forward().

    Args:
        patches: [B, C, P, L]
        d_half: target embedding dimension per stream

    Returns:
        tokens: [B, C*P, d_half]
    """
    B, C, P, L = patches.shape
    tokens = patches.reshape(B, C * P, L)
    linear = nn.Linear(L, d_half, device=patches.device, dtype=patches.dtype)
    return linear(tokens)