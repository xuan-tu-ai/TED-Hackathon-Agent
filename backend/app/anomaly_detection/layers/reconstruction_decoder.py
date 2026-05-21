"""Lightweight reconstruction decoder for the MaskFormer V7.

Predicts per-patch RevIN-normalised patch values from Transformer output
tokens.  Provides two variants:

1. **PerTokenDecoder** — ``Linear(D) → GELU → Linear(patch_size)`` per-token.
   Matches the original lightweight decoder head pattern.
2. **CNNDecoder** — adds V6-style depthwise-separable Conv1d temporal refinement
   after per-token decode.  The conv kernel slides across patches per sensor,
   giving the decoder cross-token temporal structure needed to learn waveform
   continuity.
"""

import torch
import torch.nn as nn


class PerTokenDecoder(nn.Module):
    """Per-token linear decoder (original design).

    Projects each token from [D] back to [patch_size] independently.
    Lightweight — matches the "decoder heads" pattern from dual_head.py.

    Args:
        d_model:    Model dimension D (default 128).
        patch_size: Number of timesteps per patch (default 32).
    """

    def __init__(self, d_model: int = 128, patch_size: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode all token positions to patch predictions.

        Args:
            tokens: [B, N, D] — Transformer output tokens

        Returns:
            patches_pred: [B, N, patch_size] — predicted patch values bounded to [-1, 1]
                (PerPatchRevIN targets are always in this range).
        """
        return torch.tanh(self.net(tokens))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, patch_size={self.patch_size}"


class CNNDecoder(nn.Module):
    """V6-style CNN decoder with per-token decode + temporal refinement.

    Two-stage decode:
    1. Per-token: ``Linear(D, D) → GELU → Linear(D, patch_size)`` — same as PerTokenDecoder.
    2. Temporal refinement: reshape ``[B, N, PS] → [B, C, P*PS]`` then apply
       ``Conv1d(C→4C, k=3, groups=C) → SiLU → Conv1d(4C→C, k=1)`` across
       the time dimension per sensor.

    The depthwise conv provides cross-patch interaction, letting the decoder
    learn smooth waveform shapes rather than per-token independent predictions.

    Args:
        num_sensors: Number of sensor channels (C).
        num_patches: Number of patches per sensor (P).
        d_model:     Model dimension D (default 128).
        patch_size:  Number of timesteps per patch (default 32).
    """

    def __init__(
        self,
        num_sensors: int,
        num_patches: int,
        d_model: int = 128,
        patch_size: int = 32,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_patches = num_patches
        self.d_model = d_model
        self.patch_size = patch_size

        # Stage 1: per-token decode
        self.token_decode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )

        # Stage 2: temporal refinement (V6-style depthwise-separable conv)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to patch predictions with temporal refinement.

        Args:
            tokens: [B, N, D] where N = C × P

        Returns:
            patches_pred: [B, N, patch_size]
        """
        B, N, D = tokens.shape
        C, P, PS = self.num_sensors, self.num_patches, self.patch_size

        # Stage 1: per-token decode
        patches = self.token_decode(tokens)                  # [B, N, PS]
        patches = patches.reshape(B, C, P, PS)               # [B, C, P, PS]

        # Stage 2: temporal refinement on continuous signal per sensor
        signal = patches.reshape(B, C, P * PS)               # [B, C, P*PS]
        refined = self.temporal_refine(signal)                # [B, C, P*PS]

        patches_out = refined.reshape(B, C, P, PS)
        # Tanh bounds output to [-1, 1] matching PerPatchRevIN target range
        return torch.tanh(patches_out.reshape(B, N, PS))


# Backward-compatible alias
ReconstructionDecoder = PerTokenDecoder
