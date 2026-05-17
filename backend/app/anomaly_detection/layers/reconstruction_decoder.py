"""Lightweight reconstruction decoder for the MaskFormer V7.

Predicts per-patch RevIN-normalised patch values from Transformer output
tokens.  Kept shallow (Linear → GELU → Linear) to avoid overfitting and to
keep the decoder's task well-defined — it only needs to invert the patch
encoder projection, not the full RevIN chain.
"""

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder head for masked patch prediction.

    Projects each token from [D] back to [patch_size] to predict the original
    content of the masked patch. Used exclusively during training (masked positions
    form the reconstruction target); at inference used for per-token reconstruction
    error scoring.

    Target convention: the decoder predicts the **per-patch RevIN-normalized** patch
    values (output of PerPatchRevIN applied to the local stream). This means the
    decoder must invert the local PatchEncoder projection, not the full RevIN chain,
    keeping the decoder's task well-defined and scale-free.

    Architecture: Linear(D, D) → GELU → Linear(D, patch_size)
    Matches the "lightweight decoder heads" pattern from dual_head.py.

    Args:
        d_model:    Model dimension D (default 128).
        patch_size: Number of timesteps per patch — decoder output dimension (default 32).
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
            patches_pred: [B, N, patch_size] — predicted patch values for each token
        """
        return self.net(tokens)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, patch_size={self.patch_size}"
