"""Masking utilities for MAE-style training.

Provides:
- MaskToken: learnable [MASK] embedding for MAE decoders.
- random_masking: random token dropout with shuffle/restore indices.
"""

import torch
import torch.nn as nn


class MaskToken(nn.Module):
    """Learnable [MASK] embedding used in the MAE decoder.

    A single learnable vector ``[1, 1, D]`` that is expanded and placed at
    masked token positions before the decoder sees the full sequence.

    Args:
        d_model: Embedding dimension (default 128).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, batch_size: int, n_tokens: int, device: torch.device) -> torch.Tensor:
        """Expand mask token to [B, n_tokens, D].

        Args:
            batch_size: B
            n_tokens:   Number of token positions to fill.
            device:     Target device.

        Returns:
            Tensor [B, n_tokens, D] filled with the learnable mask token.
        """
        return self.weight.to(device).expand(batch_size, n_tokens, -1)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


def random_masking(
    tokens: torch.Tensor,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly mask a fraction of tokens (per-sample independent masking).

    Implements the MAE random masking strategy (He et al., 2022) adapted for
    1-D sensor patch tokens.  For each sample in the batch a random noise
    vector is drawn, sorted to produce a shuffle permutation, and the first
    ``n_visible`` positions in the shuffled order are kept.

    Args:
        tokens:     [B, N, D] — input token sequence.
        mask_ratio: Fraction of tokens to mask, e.g. 0.45.

    Returns:
        visible:     [B, n_visible, D] — tokens kept by the encoder.
        ids_shuffle: [B, N] — argsort(noise) used to gather visible tokens.
        ids_restore: [B, N] — argsort(ids_shuffle), used to unshuffle.
        mask:        [B, N] float — 1 = masked, 0 = visible (in original order).
    """
    B, N, D = tokens.shape
    n_visible = int(N * (1.0 - mask_ratio))

    # Per-sample random shuffle via noise sort
    noise = torch.rand(B, N, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)          # [B, N]
    ids_restore = torch.argsort(ids_shuffle, dim=1)    # [B, N]

    # Gather visible tokens (first n_visible in shuffled order)
    ids_visible = ids_shuffle[:, :n_visible]           # [B, n_visible]
    visible = torch.gather(
        tokens, 1,
        ids_visible.unsqueeze(-1).expand(-1, -1, D),   # [B, n_visible, D]
    )

    # Binary mask: 0 = visible, 1 = masked (restored to original order)
    mask = torch.ones(B, N, device=tokens.device)
    mask[:, :n_visible] = 0.0
    mask = torch.gather(mask, 1, ids_restore)          # [B, N]

    return visible, ids_shuffle, ids_restore, mask
