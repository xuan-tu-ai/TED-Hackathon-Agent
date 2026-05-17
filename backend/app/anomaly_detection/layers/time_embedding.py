"""Sinusoidal time embedding for the Continuous Normalising Flow.

Maps a continuous scalar ``t ∈ [0, 1]`` to a ``dim``-dimensional vector using
sine/cosine functions at exponentially spaced frequencies (same positional
encoding formula as "Attention Is All You Need", applied to continuous time).

Both a module (:class:`SinusoidalTimeEmbedding`) and a convenience function
(:func:`get_time_embedding`) are provided.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for Flow Matching scalar t ∈ [0,1].

    Maps a scalar time value t ∈ [0, 1] to a vector of dimension `dim` using
    sine/cosine base functions at different frequencies:

        PE(t, 2i)   = sin(t * exp(-i * log_scale))
        PE(t, 2i+1) = cos(t * exp(-i * log_scale))

    This is the standard "Attention is All You Need" positional encoding applied
    to continuous time rather than discrete positions, used in CNF/ODE contexts.

    Args:
        dim: Output embedding dimension (must be even, default 32).
    """

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar time values.

        Args:
            t: Scalar time tensor, shape [B] or [B, 1] or broadcastable.
               Values expected in range [0, 1].

        Returns:
            emb: Embedded tensor shape [B, dim]
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 3:
            t = t.squeeze(-1)

        t = t.reshape(-1)
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / (half_dim - 1)
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


def get_time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Functional version of SinusoidalTimeEmbedding.forward().

    Args:
        t: Scalar time tensor [B] or [B, 1]
        dim: Embedding dimension (must be even)

    Returns:
        emb: [B, dim]
    """
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, device=t.device, dtype=t.dtype) / (half_dim - 1)
    )
    args = t * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)