"""Anomaly-Attention (Xu et al., ICLR 2022) for the MaskFormer V7.

Replaces the standard Transformer encoder with a block that computes two
attention matrices:

*   **Series Association** (A_s) — learned multi-head self-attention.
*   **Prior Association** (A_p) — a fixed Gaussian kernel with learnable
    per-head sigma.

The symmetric KL divergence between A_s and A_p per token is the **association
discrepancy** signal: normal tokens have broad A_s (global context) → high
discrepancy; anomalous tokens have narrow A_s (local focus) → low discrepancy.

.. note::
   The minimax training logic (``forward_with_detach_mode``) is the contract
   between this layer and the :class:`MaskFormerTrainer`.  The trainer
   switches ``detach`` between ``"prior"`` (minimise phase) and
   ``"series"`` (maximise phase) each step.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAttentionBlock(nn.Module):
    """Single Anomaly-Attention block (Xu et al., Anomaly Transformer, ICLR 2022).

    Computes standard multi-head self-attention (Series Association A_s) and a fixed
    Gaussian prior (Prior Association A_p), then measures the symmetric KL divergence
    between them as an Association Discrepancy signal per token.

    Key idea:
        - Normal tokens  → A_s is broadly distributed (global associations) → high discrepancy vs local A_p
        - Anomalous tokens → A_s concentrates near neighbors (local)         → low discrepancy vs A_p

    Anomaly score = -Discrepancy (lower discrepancy = more anomalous).

    Minimax training (two-phase per Xu et al.):
        Phase A (minimize): detach_mode="prior"  → sigma is frozen; Q/K/V/decoder get gradients.
                             Loss: L_recon - λ * L_assoc  (minimize both recon error and discrepancy)
        Phase B (maximize): detach_mode="series" → Q/K/V are frozen; sigma gets gradients.
                             Loss: +λ * L_assoc  (push sigma so A_p moves closer to local,
                                                  maximizing the gap with global A_s)
        Inference:          detach_mode="none"   → standard forward, no stop-grad.

    Args:
        d_model:        Model dimension.
        n_heads:        Number of attention heads.
        n_tokens:       Number of tokens in the sequence (used for prior association matrix).
        ffn_dim:        Feed-forward network hidden dimension (default 4×d_model).
        dropout:        Dropout probability.
        initial_sigma:  Initial value of learnable per-head sigma parameters (default 5.0).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_tokens: int = 84,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        initial_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # One sigma per head — learnable scale of the Gaussian prior
        self.sigma = nn.Parameter(torch.ones(n_heads) * initial_sigma)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _prior_association(self, n: int, device: torch.device) -> torch.Tensor:
        """Compute Gaussian prior association matrix [n_heads, n, n]."""
        positions = torch.arange(n, device=device).float()
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [n, n]
        sigma = self.sigma.abs().clamp(min=1e-4)  # [H] — prevent collapse to zero
        # Compute per-head kernel: exp(-dist² / 2σ²)
        kernel = torch.exp(
            -dist.unsqueeze(0) ** 2 / (2 * sigma.view(-1, 1, 1) ** 2)
        )  # [H, n, n]
        # Row-normalize to get a probability distribution
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        return kernel

    def _symmetric_kl(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Symmetric KL divergence KL(p||q) + KL(q||p) per token.

        Args:
            p: [B, H, N, N] row-normalized distributions
            q: [B, H, N, N] row-normalized distributions

        Returns:
            discrepancy: [B, N] averaged over heads
        """
        kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)  # [B, H, N]
        kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)  # [B, H, N]
        discrepancy = (kl_pq + kl_qp).mean(dim=1)  # [B, N] — average over heads
        return discrepancy

    def forward(self, x: torch.Tensor) -> dict:
        """Standard forward pass for inference (no stop-grad).

        Args:
            x: [B, N, D]

        Returns:
            dict with:
                out:         [B, N, D] — Transformer output (residual + layernorm applied)
                discrepancy: [B, N]    — per-token association discrepancy
                A_series:    [B, H, N, N] — learned series association
                A_prior:     [B, H, N, N] — Gaussian prior association
        """
        return self.forward_with_detach_mode(x, detach="none")

    def forward_with_detach_mode(
        self,
        x: torch.Tensor,
        detach: Literal["none", "series", "prior"] = "none",
    ) -> dict:
        """Forward with explicit stop-gradient mode for minimax training.

        detach="none"   — standard inference; all gradients flow freely.
        detach="prior"  — Phase A (minimize): sigma contribution is detached;
                          Q/K/V/decoder learn to reconstruct while discrepancy
                          is minimized w.r.t. the series-association parameters.
        detach="series" — Phase B (maximize): Q/K/V contribution is detached;
                          sigma learns to make A_prior as different from A_series
                          as possible (sigma → smaller → A_prior more local →
                          larger gap with the globally-distributed A_series).

        Args:
            x:      [B, N, D]
            detach: gradient routing mode

        Returns:
            dict with out, discrepancy, A_series, A_prior
        """
        B, N, D = x.shape
        H, dk = self.n_heads, self.d_k

        # Pre-norm
        x_norm = self.norm1(x)

        # Multi-head projections
        Q = self.W_q(x_norm).reshape(B, N, H, dk).transpose(1, 2)  # [B, H, N, dk]
        K = self.W_k(x_norm).reshape(B, N, H, dk).transpose(1, 2)
        V = self.W_v(x_norm).reshape(B, N, H, dk).transpose(1, 2)

        # Series Association (learned self-attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)  # [B, H, N, N]
        A_series = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        A_series = self.attn_drop(A_series)

        # Prior Association (fixed Gaussian kernel, learnable sigma)
        A_prior_base = self._prior_association(N, x.device)  # [H, N, N]
        A_prior = A_prior_base.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, N, N]

        # Stop-grad routing for minimax phases
        if detach == "series":
            # Phase B: only sigma gets gradients; Q/K are frozen
            A_series_for_kl = A_series.detach()
            A_prior_for_kl = A_prior
        elif detach == "prior":
            # Phase A: Q/K get gradients; sigma's contribution is frozen
            A_series_for_kl = A_series
            A_prior_for_kl = A_prior.detach()
        else:
            A_series_for_kl = A_series
            A_prior_for_kl = A_prior

        discrepancy = self._symmetric_kl(A_prior_for_kl, A_series_for_kl)  # [B, N]

        # Attention output
        attn_out = torch.matmul(A_series, V)  # [B, H, N, dk]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        attn_out = self.W_o(attn_out)

        # Residual + LayerNorm (pre-norm style: norm already applied above)
        x = x + attn_out

        # FFN block (pre-norm)
        x = x + self.ffn(self.norm2(x))

        return {
            "out": x,
            "discrepancy": discrepancy,
            "A_series": A_series,
            "A_prior": A_prior,
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_tokens={self.n_tokens}, d_k={self.d_k}"
        )


class AnomalyAttentionTransformer(nn.Module):
    """Stack of AnomalyAttentionBlocks producing per-layer discrepancy scores.

    Each layer produces an [B, N] discrepancy vector. The stacked result
    has shape [B, L, N]. The caller (model or trainer) averages across
    layers to get the final token-level anomaly signal.

    Args:
        num_layers:     Number of AnomalyAttentionBlock layers (default 4).
        n_heads:        Number of attention heads (default 4).
        d_model:        Model dimension (default 128).
        n_tokens:       Number of tokens in the sequence (default 84).
        ffn_dim:        FFN hidden dimension (default 512).
        dropout:        Dropout probability (default 0.1).
        initial_sigma:  Initial sigma for Gaussian prior (default 5.0).
    """

    def __init__(
        self,
        num_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 128,
        n_tokens: int = 84,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        initial_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.layers = nn.ModuleList([
            AnomalyAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_tokens=n_tokens,
                ffn_dim=ffn_dim,
                dropout=dropout,
                initial_sigma=initial_sigma,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        """Forward with standard inference mode (no stop-grad).

        Args:
            x: [B, N, D]

        Returns:
            dict with:
                out:            [B, N, D] — final layer output
                discrepancies:  [B, L, N] — per-layer discrepancy, stacked
        """
        return self.forward_with_detach_mode(x, detach="none")

    def forward_with_detach_mode(
        self,
        x: torch.Tensor,
        detach: Literal["none", "series", "prior"] = "none",
    ) -> dict:
        """Forward with stop-grad routing propagated to all layers.

        Args:
            x:      [B, N, D]
            detach: gradient routing mode passed to each block

        Returns:
            dict with out, discrepancies
        """
        all_discrepancies = []

        for layer in self.layers:
            block_out = layer.forward_with_detach_mode(x, detach=detach)
            x = block_out["out"]
            all_discrepancies.append(block_out["discrepancy"])  # [B, N]

        discrepancies = torch.stack(all_discrepancies, dim=1)  # [B, L, N]

        return {
            "out": x,
            "discrepancies": discrepancies,
        }

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, d_model={self.d_model}, n_tokens={self.n_tokens}"
