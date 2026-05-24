"""Dual-stream patch encoders and positional embeddings (V7 models).

Provides the two independent patch encoder heads for the global and local
streams:

*   :class:`PatchEncoderGlobal` — projects globally-normalised patches to D/2.
*   :class:`PatchEncoderLocal` — projects locally-normalised patches to D/2.
*   :class:`PositionalEmbedding` — learnable position encoding for the
    concatenated ``[B, 84, D]`` token sequence.

The two encoders have **independent** weights, enabling the global stream to
capture file-level drift while the local stream captures spike patterns.
"""

import torch
import torch.nn as nn


class PatchEncoderGlobal(nn.Module):
    """Global stream patch encoder — encodes [B, C*P, patch_size] → [B, C*P, D/2].

    Takes flattened patches from the global stream (after GlobalRevIN + Patchify).
    Projects from patch_size=32 to D/2=64 via Linear + LayerNorm + GELU.

    This encoder is responsible for capturing file-level drift and baseline patterns.
    It must NOT share weights with PatchEncoderLocal — each stream learns independently.

    Args:
        patch_size: Input patch dimension (default 32).
        d_half: Output embedding dimension per stream (default 64, for D=128 total).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half
        self.linear = nn.Linear(patch_size, d_half)
        self.layernorm = nn.LayerNorm(d_half)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens.

        Args:
            tokens: [B, C*P, patch_size] e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half] e.g. [B, 84, 64]
        """
        out = self.linear(tokens)
        out = self.layernorm(out)
        out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"


class PatchEncoderLocal(nn.Module):
    """Local stream patch encoder — encodes [B, C*P, patch_size] → [B, C*P, D/2].

    Takes flattened patches from the local stream (after PerPatchRevIN + Patchify).
    Projects from patch_size=32 to D/2=64 via Linear + LayerNorm + GELU.

    This encoder is responsible for capturing local spike patterns. It has
    INDEPENDENT weights from PatchEncoderGlobal — no weight sharing.
    The two streams must learn disentangled representations.

    Args:
        patch_size: Input patch dimension (default 32).
        d_half: Output embedding dimension per stream (default 64).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half
        self.linear = nn.Linear(patch_size, d_half)
        self.layernorm = nn.LayerNorm(d_half)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens.

        Args:
            tokens: [B, C*P, patch_size] e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half] e.g. [B, 84, 64]
        """
        out = self.linear(tokens)
        out = self.layernorm(out)
        out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"


class PositionalEmbedding(nn.Module):
    """2D axial positional embedding for (sensor, patch) tokens.

    Decomposes position into two independent learnable embeddings:
    - sensor_embedding: learns a vector per sensor (e.g., pressure vs flow)
    - patch_embedding: learns a vector per time patch (temporal position)

    The final positional embedding at token (s, p) is:
        pos_emb[s, p] = sensor_emb[s] + patch_emb[p]

    This 2D axial structure is more expressive than a flat 1D index because
    tokens at the same time-patch but different sensors share the same
    patch_embedding, and vice versa (ViViT-style axial attention, 2021).

    Args:
        num_sensors: Number of sensor channels (default 6).
        num_patches: Number of time positions / patches per sensor (default 14).
        d_model: Model dimension (default 128).
    """

    def __init__(self, num_sensors: int = 6, num_patches: int = 14, d_model: int = 128) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_patches = num_patches
        self.d_model = d_model
        self.sensor_embedding = nn.Embedding(num_sensors, d_model)
        self.patch_embedding = nn.Embedding(num_patches, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add 2D axial positional embedding to tokens.

        Args:
            tokens: [B, num_sensors * num_patches, d_model]

        Returns:
            tokens + pos_embedding: [B, num_sensors * num_patches, d_model]
        """
        B, N, D = tokens.shape
        S, P = self.num_sensors, self.num_patches

        sensor_ids = torch.arange(S, device=tokens.device).unsqueeze(1).expand(S, P)
        patch_ids = torch.arange(P, device=tokens.device).unsqueeze(0).expand(S, P)

        sensor_emb = self.sensor_embedding(sensor_ids)  # [S, P, D]
        patch_emb = self.patch_embedding(patch_ids)     # [S, P, D]

        pos_emb = sensor_emb + patch_emb                # [S, P, D]
        pos_emb = pos_emb.reshape(1, N, D).expand(B, -1, -1)  # [B, N, D]

        return tokens + pos_emb

    def extra_repr(self) -> str:
        return f"num_sensors={self.num_sensors}, num_patches={self.num_patches}, d_model={self.d_model}"


class PatchEncoderCNNGlobal(nn.Module):
    """CNN-based global stream patch encoder.

    Replaces the Linear projection in PatchEncoderGlobal with a
    depthwise-separable 1D CNN to capture local temporal patterns
    (ramps, trends) within each 32-step patch before Transformer.

    Architecture per patch:
        [1, patch_size]
        → Conv1d(1→16, k=3, pad=1) → GELU
        → Conv1d(16→32, k=3, pad=1) → GELU
        → flatten → Linear(32*patch_size, d_half) → LayerNorm

    Args:
        patch_size: Number of timesteps per patch (default 32).
        d_half: Output embedding dimension (default 64, for D=128 total).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * patch_size, d_half)
        self.norm = nn.LayerNorm(d_half)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens with CNN.

        Args:
            tokens: [B, C*P, patch_size]  e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half]  e.g. [B, 84, 64]
        """
        B, N, L = tokens.shape
        # Reshape to process each token independently through CNN
        x = tokens.reshape(B * N, 1, L)          # [B*N, 1, L]
        x = self.conv(x)                          # [B*N, 32, L]
        x = x.reshape(B * N, 32 * L)             # [B*N, 32*L]
        x = self.proj(x)                          # [B*N, d_half]
        x = self.norm(x)
        return x.reshape(B, N, self.d_half)       # [B, N, d_half]

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"


class PatchEncoderCNNLocal(nn.Module):
    """CNN-based local stream patch encoder.

    Same CNN architecture as PatchEncoderCNNGlobal but with INDEPENDENT
    weights. The local stream receives PerPatchRevIN-normalized patches
    where spikes are amplified — the CNN kernel naturally captures
    sharp discontinuities better than a Linear projection.

    Architecture per patch:
        [1, patch_size]
        → Conv1d(1→16, k=3, pad=1) → GELU
        → Conv1d(16→32, k=3, pad=1) → GELU
        → flatten → Linear(32*patch_size, d_half) → LayerNorm

    Args:
        patch_size: Number of timesteps per patch (default 32).
        d_half: Output embedding dimension (default 64).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * patch_size, d_half)
        self.norm = nn.LayerNorm(d_half)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens with CNN.

        Args:
            tokens: [B, C*P, patch_size]  e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half]  e.g. [B, 84, 64]
        """
        B, N, L = tokens.shape
        x = tokens.reshape(B * N, 1, L)          # [B*N, 1, L]
        x = self.conv(x)                          # [B*N, 32, L]
        x = x.reshape(B * N, 32 * L)             # [B*N, 32*L]
        x = self.proj(x)                          # [B*N, d_half]
        x = self.norm(x)
        return x.reshape(B, N, self.d_half)       # [B, N, d_half]

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"