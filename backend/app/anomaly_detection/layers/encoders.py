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
    """Learnable positional embedding for (sensor, patch) tokens.

    Adds a learnable [num_tokens, d_model] embedding to the concatenated
    global+local tokens after Concat([z_global, z_local], dim=-1).

    For 84 tokens (6 sensors × 14 patches) and D=128:
        pos_embedding: [84, 128]
        tokens: [B, 84, 128] + [B, 84, 128] = [B, 84, 256] → needs projection

    Wait — per spec, Concat produces [B, 84, D] where D = d_half*2 = 128.
    So pos_embedding should be [84, D] = [84, 128].

    Args:
        num_tokens: Number of (sensor, patch) tokens (default 84 = 6*14).
        d_model: Model dimension (default 128).
    """

    def __init__(self, num_tokens: int = 84, d_model: int = 128) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_tokens, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to tokens.

        Args:
            tokens: [B, num_tokens, d_model]

        Returns:
            tokens + pos_embedding: [B, num_tokens, d_model]
        """
        B, N, D = tokens.shape
        positions = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, N)
        pos_emb = self.embedding(positions)  # [B, N, d_model]
        return tokens + pos_emb

    def extra_repr(self) -> str:
        return f"num_tokens={self.num_tokens}, d_model={self.d_model}"