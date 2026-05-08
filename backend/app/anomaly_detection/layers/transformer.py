import math
import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """Learnable CLS token prepended to the sequence for pooled representation.

    A single learnable parameter [1, 1, D] that is expanded to [B, 1, D] and
    prepended to the patch tokens before Transformer Encoder:

        [CLS | patch_0 | patch_1 | ... | patch_83]
         [B, 1, D]   [B, 84, D] → [B, 85, D]

    After Transformer, cls_output = output[:, 0, :] captures aggregated
    file-level information through self-attention across all patches.

    Args:
        d_model: Model dimension (default 128).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Create CLS token expanded to batch dimension.

        Args:
            batch_size: Number of sequences in batch.
            device: Target device for the tensor.

        Returns:
            cls_token: [B, 1, d_model]
        """
        cls = self.token.expand(batch_size, -1, -1)  # [B, 1, D]
        if device is not None:
            cls = cls.to(device)
        return cls

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder with configurable depth and heads.

    Architecture:
        Input: [B, num_tokens, d_model] e.g. [B, 85, 128]
        4 TransformerEncoderLayers with:
            - MultiHeadSelfAttention (4 heads, d_model=128, dropout=0.1)
            - FeedForward: Linear(d_model, ffn_dim) → GELU → Linear(ffn_dim, d_model)
            - Residual + LayerNorm after each sub-layer
        Output: [B, num_tokens, d_model]

    Args:
        num_layers: Number of Transformer layers (default 4).
        num_heads: Number of attention heads (default 4).
        d_model: Model dimension (default 128).
        ffn_dim: Feed-forward network dimension (default 256).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        d_model: int = 128,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequences with Transformer.

        Args:
            x: Input tensor [B, num_tokens, d_model]

        Returns:
            output: [B, num_tokens, d_model]
        """
        return self.encoder(x)

    def extra_repr(self) -> str:
        return f"num_layers={self.encoder.num_layers}, num_heads={self.encoder.self_attn.num_heads}, d_model={self.encoder.self_attn.d_model}"


class DualPooling(nn.Module):
    """Dual pooling strategy: CLS token + Max Pool from patch tokens.

    Produces z_pooled = concat([z_cls, z_max_local]) → [B, 2D]

    This captures BOTH global drift (via CLS token, which attends to all patches)
    AND local spike (via max pooling, which preserves the strongest activations
    per dimension even after Transformer layers).

    Without dual pooling, CLS alone could "smooth away" spikes since it is
    a summary token. Max pooling ensures spike information is preserved.

    Input:  transformer_output [B, 1+84, D] (CLS token + 84 patch tokens)
    Output: z_pooled [B, 2D]

    Args:
        d_model: Model dimension (default 128).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        """Apply dual pooling to Transformer output.

        Args:
            transformer_output: [B, 85, D] — CLS token at index 0, patches at 1:85

        Returns:
            z_pooled: [B, 2D] — concatenation of CLS and max-pooled patches
        """
        z_cls = transformer_output[:, 0, :]  # [B, D]
        z_patches = transformer_output[:, 1:, :]  # [B, 84, D]
        z_max_local = z_patches.max(dim=1).values  # [B, D]
        z_pooled = torch.cat([z_cls, z_max_local], dim=-1)  # [B, 2D]
        return z_pooled

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"