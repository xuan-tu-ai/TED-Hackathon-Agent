import torch
import torch.nn as nn


class MaskToken(nn.Module):
    """Learnable [MASK] embedding for masked predictive modeling.

    A single learnable parameter [1, 1, D] that replaces masked token positions
    during training. At inference, masking is disabled (no positions are replaced).

    Convention follows CLSToken in transformer.py: the parameter is [1, 1, D]
    and is broadcast to batch dimension at call time.

    Args:
        d_model: Model dimension (default 128).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked positions with the learnable [MASK] embedding.

        Args:
            tokens: Input token sequence [B, N, D]
            mask:   Boolean mask [B, N] — True means this position is masked

        Returns:
            tokens_masked: [B, N, D] with masked positions replaced by MASK token
        """
        B, N, D = tokens.shape
        mask_emb = self.token.expand(B, N, D)  # [B, N, D]
        mask_3d = mask.unsqueeze(-1).expand_as(tokens)  # [B, N, D]
        return torch.where(mask_3d, mask_emb, tokens)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"
