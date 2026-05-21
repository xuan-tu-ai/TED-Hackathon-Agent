"""Masking layers for MaskFormer V7 training.

Consolidates two masking-related modules previously split across files:

*   :class:`MaskToken` — learnable ``[MASK]`` token that replaces masked
    positions during training.
*   :class:`SensorBlockMasking` — masks entire time columns across ALL sensors
    simultaneously to prevent cross-sensor shortcut reconstruction.
"""

import torch
import torch.nn as nn


class MaskToken(nn.Module):
    """Learnable [MASK] embedding for masked predictive modeling.

    A single learnable parameter [1, 1, D] that replaces masked token positions
    during training. At inference, masking is disabled (no positions are replaced).

    Convention follows CLSToken: the parameter is [1, 1, D]
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
            mask:   Boolean mask [B, N] - True means this position is masked

        Returns:
            tokens_masked: [B, N, D] with masked positions replaced by MASK token
        """
        B, N, D = tokens.shape
        mask_emb = self.token.expand(B, N, D)  # [B, N, D]
        mask_3d = mask.unsqueeze(-1).expand_as(tokens)  # [B, N, D]
        return torch.where(mask_3d, mask_emb, tokens)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"


class SensorBlockMasking(nn.Module):
    """Sensor-block masking strategy for masked predictive modeling.

    Masks entire time columns across ALL sensors simultaneously, preventing the
    model from learning a trivial cross-sensor shortcut. When time position p is
    masked, all 6 sensor tokens at patch p are hidden - the model must use
    temporal context from other time positions to predict them.

    Contrast with naive random token masking: if only paint_pressure at patch 7
    is masked while paint_flow at patch 7 is visible, the model can just copy
    paint_flow (they are physically correlated). Block masking eliminates this.

    Args:
        num_sensors: Number of sensor channels (default 6).
        num_patches: Number of time positions / patches per sensor (default 14).
        mask_ratio:  Fraction of time positions to mask (default 0.3 -> 4 of 14).
    """

    def __init__(
        self,
        num_sensors: int = 6,
        num_patches: int = 14,
        mask_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.n_masked = int(num_patches * mask_ratio)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sensor-block mask for a batch of token sequences.

        Does NOT apply the mask to the tokens - caller passes the returned
        boolean mask to MaskToken.forward() to perform the substitution.

        Args:
            tokens: Token sequence [B, num_sensors * num_patches, D]
                    (used only to infer B and device; content is not modified)

        Returns:
            mask:            BoolTensor [B, N] - True = masked position
            masked_time_ids: LongTensor [B, n_masked] - which time positions were masked
        """
        B, N, _ = tokens.shape
        device = tokens.device

        # Sample random time positions to mask per batch element
        noise = torch.rand(B, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        masked_time_ids = ids_shuffle[:, : self.n_masked]  # [B, n_masked]

        # Expand mask to all sensors: token index = sensor * num_patches + time_pos
        mask = torch.zeros(B, self.num_sensors, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(dim=2, index=masked_time_ids.unsqueeze(1).expand(-1, self.num_sensors, -1), value=True)
        mask = mask.reshape(B, N)  # [B, num_sensors * num_patches]

        return mask, masked_time_ids

    def extra_repr(self) -> str:
        return (
            f"num_sensors={self.num_sensors}, num_patches={self.num_patches}, "
            f"mask_ratio={self.mask_ratio}, n_masked={self.n_masked}"
        )
