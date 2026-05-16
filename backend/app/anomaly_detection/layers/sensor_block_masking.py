import torch
import torch.nn as nn


class SensorBlockMasking(nn.Module):
    """Sensor-block masking strategy for masked predictive modeling.

    Masks entire time columns across ALL sensors simultaneously, preventing the
    model from learning a trivial cross-sensor shortcut. When time position p is
    masked, all 6 sensor tokens at patch p are hidden — the model must use
    temporal context from other time positions to predict them.

    Contrast with naive random token masking: if only paint_pressure at patch 7
    is masked while paint_flow at patch 7 is visible, the model can just copy
    paint_flow (they are physically correlated). Block masking eliminates this.

    Args:
        num_sensors: Number of sensor channels (default 6).
        num_patches: Number of time positions / patches per sensor (default 14).
        mask_ratio:  Fraction of time positions to mask (default 0.3 → 4 of 14).
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

        Does NOT apply the mask to the tokens — caller passes the returned
        boolean mask to MaskToken.forward() to perform the substitution.

        Args:
            tokens: Token sequence [B, num_sensors * num_patches, D]
                    (used only to infer B and device; content is not modified)

        Returns:
            mask:           BoolTensor [B, N] — True = masked position
            masked_time_ids: LongTensor [B, n_masked] — which time positions were masked
        """
        B, N, _ = tokens.shape
        device = tokens.device

        # Sample random time positions to mask per batch element
        noise = torch.rand(B, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        masked_time_ids = ids_shuffle[:, : self.n_masked]  # [B, n_masked]

        # Expand mask to all sensors: token index = sensor * num_patches + time_pos
        mask = torch.zeros(B, self.num_sensors, self.num_patches, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, :, masked_time_ids[b]] = True
        mask = mask.reshape(B, N)  # [B, num_sensors * num_patches]

        return mask, masked_time_ids

    def extra_repr(self) -> str:
        return (
            f"num_sensors={self.num_sensors}, num_patches={self.num_patches}, "
            f"mask_ratio={self.mask_ratio}, n_masked={self.n_masked}"
        )
