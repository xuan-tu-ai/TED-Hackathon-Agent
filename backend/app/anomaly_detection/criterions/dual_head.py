import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class DualHeadLoss(BaseAnomalyDetectionLoss):
    """Dual-head reconstruction loss for auxiliary training of Encoder.

    Two lightweight decoder heads that supervise both streams during Stage 1:
    - GlobalHead: Predicts file-level statistics (mean, std, slope, peak, RMS, range)
      from z_cls → enforces learning of drift/baseline patterns
    - LocalHead: Predicts patch-level statistics (max, min per sensor)
      from z_patches → enforces learning of spike patterns before max pooling

    The heads are necessary because without explicit supervision, the Transformer
    could "smooth away" spike information (CLS token is a summary — max pooling
    without LocalHead supervision could collapse). With separate heads, the
    LocalHead loss ensures z_patches retains patch-level information before the
    max operation.

    Args:
        d_model: Model dimension (default 128).
        num_sensors: Number of sensors (default 6).
        file_stats_dim: Dimension of file-level stats (default 36 = 6*6).
        patch_stats_dim: Dimension of patch-level stats (default 12 = 6*2).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_sensors: int = 6,
        file_stats_dim: int = 36,
        patch_stats_dim: int = 2,
        n_sensors: int = 6,
        n_patches: int = 14,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sensors = num_sensors
        self.file_stats_dim = file_stats_dim
        self.patch_stats_dim = patch_stats_dim

        self.global_head = GlobalHead(d_model=d_model, n_sensors=num_sensors)
        self.local_head = LocalHead(d_model=d_model, n_sensors=n_sensors, n_patches=n_patches)

    def forward(
        self,
        z_cls: torch.Tensor,
        z_patches: torch.Tensor,
        file_stats_target: torch.Tensor,
        patch_stats_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute dual-head loss.

        Args:
            z_cls: CLS token output [B, D]
            z_patches: Patch tokens [B, 84, D]
            file_stats_target: Target file-level stats [B, C, 6]
            patch_stats_target: Target patch-level stats [B, C, N, 2]

        Returns:
            dict with 'loss', 'global_loss', 'local_loss' for logging
        """
        file_pred = self.global_head(z_cls)  # [B, C, 6]
        loss_global = ((file_pred - file_stats_target) ** 2).mean()

        patch_pred = self.local_head(z_patches)  # [B, C, N, 2]
        loss_local = ((patch_pred - patch_stats_target) ** 2).mean()

        loss = loss_global + loss_local

        return {"loss": loss, "global_loss": loss_global, "local_loss": loss_local}

    def get_metadata(self) -> dict:
        return {
            "type": "DualHeadLoss",
            "d_model": self.d_model,
            "num_sensors": self.num_sensors,
            "file_stats_dim": self.file_stats_dim,
            "patch_stats_dim": self.patch_stats_dim,
        }

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, file_stats_dim={self.file_stats_dim}, patch_stats_dim={self.patch_stats_dim}"


class GlobalHead(nn.Module):
    """Global head decoder — predicts file-level stats from CLS token.

    Projects [B, D] → [B, C*6] then reshapes to [B, C, 6], where the 6 channels
    are (mean, std, min, max, kurtosis, slope) per sensor.
    """

    def __init__(self, d_model: int = 128, n_sensors: int = 6) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_sensors * 6),
        )

    def forward(self, z_cls: torch.Tensor) -> torch.Tensor:
        out = self.net(z_cls)  # [B, C*6]
        return out.reshape(-1, self.n_sensors, 6)  # [B, C, 6]


class LocalHead(nn.Module):
    """Local head decoder — predicts per-(sensor, patch) max/min from patch tokens.

    Projects [B, C*N, D] → [B, C*N, 2] then reshapes to [B, C, N, 2], where the
    last dimension is (max, min) per (sensor, patch). No max pooling.
    """

    def __init__(self, d_model: int = 128, n_sensors: int = 6, n_patches: int = 14) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_patches = n_patches
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, z_patches: torch.Tensor) -> torch.Tensor:
        pred = self.net(z_patches)  # [B, C*N, 2]
        B = pred.shape[0]
        return pred.reshape(B, self.n_sensors, self.n_patches, 2)  # [B, C, N, 2]