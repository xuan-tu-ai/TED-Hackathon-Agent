import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class FlowMatchingLoss(BaseAnomalyDetectionLoss):
    """Conditional Flow Matching loss for training CNF.

    Objective:
        L_FM = E_{t~U[0,1], z0~N(0,I), z1~p_data} [ || v_theta(z_t, t) - (z1 - z0) ||^2 ]

    where z_t = (1-t) * z0 + t * z1 (straight-line interpolation).

    The target vector (z1 - z0) is constant for a given (z0, z1) pair — no need
    for score function estimation unlike Diffusion models.

    Args:
        reduction: 'mean' or 'none' (default 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        v_pred: torch.Tensor,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Flow Matching loss.

        Args:
            v_pred: Predicted vector field from CNF [B, D]
            z0: Starting point (noise) [B, D]
            z1: Ending point (data) [B, D]
            t: Time scalar in [0,1] [B, 1]

        Returns:
            loss: Scalar (or per-sample if reduction='none')
        """
        target = z1 - z0  # [B, D] — constant vector, not a function of t
        loss = torch.sum((v_pred - target) ** 2, dim=-1)  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def get_metadata(self) -> dict:
        return {"type": "FlowMatchingLoss", "reduction": self.reduction}


def compute_flow_matching_loss(
    cnf: nn.Module,
    z0: torch.Tensor,
    z1: torch.Tensor,
    time_embedding: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional version of FlowMatchingLoss for a complete training step.

    Args:
        cnf: CNF model that predicts vector field v(z_t, t)
        z0: Noise samples [B, D]
        z1: Data samples [B, D]
        time_embedding: SinusoidalTimeEmbedding module

    Returns:
        loss: Scalar flow matching loss
        v_pred: Predicted vector field for logging
    """
    B = z0.shape[0]
    t = torch.rand(B, device=z0.device, dtype=z0.dtype).unsqueeze(-1)  # [B, 1]
    z_t = (1 - t) * z0 + t * z1  # straight-line interpolation

    t_emb = time_embedding(t)  # [B, time_dim]
    v_pred = cnf(torch.cat([z_t, t_emb], dim=-1))  # [B, D]

    target = z1 - z0
    loss = torch.sum((v_pred - target) ** 2, dim=-1).mean()

    return loss, v_pred