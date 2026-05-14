import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class OrthogonalityLoss(BaseAnomalyDetectionLoss):
    """Orthogonality regularization between global and local subspaces in z_pooled.

    Loss = || Z_g^T Z_l ||_F^2

    where Z_g = z_pooled[:, :D/2] (global half) and Z_l = z_pooled[:, D/2:] (local half).

    This loss enforces that the global and local subspaces are orthogonal —
    i.e., they encode truly different information. Without this, the Transformer
    could allow the two halves to become correlated (information "leaking" between
    streams), making disentangled spike/drift detection less reliable.

    The Gram matrix Z_g^T Z_l has shape [D/2, D/2]. Its Frobenius norm squared
    measures total correlation between the two subspaces.

    Args:
        d_model: Model dimension (default 128).
        lambda_orth: Weight for orthogonality loss (default 0.05).
    """

    def __init__(self, d_model: int = 128, lambda_orth: float = 0.05) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_half = d_model // 2
        self.lambda_orth = lambda_orth

    def forward(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality loss.

        Args:
            z_pooled: Pooled representation [B, 2D]

        Returns:
            loss: Scalar orthogonality loss (weighted by lambda_orth)
        """
        z_g = z_pooled[:, : self.d_half]  # [B, D/2]
        z_l = z_pooled[:, self.d_half :]  # [B, D/2]

        gram = z_g.T @ z_l  # [D/2, D/2]
        loss = (gram ** 2).sum()

        return self.lambda_orth * loss

    def get_metadata(self) -> dict:
        return {
            "type": "OrthogonalityLoss",
            "d_model": self.d_model,
            "lambda_orth": self.lambda_orth,
        }

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_half={self.d_half}, lambda_orth={self.lambda_orth}"


def orthogonality_loss(z_pooled: torch.Tensor, d_model: int = 128) -> torch.Tensor:
    """Functional version of OrthogonalityLoss.forward().

    Args:
        z_pooled: Pooled representation [B, 2D]
        d_model: Model dimension

    Returns:
        loss: Scalar orthogonality loss without lambda weighting
    """
    d_half = d_model // 2
    z_g = z_pooled[:, :d_half]
    z_l = z_pooled[:, d_half:]
    gram = z_g.T @ z_l
    return (gram ** 2).sum()