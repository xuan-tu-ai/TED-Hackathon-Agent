import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class AssociationDiscrepancyLoss(BaseAnomalyDetectionLoss):
    """Mean association discrepancy over batch, layers, and tokens.

    Computes the average symmetric KL divergence (already computed per-token
    by AnomalyAttentionTransformer) as a scalar loss value.

    This module ONLY measures the magnitude of the discrepancy.
    The minimax sign and stop-gradient mechanics live in MaskFormerTrainer:

        Phase A (minimize): trainer uses  -lambda_assoc * L_assoc
                            with detach_mode="prior" on the transformer,
                            so sigma's gradient is detached during this phase.
        Phase B (maximize): trainer uses  +lambda_assoc * L_assoc
                            with detach_mode="series", so only sigma gets
                            gradients and Q/K/V are frozen.

    Args:
        lambda_assoc: Weight coefficient applied externally by the trainer
                      (stored here for metadata serialization only).
    """

    def __init__(self, lambda_assoc: float = 2.0) -> None:
        super().__init__()
        self.lambda_assoc = lambda_assoc

    def forward(self, discrepancies: torch.Tensor) -> dict:
        """Compute mean discrepancy loss.

        Args:
            discrepancies: FloatTensor [B, L, N] — per-layer, per-token discrepancy

        Returns:
            dict with 'loss' scalar — average discrepancy (always positive)
        """
        loss = discrepancies.mean()
        return {"loss": loss}

    def get_metadata(self) -> dict:
        return {
            "type": "AssociationDiscrepancyLoss",
            "lambda_assoc": self.lambda_assoc,
        }

    def extra_repr(self) -> str:
        return f"lambda_assoc={self.lambda_assoc}"
