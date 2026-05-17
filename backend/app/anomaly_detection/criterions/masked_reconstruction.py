"""Masked-patch MSE reconstruction loss for anomaly-attention training.

Computes the mean squared error exclusively on masked token positions.  The
model is penalised only for tokens it was asked to reconstruct; unmasked
(visible) positions are excluded from the gradient:

.. math::

    \\mathcal{L}_{\\text{mask}} =
        \\frac{\\sum_{b,n} \\mathbb{1}_{\\text{masked}}(b,n) \\cdot
              \\| \\hat{x}_{b,n} - x_{b,n} \\|^2}
             {\\bigl(\\sum_{b,n} \\mathbb{1}_{\\text{masked}}(b,n)\\bigr)
              \\cdot L + \\varepsilon}

where :math:`\\mathbb{1}_{\\text{masked}}` is the boolean mask,
:math:`\\hat{x}` are the decoder predictions, :math:`x` are the ground-truth
patches, and :math:`L` is the patch length.

Training paradigm
-----------------
This loss is used in **single-stage** (MaskFormer) training where the
anomaly-attention mechanism jointly handles representation learning and
anomaly separation via a minimax objective.
"""

import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class MaskedReconstructionLoss(BaseAnomalyDetectionLoss):
    """MSE reconstruction loss computed only on masked token positions.

    Computes mean squared error between predicted and target patches
    exclusively for the positions that were masked during training.
    Unmasked positions are excluded from the gradient — the model is
    only penalized for what it was asked to reconstruct.

    Loss formula:
        L = sum(MSE(pred_i, target_i) * mask_i) / (sum(mask_i) * patch_size + eps)

    Args:
        eps: Small constant to prevent division by zero on all-visible batches.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        patches_pred: torch.Tensor,
        patches_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """Compute masked reconstruction loss.

        Args:
            patches_pred:   [B, N, patch_size] — decoder predictions for all tokens
            patches_target: [B, N, patch_size] — ground-truth per-patch-RevIN'd patches
            mask:           BoolTensor [B, N]  — True = position was masked (included in loss)

        Returns:
            dict with 'loss' scalar
        """
        sq_err = (patches_pred - patches_target) ** 2  # [B, N, patch_size]
        mask_float = mask.float().unsqueeze(-1)         # [B, N, 1] — broadcast to patch_size
        masked_sq_err = sq_err * mask_float             # zero out unmasked positions

        num_masked_elements = mask.float().sum() * patches_pred.shape[-1]
        loss = masked_sq_err.sum() / (num_masked_elements + self.eps)

        return {"loss": loss}

    def get_metadata(self) -> dict:
        return {"type": "MaskedReconstructionLoss", "eps": self.eps}

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
