"""Association discrepancy loss for minimax anomaly-attention training.

Computes the mean Jensen-Shannon divergence between the prior and series
association matrices, averaged over the batch, layers, and tokens.  This
module **only** measures the magnitude of the discrepancy; the sign and
stop-gradient logic are applied by the trainer (see below).

Minimax training protocol
-------------------------
The Anomaly-Transformer uses a two-phase minimax loop controlled by the
trainer (not this module):

*   **Phase A (minimise)** — the trainer applies
    :math:`-\\lambda_{\\text{assoc}} \\cdot \\mathcal{L}_{\\text{assoc}}`
    with ``detach_mode="prior"``, so the prior's gradient is detached and only
    the series branch (Q/K/V) is updated as a **discriminator**.
*   **Phase B (maximise)** — the trainer applies
    :math:`+\\lambda_{\\text{assoc}} \\cdot \\mathcal{L}_{\\text{assoc}}`
    with ``detach_mode="series"``, so only the prior (sigma) receives
    gradients as a **generator**.

This adversarial game encourages the model to separate normal and anomalous
patterns in the association space.

Taxonomy
--------
:class:`AssociationDiscrepancyLoss` is a **single-stage, minimax** loss.
"""

import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss


class AssociationDiscrepancyLoss(BaseAnomalyDetectionLoss):
    """Mean association discrepancy over batch, layers, and tokens.

    Computes the average Jensen-Shannon divergence (already computed per-token
    by AnomalyAttentionTransformer) as a scalar loss value.

    JS divergence is bounded in [0, log(2)] ≈ [0, 0.69] for any pair of
    normalized distributions, eliminating the eps-floor pathology that
    inflated symmetric KL and made λ impossible to reason about.

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
