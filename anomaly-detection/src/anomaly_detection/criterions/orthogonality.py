"""Orthogonality regularisation between global and local latent subspaces.

Let the pooled representation :math:`z_{\\text{pooled}} \\in \\mathbb{R}^{B
\\times 2D}` be split into two halves:

*   :math:`Z_g = z_{\\text{pooled}}[:, :D/2]` — **global** subspace (drift /
    baseline information).
*   :math:`Z_l = z_{\\text{pooled}}[:, D/2:]` — **local** subspace (spike /
    transient information).

The orthogonality loss penalises correlation between the two subspaces:

.. math::

    \\mathcal{L}_{\\text{orth}} =
        \\lambda_{\\text{orth}} \\cdot \\| Z_g^\\top Z_l \\|_F^2

where :math:`\\|\\cdot\\|_F` denotes the Frobenius norm and the Gram matrix
:math:`Z_g^\\top Z_l \\in \\mathbb{R}^{D/2 \\times D/2}`.

**Why this is needed:** Without orthogonality regularisation, the Transformer
could allow information to "leak" between the two halves of the pooled vector,
making disentangled spike-vs-drift detection unreliable.

Taxonomy
--------
:class:`OrthogonalityLoss` is a **Stage-1, auxiliary regularisation** loss.
"""

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

        # Normalize by B so the Gram matrix measures per-sample cross-correlation
        # and use mean (not sum) over elements to keep scale independent of D.
        # Without this, loss scales as B^2 * D^2, causing ~100k initial values that
        # swamp the reconstruction signal for the first 15+ epochs.
        B = z_g.shape[0]
        gram = (z_g.T @ z_l) / B   # [D/2, D/2] — per-sample average
        loss = (gram ** 2).mean()

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
    """Functional version of OrthogonalityLoss.forward() without lambda weighting.

    Args:
        z_pooled: Pooled representation [B, 2D]
        d_model: Model dimension

    Returns:
        loss: Scalar orthogonality loss without lambda weighting
    """
    d_half = d_model // 2
    z_g = z_pooled[:, :d_half]
    z_l = z_pooled[:, d_half:]
    B = z_g.shape[0]
    gram = (z_g.T @ z_l) / B
    return (gram ** 2).mean()