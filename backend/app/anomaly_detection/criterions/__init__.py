"""Loss function module for anomaly detection.

All criterion classes inherit from :class:`BaseAnomalyDetectionLoss` and
implement a :meth:`forward` returning a dict with a ``"loss"`` key.

Exports
-------
    BaseAnomalyDetectionLoss
        Abstract base class for all anomaly detection losses.
    FlowMatchingLoss
        Conditional flow matching MSE for CNF training (Stage 2).
    DualHeadLoss, GlobalHead, LocalHead
        Dual-head decoder supervision for encoder pre-training (Stage 1).
    OrthogonalityLoss, orthogonality_loss
        Orthogonality regularisation between global/local subspaces (Stage 1).
    MaskedReconstructionLoss
        Masked-patch MSE for anomaly-attention training (single-stage).
    AssociationDiscrepancyLoss
        Mean association discrepancy for minimax anomaly-attention (single-stage).
"""

from .base import BaseAnomalyDetectionLoss
from .flow_matching import FlowMatchingLoss
from .dual_head import DualHeadLoss, GlobalHead, LocalHead
from .orthogonality import OrthogonalityLoss, orthogonality_loss
from .masked_reconstruction import MaskedReconstructionLoss
from .association_discrepancy import AssociationDiscrepancyLoss

__all__ = [
    "BaseAnomalyDetectionLoss",
    "FlowMatchingLoss",
    "DualHeadLoss",
    "GlobalHead",
    "LocalHead",
    "OrthogonalityLoss",
    "orthogonality_loss",
    "MaskedReconstructionLoss",
    "AssociationDiscrepancyLoss",
]
