import pytest
import torch

from app.anomaly_detection.criterions.masked_reconstruction import MaskedReconstructionLoss
from app.anomaly_detection.criterions.association_discrepancy import AssociationDiscrepancyLoss


class TestMaskedReconstructionLoss:
    def setup_method(self):
        self.loss_fn = MaskedReconstructionLoss()

    def test_returns_dict_with_loss(self):
        pred = torch.randn(2, 84, 32)
        target = torch.randn(2, 84, 32)
        mask = torch.ones(2, 84, dtype=torch.bool)
        out = self.loss_fn(pred, target, mask)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar

    def test_zero_loss_when_pred_equals_target_on_masked(self):
        pred = torch.randn(2, 84, 32)
        target = pred.clone()
        mask = torch.zeros(2, 84, dtype=torch.bool)
        mask[:, :10] = True
        out = self.loss_fn(pred, target, mask)
        assert out["loss"].item() < 1e-6

    def test_gradient_flows_to_pred(self):
        pred = torch.randn(2, 84, 32, requires_grad=True)
        target = torch.randn(2, 84, 32)
        mask = torch.ones(2, 84, dtype=torch.bool)
        out = self.loss_fn(pred, target, mask)
        out["loss"].backward()
        assert pred.grad is not None

    def test_unmasked_positions_excluded(self):
        """Loss should be near zero if pred=target on masked positions,
        even if they differ on unmasked positions."""
        B, N, L = 2, 84, 32
        pred = torch.zeros(B, N, L)
        target = torch.zeros(B, N, L)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :5] = True  # only first 5 positions masked and pred==target there

        # Make unmasked positions very different — should NOT affect loss
        pred[:, 5:, :] = 1000.0
        target[:, 5:, :] = -1000.0

        out = self.loss_fn(pred, target, mask)
        assert out["loss"].item() < 1e-6, "Loss should be ~0 when pred==target on masked positions"

    def test_all_unmasked_gives_near_zero_loss(self):
        """When no positions are masked, loss should be near zero (no denominator)."""
        pred = torch.randn(2, 84, 32)
        target = torch.randn(2, 84, 32)
        mask = torch.zeros(2, 84, dtype=torch.bool)
        out = self.loss_fn(pred, target, mask)
        assert out["loss"].item() < 1e-6

    def test_get_metadata(self):
        meta = self.loss_fn.get_metadata()
        assert meta["type"] == "MaskedReconstructionLoss"


class TestAssociationDiscrepancyLoss:
    def setup_method(self):
        self.loss_fn = AssociationDiscrepancyLoss(lambda_assoc=2.0)

    def test_returns_dict_with_loss(self):
        disc = torch.rand(2, 4, 84)  # [B, L, N]
        out = self.loss_fn(disc)
        assert "loss" in out
        assert out["loss"].ndim == 0

    def test_loss_equals_mean(self):
        disc = torch.ones(2, 4, 84) * 3.0
        out = self.loss_fn(disc)
        assert abs(out["loss"].item() - 3.0) < 1e-5

    def test_gradient_flows(self):
        disc = torch.rand(2, 4, 84, requires_grad=True)
        out = self.loss_fn(disc)
        out["loss"].backward()
        assert disc.grad is not None

    def test_get_metadata(self):
        meta = self.loss_fn.get_metadata()
        assert meta["type"] == "AssociationDiscrepancyLoss"
        assert meta["lambda_assoc"] == 2.0
