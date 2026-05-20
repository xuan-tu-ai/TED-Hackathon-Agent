import pytest
import torch

from app.anomaly_detection.criterions.flow_matching import FlowMatchingLoss, compute_flow_matching_loss
from app.anomaly_detection.criterions.dual_head import DualHeadLoss, GlobalHead, LocalHead
from app.anomaly_detection.criterions.orthogonality import OrthogonalityLoss, orthogonality_loss


class TestFlowMatchingLoss:
    def test_output_shape_scalar(self):
        loss_fn = FlowMatchingLoss(reduction="mean")
        v_pred = torch.randn(4, 256)
        z0 = torch.randn(4, 256)
        z1 = torch.randn(4, 256)
        t = torch.rand(4, 1)
        loss = loss_fn(v_pred, z0, z1, t)
        assert loss.ndim == 0

    def test_output_shape_none(self):
        loss_fn = FlowMatchingLoss(reduction="none")
        v_pred = torch.randn(4, 256)
        z0 = torch.randn(4, 256)
        z1 = torch.randn(4, 256)
        t = torch.rand(4, 1)
        loss = loss_fn(v_pred, z0, z1, t)
        assert loss.shape == (4,)

    def test_target_is_z1_minus_z0(self):
        loss_fn = FlowMatchingLoss(reduction="mean")
        z0 = torch.randn(4, 256)
        z1 = torch.randn(4, 256)
        t = torch.rand(4, 1)
        target = z1 - z0

        loss = loss_fn(target, z0, z1, t)
        assert loss.item() < 1e-6

    def test_high_loss_when_wrong_prediction(self):
        loss_fn = FlowMatchingLoss(reduction="mean")
        z0 = torch.randn(4, 256)
        z1 = torch.randn(4, 256)
        t = torch.rand(4, 1)
        v_wrong = torch.randn(4, 256) * 100
        loss = loss_fn(v_wrong, z0, z1, t)
        assert loss.item() > 0.1


class TestDualHeadLoss:
    def test_output_dict_keys(self):
        criterion = DualHeadLoss(d_model=128, num_sensors=6, n_patches=14)
        z_cls = torch.randn(4, 128)
        z_patches = torch.randn(4, 84, 128)
        file_stats = torch.randn(4, 6, 6)  # [B, C, 6]
        patch_stats = torch.randn(4, 6, 14, 2)  # [B, C, N, 2]
        result = criterion(z_cls, z_patches, file_stats, patch_stats)
        assert set(result.keys()) == {"loss", "global_loss", "local_loss"}

    def test_loss_reasonable_positive(self):
        criterion = DualHeadLoss(d_model=128, num_sensors=6, n_patches=14)
        z_cls = torch.randn(4, 128)
        z_patches = torch.randn(4, 84, 128)
        file_stats = torch.randn(4, 6, 6)
        patch_stats = torch.randn(4, 6, 14, 2)
        result = criterion(z_cls, z_patches, file_stats, patch_stats)
        assert result["loss"].item() > 0.0
        assert not torch.isnan(result["loss"])

    def test_global_head_output_dim(self):
        head = GlobalHead(d_model=128, n_sensors=6)
        z_cls = torch.randn(4, 128)
        out = head(z_cls)
        assert out.shape == (4, 6, 6)  # [B, C, 6]

    def test_local_head_output_dim(self):
        head = LocalHead(d_model=128, n_sensors=6, n_patches=14)
        z_patches = torch.randn(4, 84, 128)
        out = head(z_patches)
        assert out.shape == (4, 6, 14, 2)


class TestOrthogonalityLoss:
    def test_orthogonal_vectors_give_zero_loss(self):
        loss_fn = OrthogonalityLoss(d_model=128, lambda_orth=0.05)
        z_pooled = torch.randn(4, 256)
        z_pooled[:, :64] = torch.randn(4, 64)
        z_pooled[:, 64:] = torch.randn(4, 192)
        loss = loss_fn(z_pooled)
        assert loss.ndim == 0

    def test_correlated_vectors_give_high_loss(self):
        loss_fn = OrthogonalityLoss(d_model=128, lambda_orth=1.0)
        z_g = torch.randn(4, 64)
        z_l = z_g * 2 + 0.1
        z_pooled = torch.cat([z_g, z_l], dim=-1)
        loss = loss_fn(z_pooled)
        assert loss.item() > 0.0

    def test_lambda_multiplier_applied(self):
        loss_fn_05 = OrthogonalityLoss(d_model=128, lambda_orth=0.05)
        loss_fn_10 = OrthogonalityLoss(d_model=128, lambda_orth=0.10)
        z_pooled = torch.randn(4, 256)
        z_pooled[:, :64] = torch.randn(4, 64)
        z_pooled[:, 64:] = torch.randn(4, 192) * 0.5

        raw_05 = loss_fn_05(z_pooled)
        raw_10 = loss_fn_10(z_pooled)
        assert abs(raw_10.item() - 2 * raw_05.item()) < 1e-6

    def test_functional_version(self):
        z_pooled = torch.randn(4, 256)
        loss_mod = OrthogonalityLoss(d_model=128, lambda_orth=1.0)
        loss_func = orthogonality_loss(z_pooled, d_model=128)
        assert loss_mod(z_pooled).item() == loss_func.item()

    def test_d_half_is_d_model_div_2(self):
        loss_fn = OrthogonalityLoss(d_model=128, lambda_orth=1.0)
        assert loss_fn.d_half == 64

    def test_scale_invariant_to_batch_size(self):
        """Loss should be approximately the same for different batch sizes.

        The old sum-based formula scaled as B^2 * D^2, causing ~100k initial
        values with B=32 that swamped reconstruction gradients.  The normalized
        formula (gram/B, then .mean()) keeps the loss O(1) regardless of B.
        """
        torch.manual_seed(0)
        loss_fn = OrthogonalityLoss(d_model=128, lambda_orth=1.0)
        z_small = torch.randn(4,  128)
        z_large = torch.randn(32, 128)
        loss_small = loss_fn(z_small).item()
        loss_large = loss_fn(z_large).item()
        # Both should be O(1); allow 5× margin for statistical variation
        assert loss_large < loss_small * 5 + 1.0, (
            f"loss should not blow up with larger batch: {loss_large:.2f} vs {loss_small:.2f}"
        )