import pytest
import torch

from app.anomaly_detection.layers.anomaly_attention import (
    AnomalyAttentionBlock,
    AnomalyAttentionTransformer,
)


class TestAnomalyAttentionBlock:
    def setup_method(self):
        self.block = AnomalyAttentionBlock(
            d_model=64, n_heads=4, n_tokens=84, ffn_dim=128, dropout=0.0
        )

    def test_output_keys(self):
        x = torch.randn(2, 84, 64)
        out = self.block(x)
        assert set(out.keys()) == {"out", "discrepancy", "A_series", "A_prior"}

    def test_output_shapes(self):
        B, N, D = 2, 84, 64
        x = torch.randn(B, N, D)
        out = self.block(x)
        assert out["out"].shape == (B, N, D)
        assert out["discrepancy"].shape == (B, N)
        assert out["A_series"].shape == (B, 4, N, N)  # n_heads=4
        assert out["A_prior"].shape == (B, 4, N, N)

    def test_a_prior_rows_sum_to_one(self):
        x = torch.randn(2, 84, 64)
        out = self.block(x)
        row_sums = out["A_prior"].sum(dim=-1)  # [B, H, N]
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_a_series_rows_sum_to_one(self):
        x = torch.randn(2, 84, 64)
        out = self.block(x)
        row_sums = out["A_series"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_discrepancy_non_negative(self):
        x = torch.randn(2, 84, 64)
        out = self.block(x)
        assert (out["discrepancy"] >= 0).all(), "Symmetric KL must be non-negative"

    def test_detach_series_freezes_qkv_gradients(self):
        """With detach_mode='series', Q/K/V should NOT receive gradients."""
        block = AnomalyAttentionBlock(
            d_model=32, n_heads=2, n_tokens=10, ffn_dim=64, dropout=0.0
        )
        x = torch.randn(2, 10, 32, requires_grad=False)
        out = block.forward_with_detach_mode(x, detach="series")
        discrepancy = out["discrepancy"]
        discrepancy.mean().backward()
        # sigma must have gradient (it's in the maximize path)
        assert block.sigma.grad is not None
        # Q/K/V linear layers must NOT have gradients from this pass
        assert block.W_q.weight.grad is None or (block.W_q.weight.grad == 0).all()

    def test_detach_prior_freezes_sigma_gradient(self):
        """With detach_mode='prior', sigma should NOT receive gradients."""
        block = AnomalyAttentionBlock(
            d_model=32, n_heads=2, n_tokens=10, ffn_dim=64, dropout=0.0
        )
        x = torch.randn(2, 10, 32, requires_grad=False)
        out = block.forward_with_detach_mode(x, detach="prior")
        out["discrepancy"].mean().backward()
        # sigma must NOT have gradient from this pass
        assert block.sigma.grad is None or (block.sigma.grad == 0).all()


class TestAnomalyAttentionTransformer:
    def test_output_keys(self):
        transformer = AnomalyAttentionTransformer(
            num_layers=2, n_heads=2, d_model=32, n_tokens=10, ffn_dim=64, dropout=0.0
        )
        x = torch.randn(2, 10, 32)
        out = transformer(x)
        assert set(out.keys()) == {"out", "discrepancies"}

    def test_output_shapes(self):
        L, B, N, D = 3, 2, 14, 64
        transformer = AnomalyAttentionTransformer(
            num_layers=L, n_heads=4, d_model=D, n_tokens=N, ffn_dim=128, dropout=0.0
        )
        x = torch.randn(B, N, D)
        out = transformer(x)
        assert out["out"].shape == (B, N, D)
        assert out["discrepancies"].shape == (B, L, N)

    def test_discrepancies_non_negative(self):
        transformer = AnomalyAttentionTransformer(
            num_layers=2, n_heads=2, d_model=32, n_tokens=10, ffn_dim=64, dropout=0.0
        )
        x = torch.randn(3, 10, 32)
        out = transformer(x)
        assert (out["discrepancies"] >= 0).all()

    def test_forward_with_detach_mode_none(self):
        transformer = AnomalyAttentionTransformer(
            num_layers=2, n_heads=2, d_model=32, n_tokens=10, ffn_dim=64, dropout=0.0
        )
        x = torch.randn(2, 10, 32)
        out = transformer.forward_with_detach_mode(x, detach="none")
        assert out["out"].shape == (2, 10, 32)
        assert out["discrepancies"].shape == (2, 2, 10)
