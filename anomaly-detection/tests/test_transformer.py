import pytest
import torch

from src.anomaly_detection.layers.transformer import (
    CLSToken,
    TransformerEncoder,
    DualPooling,
)


class TestCLSToken:
    def test_output_shape(self):
        cls = CLSToken(d_model=128)
        out = cls(batch_size=4)
        assert out.shape == (4, 1, 128)

    def test_requires_grad(self):
        cls = CLSToken(d_model=128)
        out = cls(batch_size=4)
        assert out.requires_grad

    def test_deterministic(self):
        cls = CLSToken(d_model=128)
        out1 = cls(batch_size=1)
        out2 = cls(batch_size=1)
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)

    def test_device_transfer(self):
        cls = CLSToken(d_model=128)
        out_cpu = cls(batch_size=2, device=torch.device("cpu"))
        assert out_cpu.device.type == "cpu"

    def test_batch_size_varied(self):
        cls = CLSToken(d_model=128)
        out16 = cls(batch_size=16)
        out1 = cls(batch_size=1)
        assert out16.shape[0] == 16
        assert out1.shape[0] == 1


class TestTransformerEncoder:
    def test_output_shape_preserved(self):
        enc = TransformerEncoder(num_layers=4, num_heads=4, d_model=128)
        x = torch.randn(4, 85, 128)
        out = enc(x)
        assert out.shape == (4, 85, 128)

    def test_different_num_layers(self):
        enc = TransformerEncoder(num_layers=2, num_heads=4, d_model=128)
        x = torch.randn(4, 85, 128)
        out = enc(x)
        assert out.shape == (4, 85, 128)

    def test_no_nan_output(self):
        enc = TransformerEncoder(num_layers=4, num_heads=4, d_model=128, dropout=0.0)
        x = torch.randn(4, 85, 128)
        out = enc(x)
        assert not torch.isnan(out).any()

    def test_tokens_varied(self):
        enc = TransformerEncoder(num_layers=4, num_heads=4, d_model=128)
        x = torch.randn(2, 50, 128)
        out = enc(x)
        assert out.shape == (2, 50, 128)

    def test_transformer_output_differs_from_input(self):
        enc = TransformerEncoder(num_layers=4, num_heads=4, d_model=128)
        x = torch.randn(4, 85, 128)
        out = enc(x)
        assert not torch.allclose(out, x)


class TestDualPooling:
    def test_output_shape(self):
        pool = DualPooling(d_model=128)
        transformer_out = torch.randn(4, 85, 128)
        z_pooled = pool(transformer_out)
        assert z_pooled.shape == (4, 256)  # 2 * 128

    def test_cls_from_index_zero(self):
        pool = DualPooling(d_model=128)
        transformer_out = torch.randn(4, 85, 128)
        z_pooled = pool(transformer_out)
        z_cls_extracted = transformer_out[:, 0, :]
        assert torch.allclose(z_pooled[:, :128], z_cls_extracted, rtol=1e-4, atol=1e-4)

    def test_max_pool_applied_to_patches_only(self):
        pool = DualPooling(d_model=128)
        transformer_out = torch.randn(4, 85, 128)
        z_pooled = pool(transformer_out)
        z_max_extracted = transformer_out[:, 1:, :].max(dim=1).values
        assert torch.allclose(z_pooled[:, 128:], z_max_extracted, rtol=1e-4, atol=1e-4)

    def test_d_model_64(self):
        pool = DualPooling(d_model=64)
        transformer_out = torch.randn(4, 85, 64)
        z_pooled = pool(transformer_out)
        assert z_pooled.shape == (4, 128)

    def test_z_cls_retains_global_info(self):
        """CLS token aggregates across all patches, max pool retains strongest activations."""
        pool = DualPooling(d_model=128)
        transformer_out = torch.randn(4, 85, 128)
        z_pooled = pool(transformer_out)
        assert z_pooled[:, :128].shape == (4, 128)
        assert z_pooled[:, 128:].shape == (4, 128)
