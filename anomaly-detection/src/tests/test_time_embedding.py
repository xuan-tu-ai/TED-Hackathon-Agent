import pytest
import torch
import math

from src.anomaly_detection.layers.time_embedding import (
    SinusoidalTimeEmbedding,
    get_time_embedding,
)


class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 32)

    def test_t_1d(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 32)

    def test_t_2d(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.rand(4, 1)
        out = emb(t)
        assert out.shape == (4, 32)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="must be even"):
            SinusoidalTimeEmbedding(dim=33)

    def test_deterministic(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out1 = emb(t)
        out2 = emb(t)
        torch.testing.assert_close(out1, out2)

    def test_t_values_different_outputs(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(out1, out2)

    def test_0_and_1_symmetry(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t_0 = torch.tensor([0.0])
        t_1 = torch.tensor([1.0])
        out_0 = emb(t_0)
        out_1 = emb(t_1)
        assert not torch.allclose(out_0, out_1)

    def test_functional_same_as_module(self):
        t = torch.rand(4)
        mod_out = SinusoidalTimeEmbedding(dim=32)(t)
        func_out = get_time_embedding(t, dim=32)
        torch.testing.assert_close(mod_out, func_out)

    def test_dim_varied(self):
        for d in [16, 32, 64, 128]:
            emb = SinusoidalTimeEmbedding(dim=d)
            t = torch.rand(4)
            out = emb(t)
            assert out.shape == (4, d)

    def test_freq_range(self):
        """Test that output values are within [-1, 1] (sin/cos property)."""
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.rand(32)
        out = emb(t)
        assert out.min() >= -1.0 and out.max() <= 1.0
