import pytest
import torch

from app.anomaly_detection.layers.encoders import (
    PatchEncoderGlobal,
    PatchEncoderLocal,
    PositionalEmbedding,
)


class TestPatchEncoderGlobal:
    def test_output_shape(self):
        enc = PatchEncoderGlobal(patch_size=32, d_half=64)
        tokens = torch.randn(4, 84, 32)
        out = enc(tokens)
        assert out.shape == (4, 84, 64)

    def test_independent_weights(self):
        """Verify that global and local encoders have different weights."""
        enc_g = PatchEncoderGlobal(patch_size=32, d_half=64)
        enc_l = PatchEncoderLocal(patch_size=32, d_half=64)

        w_g = enc_g.linear.weight.data.clone()
        w_l = enc_l.linear.weight.data.clone()
        assert not torch.allclose(w_g, w_l)

    def test_shape_varied(self):
        enc = PatchEncoderGlobal(patch_size=32, d_half=128)
        tokens = torch.randn(2, 30, 32)
        out = enc(tokens)
        assert out.shape == (2, 30, 128)


class TestPatchEncoderLocal:
    def test_output_shape(self):
        enc = PatchEncoderLocal(patch_size=32, d_half=64)
        tokens = torch.randn(4, 84, 32)
        out = enc(tokens)
        assert out.shape == (4, 84, 64)

    def test_layernorm_applied(self):
        enc = PatchEncoderLocal(patch_size=32, d_half=64)
        tokens = torch.randn(4, 84, 32) * 10 + 5
        out = enc(tokens)
        assert out.std() < tokens.std() * 2  # layernorm should reduce extreme values

    def test_activation_applied(self):
        enc = PatchEncoderLocal(patch_size=32, d_half=64)
        tokens = torch.randn(4, 84, 32)
        out = enc(tokens)
        assert not torch.isnan(out).any()


class TestPositionalEmbedding:
    def test_output_shape_preserved(self):
        pos = PositionalEmbedding(num_tokens=84, d_model=128)
        tokens = torch.randn(4, 84, 128)
        out = pos(tokens)
        assert out.shape == (4, 84, 128)

    def test_pos_embedding_added(self):
        pos = PositionalEmbedding(num_tokens=84, d_model=128)
        tokens = torch.zeros(4, 84, 128)
        out = pos(tokens)
        assert not torch.allclose(out, tokens)  # pos embedding changed zeros

    def test_same_input_same_output_across_batches(self):
        pos = PositionalEmbedding(num_tokens=84, d_model=128)
        tokens = torch.randn(1, 84, 128).expand(8, -1, -1).clone()  # same input for all batches
        out = pos(tokens)
        assert torch.allclose(out[0], out[1], atol=1e-6)  # same pos embedding for all batches

    def test_num_tokens_varied(self):
        pos = PositionalEmbedding(num_tokens=30, d_model=64)
        tokens = torch.randn(4, 30, 64)
        out = pos(tokens)
        assert out.shape == (4, 30, 64)