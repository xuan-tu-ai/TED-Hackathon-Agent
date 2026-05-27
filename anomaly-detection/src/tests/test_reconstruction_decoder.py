import pytest
import torch

from src.anomaly_detection.layers.reconstruction import CNNDecoder

ReconstructionDecoder = CNNDecoder


class TestReconstructionDecoder:
    def test_output_shape(self):
        dec = ReconstructionDecoder(d_model=128, patch_size=32)
        tokens = torch.randn(2, 84, 128)
        out = dec(tokens)
        assert out.shape == (2, 84, 32)

    def test_output_shape_small(self):
        dec = ReconstructionDecoder(d_model=64, patch_size=16)
        tokens = torch.randn(3, 10, 64)
        out = dec(tokens)
        assert out.shape == (3, 10, 16)

    def test_gradient_flows(self):
        dec = ReconstructionDecoder(d_model=32, patch_size=8)
        tokens = torch.randn(2, 5, 32, requires_grad=True)
        out = dec(tokens)
        out.mean().backward()
        assert tokens.grad is not None
        assert tokens.grad.shape == tokens.shape

    def test_all_positions_decoded(self):
        """Decoder must produce predictions for ALL tokens, not just masked ones."""
        dec = ReconstructionDecoder(d_model=64, patch_size=32)
        tokens = torch.randn(1, 84, 64)
        out = dec(tokens)
        assert out.shape[1] == 84
