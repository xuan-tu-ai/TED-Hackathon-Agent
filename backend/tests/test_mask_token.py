import pytest
import torch

from app.anomaly_detection.layers.mask_token import MaskToken


class TestMaskToken:
    def test_output_shape(self):
        mt = MaskToken(d_model=64)
        tokens = torch.randn(2, 84, 64)
        mask = torch.zeros(2, 84, dtype=torch.bool)
        mask[:, :4] = True
        out = mt(tokens, mask)
        assert out.shape == (2, 84, 64)

    def test_unmasked_positions_unchanged(self):
        mt = MaskToken(d_model=64)
        tokens = torch.randn(2, 84, 64)
        mask = torch.zeros(2, 84, dtype=torch.bool)
        mask[:, :4] = True
        out = mt(tokens, mask)
        # Unmasked positions must be identical
        assert torch.allclose(out[:, 4:, :], tokens[:, 4:, :])

    def test_masked_positions_replaced(self):
        mt = MaskToken(d_model=64)
        tokens = torch.zeros(2, 84, 64)  # all zeros
        mask = torch.zeros(2, 84, dtype=torch.bool)
        mask[:, 0] = True  # mask only position 0
        # Give the mask token a known non-zero value
        with torch.no_grad():
            mt.token.fill_(1.0)
        out = mt(tokens, mask)
        # Position 0 should be filled with 1.0 (mask token value)
        assert torch.allclose(out[:, 0, :], torch.ones(2, 64))
        # Positions 1+ should still be 0
        assert torch.allclose(out[:, 1:, :], torch.zeros(2, 83, 64))

    def test_token_is_learnable_parameter(self):
        mt = MaskToken(d_model=32)
        params = list(mt.parameters())
        assert len(params) == 1
        assert params[0].requires_grad

    def test_all_masked(self):
        mt = MaskToken(d_model=16)
        tokens = torch.randn(3, 10, 16)
        mask = torch.ones(3, 10, dtype=torch.bool)
        out = mt(tokens, mask)
        expected = mt.token.expand(3, 10, 16)
        assert torch.allclose(out, expected)

    def test_none_masked(self):
        mt = MaskToken(d_model=16)
        tokens = torch.randn(3, 10, 16)
        mask = torch.zeros(3, 10, dtype=torch.bool)
        out = mt(tokens, mask)
        assert torch.allclose(out, tokens)
