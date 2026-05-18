import pytest
import torch

from app.anomaly_detection.layers.revin import (
    GlobalRevIN,
    PerPatchRevIN,
    compute_file_stats,
    compute_patch_stats,
)


class TestGlobalRevIN:
    def test_forward_shape_preserved(self):
        revin = GlobalRevIN(num_sensors=6)
        x = torch.randn(4, 6, 240)
        x_norm, stats = revin(x)
        assert x_norm.shape == x.shape
        mu, sigma = stats
        assert mu.shape == (4, 6, 1)
        assert sigma.shape == (4, 6, 1)

    def test_inverse_recovers_original(self):
        revin = GlobalRevIN(num_sensors=6)
        x = torch.randn(4, 6, 240)
        x_norm, stats = revin(x)
        x_rec = revin.inverse(x_norm, stats)
        torch.testing.assert_close(x_rec, x, rtol=1e-5, atol=1e-5)

    def test_normalization_makes_zero_mean(self):
        revin = GlobalRevIN(num_sensors=6)
        x = torch.randn(4, 6, 240)
        x_norm, _ = revin(x)
        for c in range(6):
            channel_mean = x_norm[:, c, :].mean()
            assert abs(channel_mean) < 1e-4

    def test_stats_are_tensors(self):
        revin = GlobalRevIN(num_sensors=6)
        x = torch.randn(4, 6, 240)
        _, stats = revin(x)
        mu, sigma = stats
        assert isinstance(mu, torch.Tensor)
        assert isinstance(sigma, torch.Tensor)
        assert mu.shape == sigma.shape


class TestPerPatchRevIN:
    def test_forward_shape_preserved(self):
        revin = PerPatchRevIN(num_sensors=6, num_patches=14, patch_size=32)
        x = torch.randn(4, 6, 14, 32)
        x_norm, stats = revin(x)
        assert x_norm.shape == x.shape
        p_min, p_max = stats
        assert p_min.shape == (4, 6, 14, 1)
        assert p_max.shape == (4, 6, 14, 1)

    def test_inverse_recovers_original(self):
        revin = PerPatchRevIN(num_sensors=6, num_patches=14, patch_size=32)
        x = torch.randn(4, 6, 14, 32)
        x_norm, stats = revin(x)
        x_rec = revin.inverse(x_norm, stats)
        torch.testing.assert_close(x_rec, x, rtol=1e-4, atol=1e-4)

    def test_per_patch_independent_normalization(self):
        revin = PerPatchRevIN(num_sensors=6, num_patches=14, patch_size=32)
        x = torch.randn(4, 6, 14, 32)
        x_norm, stats = revin(x)
        for c in range(6):
            for p in range(14):
                patch_min = x_norm[:, c, p, :].amin(dim=-1)
                patch_max = x_norm[:, c, p, :].amax(dim=-1)
                assert torch.all(patch_max <= 1.0 + 1e-4)
                assert torch.all(patch_min >= -1.0 - 1e-4)

    def test_stats_are_tensors(self):
        revin = PerPatchRevIN(num_sensors=6, num_patches=14, patch_size=32)
        x = torch.randn(4, 6, 14, 32)
        _, stats = revin(x)
        p_min, p_max = stats
        assert isinstance(p_min, torch.Tensor)
        assert isinstance(p_max, torch.Tensor)


class TestComputeFileStats:
    def test_output_shape(self):
        x = torch.randn(4, 6, 240)
        stats = compute_file_stats(x)
        assert stats.shape == (4, 6, 6)  # [B, C, 6]

    def test_stats_mean_nonzero_input(self):
        x = torch.ones(4, 6, 240) * 5.0
        stats = compute_file_stats(x)
        mean_vals = stats[:, :, 0]  # [B, C] — first stat is mean
        torch.testing.assert_close(mean_vals, torch.ones(4, 6) * 5.0, rtol=1e-4, atol=1e-4)

    def test_stats_std_reasonable_range(self):
        x = torch.randn(4, 6, 240) * 2.0 + 1.0
        stats = compute_file_stats(x)
        std_vals = stats[:, :, 1]  # [B, C] — second stat is std
        assert (std_vals > 1.0).all() and (std_vals < 3.0).all()

    def test_stats_contain_min_max(self):
        x = torch.randn(4, 6, 240)
        stats = compute_file_stats(x)
        assert stats.shape == (4, 6, 6)
        mn = stats[:, :, 2]
        mx = stats[:, :, 3]
        assert (mx >= mn).all()


class TestComputePatchStats:
    def test_output_shape(self):
        x = torch.randn(4, 6, 240)
        stats = compute_patch_stats(x, num_patches=14, patch_size=32, patch_stride=16)
        assert stats.shape == (4, 6, 14, 2)  # [B, C, N, 2]

    def test_max_greater_than_min(self):
        x = torch.randn(4, 6, 240)
        stats = compute_patch_stats(x, num_patches=14, patch_size=32, patch_stride=16)
        max_vals = stats[:, :, :, 0]  # [B, C, N]
        min_vals = stats[:, :, :, 1]  # [B, C, N]
        assert (max_vals >= min_vals).all()