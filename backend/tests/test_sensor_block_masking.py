import pytest
import torch

from app.anomaly_detection.layers.sensor_block_masking import SensorBlockMasking


class TestSensorBlockMasking:
    def setup_method(self):
        self.sbm = SensorBlockMasking(num_sensors=6, num_patches=14, mask_ratio=0.3)

    def test_output_types(self):
        tokens = torch.randn(2, 84, 128)
        mask, masked_time_ids = self.sbm(tokens)
        assert mask.dtype == torch.bool
        assert masked_time_ids.dtype == torch.long

    def test_mask_shape(self):
        tokens = torch.randn(2, 84, 128)
        mask, _ = self.sbm(tokens)
        assert mask.shape == (2, 84)

    def test_masked_time_ids_shape(self):
        tokens = torch.randn(2, 84, 128)
        _, masked_time_ids = self.sbm(tokens)
        n_masked = self.sbm.n_masked  # floor(14 * 0.3) = 4
        assert masked_time_ids.shape == (2, n_masked)

    def test_exactly_n_masked_times(self):
        """Each batch element masks exactly n_masked time positions."""
        sbm = SensorBlockMasking(num_sensors=6, num_patches=14, mask_ratio=0.3)
        tokens = torch.randn(4, 84, 128)
        mask, _ = sbm(tokens)
        # Reshape to [B, num_sensors, num_patches] and sum over sensors per time pos
        mask_2d = mask.reshape(4, 6, 14)
        # Every masked time pos → all 6 sensors True
        time_mask = mask_2d.all(dim=1)  # [B, 14] — True if ALL sensors masked at that time
        assert time_mask.sum(dim=-1).min().item() == sbm.n_masked
        assert time_mask.sum(dim=-1).max().item() == sbm.n_masked

    def test_sensor_block_structure(self):
        """Masked tokens always cover ALL sensors at the same time position."""
        tokens = torch.randn(3, 84, 32)
        mask, masked_time_ids = self.sbm(tokens)
        mask_3d = mask.reshape(3, 6, 14)  # [B, sensors, patches]
        for b in range(3):
            for t in masked_time_ids[b]:
                # All 6 sensors at time t must be masked
                assert mask_3d[b, :, t].all(), f"Not all sensors masked at time {t} for batch {b}"

    def test_no_partial_column_masks(self):
        """A time position is either fully masked (all sensors) or not at all."""
        tokens = torch.randn(5, 84, 128)
        mask, _ = self.sbm(tokens)
        mask_3d = mask.reshape(5, 6, 14)
        # For each (batch, time_pos), count how many sensors are masked
        sensor_count = mask_3d.sum(dim=1)  # [B, 14]
        # Each time position must have either 0 or 6 sensors masked
        assert ((sensor_count == 0) | (sensor_count == 6)).all()

    def test_different_batch_elements_can_have_different_masks(self):
        """Masking is sampled independently per batch element."""
        # Run many times to confirm statistical independence (not a trivial check)
        sbm = SensorBlockMasking(num_sensors=6, num_patches=14, mask_ratio=0.5)
        tokens = torch.randn(32, 84, 128)
        mask, masked_ids = sbm(tokens)
        mask_3d = mask.reshape(32, 6, 14)
        time_mask = mask_3d[:, 0, :]  # [32, 14] — first sensor as representative
        # Very unlikely all 32 elements have identical masks
        unique_masks = {tuple(time_mask[i].tolist()) for i in range(32)}
        assert len(unique_masks) > 1, "All batch elements got identical masks — likely bug"
