import pytest
from app.anomaly_detection.schemas import GLCFlowConfig, RevINStat, PatchStats


class TestGLCFlowConfig:
    def test_default_values(self):
        config = GLCFlowConfig()
        assert config.num_sensors == 6
        assert config.seq_len == 240
        assert config.patch_size == 32
        assert config.patch_stride == 16
        assert config.d_model == 128

    def test_d_half_computed(self):
        config = GLCFlowConfig()
        assert config.d_half == 64

    def test_num_patches_computed(self):
        config = GLCFlowConfig()
        expected = (240 - 32) // 16 + 1
        assert config.num_patches == expected

    def test_custom_values(self):
        config = GLCFlowConfig(d_model=256, transformer_layers=6, dropout=0.2)
        assert config.d_model == 256
        assert config.d_half == 128
        assert config.transformer_layers == 6
        assert config.dropout == 0.2

    def test_d_half_d_model_even(self):
        config = GLCFlowConfig(d_model=128)
        assert config.d_model % 2 == 0

    def test_rev_in_stat(self):
        stat = RevINStat(mean=1.5, std=0.8)
        assert stat.mean == 1.5
        assert stat.std == 0.8

    def test_patch_stats(self):
        stats = PatchStats(
            file_mean=0.5,
            file_std=0.1,
            file_slope=0.02,
            file_peak=1.0,
            file_rms=0.7,
            file_range=0.8,
            patch_max=0.9,
            patch_min=0.1,
        )
        assert stats.file_mean == 0.5
        assert stats.patch_max == 0.9