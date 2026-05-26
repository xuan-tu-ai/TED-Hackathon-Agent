import pytest
from src.anomaly_detection.models.glc_flow_v7 import GLCFlowConfig


_BASE = {"num_sensors": 6, "seq_len": 240}


class TestGLCFlowConfig:
    def test_default_values(self):
        config = GLCFlowConfig(**_BASE)
        assert config.num_sensors == 6
        assert config.seq_len == 240
        assert config.patch_size == 32
        assert config.patch_stride == 16
        assert config.d_model == 128

    def test_d_half_computed(self):
        config = GLCFlowConfig(**_BASE)
        assert config.d_half == 64

    def test_num_patches_computed(self):
        config = GLCFlowConfig(**_BASE)
        expected = (240 - 32) // 16 + 1
        assert config.num_patches == expected

    def test_custom_values(self):
        config = GLCFlowConfig(**_BASE, d_model=256, transformer_layers=6, dropout=0.2)
        assert config.d_model == 256
        assert config.d_half == 128
        assert config.transformer_layers == 6
        assert config.dropout == 0.2

    def test_d_half_d_model_even(self):
        config = GLCFlowConfig(**_BASE, d_model=128)
        assert config.d_model % 2 == 0