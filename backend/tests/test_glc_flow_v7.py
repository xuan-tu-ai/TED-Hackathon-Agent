import pytest
import torch

from app.anomaly_detection.models.factory import create_glc_flow_v7, count_parameters, get_model_summary
from app.anomaly_detection.schemas import GLCFlowConfig


class TestFactory:
    def test_create_default_model(self):
        model = create_glc_flow_v7()
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_with_config(self):
        config = GLCFlowConfig(d_model=64, transformer_layers=2)
        model = create_glc_flow_v7(config)
        assert model.config.d_model == 64
        assert model.config.transformer_layers == 2

    def test_count_parameters_positive(self):
        model = create_glc_flow_v7()
        n = count_parameters(model)
        assert n > 0

    def test_model_summary_keys(self):
        model = create_glc_flow_v7()
        summary = get_model_summary(model)
        assert "total_params" in summary
        assert "trainable_params" in summary
        assert "by_component" in summary

    def test_model_summary_total_equals_sum(self):
        model = create_glc_flow_v7()
        summary = get_model_summary(model)
        component_sum = sum(summary["by_component"].values())
        assert component_sum == summary["total_params"]


class TestGLCFlowV7Forward:
    def test_forward_output_dict_keys(self):
        model = create_glc_flow_v7()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        expected_keys = [
            "z_pooled", "z_cls", "z_patches", "z_max_local",
            "file_stats_pred", "patch_stats_pred", "transformer_output",
        ]
        assert set(out.keys()) == set(expected_keys)

    def test_forward_shapes(self):
        config = GLCFlowConfig()
        model = create_glc_flow_v7(config)
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        assert out["z_pooled"].shape == (2, 256)  # [B, 2D=2*128]
        assert out["z_cls"].shape == (2, 128)       # [B, D]
        assert out["z_patches"].shape == (2, 84, 128)  # [B, 84, D]
        assert out["z_max_local"].shape == (2, 128)    # [B, D]
        assert out["file_stats_pred"].shape == (2, 6, 6)  # [B, C, 6]
        assert out["patch_stats_pred"].shape == (2, 6, 14, 2)  # [B, C, N, 2]
        assert out["transformer_output"].shape == (2, 85, 128)  # [B, 85, D]

    def test_forward_single_sample(self):
        model = create_glc_flow_v7()
        x = {"input": torch.randn(1, 6, 240)}
        out = model(x)
        assert out["z_pooled"].shape == (1, 256)

    def test_forward_different_batch_sizes(self):
        model = create_glc_flow_v7()
        for B in [1, 4, 8]:
            x = {"input": torch.randn(B, 6, 240)}
            out = model(x)
            assert out["z_pooled"].shape[0] == B

    def test_detect_anomaly_returns_dict(self):
        model = create_glc_flow_v7()
        raw = torch.randn(1, 6, 240)
        result = model.detect_anomaly(raw)
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "z_pooled" in result

    def test_detect_anomaly_single_file(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240)  # 2D input (no batch dim)
        result = model.detect_anomaly(raw)
        assert result["anomaly_score"].ndim == 0  # scalar
        assert isinstance(result["is_anomaly"], bool)

    def test_stage1_loss_dict(self):
        config = GLCFlowConfig()
        model = create_glc_flow_v7(config)
        x = {"input": torch.randn(2, 6, 240)}
        file_stats = torch.randn(2, 6, 6)  # [B, C, 6]
        patch_stats = torch.randn(2, 6, 14, 2)  # [B, C, N, 2]
        loss_dict = model.stage1_loss(x, file_stats, patch_stats, lambda_orth=0.05)
        assert "loss" in loss_dict
        assert "global_loss" in loss_dict
        assert "local_loss" in loss_dict
        assert "orth_loss" in loss_dict
        assert loss_dict["loss"].ndim == 0

    def test_extra_repr(self):
        model = create_glc_flow_v7()
        repr_str = model.extra_repr()
        assert "GLCFlowV7" in repr_str
        assert "D=128" in repr_str