"""Tests for GLCReconV8 model — forward pass, loss, anomaly scoring, checkpoint.

Smoke test: forward pass [B=2, C=6, T=240] → loss scalar.
"""

import os
import tempfile

import pytest
import torch

from app.anomaly_detection.models.factory import create_model, count_parameters, get_model_summary
from app.anomaly_detection.models.glc_recon_v8 import GLCReconV8, GLCReconV8Config


_BASE_META = {"num_sensors": 6, "seq_len": 240}


def create_glc_recon_v8(metadata=None):
    meta = dict(_BASE_META)
    if metadata:
        meta.update(metadata)
    return create_model("glc_recon_v8", metadata=meta)


class TestFactory:
    def test_create_default_model(self):
        model = create_glc_recon_v8()
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_with_metadata(self):
        meta = {"d_model": 64, "transformer_layers": 2}
        model = create_glc_recon_v8(meta)
        assert model.d_model == 64
        assert model.transformer_layers == 2

    def test_count_parameters_positive(self):
        model = create_glc_recon_v8()
        n = count_parameters(model)
        assert n > 0

    def test_model_summary_keys(self):
        model = create_glc_recon_v8()
        summary = get_model_summary(model)
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "by_component" in summary

    def test_model_summary_total_equals_sum(self):
        model = create_glc_recon_v8()
        summary = get_model_summary(model)
        component_sum = sum(summary["by_component"].values())
        assert component_sum == summary["total_parameters"]

    def test_registered_in_reconstruction_models(self):
        """GLCReconV8 must be registered in ReconstructionModels enum."""
        from app.anomaly_detection.models.factory import ReconstructionModels
        assert hasattr(ReconstructionModels, "GLC_RECON_V8")
        assert ReconstructionModels.GLC_RECON_V8.value == GLCReconV8


class TestGLCReconV8Forward:
    def test_forward_output_dict_keys(self):
        model = create_glc_recon_v8()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        expected_keys = {"recon", "target", "tokens"}
        assert set(out.keys()) == expected_keys

    def test_forward_shapes(self):
        """Smoke test: forward pass [B=2, C=6, T=240]."""
        model = create_glc_recon_v8()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        B, C, T = 2, 6, 240
        patch_size = 32
        patch_stride = 16
        num_patches = (T - patch_size) // patch_stride + 1  # 14
        num_tokens = C * num_patches  # 84

        assert out["recon"].shape == (B, num_tokens, patch_size)
        assert out["target"].shape == (B, num_tokens, patch_size)
        assert out["tokens"].shape == (B, num_tokens, model.d_model)

    def test_forward_single_sample(self):
        model = create_glc_recon_v8()
        x = {"input": torch.randn(1, 6, 240)}
        out = model(x)
        assert out["recon"].shape[0] == 1

    def test_forward_different_batch_sizes(self):
        model = create_glc_recon_v8()
        for B in [1, 4, 8]:
            x = {"input": torch.randn(B, 6, 240)}
            out = model(x)
            assert out["recon"].shape[0] == B

    def test_detect_anomaly_returns_dict(self):
        model = create_glc_recon_v8()
        raw = torch.randn(1, 6, 240)
        result = model.detect_anomaly(raw)
        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "heatmap" in result

    def test_detect_anomaly_single_file(self):
        model = create_glc_recon_v8()
        raw = torch.randn(6, 240)
        result = model.detect_anomaly(raw)
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["is_anomaly"], bool)

    def test_compute_loss_returns_scalar(self):
        """T13: Smoke test — forward pass [B=2, C=6, T=240] -> loss scalar."""
        model = create_glc_recon_v8()
        raw = torch.randn(2, 6, 240)
        out = model.compute_loss(raw)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar
        assert out["loss"].item() > 0
        assert "recon_error" in out
        assert out["recon_error"].shape == (2, 84)  # [B, C*P]

    def test_compute_loss_gradients_flow(self):
        """Loss backward should produce gradients for all parameters."""
        model = create_glc_recon_v8()
        raw = torch.randn(2, 6, 240)
        out = model.compute_loss(raw)
        loss = out["loss"]
        loss.backward()
        has_grad = [p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters()]
        assert any(has_grad), "No gradients flowing — model may not be training"

    def test_compute_anomaly_score_shape(self):
        model = create_glc_recon_v8()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        scores = model.compute_anomaly_score(x)
        assert scores.shape == (2,)

    def test_compute_heatmap_shape(self):
        model = create_glc_recon_v8()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        heatmap = model.compute_heatmap(x)
        assert heatmap.shape == (2, 6, 14)  # [B, C, P]

    def test_heatmap_values_non_negative(self):
        model = create_glc_recon_v8()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        heatmap = model.compute_heatmap(x)
        assert heatmap.min() >= 0.0 - 1e-6


class TestGLCReconV8Checkpoint:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_glc_recon_v8()
            model.eval()
            model.save_manual_checkpoint(
                epoch=1, val_loss=0.05, folder_path=tmpdir, is_best=True
            )
            # Load from checkpoint
            model2 = GLCReconV8.load_from_checkpoint(
                checkpoint=tmpdir,
                model_file=f"model_GLCReconV8_best.pt",
                metadata_file=f"metadata_GLCReconV8_best.json",
            )
            # Both should be in eval mode for comparison
            model2.eval()
            # Verify state dict keys match
            sd1 = model.state_dict()
            sd2 = model2.state_dict()
            assert set(sd1.keys()) == set(sd2.keys()), "State dict keys differ"
            # Verify weights are identical
            for key in sd1:
                assert torch.allclose(sd1[key], sd2[key], atol=1e-6), \
                    f"Weight mismatch in {key}"
            # Check same output with fixed input
            x = {"input": torch.randn(2, 6, 240)}
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)
            diff = (out1["recon"] - out2["recon"]).abs().max().item()
            assert diff < 0.02, f"Checkpoint reload diff too large: {diff}"

    def test_load_from_metadata(self):
        meta = {"num_sensors": 6, "seq_len": 240, "d_model": 64}
        model = GLCReconV8.load_from_metadata(meta)
        assert model.d_model == 64
        assert model.num_sensors == 6

    def test_metadata_roundtrip(self):
        model = create_glc_recon_v8()
        meta = model.get_metadata()
        model2 = GLCReconV8.load_from_metadata(meta)
        assert model2.num_sensors == model.num_sensors
        assert model2.seq_len == model.seq_len
        assert model2.d_model == model.d_model
        assert model2.patch_size == model.patch_size
        assert model2.patch_stride == model.patch_stride


class TestGLCReconV8Config:
    def test_config_post_init(self):
        cfg = GLCReconV8Config(num_sensors=6, seq_len=240)
        assert cfg.d_half == 64  # d_model // 2
        assert cfg.num_patches == 14  # (240 - 32) // 16 + 1
        assert cfg.num_tokens == 84  # 6 * 14

    def test_config_to_dict(self):
        cfg = GLCReconV8Config(num_sensors=6, seq_len=240)
        d = cfg.to_dict()
        assert d["num_sensors"] == 6
        assert d["seq_len"] == 240
        assert "d_half" not in d  # non-init fields excluded

    def test_config_from_dict(self):
        d = {"num_sensors": 6, "seq_len": 240, "d_model": 64}
        cfg = GLCReconV8Config.from_dict(d)
        assert cfg.num_sensors == 6
        assert cfg.d_model == 64

    def test_config_name(self):
        model = create_glc_recon_v8()
        assert model.name == "GLCReconV8"
