import os
import tempfile

import pytest
import torch

from app.anomaly_detection.models.factory import create_model, count_parameters
from app.anomaly_detection.models.glc_anomaly_maskformer import (
    GLCAnomalyMaskFormer,
    GLCAnomalyMaskFormerConfig,
)


def small_model(**kwargs) -> GLCAnomalyMaskFormer:
    """Create a small model for fast tests."""
    defaults = dict(d_model=32, transformer_layers=1, transformer_heads=2, ffn_dim=64, use_context_embeddings=False)
    defaults.update(kwargs)
    return create_model("glc_anomaly_maskformer", defaults)


class TestGLCAnomalyMaskFormerFactory:
    def test_create_default(self):
        model = create_model("glc_anomaly_maskformer")
        assert model is not None
        assert isinstance(model, GLCAnomalyMaskFormer)

    def test_create_with_overrides(self):
        model = small_model()
        assert model.d_model == 32
        assert model.transformer_layers == 1

    def test_count_parameters(self):
        model = small_model()
        n = count_parameters(model)
        assert n > 0


class TestGLCAnomalyMaskFormerForward:
    def test_forward_in_training_mode_output_keys(self):
        model = small_model()
        model.train()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        expected = {"out", "discrepancies", "patches_pred", "patches_target", "mask", "z_global", "z_local"}
        assert set(out.keys()) == expected

    def test_forward_training_shapes(self):
        model = small_model(d_model=32, transformer_layers=1, transformer_heads=2)
        model.train()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        assert out["out"].shape == (2, 84, 32)
        assert out["discrepancies"].shape == (2, 1, 84)  # L=1
        assert out["patches_pred"].shape == (2, 84, 32)
        assert out["patches_target"].shape == (2, 84, 32)
        assert out["mask"].shape == (2, 84)
        assert out["z_global"].shape == (2, 84, 16)  # d_half=16
        assert out["z_local"].shape == (2, 84, 16)

    def test_forward_eval_mode_mask_all_false(self):
        """In eval mode, no positions should be masked."""
        model = small_model()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        assert not out["mask"].any(), "eval mode must have all-False mask"

    def test_compute_anomaly_score_shape(self):
        model = small_model()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        scores = model.compute_anomaly_score(x)
        assert scores.shape == (2,)

    def test_compute_combined_heatmap_shape(self):
        model = small_model()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        heatmap = model.compute_combined_heatmap(x, fast=True)
        assert heatmap.shape == (2, 6, 14)

    def test_combined_heatmap_values_in_01(self):
        model = small_model()
        model.eval()
        x = {"input": torch.randn(2, 6, 240)}
        heatmap = model.compute_combined_heatmap(x, fast=True)
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_detect_anomaly_keys(self):
        model = small_model()
        raw = torch.randn(1, 6, 240)
        result = model.detect_anomaly(raw)
        expected_keys = {"anomaly_score", "is_anomaly", "heatmap", "assoc_heatmap", "recon_heatmap"}
        assert set(result.keys()) == expected_keys

    def test_detect_anomaly_single_2d_file(self):
        model = small_model()
        raw = torch.randn(6, 240)  # no batch dim
        result = model.detect_anomaly(raw)
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["is_anomaly"], bool)
        assert result["heatmap"].shape == (6, 14)

    def test_detect_anomaly_with_threshold(self):
        model = small_model()
        raw = torch.randn(1, 6, 240)
        result = model.detect_anomaly(raw, threshold=1e10)  # extremely high → not anomaly
        assert result["is_anomaly"] is False
        result2 = model.detect_anomaly(raw, threshold=-1e10)  # extremely low → anomaly
        assert result2["is_anomaly"] is True


class TestGLCAnomalyMaskFormerContextEmbeddings:
    def test_no_context_embeddings_flag(self):
        model = GLCAnomalyMaskFormer(
            d_model=32, transformer_layers=1, transformer_heads=2,
            ffn_dim=64, use_context_embeddings=False
        )
        assert model.required_input_args == {"input"}
        x = {"input": torch.randn(2, 6, 240)}
        out = model(x)
        assert out["out"].shape == (2, 84, 32)

    def test_with_context_embeddings(self):
        model = GLCAnomalyMaskFormer(
            d_model=32, transformer_layers=1, transformer_heads=2,
            ffn_dim=64, use_context_embeddings=True,
            num_robots=10, num_programs=5
        )
        assert "robot_idx" in model.required_input_args
        assert "program_idx" in model.required_input_args
        x = {
            "input": torch.randn(2, 6, 240),
            "robot_idx": torch.zeros(2, dtype=torch.long),
            "program_idx": torch.zeros(2, dtype=torch.long),
        }
        out = model(x)
        assert out["out"].shape == (2, 84, 32)


class TestGLCAnomalyMaskFormerCheckpoint:
    def test_save_and_load_roundtrip(self):
        model = small_model()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_manual_checkpoint(epoch=0, val_loss=0.5, folder_path=tmpdir, is_best=True)

            model2 = GLCAnomalyMaskFormer.load_from_checkpoint(
                checkpoint=tmpdir,
                model_file=f"model_{model.name}_best.pt",
                metadata_file=f"metadata_{model.name}_best.json",
                device="cpu",
            )

        # Check state dicts match
        sd1 = model.state_dict()
        sd2 = model2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for k in sd1:
            assert torch.allclose(sd1[k].float(), sd2[k].float())

    def test_get_metadata_keys(self):
        model = small_model()
        meta = model.get_metadata()
        required_keys = {
            "name", "num_sensors", "seq_len", "patch_size", "patch_stride",
            "d_model", "d_half", "num_patches", "num_tokens",
            "transformer_layers", "transformer_heads", "ffn_dim", "dropout",
            "mask_ratio", "initial_sigma", "lambda_assoc", "lambda_orth",
            "num_robots", "num_programs", "use_context_embeddings",
        }
        assert required_keys.issubset(set(meta.keys()))
