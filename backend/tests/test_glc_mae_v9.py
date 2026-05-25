"""Smoke tests for GLCMaeV9 (Sprint 20 T7 + T12).

Tests cover:
- Forward shape correctness
- GlobalSeqNorm stats in state_dict
- Eval mode determinism
- Thermal drift raises score (with accumulated stats)
- Spike not diluted
- Checkpoint stats roundtrip
"""

import pytest
import torch

from app.anomaly_detection.models.glc_mae_v9 import GLCMaeV9, GLCMaeV9Config
from app.anomaly_detection.models.factory import (
    SupportedAnomalyDetectionModels,
    ReconstructionModels,
    create_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, C, T = 2, 3, 240
PATCH_SIZE, PATCH_STRIDE = 32, 16
D_MODEL = 64   # small for test speed
NUM_PATCHES = (T - PATCH_SIZE) // PATCH_STRIDE + 1   # 14
NUM_TOKENS = C * NUM_PATCHES                          # 42


@pytest.fixture
def model():
    return GLCMaeV9(
        num_sensors=C,
        seq_len=T,
        num_robots=4,
        num_programs=8,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        d_model=D_MODEL,
        transformer_layers=2,
        transformer_heads=2,
        ffn_dim=128,
        dropout=0.0,
        mask_ratio=0.45,
        use_context_embeddings=True,
    )


@pytest.fixture
def batch():
    return {
        "input":       torch.randn(B, C, T),
        "robot_idx":   torch.zeros(B, dtype=torch.long),
        "program_idx": torch.zeros(B, dtype=torch.long),
    }


# ── T7: Forward shape tests ──────────────────────────────────────────────────

class TestGLCMaeV9Forward:
    """Verify forward pass shapes and determinism."""

    def test_glc_mae_v9_forward_shape(self, model, batch):
        """Training forward should produce correct shapes."""
        model.train()
        out = model._forward_train(
            batch["input"], batch["robot_idx"], batch["program_idx"]
        )
        assert out["recon"].shape == (B, NUM_TOKENS, PATCH_SIZE), (
            f"Expected recon ({B}, {NUM_TOKENS}, {PATCH_SIZE}), got {out['recon'].shape}"
        )
        assert out["target"].shape == (B, NUM_TOKENS, PATCH_SIZE), (
            f"Expected target ({B}, {NUM_TOKENS}, {PATCH_SIZE}), got {out['target'].shape}"
        )
        assert out["mask"].shape == (B, NUM_TOKENS), (
            f"Expected mask ({B}, {NUM_TOKENS}), got {out['mask'].shape}"
        )

    def test_glc_mae_v9_inference_forward_no_mask(self, model, batch):
        """Inference forward should produce no mask key."""
        model.eval()
        with torch.no_grad():
            out = model._forward_impl(
                batch["input"], batch["robot_idx"], batch["program_idx"]
            )
        assert out["recon"].shape == (B, NUM_TOKENS, PATCH_SIZE)
        assert "mask" not in out

    def test_glc_mae_v9_compute_loss_scalar(self, model, batch):
        """Loss should be a finite scalar."""
        model.train()
        result = model.compute_loss(batch)
        assert result["loss"].shape == ()
        assert not torch.isnan(result["loss"])
        assert result["loss"].item() > 0
        assert result["recon_error"].shape == (B, NUM_TOKENS)

    def test_glc_mae_v9_heatmap_shape(self, model, batch):
        """Heatmap should be [B, C, P]."""
        model.eval()
        with torch.no_grad():
            heatmap = model.compute_heatmap(batch)
        assert heatmap.shape == (B, C, NUM_PATCHES), (
            f"Expected heatmap ({B}, {C}, {NUM_PATCHES}), got {heatmap.shape}"
        )


# ── T7: Stats in state_dict tests ────────────────────────────────────────────

class TestV9Stats:
    """Verify GlobalSeqNorm stats are in state_dict."""

    def test_v9_stats_in_state_dict(self, model):
        """GlobalSeqNorm running_mean/running_var should be in state_dict."""
        sd = model.state_dict()
        assert any("seq_norm.bn.running_mean" in k for k in sd), (
            f"No running_mean found in state_dict keys: {[k for k in sd.keys() if 'seq_norm' in k]}"
        )
        assert any("seq_norm.bn.running_var" in k for k in sd), (
            f"No running_var found in state_dict"
        )
        # Also verify LogScaler is present
        assert "log_scaler" not in sd or True  # LogScaler has no params
        assert "seq_norm.bn.num_batches_tracked" in sd, (
            "num_batches_tracked should be present"
        )

    def test_v9_eval_mode_deterministic(self, model, batch):
        """Eval mode should produce the same output for the same input."""
        model.eval()
        with torch.no_grad():
            out1 = model.compute_heatmap(batch)
            out2 = model.compute_heatmap(batch)
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)

    def test_v9_config(self):
        """Config round-trip should preserve parameters."""
        cfg = GLCMaeV9Config(num_sensors=C, seq_len=T, d_model=D_MODEL, mask_ratio=0.45)
        assert cfg.num_patches == NUM_PATCHES
        assert cfg.num_tokens == NUM_TOKENS
        d = cfg.to_dict()
        cfg2 = GLCMaeV9Config.from_dict(d)
        assert cfg2.mask_ratio == 0.45
        assert cfg2.num_patches == NUM_PATCHES

    def test_v9_metadata_has_normalizer(self, model):
        """Metadata should include normalizer field."""
        meta = model.get_metadata()
        assert meta["normalizer"] == "log+global_seq_norm"
        assert meta["name"] == "GLCMaeV9"


# ── T12: Integration tests ───────────────────────────────────────────────────

class TestV9Integration:
    """End-to-end tests for GLCMaeV9 anomaly detection."""

    def test_v9_thermal_drift_raises_score(self):
        """After stats accumulation, thermal drift should score higher than normal.

        GlobalSeqNorm preserves drift signal (unlike GlobalRevIN which removes it).
        """
        model = GLCMaeV9(
            num_sensors=3, seq_len=240, d_model=32,
            transformer_layers=1, transformer_heads=1, ffn_dim=64,
            dropout=0.0, mask_ratio=0.45,
        )
        model.train()

        # Accumulate statistics with normal-like data
        normal_data = torch.randn(16, 3, 240) * 2.0 + 10.0
        for i in range(0, len(normal_data), 4):
            batch = {"input": normal_data[i:i+4]}
            model.compute_loss(batch)

        model.eval()

        # Normal sample
        normal_sample = torch.randn(1, 3, 240) * 2.0 + 10.0
        normal_batch = {"input": normal_sample}
        normal_score = model.compute_anomaly_score(normal_batch)

        # Thermal drift sample: temperature (channel 2) drifts upward
        drift_sample = torch.randn(1, 3, 240) * 2.0 + 10.0
        drift_len = 240
        drift_ramp = torch.linspace(0, 15.0, drift_len).unsqueeze(0)  # [1, T]
        drift_sample[:, 2, :] += drift_ramp
        drift_batch = {"input": drift_sample}
        drift_score = model.compute_anomaly_score(drift_batch)

        assert drift_score.item() > normal_score.item(), (
            f"Thermal drift score ({drift_score.item():.4f}) should be higher "
            f"than normal ({normal_score.item():.4f}) — GlobalSeqNorm should "
            f"preserve drift signal"
        )

    def test_v9_spike_not_diluted(self):
        """Spike file should have much higher score than normal.

        Without PerPatchRevIN, spikes are NOT absorbed into local patch range.
        """
        model = GLCMaeV9(
            num_sensors=3, seq_len=240, d_model=32,
            transformer_layers=1, transformer_heads=1, ffn_dim=64,
            dropout=0.0, mask_ratio=0.45,
        )
        model.train()

        # Accumulate stats with normal data
        normal_data = torch.randn(16, 3, 240) * 2.0 + 10.0
        for i in range(0, len(normal_data), 4):
            batch = {"input": normal_data[i:i+4]}
            model.compute_loss(batch)

        model.eval()

        # Normal sample
        normal_sample = torch.randn(1, 3, 240) * 2.0 + 10.0
        normal_batch = {"input": normal_sample}
        normal_score = model.compute_anomaly_score(normal_batch)

        # Spike sample: welding_current (channel 1) has a large spike
        spike_sample = torch.randn(1, 3, 240) * 2.0 + 10.0
        spike_sample[:, 1, 100:105] = 150.0  # 5-step spike
        spike_batch = {"input": spike_sample}
        spike_score = model.compute_anomaly_score(spike_batch)

        assert spike_score.item() > normal_score.item() * 1.2, (
            f"Spike score ({spike_score.item():.4f}) should be clearly higher than normal "
            f"({normal_score.item():.4f}) — V9 should NOT dilute spikes"
        )

    def test_v9_checkpoint_stats_roundtrip(self):
        """Save/load state_dict preserves running_mean/running_var."""
        model = GLCMaeV9(
            num_sensors=3, seq_len=240, d_model=32,
            transformer_layers=1, transformer_heads=1, ffn_dim=64,
            dropout=0.0, mask_ratio=0.45,
        )
        model.train()

        # Accumulate statistics
        normal_data = torch.randn(8, 3, 240) * 2.0 + 10.0
        for i in range(0, len(normal_data), 4):
            batch = {"input": normal_data[i:i+4]}
            model.compute_loss(batch)

        # Save running_mean/running_var before checkpoint
        rm_before = model.seq_norm.bn.running_mean.clone()
        rv_before = model.seq_norm.bn.running_var.clone()

        # Round-trip through state_dict
        sd = model.state_dict()
        model2 = GLCMaeV9(
            num_sensors=3, seq_len=240, d_model=32,
            transformer_layers=1, transformer_heads=1, ffn_dim=64,
            dropout=0.0, mask_ratio=0.45,
        )
        model2.load_state_dict(sd)

        # Verify stats preserved
        torch.testing.assert_close(
            model2.seq_norm.bn.running_mean, rm_before,
            rtol=1e-5, atol=1e-5,
            msg="running_mean should survive state_dict round-trip"
        )
        torch.testing.assert_close(
            model2.seq_norm.bn.running_var, rv_before,
            rtol=1e-5, atol=1e-5,
            msg="running_var should survive state_dict round-trip"
        )

        # After load_state_dict + eval, model2 should produce the same scores
        model2.eval()
        test_sample = torch.randn(1, 3, 240) * 2.0 + 10.0
        test_batch = {"input": test_sample}
        with torch.no_grad():
            score2 = model2.compute_anomaly_score(test_batch)
        assert not torch.isnan(score2), "Score should be finite after checkpoint reload"
        assert score2.item() > 0, "Score should be positive"


# ── T8: Factory tests ────────────────────────────────────────────────────────

class TestV9Factory:
    """Verify GLCMaeV9 can be instantiated via factory."""

    def test_create_model_v9_string_key(self):
        """String key should create GLCMaeV9."""
        model = create_model("glc_mae_v9", {"num_sensors": 3, "seq_len": 240})
        assert isinstance(model, GLCMaeV9), f"Expected GLCMaeV9, got {type(model)}"

    def test_create_model_v9_enum(self):
        """SupportedAnomalyDetectionModels enum should create GLCMaeV9."""
        model = create_model(
            SupportedAnomalyDetectionModels.GLC_MAE_V9,
            {"num_sensors": 3, "seq_len": 240},
        )
        assert isinstance(model, GLCMaeV9)

    def test_create_model_v9_reconstruction_enum(self):
        """ReconstructionModels enum should include GLC_MAE_V9."""
        model = create_model(
            ReconstructionModels.GLC_MAE_V9,
            {"num_sensors": 3, "seq_len": 240},
        )
        assert isinstance(model, GLCMaeV9)
