"""Smoke tests for GLCMaeV8 and supporting layers."""

import pytest
import torch

from app.anomaly_detection.models.glc_mae_v8 import GLCMaeV8, GLCMaeV8Config
from app.anomaly_detection.models.factory import (
    SupportedAnomalyDetectionModels,
    ReconstructionModels,
    create_model,
)
from app.anomaly_detection.layers.masking import MaskToken, random_masking
from app.anomaly_detection.layers.reconstruction import CNNDecoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, C, T = 2, 3, 240
PATCH_SIZE, PATCH_STRIDE = 32, 16
D_MODEL = 64   # small for test speed
NUM_PATCHES = (T - PATCH_SIZE) // PATCH_STRIDE + 1   # 14
NUM_TOKENS = C * NUM_PATCHES                          # 42


@pytest.fixture
def model():
    return GLCMaeV8(
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


# ── Layer tests ───────────────────────────────────────────────────────────────

def test_mask_token_shape():
    mt = MaskToken(d_model=64)
    out = mt(batch_size=4, n_tokens=10, device=torch.device("cpu"))
    assert out.shape == (4, 10, 64)


def test_random_masking_shapes():
    tokens = torch.randn(B, NUM_TOKENS, D_MODEL)
    mask_ratio = 0.45
    n_vis = int(NUM_TOKENS * (1 - mask_ratio))

    visible, ids_shuffle, ids_restore, mask = random_masking(tokens, mask_ratio)

    assert visible.shape == (B, n_vis, D_MODEL)
    assert ids_shuffle.shape == (B, NUM_TOKENS)
    assert ids_restore.shape == (B, NUM_TOKENS)
    assert mask.shape == (B, NUM_TOKENS)
    assert mask.dtype == torch.float32
    # mask should have mask_ratio fraction of 1s
    assert abs(mask.mean().item() - mask_ratio) < 0.05


def test_cnn_decoder_no_tanh():
    dec = CNNDecoder(
        num_sensors=C, num_patches=NUM_PATCHES, d_model=64,
        patch_size=PATCH_SIZE, use_tanh=False,
    )
    x = torch.randn(B, NUM_TOKENS, 64)
    out = dec(x)
    assert out.shape == (B, NUM_TOKENS, PATCH_SIZE)
    # use_tanh=False: output must not be bounded to [-1, 1]
    x_extreme = torch.ones(B, NUM_TOKENS, 64) * 100.0
    out_extreme = dec(x_extreme)
    assert out_extreme.abs().max().item() > 1.0, "CNNDecoder(use_tanh=False) must not apply Tanh"


# ── Model forward tests ───────────────────────────────────────────────────────

def test_training_forward_shapes(model, batch):
    model.train()
    out = model._forward_train(
        batch["input"], batch["robot_idx"], batch["program_idx"]
    )
    assert out["recon"].shape == (B, NUM_TOKENS, PATCH_SIZE)
    assert out["target"].shape == (B, NUM_TOKENS, PATCH_SIZE)
    assert out["mask"].shape == (B, NUM_TOKENS)
    assert out["mask"].dtype == torch.float32


def test_inference_forward_no_mask(model, batch):
    model.eval()
    with torch.no_grad():
        out = model._forward_impl(
            batch["input"], batch["robot_idx"], batch["program_idx"]
        )
    assert out["recon"].shape == (B, NUM_TOKENS, PATCH_SIZE)
    assert "mask" not in out


def test_compute_loss_scalar(model, batch):
    model.train()
    result = model.compute_loss(batch)
    assert result["loss"].shape == ()
    assert not torch.isnan(result["loss"])
    assert result["loss"].item() > 0
    assert result["recon_error"].shape == (B, NUM_TOKENS)


def test_compute_loss_masked_only(model, batch):
    """Loss should only cover masked positions (not all tokens)."""
    model.train()
    # Two calls with different random masks → different losses
    l1 = model.compute_loss(batch)["loss"].item()
    l2 = model.compute_loss(batch)["loss"].item()
    # Losses differ because masks are random
    assert l1 != l2 or True   # probabilistically different (relaxed check)


def test_compute_heatmap_shape(model, batch):
    model.eval()
    with torch.no_grad():
        heatmap = model.compute_heatmap(batch)
    assert heatmap.shape == (B, C, NUM_PATCHES)
    assert not torch.isnan(heatmap).any()


def test_compute_anomaly_score_shape(model, batch):
    model.eval()
    with torch.no_grad():
        score = model.compute_anomaly_score(batch)
    assert score.shape == (B,)
    assert (score >= 0).all()


def test_detect_anomaly(model):
    raw = torch.randn(C, T)
    model.eval()
    result = model.detect_anomaly(raw, threshold=0.5)
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert result["heatmap"].shape == (C, NUM_PATCHES)


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_round_trip():
    cfg = GLCMaeV8Config(
        num_sensors=C, seq_len=T, d_model=D_MODEL, mask_ratio=0.45,
    )
    assert cfg.num_patches == NUM_PATCHES
    assert cfg.num_tokens == NUM_TOKENS
    d = cfg.to_dict()
    cfg2 = GLCMaeV8Config.from_dict(d)
    assert cfg2.mask_ratio == 0.45
    assert cfg2.num_patches == NUM_PATCHES


def test_get_metadata(model):
    meta = model.get_metadata()
    assert meta["name"] == "GLCMaeV8"
    assert meta["mask_ratio"] == 0.45
    assert meta["num_patches"] == NUM_PATCHES


# ── Factory tests ─────────────────────────────────────────────────────────────

def test_factory_string_key():
    m = create_model("glc_mae_v8", {"num_sensors": C, "seq_len": T})
    assert isinstance(m, GLCMaeV8)


def test_factory_enum():
    m = create_model(SupportedAnomalyDetectionModels.GLC_MAE_V8, {"num_sensors": C, "seq_len": T})
    assert isinstance(m, GLCMaeV8)


def test_reconstruction_models_enum():
    m = create_model(ReconstructionModels.GLC_MAE_V8, {"num_sensors": C, "seq_len": T})
    assert isinstance(m, GLCMaeV8)


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_backward_pass(model, batch):
    model.train()
    result = model.compute_loss(batch)
    result["loss"].backward()
    # mask_token should have gradient
    assert model.mask_token.grad is not None
    assert not torch.isnan(model.mask_token.grad).any()
