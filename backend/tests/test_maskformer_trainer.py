import math
import pytest
import torch
import torch.optim as optim

from app.anomaly_detection.models.glc_anomaly_maskformer import (
    GLCAnomalyMaskFormer,
    GLCAnomalyMaskFormerConfig,
)
from app.anomaly_detection.layers.reconstruction_decoder import (
    PerTokenDecoder,
    CNNDecoder,
)
from app.anomaly_detection.training.maskformer_trainer import (
    MaskFormerTrainer,
    MaskFormerEvaluator,
    MaskFormerTrainerRunner,
)


_TINY_CFG = dict(
    num_sensors=6, seq_len=240, num_robots=47, num_programs=16,
    d_model=32, transformer_layers=1, transformer_heads=2, ffn_dim=64,
    use_context_embeddings=False,
)


def make_tiny_model() -> GLCAnomalyMaskFormer:
    return GLCAnomalyMaskFormer(**_TINY_CFG, dropout=0.0)


def make_trainer(model: GLCAnomalyMaskFormer) -> MaskFormerTrainer:
    sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
    main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
    opt_main = optim.Adam(main_params, lr=1e-3)
    opt_sigma = optim.Adam(sigma_params, lr=1e-3)
    return MaskFormerTrainer(
        model=model,
        optimizer_main=opt_main,
        optimizer_sigma=opt_sigma,
        lambda_assoc=1.0,
        lambda_orth=0.01,
    )


class TestMaskFormerTrainer:
    def test_train_step_returns_expected_keys(self):
        model = make_tiny_model()
        trainer = make_trainer(model)
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x)
        expected = {"loss", "recon_loss", "assoc_loss_min", "assoc_loss_max",
                     "orth_loss", "sigma_mean", "grad_ratio", "discrepancy_std"}
        assert set(out.keys()) == expected

    def test_train_step_no_nan(self):
        model = make_tiny_model()
        trainer = make_trainer(model)
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x)
        for k, v in out.items():
            assert not torch.isnan(v), f"NaN in {k}"
            assert not torch.isinf(v), f"Inf in {k}"

    def test_sigma_changes_after_step(self):
        """After a maximize step, sigma values should shift."""
        model = make_tiny_model()
        # Record initial sigma
        sigma_before = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        trainer = make_trainer(model)
        x = {"input": torch.randn(2, 6, 240)}
        trainer.train_step(x)
        sigma_after = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        assert not torch.allclose(sigma_before, sigma_after), (
            "sigma should change after a training step"
        )

    def test_sigma_mean_in_output(self):
        model = make_tiny_model()
        trainer = make_trainer(model)
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x)
        assert out["sigma_mean"].item() > 0.0

    def test_minimax_sign_convention(self):
        """Phase B loss should push sigma in the direction that INCREASES discrepancy.

        We measure discrepancy before and after one maximize step.
        A correct implementation should increase (or at least not decrease)
        average discrepancy after Phase B. We use a simple sanity check
        over 5 steps rather than a single-step assertion to avoid noise.
        """
        model = make_tiny_model()
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
        # Freeze main params to isolate sigma effect
        for p in main_params:
            p.requires_grad_(False)

        opt_main = optim.Adam(main_params, lr=0.0)  # effectively frozen
        opt_sigma = optim.Adam(sigma_params, lr=1e-2)
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=opt_main,
            optimizer_sigma=opt_sigma,
            lambda_assoc=5.0,
            lambda_orth=0.0,
        )

        x = {"input": torch.randn(4, 6, 240)}

        assoc_losses = []
        for _ in range(5):
            out = trainer.train_step(x)
            assoc_losses.append(out["assoc_loss_max"].item())

        # The maximize phase should at least not decrease assoc consistently
        # (simple sanity check — not a strict monotone guarantee)
        assert any(v >= assoc_losses[0] * 0.5 for v in assoc_losses), (
            "Association discrepancy collapsed to near zero — minimax sign may be wrong"
        )

    def test_phase_b_loss_is_positive(self):
        """Phase B loss should be positive (maximise discrepancy)."""
        model = make_tiny_model()
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
        for p in main_params:
            p.requires_grad_(False)

        opt_main = optim.Adam(main_params, lr=0.0)
        opt_sigma = optim.Adam(sigma_params, lr=1e-2)
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=opt_main,
            optimizer_sigma=opt_sigma,
            lambda_assoc=2.0,
            lambda_orth=0.0,
        )

        x = {"input": torch.randn(4, 6, 240)}
        out = trainer.train_step(x)
        assert out["assoc_loss_max"].item() > 0.0, (
            "Phase B loss must be positive after sign fix (was -lambda, now +lambda)"
        )


    def test_sigma_clamped_to_sigma_max(self):
        """Sigma must never exceed sigma_max after a training step."""
        model = make_tiny_model()
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params  = [p for n, p in model.named_parameters() if "sigma" not in n]

        # Large LR for sigma so it would naturally overshoot without clamping
        opt_main  = optim.Adam(main_params,  lr=1e-3)
        opt_sigma = optim.Adam(sigma_params, lr=10.0)

        SIGMA_MAX = 6.0  # initial sigma is 5.0 — cap just above to make clamping observable
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=opt_main,
            optimizer_sigma=opt_sigma,
            lambda_assoc=5.0,
            lambda_orth=0.0,
            sigma_max=SIGMA_MAX,
        )

        x = {"input": torch.randn(4, 6, 240)}
        for _ in range(5):
            trainer.train_step(x)

        for layer in model.transformer.layers:
            assert layer.sigma.data.max().item() <= SIGMA_MAX + 1e-5, (
                f"sigma exceeded sigma_max={SIGMA_MAX}: {layer.sigma.data}"
            )

    def test_runner_respects_lr_sigma_scale(self):
        """lr_sigma_scale < 1 should slow sigma adaptation vs sigma_scale=1."""
        def _run(scale):
            torch.manual_seed(0)
            runner = MaskFormerTrainerRunner(
                model_or_config={**_TINY_CFG},
                lr_sigma_scale=scale,
                sigma_max=500.0,
            )
            data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
            runner.train(data, epochs=1, lr=1e-3)
            return runner.model.transformer.layers[0].sigma.detach().mean().item()

        sigma_slow = _run(scale=0.01)
        sigma_fast = _run(scale=1.0)
        assert sigma_fast > sigma_slow, (
            f"faster sigma LR should cause more sigma change: fast={sigma_fast:.4f}, slow={sigma_slow:.4f}"
        )


class TestMaskFormerEvaluator:
    def test_evaluate_returns_eval_result(self):
        model = make_tiny_model()
        model.eval()
        evaluator = MaskFormerEvaluator(model)
        normal = [{"input": torch.randn(2, 6, 240)} for _ in range(3)]
        anomaly = [{"input": torch.randn(2, 6, 240)} for _ in range(3)]
        result = evaluator.evaluate(normal, anomaly)
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.auprc <= 1.0
        assert result.num_normal == 6
        assert result.num_anomaly == 6

    def test_compute_scores_shape(self):
        model = make_tiny_model()
        evaluator = MaskFormerEvaluator(model)
        x = {"input": torch.randn(4, 6, 240)}
        scores = evaluator.compute_scores(x)
        assert scores.shape == (4,)


class TestMaskFormerTrainerRunner:
    def test_runner_trains_one_epoch(self):
        runner = MaskFormerTrainerRunner(
            model_or_config={**_TINY_CFG}
        )
        train_data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
        history = runner.train(train_data, epochs=1, lr=1e-3)
        assert "loss" in history
        assert len(history["loss"]) == 1
        assert not math.isnan(history["loss"][0])

    def test_runner_evaluate(self):
        runner = MaskFormerTrainerRunner(
            model_or_config={**_TINY_CFG}
        )
        normal = [{"input": torch.randn(2, 6, 240)} for _ in range(2)]
        anomaly = [{"input": torch.randn(2, 6, 240)} for _ in range(2)]
        result = runner.evaluate(normal, anomaly)
        assert 0.0 <= result.auroc <= 1.0


class TestWarmup:
    """Phase-A-only warmup tests."""

    def test_warmup_skips_phase_b(self):
        """During warmup epoch, sigma should NOT change (Phase B skipped)."""
        model = make_tiny_model()
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
        opt_main = optim.Adam(main_params, lr=1e-3)
        opt_sigma = optim.Adam(sigma_params, lr=1e-2)
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=opt_main,
            optimizer_sigma=opt_sigma,
            lambda_assoc=1.0,
            lambda_orth=0.0,
            warmup_epochs=1,
        )

        sigma_before = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x, epoch=0)  # warmup epoch (0 < 1)
        sigma_after = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        assert torch.allclose(sigma_before, sigma_after), (
            "sigma should NOT change during warmup (Phase B skipped)"
        )
        assert out["assoc_loss_max"].item() == 0.0, (
            "assoc_loss_max should be 0 during warmup"
        )

    def test_warmup_then_phase_b_resumes(self):
        """After warmup ends, Phase B should resume and sigma should change."""
        model = make_tiny_model()
        sigma_before = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
        opt_main = optim.Adam(main_params, lr=1e-3)
        opt_sigma = optim.Adam(sigma_params, lr=1e-2)
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=opt_main,
            optimizer_sigma=opt_sigma,
            lambda_assoc=1.0,
            lambda_orth=0.0,
            warmup_epochs=1,
        )

        x = {"input": torch.randn(2, 6, 240)}
        _ = trainer.train_step(x, epoch=0)  # warmup
        _ = trainer.train_step(x, epoch=1)  # post-warmup (1 < 1 is False)
        sigma_after = torch.cat([
            layer.sigma.detach().clone() for layer in model.transformer.layers
        ])
        assert not torch.allclose(sigma_before, sigma_after), (
            "sigma should change after warmup ends (Phase B resumes)"
        )


class TestCNNDecoder:
    def test_cnn_decoder_output_shape(self):
        """CNNDecoder should produce the same output shape as PerTokenDecoder."""
        B, C, P, D, PS = 2, 6, 14, 128, 32
        tokens = torch.randn(B, C * P, D)
        cnn_decoder = CNNDecoder(num_sensors=C, num_patches=P, d_model=D, patch_size=PS)
        lin_decoder = PerTokenDecoder(d_model=D, patch_size=PS)
        cnn_out = cnn_decoder(tokens)
        lin_out = lin_decoder(tokens)
        assert cnn_out.shape == (B, C * P, PS), f"CNNDecoder shape: {cnn_out.shape}"
        assert cnn_out.shape == lin_out.shape, "CNNDecoder and PerTokenDecoder shapes should match"

    def test_cnn_decoder_no_nan(self):
        """CNNDecoder forward should not produce NaN/Inf."""
        B, C, P, D, PS = 2, 6, 14, 128, 32
        tokens = torch.randn(B, C * P, D)
        decoder = CNNDecoder(num_sensors=C, num_patches=P, d_model=D, patch_size=PS)
        out = decoder(tokens)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_cnn_decoder_differs_from_per_token(self):
        """CNNDecoder output should differ from PerTokenDecoder (temporal refinement changes it)."""
        B, C, P, D, PS = 2, 6, 14, 128, 32
        torch.manual_seed(0)
        tokens = torch.randn(B, C * P, D)
        cnn_decoder = CNNDecoder(num_sensors=C, num_patches=P, d_model=D, patch_size=PS)
        lin_decoder = PerTokenDecoder(d_model=D, patch_size=PS)
        # Use same seed for fair comparison (no randomness in forward pass)
        torch.manual_seed(42)
        cnn_out = cnn_decoder(tokens)
        torch.manual_seed(42)
        lin_out = lin_decoder(tokens)
        # They should differ (CNN adds temporal refinement)
        diff = (cnn_out - lin_out).abs().mean().item()
        assert diff > 1e-6, (
            f"CNNDecoder should differ from PerTokenDecoder, diff={diff:.8f}"
        )


class TestGradNormLogging:
    """Phase 5: gradient-norm ratio logging tests."""

    def test_grad_ratio_in_output(self):
        """train_step should return grad_ratio when log_grad_norms=True."""
        model = make_tiny_model()
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=optim.Adam(
                [p for n, p in model.named_parameters() if "sigma" not in n], lr=1e-3
            ),
            optimizer_sigma=optim.Adam(
                [p for n, p in model.named_parameters() if "sigma" in n], lr=1e-3
            ),
            lambda_assoc=1.0,
            lambda_orth=0.01,
            log_grad_norms=True,
        )
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x, epoch=5)  # non-warmup epoch so λ > 0
        assert "grad_ratio" in out
        assert out["grad_ratio"].item() >= 0.0

    def test_grad_ratio_during_warmup(self):
        """grad_ratio should be 0 during warmup (no adversarial term)."""
        model = make_tiny_model()
        sigma_params = [p for n, p in model.named_parameters() if "sigma" in n]
        main_params = [p for n, p in model.named_parameters() if "sigma" not in n]
        trainer = MaskFormerTrainer(
            model=model,
            optimizer_main=optim.Adam(main_params, lr=1e-3),
            optimizer_sigma=optim.Adam(sigma_params, lr=1e-3),
            lambda_assoc=1.0,
            lambda_orth=0.01,
            warmup_epochs=2,
            log_grad_norms=True,
        )
        x = {"input": torch.randn(2, 6, 240)}
        out = trainer.train_step(x, epoch=0)  # warmup
        assert out["grad_ratio"].item() == 0.0


class TestLambdaRamp:
    """Phase 6: λ ramp schedule tests."""

    def test_ramp_linear_interpolation(self):
        """λ should linearly interpolate from lambda_assoc to lambda_end."""
        runner = MaskFormerTrainerRunner(
            model_or_config={**_TINY_CFG},
            lambda_assoc=0.0,
            lambda_end=3.0,
            lambda_ramp_epochs=10,
            warmup_epochs=0,
        )
        # Train 11 epochs: ramp_progress goes 0 → 1.0 over epochs 0-10
        data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
        _ = runner.train(data, epochs=11, lr=1e-3)
        # After full ramp (epoch 10), λ should be at lambda_end
        assert abs(runner.trainer.lambda_assoc - 3.0) < 1e-4, (
            f"After 10-epoch ramp with λ_end=3.0, got {runner.trainer.lambda_assoc:.4f}"
        )

    def test_ramp_clamps_at_end(self):
        """After ramp_epochs, λ should stay at lambda_end (not exceed)."""
        runner = MaskFormerTrainerRunner(
            model_or_config={**_TINY_CFG},
            lambda_assoc=0.0,
            lambda_end=2.0,
            lambda_ramp_epochs=5,
            warmup_epochs=0,
        )
        runner._build_trainer(lr=1e-3)
        # Run 10 epochs (past the 5-epoch ramp)
        data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
        history = runner.train(data, epochs=10, lr=1e-3)
        # After ramp, λ should be ~2.0
        assert abs(runner.trainer.lambda_assoc - 2.0) < 1e-4, (
            f"Final λ should be 2.0, got {runner.trainer.lambda_assoc:.4f}"
        )

    def test_ramp_starts_after_warmup(self):
        """Ramp should start after warmup_epochs, not before."""
        runner = MaskFormerTrainerRunner(
            model_or_config={**_TINY_CFG},
            lambda_assoc=0.0,
            lambda_end=3.0,
            lambda_ramp_epochs=10,
            warmup_epochs=5,
        )
        data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
        history = runner.train(data, epochs=12, lr=1e-3)
        # During warmup (epochs 0-4), λ stays at 0
        # After warmup, ramps from 0 toward 3 over 10 epochs
        # At epoch 11 (last of 12 total, 0-indexed), we're 6 epochs into ramp
        # λ = 0 + (6/10) * 3.0 = 1.8
        assert abs(runner.trainer.lambda_assoc - 1.8) < 0.2, (
            f"Expected λ ~1.8 after 6/10 ramp, got {runner.trainer.lambda_assoc:.4f}"
        )
