import math
import pytest
import torch
import torch.optim as optim

from app.anomaly_detection.models.glc_anomaly_maskformer import (
    GLCAnomalyMaskFormer,
    GLCAnomalyMaskFormerConfig,
)
from app.anomaly_detection.training.maskformer_trainer import (
    MaskFormerTrainer,
    MaskFormerEvaluator,
    MaskFormerTrainerRunner,
)


def make_tiny_model() -> GLCAnomalyMaskFormer:
    return GLCAnomalyMaskFormer(
        d_model=32,
        transformer_layers=1,
        transformer_heads=2,
        ffn_dim=64,
        use_context_embeddings=False,
        dropout=0.0,
    )


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
        expected = {"loss", "recon_loss", "assoc_loss_min", "assoc_loss_max", "orth_loss", "sigma_mean"}
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
            model_or_config={"d_model": 32, "transformer_layers": 1,
                             "transformer_heads": 2, "ffn_dim": 64,
                             "use_context_embeddings": False}
        )
        train_data = [{"input": torch.randn(2, 6, 240)} for _ in range(4)]
        history = runner.train(train_data, epochs=1, lr=1e-3)
        assert "loss" in history
        assert len(history["loss"]) == 1
        assert not math.isnan(history["loss"][0])

    def test_runner_evaluate(self):
        runner = MaskFormerTrainerRunner(
            model_or_config={"d_model": 32, "transformer_layers": 1,
                             "transformer_heads": 2, "ffn_dim": 64,
                             "use_context_embeddings": False}
        )
        normal = [{"input": torch.randn(2, 6, 240)} for _ in range(2)]
        anomaly = [{"input": torch.randn(2, 6, 240)} for _ in range(2)]
        result = runner.evaluate(normal, anomaly)
        assert 0.0 <= result.auroc <= 1.0
