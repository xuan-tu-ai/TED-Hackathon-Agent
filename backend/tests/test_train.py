import pytest
import torch

from app.anomaly_detection.training import (
    Stage1Trainer,
    Stage2Trainer,
    Evaluator,
    FlowMatchingTrainer,
    EvalResult,
)
from app.anomaly_detection.models.factory import create_model
from app.anomaly_detection.layers.scorer import MADThreshold


_BASE_META = {"num_sensors": 6, "seq_len": 240}


def create_glc_flow_v7():
    return create_model("glc_flow_v7", _BASE_META)


class TestStage1Trainer:
    def test_train_step_updates_weights(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Stage1Trainer(model, optimizer, lambda_orth=0.05)

        x = {"input": torch.randn(4, 6, 240)}

        before = model.encoder_global.linear.weight.clone()

        losses = trainer.train_step(x)

        assert "loss" in losses
        assert "global_loss" in losses
        assert "local_loss" in losses
        assert "orth_loss" in losses
        assert losses["loss"].ndim == 0

        after = model.encoder_global.linear.weight
        assert not torch.allclose(before, after), "weights should update"

    def test_cnf_is_frozen(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Stage1Trainer(model, optimizer, lambda_orth=0.05)

        for param in model.cnf.parameters():
            assert not param.requires_grad

    def test_loss_reasonable_magnitude(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Stage1Trainer(model, optimizer, lambda_orth=0.05)

        x = {"input": torch.randn(4, 6, 240)}
        losses = trainer.train_step(x)

        assert losses["loss"].item() > 0
        assert losses["global_loss"].item() > 0
        assert losses["local_loss"].item() > 0

    def test_orth_loss_is_scalar(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Stage1Trainer(model, optimizer, lambda_orth=0.05)

        x = {"input": torch.randn(4, 6, 240)}
        losses = trainer.train_step(x)

        assert losses["orth_loss"].ndim == 0
        assert losses["orth_loss"].item() >= 0


class TestStage2Trainer:
    def test_train_step_updates_cnf(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.cnf.parameters(), lr=1e-3)
        trainer = Stage2Trainer(model, optimizer)

        x = {"input": torch.randn(4, 6, 240)}

        before = model.cnf.vf_net.net[0].weight.clone()

        losses = trainer.train_step(x)

        assert "loss" in losses
        assert "v_pred_norm" in losses
        assert losses["loss"].ndim == 0

        after = model.cnf.vf_net.net[0].weight
        assert not torch.allclose(before, after)

    def test_encoder_is_frozen(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.cnf.parameters(), lr=1e-3)
        trainer = Stage2Trainer(model, optimizer)

        for name, param in model.named_parameters():
            if "cnf" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_loss_positive(self):
        model = create_glc_flow_v7()
        optimizer = torch.optim.Adam(model.cnf.parameters(), lr=1e-3)
        trainer = Stage2Trainer(model, optimizer)

        x = {"input": torch.randn(4, 6, 240)}
        losses = trainer.train_step(x)

        assert losses["loss"].item() > 0


class TestEvaluator:
    def test_compute_scores_shape(self):
        model = create_glc_flow_v7()
        evaluator = Evaluator(model)

        x = {"input": torch.randn(4, 6, 240)}
        scores = evaluator.compute_scores(x)

        assert scores.shape == (4,)

    def test_compute_scores_finite(self):
        model = create_glc_flow_v7()
        evaluator = Evaluator(model)

        x = {"input": torch.randn(4, 6, 240)}
        scores = evaluator.compute_scores(x)

        assert torch.isfinite(scores).all()

    def test_evaluate_returns_eval_result(self):
        model = create_glc_flow_v7()
        threshold = MADThreshold()
        normal_data = [{"input": torch.randn(1, 6, 240)} for _ in range(10)]
        anomaly_data = [{"input": torch.randn(1, 6, 240)} for _ in range(10)]

        threshold.fit(torch.randn(10))
        evaluator = Evaluator(model, threshold=threshold)

        result = evaluator.evaluate(normal_data, anomaly_data)

        assert isinstance(result, EvalResult)
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.auprc <= 1.0
        assert result.num_normal == 10
        assert result.num_anomaly == 10
        assert result.num_files == 20

    def test_evaluate_without_threshold(self):
        model = create_glc_flow_v7()
        evaluator = Evaluator(model)

        normal_data = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]
        anomaly_data = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]

        result = evaluator.evaluate(normal_data, anomaly_data)

        assert isinstance(result, EvalResult)
        assert result.threshold > 0


class TestFlowMatchingTrainer:
    def test_initialization(self):
        trainer = FlowMatchingTrainer(_BASE_META)
        assert trainer.model is not None
        assert trainer.stage1_trainer is None
        assert trainer.stage2_trainer is None

    def test_initialization_with_metadata(self):
        from app.anomaly_detection.models.factory import create_model
        trainer = FlowMatchingTrainer(_BASE_META)
        trainer.model = create_model("glc_flow_v7", metadata={**_BASE_META, "d_model": 64, "transformer_layers": 2})
        assert trainer.model.d_model == 64
        assert trainer.model.transformer_layers == 2

    def test_train_stage1_history_keys(self):
        trainer = FlowMatchingTrainer(_BASE_META)

        train_data = [
            {"input": torch.randn(4, 6, 240)}
            for _ in range(4)
        ]

        history = trainer.train_stage1(train_data, epochs=2, lr=1e-3)

        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_train_stage2_history_keys(self):
        trainer = FlowMatchingTrainer(_BASE_META)

        train_data = [
            {"input": torch.randn(4, 6, 240)}
            for _ in range(4)
        ]

        history = trainer.train_stage2(train_data, epochs=2, lr=1e-3)

        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_evaluate_returns_result(self):
        trainer = FlowMatchingTrainer(_BASE_META)

        normal_files = [{"input": torch.randn(1, 6, 240)} for _ in range(6)]
        anomaly_files = [{"input": torch.randn(1, 6, 240)} for _ in range(6)]

        result = trainer.evaluate(normal_files, anomaly_files)

        assert isinstance(result, EvalResult)
        assert result.num_normal == 6
        assert result.num_anomaly == 6

    def test_full_two_stage_pipeline(self):
        trainer = FlowMatchingTrainer(_BASE_META)

        train_data = [
            {"input": torch.randn(4, 6, 240)}
            for _ in range(8)
        ]

        trainer.train_stage1(train_data, epochs=2, lr=1e-3)
        trainer.train_stage2(train_data, epochs=2, lr=1e-3)

        result = trainer.evaluate(train_data[:3], train_data[3:6])

        assert isinstance(result, EvalResult)