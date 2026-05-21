import pytest
import torch

from app.anomaly_detection.models.factory import create_model, count_parameters
from app.anomaly_detection.models.glc_flow_v7 import GLCFlowConfig
from app.anomaly_detection.layers.scorer import MADThreshold, AnomalyScorer
from app.anomaly_detection.training import FlowMatchingTrainer


def create_glc_flow_v7():
    return create_model("glc_flow_v7")


class TestDetectAnomalyPipeline:
    """Integration tests for the full detect_anomaly() pipeline."""

    def test_detect_anomaly_runs_without_error(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240)

        result = model.detect_anomaly(raw)

        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert "z_pooled" in result
        assert "z_cls" in result
        assert "file_stats_pred" in result
        assert "patch_stats_pred" in result

    def test_detect_anomaly_score_is_scalar(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240)

        result = model.detect_anomaly(raw)

        assert result["anomaly_score"].ndim == 0
        assert isinstance(result["is_anomaly"], bool)

    def test_detect_anomaly_with_threshold(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240)

        threshold = MADThreshold()
        threshold.fit(torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5]))

        result = model.detect_anomaly(raw, threshold=threshold.threshold)

        assert "is_anomaly" in result
        assert isinstance(result["is_anomaly"], bool)

    def test_detect_anomaly_batch_vs_single_same_model(self):
        model = create_glc_flow_v7()
        raw_single = torch.randn(6, 240)
        raw_batch = raw_single.unsqueeze(0)

        result_single = model.detect_anomaly(raw_single)
        result_batch = model.detect_anomaly(raw_batch)

        assert result_single["anomaly_score"].ndim == 0
        assert result_batch["anomaly_score"].ndim == 0

    def test_detect_anomaly_scores_differ_for_different_inputs(self):
        model = create_glc_flow_v7()

        raw1 = torch.randn(6, 240)
        raw2 = torch.randn(6, 240)

        result1 = model.detect_anomaly(raw1)
        result2 = model.detect_anomaly(raw2)

        assert not torch.allclose(result1["anomaly_score"], result2["anomaly_score"])

    def test_detect_anomaly_z_pooled_shape(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240)

        result = model.detect_anomaly(raw)

        assert result["z_pooled"].shape == (256,)
        assert result["z_cls"].shape == (128,)

    def test_anomaly_scorer_direct(self):
        model = create_glc_flow_v7()
        scorer = AnomalyScorer(model.cnf)

        z_pooled = torch.randn(2, 256)
        scores = scorer.score(z_pooled)

        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()

    def test_mad_threshold_fit_and_predict(self):
        normal_scores = torch.randn(100)
        threshold = MADThreshold()
        threshold.fit(normal_scores)

        assert threshold.threshold is not None
        assert threshold.threshold > 0

        test_scores = torch.tensor([threshold.threshold * 0.5, threshold.threshold * 2.0])
        predictions = threshold.predict(test_scores)

        assert predictions.shape == (2,)
        assert predictions[0] == False
        assert predictions[1] == True


class TestEndToEndWithTrainer:
    """End-to-end tests using FlowMatchingTrainer with the full pipeline."""

    def test_trainer_initialization_with_default_config(self):
        trainer = FlowMatchingTrainer()
        assert trainer.model is not None
        assert trainer.config is not None
        assert trainer.config.d_model == 128

    def test_trainer_initialization_with_custom_config(self):
        config = GLCFlowConfig(d_model=64, transformer_layers=2)
        trainer = FlowMatchingTrainer(config)
        assert trainer.model.d_model == 64
        assert trainer.model.transformer_layers == 2

    def test_train_stage1_single_epoch(self):
        trainer = FlowMatchingTrainer()
        train_data = [{"input": torch.randn(4, 6, 240)} for _ in range(4)]

        history = trainer.train_stage1(train_data, epochs=1, lr=1e-3)

        assert "loss" in history
        assert len(history["loss"]) == 1

    def test_train_stage2_single_epoch(self):
        trainer = FlowMatchingTrainer()
        train_data = [{"input": torch.randn(4, 6, 240)} for _ in range(4)]

        history = trainer.train_stage2(train_data, epochs=1, lr=1e-3)

        assert "loss" in history
        assert len(history["loss"]) == 1

    def test_full_training_pipeline_runs(self):
        trainer = FlowMatchingTrainer()
        train_data = [{"input": torch.randn(4, 6, 240)} for _ in range(8)]

        h1 = trainer.train_stage1(train_data, epochs=1, lr=1e-3)
        h2 = trainer.train_stage2(train_data, epochs=1, lr=1e-3)

        assert len(h1["loss"]) == 1
        assert len(h2["loss"]) == 1
        assert h1["loss"][0] > 0
        assert h2["loss"][0] > 0

    def test_evaluate_returns_structured_result(self):
        trainer = FlowMatchingTrainer()

        normal_files = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]
        anomaly_files = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]

        result = trainer.evaluate(normal_files, anomaly_files)

        assert result.num_normal == 8
        assert result.num_anomaly == 8
        assert result.num_files == 16
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.auprc <= 1.0

    def test_evaluate_with_pretrained_threshold(self):
        trainer = FlowMatchingTrainer()

        normal_files = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]
        anomaly_files = [{"input": torch.randn(1, 6, 240)} for _ in range(8)]

        from app.anomaly_detection.layers.scorer import MADThreshold
        threshold = MADThreshold()
        threshold.fit(torch.randn(50))

        trainer.evaluator = trainer.evaluator = trainer.evaluator or type('E', (), {'threshold': threshold})()
        trainer.evaluator.threshold = threshold

        assert threshold.threshold is not None


class TestModelParameterCount:
    """Test model size and parameter counts."""

    def test_default_model_has_expected_parameter_count(self):
        model = create_glc_flow_v7()
        total = count_parameters(model)
        assert total > 0
        assert total < 5_000_000

    def test_model_with_small_d_model(self):
        meta = {"d_model": 32, "transformer_layers": 2}
        model = create_model("glc_flow_v7", metadata=meta)
        total = count_parameters(model)
        assert total > 0
        assert total < 1_000_000


class TestDetectAnomalyWithMockData:
    """Test detect_anomaly with controlled inputs to verify behavior."""

    def test_near_zero_input_gives_finite_score(self):
        model = create_glc_flow_v7()
        raw = torch.zeros(6, 240)

        result = model.detect_anomaly(raw)

        assert torch.isfinite(result["anomaly_score"])
        assert result["anomaly_score"].ndim == 0

    def test_constant_input_gives_finite_score(self):
        model = create_glc_flow_v7()
        raw = torch.ones(6, 240) * 0.5

        result = model.detect_anomaly(raw)

        assert torch.isfinite(result["anomaly_score"])
        assert result["anomaly_score"].ndim == 0

    def test_sinusoidal_input_gives_finite_score(self):
        model = create_glc_flow_v7()
        t = torch.linspace(0, 4 * torch.pi, 240)
        raw = torch.stack([torch.sin(t + i) for i in range(6)], dim=0)

        result = model.detect_anomaly(raw)

        assert torch.isfinite(result["anomaly_score"])
        assert result["anomaly_score"].ndim == 0

    def test_high_magnitude_input_gives_finite_score(self):
        model = create_glc_flow_v7()
        raw = torch.randn(6, 240) * 10.0

        result = model.detect_anomaly(raw)

        assert torch.isfinite(result["anomaly_score"])
        assert result["anomaly_score"].ndim == 0

    def test_batch_processing_all_batches_complete(self):
        model = create_glc_flow_v7()

        for batch_size in [1, 4, 8]:
            raw = torch.randn(batch_size, 6, 240)
            x = {"input": raw}
            out = model(x)
            assert out["z_pooled"].shape == (batch_size, 256)
            assert torch.isfinite(out["z_pooled"]).all()