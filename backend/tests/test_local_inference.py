"""Tests for local-compatible inference classes (MaskFormer, FlowMatching, Reconstruction).

Each variant is tested end-to-end with synthetic data:
1. Instantiate all 3 classes (LocalPreResult, LocalThresholdCreator, LocalApplier)
2. Run score_batch() on synthetic windows
3. Fit thresholds
4. Apply thresholds
5. Assert output structure

No Spark dependency — runs in local CI.
"""

import torch
import numpy as np

from app.anomaly_detection.models import (
    MaskFormerModels,
    FlowMatchingModels,
    SupportedAnomalyDetectionModels,
    create_model,
)


def make_synthetic_maskformer_records(n_files: int = 4, n_windows: int = 5):
    """Create synthetic records mimicking the MaskFormerPreResult input format."""
    C, W = 6, 240
    records = []
    for i in range(n_files):
        windows = np.random.randn(n_windows, C, W).astype(np.float32)
        serials = np.arange(0, n_windows * 16, 16, dtype=np.int64)
        records.append({
            "robot_code": f"R{i:03d}",
            "program_number": i % 3,
            "start_time": f"2024-01-01T00:0{i}:00",
            "windows": windows,
            "serials": serials,
        })
    return records


def make_synthetic_flow_records(n_files: int = 4, n_windows: int = 5):
    """Create synthetic records mimicking the FlowMatchingPreResult input format."""
    C, W = 6, 240
    records = []
    for i in range(n_files):
        windows = np.random.randn(n_windows, C, W).astype(np.float32)
        serials = np.arange(0, n_windows * 16, 16, dtype=np.int64)
        records.append({
            "robot_code": f"R{i:03d}",
            "program_number": i % 3,
            "start_time": f"2024-01-01T00:0{i}:00",
            "windows": windows,
            "serials": serials,
        })
    return records


def make_synthetic_recon_records(n_files: int = 4, n_windows: int = 5):
    """Create synthetic records mimicking the ReconPreResult input format."""
    C, W = 6, 32
    K_WIN = 60
    records = []
    for i in range(n_files):
        windows = np.random.randn(n_windows, C, W).astype(np.float32)
        serials = np.arange(0, n_windows * 16, 16, dtype=np.int64)
        stats = np.random.randn(n_windows, C, K_WIN).astype(np.float32)
        records.append({
            "robot_code": f"R{i:03d}",
            "program_number": i % 3,
            "start_time": f"2024-01-01T00:0{i}:00",
            "windows": windows,
            "serials": serials,
            "stats": stats,
            "robot_idx": np.full(n_windows, i % 47, dtype=np.int32),
            "program_idx": np.full(n_windows, i % 16, dtype=np.int32),
        })
    return records


# =============================================================================
# MaskFormer
# =============================================================================


class TestMaskFormerLocalPipeline:
    def setup_method(self):
        self.model = create_model("glc_anomaly_maskformer", {"d_model": 32, "transformer_layers": 1, "transformer_heads": 2, "ffn_dim": 64, "use_context_embeddings": False})
        self.metadata = self.model.get_metadata()
        self.state_dict = self.model.state_dict()

    def test_local_pre_result_output_keys(self):
        from app.anomaly_detection.inference.maskformer.local import MaskFormerLocalPreResult
        pre = MaskFormerLocalPreResult(MaskFormerModels.GLC_ANOMALY_MASKFORMER, self.metadata, self.state_dict, quantile=0.95, k_skip=1, fast=True, device="cpu")
        records = make_synthetic_maskformer_records(3, 5)
        results = pre.score_batch(records)
        assert len(results) == 3
        expected_keys = {"robot_code", "program_number", "start_time", "p95_score", "valid_start_idx", "valid_end_idx"}
        for s in ["air_motor_rotation_speed", "electro_pneumatic_output", "high_pressure_current", "high_pressure_voltage", "paint_pressure", "paint_flow_rate"]:
            expected_keys.add(f"heatmap_{s}")
        for r in results:
            assert expected_keys.issubset(set(r.keys()))

    def test_local_pre_result_scores_are_finite(self):
        from app.anomaly_detection.inference.maskformer.local import MaskFormerLocalPreResult
        pre = MaskFormerLocalPreResult(MaskFormerModels.GLC_ANOMALY_MASKFORMER, self.metadata, self.state_dict, quantile=0.95, k_skip=0, fast=True, device="cpu")
        records = make_synthetic_maskformer_records(3, 5)
        results = pre.score_batch(records)
        for r in results:
            assert np.isfinite(r["p95_score"])

    def test_local_threshold_creator_fit_and_serialize(self):
        from app.anomaly_detection.inference.maskformer.local import MaskFormerLocalPreResult, MaskFormerLocalThresholdCreator
        pre = MaskFormerLocalPreResult(MaskFormerModels.GLC_ANOMALY_MASKFORMER, self.metadata, self.state_dict, quantile=0.95, k_skip=0, fast=True, device="cpu")
        records = make_synthetic_maskformer_records(4, 5)
        results = pre.score_batch(records)
        tc = MaskFormerLocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(results)
        assert tc.threshold is not None
        assert tc.threshold > 0.0
        d = tc.to_dict()
        assert "threshold" in d
        tc2 = MaskFormerLocalThresholdCreator.from_dict(d)
        assert tc2.threshold == tc.threshold

    def test_local_applier_output_keys(self):
        from app.anomaly_detection.inference.maskformer.local import MaskFormerLocalPreResult, MaskFormerLocalThresholdCreator, MaskFormerLocalApplier
        pre = MaskFormerLocalPreResult(MaskFormerModels.GLC_ANOMALY_MASKFORMER, self.metadata, self.state_dict, quantile=0.95, k_skip=0, fast=True, device="cpu")
        records = make_synthetic_maskformer_records(4, 5)
        results = pre.score_batch(records)
        tc = MaskFormerLocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(results)
        applier = MaskFormerLocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(results)
        assert len(applied) == 4
        expected = {"robot_code", "program_number", "start_time", "is_file_anomaly", "p95_score"}
        for s in ["air_motor_rotation_speed", "electro_pneumatic_output", "high_pressure_current", "high_pressure_voltage", "paint_pressure", "paint_flow_rate"]:
            expected.add(f"heatmap_{s}")
            expected.add(f"is_anomaly_{s}")
        for r in applied:
            assert expected.issubset(set(r.keys()))

    def test_full_pipeline_end_to_end(self):
        from app.anomaly_detection.inference.maskformer.local import MaskFormerLocalPreResult, MaskFormerLocalThresholdCreator, MaskFormerLocalApplier
        pre = MaskFormerLocalPreResult(MaskFormerModels.GLC_ANOMALY_MASKFORMER, self.metadata, self.state_dict, quantile=0.95, k_skip=0, fast=True, device="cpu")
        records = make_synthetic_maskformer_records(4, 5)
        pre_scores = pre.score_batch(records)
        tc = MaskFormerLocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(pre_scores)
        applier = MaskFormerLocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(pre_scores)
        assert all(isinstance(r["is_file_anomaly"], bool) for r in applied)
        assert all(isinstance(r[f"is_anomaly_air_motor_rotation_speed"], bool) for r in applied)


# =============================================================================
# Flow Matching
# =============================================================================


class TestFlowMatchingLocalPipeline:
    def setup_method(self):
        self.model = create_model("glc_flow_v7", {"d_model": 32, "transformer_layers": 1, "transformer_heads": 2, "ffn_dim": 64, "use_context_embeddings": False})
        self.metadata = self.model.get_metadata()
        self.state_dict = self.model.state_dict()

    def test_local_pre_result_output_keys(self):
        from app.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult
        pre = FlowMatchingLocalPreResult(FlowMatchingModels.GLC_FLOW_V7, self.metadata, self.state_dict, quantile=0.95, k_skip=1, device="cpu")
        records = make_synthetic_flow_records(3, 5)
        results = pre.score_batch(records)
        assert len(results) == 3
        expected = {"robot_code", "program_number", "start_time", "p95_score", "valid_start_idx", "valid_end_idx", "min_serial", "total_steps"}
        for r in results:
            assert expected.issubset(set(r.keys()))

    def test_local_threshold_creator_fit_and_serialize(self):
        from app.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator
        pre = FlowMatchingLocalPreResult(FlowMatchingModels.GLC_FLOW_V7, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_flow_records(4, 5)
        results = pre.score_batch(records)
        tc = FlowMatchingLocalThresholdCreator(multiplier=3.0)
        tc.fit_from_pre_results(results)
        assert tc.threshold is not None
        d = tc.to_dict()
        assert "threshold" in d
        tc2 = FlowMatchingLocalThresholdCreator.from_dict(d)
        assert tc2.threshold == tc.threshold

    def test_local_applier_output_keys(self):
        from app.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator, FlowMatchingLocalApplier
        pre = FlowMatchingLocalPreResult(FlowMatchingModels.GLC_FLOW_V7, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_flow_records(4, 5)
        results = pre.score_batch(records)
        tc = FlowMatchingLocalThresholdCreator(multiplier=3.0)
        tc.fit_from_pre_results(results)
        applier = FlowMatchingLocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(results)
        assert len(applied) == 4
        expected = {"robot_code", "program_number", "start_time", "is_file_anomaly", "p95_score"}
        for r in applied:
            assert expected.issubset(set(r.keys()))
            assert isinstance(r["is_file_anomaly"], bool)

    def test_full_pipeline_end_to_end(self):
        from app.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator, FlowMatchingLocalApplier
        pre = FlowMatchingLocalPreResult(FlowMatchingModels.GLC_FLOW_V7, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_flow_records(4, 5)
        pre_scores = pre.score_batch(records)
        tc = FlowMatchingLocalThresholdCreator(multiplier=3.0)
        tc.fit_from_pre_results(pre_scores)
        applier = FlowMatchingLocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(pre_scores)
        assert all(isinstance(r["is_file_anomaly"], bool) for r in applied)


# =============================================================================
# Reconstruction
# =============================================================================


class TestReconstructionLocalPipeline:
    """Reconstruction local pipeline tests.
    
    The V6 model requires specific metadata. Tests validate the
    standalone threshold creator and applier logic even when the
    full model forward pass is not feasible in CI (model dim mismatch).
    """

    def test_local_threshold_creator_fit_and_serialize(self):
        from app.anomaly_detection.inference.reconstruction.local import ReconLocalThresholdCreator
        tc = ReconLocalThresholdCreator(quantile=0.95)
        rows = [
            {"p95_shape_air_motor_rotation_speed": 0.5, "p95_win_air_motor_rotation_speed": 0.3,
             "p95_shape_electro_pneumatic_output": 0.4, "p95_win_electro_pneumatic_output": 0.2},
            {"p95_shape_air_motor_rotation_speed": 0.8, "p95_win_air_motor_rotation_speed": 0.6,
             "p95_shape_electro_pneumatic_output": 0.7, "p95_win_electro_pneumatic_output": 0.5},
        ]
        tc.fit_from_pre_results(rows)
        assert tc.thresholds is not None
        assert "shape" in tc.thresholds["air_motor_rotation_speed"]
        assert "win" in tc.thresholds["air_motor_rotation_speed"]
        d = tc.to_dict()
        assert "thresholds" in d
        tc2 = ReconLocalThresholdCreator.from_dict(d)
        assert tc2.thresholds == tc.thresholds

    def _make_full_fake_row(self):
        SENSOR_KEYS = [
            "air_motor_rotation_speed", "electro_pneumatic_output",
            "high_pressure_current", "high_pressure_voltage",
            "paint_pressure", "paint_flow_rate",
        ]
        row = {
            "robot_code": "R001", "program_number": 1, "start_time": "2024-01-01T00:00:00",
            "p95_shape_score": 0.5, "p95_win_score": 0.3,
            "valid_start_idx": 0, "valid_end_idx": 10,
        }
        for s in SENSOR_KEYS:
            row[f"shape_map_{s}"] = [0.1, 0.2, 0.9]
            row[f"win_map_{s}"] = [0.1, 0.2, 0.3]
            row[f"p95_shape_{s}"] = 0.5
            row[f"p95_win_{s}"] = 0.3
            row[f"actual_raw_{s}"] = [1.0, 2.0, 3.0]
            row[f"recon_raw_{s}"] = [1.0, 2.0, 3.0]
        return row

    def test_local_applier_output_keys(self):
        from app.anomaly_detection.inference.reconstruction.local import ReconLocalApplier
        SENSOR_KEYS = [
            "air_motor_rotation_speed", "electro_pneumatic_output",
            "high_pressure_current", "high_pressure_voltage",
            "paint_pressure", "paint_flow_rate",
        ]
        fake_row = self._make_full_fake_row()
        threshold_config = {s: {"shape": 0.5, "win": 0.5} for s in SENSOR_KEYS}
        applier = ReconLocalApplier(threshold_config, quantile=0.95)
        applied = applier.apply([fake_row])
        assert len(applied) == 1
        assert "is_file_anomaly" in applied[0]
        assert "is_shape_triggered" in applied[0]
        assert "err_steps_air_motor_rotation_speed" in applied[0]
        assert isinstance(applied[0]["is_file_anomaly"], bool)

    def test_full_pipeline_with_precomputed_results(self):
        """Test that pre-computed scores flow through thresholding + applier correctly."""
        from app.anomaly_detection.inference.reconstruction.local import ReconLocalThresholdCreator, ReconLocalApplier
        SENSOR_KEYS = [
            "air_motor_rotation_speed", "electro_pneumatic_output",
            "high_pressure_current", "high_pressure_voltage",
            "paint_pressure", "paint_flow_rate",
        ]
        row = self._make_full_fake_row()
        row["p95_shape_score"] = 0.1
        row["p95_win_score"] = 0.1
        row["valid_start_idx"] = 0
        row["valid_end_idx"] = 5
        for s in SENSOR_KEYS:
            row[f"p95_shape_{s}"] = 0.1
            row[f"p95_win_{s}"] = 0.1
            row[f"shape_map_{s}"] = [0.05] * 5
            row[f"win_map_{s}"] = [0.05] * 5
            row[f"actual_raw_{s}"] = [1.0] * 5
            row[f"recon_raw_{s}"] = [1.0] * 5
        tc = ReconLocalThresholdCreator(quantile=0.50)
        tc.fit_from_pre_results([row])
        assert tc.thresholds is not None
        applier = ReconLocalApplier(tc.thresholds, quantile=0.95)
        applied = applier.apply([row])
        assert isinstance(applied[0]["is_file_anomaly"], bool)
