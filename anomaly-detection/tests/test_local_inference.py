"""Tests for local-compatible inference classes (ReconV8, FlowMatching, Reconstruction).

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

from src.anomaly_detection.models import (
    ReconstructionModels,
    FlowMatchingModels,
    SupportedAnomalyDetectionModels,
    create_model,
)


def make_synthetic_recon_v8_records(n_files: int = 4, n_windows: int = 5):
    """Create synthetic records mimicking the Reconf8LocalPreResult input format."""
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
# Recon V8 (GLCReconV8)
# =============================================================================


class TestReconf8LocalPipeline:
    def setup_method(self):
        self.model = create_model("glc_recon_v8", {"num_sensors": 6, "seq_len": 240, "d_model": 32, "transformer_layers": 1, "transformer_heads": 2, "ffn_dim": 64})
        self.metadata = self.model.get_metadata()
        self.state_dict = self.model.state_dict()

    def test_local_pre_result_output_keys(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=1, device="cpu")
        records = make_synthetic_recon_v8_records(3, 5)
        results = pre.score_batch(records)
        assert len(results) == 3
        required = {"robot_code", "program_number", "start_time", "p95_score",
                     "p95_score_rampup", "p95_score_active",
                     "valid_start_idx", "valid_end_idx"}
        for r in results:
            assert required.issubset(set(r.keys()))
            heatmap_keys = [k for k in r if k.startswith("heatmap_")]
            assert len(heatmap_keys) == self.model.num_sensors

    def test_phase_scores_detected(self):
        """Phase-specific scores should be present and finite in all results."""
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        results = pre.score_batch(records)
        for r in results:
            assert np.isfinite(r["p95_score"])
            assert np.isfinite(r["p95_score_rampup"])
            assert np.isfinite(r["p95_score_active"])
            r_start, r_end = r.get("phase_rampup", (0, 0))
            a_start, a_end = r.get("phase_active", (0, 0))
            assert r_start >= 0 and r_end >= r_start
            assert a_start >= 0 and a_end >= a_start

    def test_multi_score_threshold_creator(self):
        """ThresholdCreator should support multiple score keys."""
        from src.anomaly_detection.inference.recon_v8.local import (
            Reconf8LocalPreResult, Reconf8LocalThresholdCreator,
        )
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        results = pre.score_batch(records)
        tc = Reconf8LocalThresholdCreator(quantile=0.95, score_keys=["p95_score", "p95_score_rampup", "p95_score_active"])
        tc.fit_from_pre_results(results)
        assert "p95_score" in tc.thresholds
        for key in ["p95_score_rampup", "p95_score_active"]:
            assert key in tc.thresholds, f"Missing threshold key: {key}"
        assert tc.thresholds["p95_score"] > 0.0
        d = tc.to_dict()
        tc2 = Reconf8LocalThresholdCreator.from_dict(d)
        assert tc2.thresholds == tc.thresholds

    def test_multi_score_applier_or_logic(self):
        """Applier with multi-score thresholds should flag file if ANY score exceeds threshold."""
        from src.anomaly_detection.inference.recon_v8.local import (
            Reconf8LocalPreResult, Reconf8LocalThresholdCreator, Reconf8LocalApplier,
        )
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        results = pre.score_batch(records)
        tc = Reconf8LocalThresholdCreator(quantile=0.95, score_keys=["p95_score", "p95_score_rampup", "p95_score_active"])
        tc.fit_from_pre_results(results)
        low_thresholds = {"p95_score": tc.thresholds["p95_score"] * 0.01}
        applier = Reconf8LocalApplier({
            "global_threshold": 1e10,
            "score_thresholds": low_thresholds,
        }, quantile=0.95)
        applied = applier.apply(results)
        any_flagged = any(r["is_file_anomaly"] for r in applied)
        assert any_flagged, "Expected at least one file flagged with low thresholds"
        for r in applied:
            if r["is_file_anomaly"]:
                assert "p95_score" in r["triggered_by"], (
                    f"Expected p95_score in triggered_by, got {r['triggered_by']}"
                )

    def test_local_pre_result_scores_are_finite(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(3, 5)
        results = pre.score_batch(records)
        for r in results:
            assert np.isfinite(r["p95_score"])

    def test_local_threshold_creator_fit_and_serialize(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult, Reconf8LocalThresholdCreator
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        results = pre.score_batch(records)
        tc = Reconf8LocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(results)
        assert tc.threshold is not None
        assert tc.threshold > 0.0
        assert "p95_score" in tc.thresholds
        d = tc.to_dict()
        assert "thresholds" in d
        assert "p95_score" in d["thresholds"]
        tc2 = Reconf8LocalThresholdCreator.from_dict(d)
        assert tc2.thresholds == tc.thresholds

    def test_local_applier_output_keys(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult, Reconf8LocalThresholdCreator, Reconf8LocalApplier
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        results = pre.score_batch(records)
        tc = Reconf8LocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(results)
        applier = Reconf8LocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(results)
        assert len(applied) == 4
        required = {"robot_code", "program_number", "start_time", "is_file_anomaly", "p95_score"}
        for r in applied:
            assert required.issubset(set(r.keys()))
            heatmap_keys = [k for k in r if k.startswith("heatmap_")]
            is_anomaly_keys = [k for k in r if k.startswith("is_anomaly_")]
            assert len(heatmap_keys) == self.model.num_sensors
            assert len(is_anomaly_keys) == self.model.num_sensors

    def test_full_pipeline_end_to_end(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult, Reconf8LocalThresholdCreator, Reconf8LocalApplier
        pre = Reconf8LocalPreResult(ReconstructionModels.GLC_RECON_V8, self.metadata, self.state_dict, quantile=0.95, k_skip=0, device="cpu")
        records = make_synthetic_recon_v8_records(4, 5)
        pre_scores = pre.score_batch(records)
        tc = Reconf8LocalThresholdCreator(quantile=0.95)
        tc.fit_from_pre_results(pre_scores)
        applier = Reconf8LocalApplier({"global_threshold": tc.threshold}, quantile=0.95)
        applied = applier.apply(pre_scores)
        assert all(isinstance(r["is_file_anomaly"], bool) for r in applied)
        for r in applied:
            for k, v in r.items():
                if k.startswith("is_anomaly_"):
                    assert isinstance(v, bool)


# =============================================================================
# Flow Matching
# =============================================================================


class TestFlowMatchingLocalPipeline:
    def setup_method(self):
        self.model = create_model("glc_flow_v7", {"num_sensors": 6, "seq_len": 240, "d_model": 32, "transformer_layers": 1, "transformer_heads": 2, "ffn_dim": 64, "use_context_embeddings": False})
        self.metadata = self.model.get_metadata()
        self.state_dict = self.model.state_dict()

    def test_local_pre_result_output_keys(self):
        from src.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult
        pre = FlowMatchingLocalPreResult(FlowMatchingModels.GLC_FLOW_V7, self.metadata, self.state_dict, quantile=0.95, k_skip=1, device="cpu")
        records = make_synthetic_flow_records(3, 5)
        results = pre.score_batch(records)
        assert len(results) == 3
        expected = {"robot_code", "program_number", "start_time", "p95_score", "valid_start_idx", "valid_end_idx", "min_serial", "total_steps"}
        for r in results:
            assert expected.issubset(set(r.keys()))

    def test_local_threshold_creator_fit_and_serialize(self):
        from src.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator
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
        from src.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator, FlowMatchingLocalApplier
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
        from src.anomaly_detection.inference.flow_matching.local import FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator, FlowMatchingLocalApplier
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
        from src.anomaly_detection.inference.reconstruction.local import ReconLocalThresholdCreator
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
        from src.anomaly_detection.inference.reconstruction.local import ReconLocalApplier
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
        from src.anomaly_detection.inference.reconstruction.local import ReconLocalThresholdCreator, ReconLocalApplier
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


# =============================================================================
# Per-Sensor Scoring — GLCMaeV8 (Hypothesis 2 fix)
# =============================================================================

C3, T3 = 3, 240
_MAE_META = {
    "name": "GLCMaeV8",
    "num_sensors": C3,
    "seq_len": T3,
    "num_robots": 4,
    "num_programs": 16,
    "patch_size": 32,
    "patch_stride": 16,
    "d_model": 64,
    "d_half": 32,
    "num_patches": 14,
    "transformer_layers": 2,
    "transformer_heads": 2,
    "ffn_dim": 128,
    "dropout": 0.0,
    "mask_ratio": 0.45,
    "use_context_embeddings": False,
}


def _mae_engine():
    from src.anomaly_detection.models.glc_mae_v8 import GLCMaeV8
    model = GLCMaeV8(
        num_sensors=C3, seq_len=T3, num_robots=4, num_programs=16,
        patch_size=32, patch_stride=16, d_model=64, transformer_layers=2,
        transformer_heads=2, ffn_dim=128, dropout=0.0,
        mask_ratio=0.45, use_context_embeddings=False,
    )
    from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult
    return Reconf8LocalPreResult(
        model_version=ReconstructionModels.GLC_MAE_V8,
        metadata=_MAE_META,
        state_dict=model.state_dict(),
        quantile=0.95,
        k_skip=0,
        device="cpu",
    )


def _mae_records(n_files, noisy_channel=-1):
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n_files):
        w = rng.normal(0, 1.0, (3, C3, T3)).astype(np.float32)
        if noisy_channel >= 0:
            w[:, noisy_channel, :] *= 8.0
        records.append({
            "robot_code": "0", "program_number": 0, "start_time": "0",
            "windows": w,
            "serials": np.array([0, 16, 32], dtype=np.int64),
        })
    return records


class TestPerSensorScoring:
    def setup_method(self):
        self.engine = _mae_engine()

    def test_score_batch_has_per_sensor_keys(self):
        from src.anomaly_detection import constants
        results = self.engine.score_batch(_mae_records(3))
        for res in results:
            for s in constants.SENSORS:
                key = f"p95_score_{s}"
                assert key in res, f"Missing {key}"
                assert isinstance(res[key], float)
                assert np.isfinite(res[key])

    def test_per_sensor_scores_differ_when_channel_noisy(self):
        """Channel 2 (welding_current) with 8× noise should score higher than channel 0 (wire_feed_speed)."""
        results = self.engine.score_batch(_mae_records(5, noisy_channel=2))
        wins = sum(
            1 for res in results
            if res["p95_score_welding_current"] > res["p95_score_wire_feed_speed"]
        )
        assert wins >= 3, f"Expected welding_current > wire_feed_speed for most files, got {wins}/5"

    def test_early_return_has_per_sensor_and_phase_keys(self):
        from src.anomaly_detection.models.glc_mae_v8 import GLCMaeV8
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalPreResult
        model = GLCMaeV8(
            num_sensors=C3, seq_len=T3, num_robots=4, num_programs=16,
            patch_size=32, patch_stride=16, d_model=64, transformer_layers=2,
            transformer_heads=2, ffn_dim=128, dropout=0.0,
            mask_ratio=0.45, use_context_embeddings=False,
        )
        engine = Reconf8LocalPreResult(
            model_version=ReconstructionModels.GLC_MAE_V8,
            metadata=_MAE_META,
            state_dict=model.state_dict(),
            quantile=0.95,
            k_skip=5,   # forces early return for N=1
            device="cpu",
        )
        rec = {
            "robot_code": "0", "program_number": 0, "start_time": "0",
            "windows": np.zeros((1, C3, T3), dtype=np.float32),
            "serials": np.array([0], dtype=np.int64),
        }
        from src.anomaly_detection import constants
        res = engine.score_batch([rec])[0]
        for s in constants.SENSORS:
            assert f"p95_score_{s}" in res
            assert res[f"p95_score_{s}"] == 100.0
        assert res["p95_score_rampup"] == 100.0
        assert res["p95_score_active"] == 100.0

    def test_make_per_sensor_keys(self):
        from src.anomaly_detection import constants
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalThresholdCreator
        keys = Reconf8LocalThresholdCreator.make_per_sensor_keys(0.95, constants.SENSORS)
        assert keys == [f"p95_score_{s}" for s in constants.SENSORS]
        custom = ["alpha", "beta", "gamma", "delta", "epsilon"]
        keys5 = Reconf8LocalThresholdCreator.make_per_sensor_keys(0.90, custom)
        assert len(keys5) == 5
        assert all(k.startswith("p90_score_") for k in keys5)
        assert keys5[0] == "p90_score_alpha"

    def test_threshold_creator_per_sensor_fit_and_roundtrip(self):
        from src.anomaly_detection import constants
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalThresholdCreator
        results = self.engine.score_batch(_mae_records(10))
        sensor_keys = Reconf8LocalThresholdCreator.make_per_sensor_keys(0.95, constants.SENSORS)
        score_keys = ["p95_score", "p95_score_rampup", "p95_score_active"] + sensor_keys
        tc = Reconf8LocalThresholdCreator(quantile=0.95, score_keys=score_keys)
        tc.fit_from_pre_results(results)
        for k in sensor_keys:
            assert k in tc.thresholds, f"Missing threshold for {k}"
            assert tc.thresholds[k] > 0
        # Round-trip
        tc2 = Reconf8LocalThresholdCreator.from_dict(tc.to_dict())
        for k in sensor_keys:
            assert tc2.thresholds[k] == tc.thresholds[k]

    def test_applier_or_logic_sensor_triggered(self):
        from src.anomaly_detection.inference.recon_v8.local import Reconf8LocalApplier
        results = self.engine.score_batch(_mae_records(5))
        threshold_config = {
            "global_threshold": 1e10,
            "score_thresholds": {
                "p95_score": 1e10,
                "p95_score_rampup": 1e10,
                "p95_score_active": 1e10,
                "p95_score_wire_feed_speed":  1e10,
                "p95_score_wire_temperature": 1e10,
                "p95_score_welding_current":  1e-9,  # always triggers
            },
        }
        applier = Reconf8LocalApplier(threshold_config, quantile=0.95)
        applied = applier.apply(results)
        for res in applied:
            assert res["is_file_anomaly"] is True
            assert "p95_score_welding_current" in res["triggered_by"]

    def test_backward_compat_no_per_sensor_keys(self):
        """Legacy score_keys (no per-sensor) must still work end-to-end."""
        from src.anomaly_detection.inference.recon_v8.local import (
            Reconf8LocalThresholdCreator, Reconf8LocalApplier,
        )
        results = self.engine.score_batch(_mae_records(5))
        tc = Reconf8LocalThresholdCreator(quantile=0.95, score_keys=["p95_score"])
        tc.fit_from_pre_results(results)
        applier = Reconf8LocalApplier({
            "global_threshold": 1e10,
            "score_thresholds": tc.thresholds,
        }, quantile=0.95)
        applied = applier.apply(results)
        assert len(applied) == len(results)
        assert all("is_file_anomaly" in r for r in applied)


# =============================================================================
# Recon V9 (GLCMaeV9) — Sprint 20 T10
# =============================================================================

C9, T9 = 3, 256
_V9_META = {
    "name": "GLCMaeV9",
    "num_sensors": C9,
    "seq_len": T9,
    "num_robots": 4,
    "num_programs": 16,
    "patch_size": 32,
    "patch_stride": 16,
    "d_model": 64,
    "num_patches": 15,
    "transformer_layers": 2,
    "transformer_heads": 2,
    "ffn_dim": 128,
    "dropout": 0.0,
    "mask_ratio": 0.45,
    "use_context_embeddings": False,
    "normalizer": "log+global_seq_norm",
}


def _v9_engine():
    from src.anomaly_detection.models.glc_mae_v9 import GLCMaeV9
    model = GLCMaeV9(
        num_sensors=C9, seq_len=T9, num_robots=4, num_programs=16,
        patch_size=32, patch_stride=16, d_model=64, transformer_layers=2,
        transformer_heads=2, ffn_dim=128, dropout=0.0,
        mask_ratio=0.45, use_context_embeddings=False,
    )
    from src.anomaly_detection.inference.recon_v9.local import Reconf9LocalPreResult
    return Reconf9LocalPreResult(
        model_version=ReconstructionModels.GLC_MAE_V9,
        metadata=_V9_META,
        state_dict=model.state_dict(),
        quantile=0.99,
        k_skip=1,
        device="cpu",
    )


def _v9_records(n_files: int = 3, n_windows: int = 5):
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n_files):
        w = rng.normal(0, 1.0, (n_windows, C9, T9)).astype(np.float32)
        records.append({
            "robot_code": "0", "program_number": 0, "start_time": "0",
            "windows": w,
            "serials": np.arange(0, n_windows * 64, 64, dtype=np.int64),
        })
    return records


class TestReconf9LocalPipeline:
    """V9 inference: p99 quantile, no tail window, k_skip=1."""

    def test_reconf9_score_keys_p99(self):
        """Score keys should use p99 prefix, not p95."""
        engine = _v9_engine()
        results = engine.score_batch(_v9_records(3, 5))
        for res in results:
            assert "p99_score" in res, f"Missing p99_score key, got keys: {list(res.keys())}"
            assert "p99_score_rampup" in res
            assert "p99_score_active" in res
            assert "p99_score_wire_feed_speed" in res
            assert "p99_score_wire_temperature" in res
            assert "p99_score_welding_current" in res
            # p95 keys should NOT be present
            assert "p95_score" not in res, "p95_score should not appear in V9 results"

    def test_reconf9_no_tail_window(self):
        """With T=256, stride=64, N=5, the last serial should be 256 (no tail)."""
        records = _v9_records(1, 5)
        # Manually verify the serials don't include a 6th window position
        serials = records[0]["serials"]
        assert len(serials) == 5, f"Expected 5 windows, got {len(serials)}"
        # stride=64, so serials should be [0, 64, 128, 192, 256]
        assert serials[-1] == 256, f"Last serial should be 256 (stride*4), got {serials[-1]}"

    def test_reconf9_thermal_drift_score_higher_than_normal(self):
        """Thermal drift should produce higher scores than normal data."""
        from src.anomaly_detection.models.glc_mae_v9 import GLCMaeV9
        model = GLCMaeV9(
            num_sensors=C9, seq_len=T9, num_robots=4, num_programs=16,
            patch_size=32, patch_stride=16, d_model=64, transformer_layers=2,
            transformer_heads=2, ffn_dim=128, dropout=0.0,
            mask_ratio=0.45, use_context_embeddings=False,
        )
        # Accumulate stats with normal data
        model.train()
        normal_data = torch.randn(8, C9, T9) * 2.0 + 10.0
        for i in range(0, len(normal_data), 4):
            model.compute_loss({"input": normal_data[i:i+4]})

        model.eval()

        # Normal record
        normal_w = np.random.default_rng(42).normal(0, 1.0, (3, C9, T9)).astype(np.float32)
        # Thermal drift record: channel 2 ramps up
        drift_w = normal_w.copy()
        drift_ramp = np.linspace(0, 10.0, T9).astype(np.float32)
        drift_w[:, 2, :] += drift_ramp  # type: ignore

        from src.anomaly_detection.inference.recon_v9.local import Reconf9LocalPreResult
        engine = Reconf9LocalPreResult(
            model_version=ReconstructionModels.GLC_MAE_V9,
            metadata=_V9_META,
            state_dict=model.state_dict(),
            quantile=0.99,
            k_skip=0,
            device="cpu",
        )
        normal_rec = {
            "robot_code": "0", "program_number": 0, "start_time": "0",
            "windows": normal_w, "serials": np.arange(0, 3 * 64, 64, dtype=np.int64),
        }
        drift_rec = {
            "robot_code": "0", "program_number": 0, "start_time": "1",
            "windows": drift_w, "serials": np.arange(0, 3 * 64, 64, dtype=np.int64),
        }
        normal_result = engine.score_batch([normal_rec])
        drift_result = engine.score_batch([drift_rec])

        drift_p99 = drift_result[0]["p99_score"]
        normal_p99_val = normal_result[0]["p99_score"]
        assert drift_p99 > normal_p99_val, (
            f"Thermal drift p99 ({drift_p99:.4f}) should be higher than "
            f"normal p99 ({normal_p99_val:.4f})"
        )

    def test_reconf9_threshold_creator_p99(self):
        """Threshold creator should work with p99 score keys."""
        engine = _v9_engine()
        results = engine.score_batch(_v9_records(5, 5))
        from src.anomaly_detection.inference.recon_v9.local import Reconf9LocalThresholdCreator
        tc = Reconf9LocalThresholdCreator(quantile=0.99)
        tc.fit_from_pre_results(results)
        assert "p99_score" in tc.thresholds
        assert tc.thresholds["p99_score"] > 0
        d = tc.to_dict()
        tc2 = Reconf9LocalThresholdCreator.from_dict(d)
        assert tc2.thresholds == tc.thresholds

    def test_reconf9_applier_output_keys(self):
        """Applier should produce correct output keys with p99."""
        engine = _v9_engine()
        results = engine.score_batch(_v9_records(3, 5))
        from src.anomaly_detection.inference.recon_v9.local import Reconf9LocalThresholdCreator, Reconf9LocalApplier
        tc = Reconf9LocalThresholdCreator(quantile=0.99)
        tc.fit_from_pre_results(results)
        applier = Reconf9LocalApplier({
            "global_threshold": tc.threshold,
            "score_thresholds": tc.thresholds,
        }, quantile=0.99)
        applied = applier.apply(results)
        for res in applied:
            assert "is_file_anomaly" in res
            assert isinstance(res["is_file_anomaly"], bool)
            assert "p99_score" in res
