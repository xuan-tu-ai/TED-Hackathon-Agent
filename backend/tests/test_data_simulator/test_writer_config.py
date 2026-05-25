"""Tests for CSV writer and config (Sprint 14, T11, T13)."""

import csv
import os
import tempfile
import json

import pytest
from data_simulator.batch_generator import BatchGenerator
from data_simulator.writer import CSVWriter
from data_simulator.config import (
    SimulatorConfig,
    PhysicsConfig,
    NoiseConfig,
    AnomalyConfig,
    OutputConfig,
)


class TestSimulatorConfig:
    """T13: Config tests."""

    def test_default_config(self):
        cfg = SimulatorConfig()
        assert cfg.seed == 42
        assert cfg.num_days == 30
        assert cfg.robots == [0, 1, 2, 3]
        assert isinstance(cfg.physics, PhysicsConfig)
        assert isinstance(cfg.noise, NoiseConfig)
        assert isinstance(cfg.anomaly, AnomalyConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_to_dict_round_trip(self):
        cfg = SimulatorConfig(seed=99, num_days=10)
        d = cfg.to_dict()
        assert d["seed"] == 99
        assert d["num_days"] == 10
        restored = SimulatorConfig.from_dict(d)
        assert restored.seed == 99
        assert restored.num_days == 10

    def test_to_json_round_trip(self):
        cfg = SimulatorConfig(seed=123)
        json_str = cfg.to_json()
        restored = SimulatorConfig.from_json(json_str)
        assert restored.seed == 123

    def test_custom_physics(self):
        cfg = SimulatorConfig(physics=PhysicsConfig(current_damping_ratio=0.7))
        assert cfg.physics.current_damping_ratio == 0.7

    def test_custom_anomaly(self):
        cfg = SimulatorConfig(anomaly=AnomalyConfig(spike_magnitude_multiplier=5.0))
        assert cfg.anomaly.spike_magnitude_multiplier == 5.0


class TestCSVWriter:
    """T11: CSV writer tests."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_train_csv_header(self, tmpdir):
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir, chunk_size=5000)
        path, _ = writer.write_train(gen, num_days=1, robots=[0])

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        expected = [
            "cycle_id", "robot_code", "start_time", "program_number",
            "serial_number", "phase",
            "wire_feed_speed_mm_min", "wire_temperature_c",
            "welding_current_a", "anomaly_type", "timestamp",
        ]
        assert header == expected

    def test_train_csv_data_types(self, tmpdir):
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        path, _ = writer.write_train(gen, num_days=1, robots=[0])

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        # Check column values are valid
        assert row["cycle_id"].isdigit()
        assert row["robot_code"].isdigit()
        assert row["program_number"].isdigit()
        assert row["serial_number"].isdigit()
        assert row["phase"] in ("idle", "rampup", "active", "winddown", "shutdown")
        assert row["anomaly_type"] == "none"
        assert row["timestamp"].endswith("+00:00")

    def test_train_file_has_data(self, tmpdir):
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        path, count = writer.write_train(gen, num_days=1, robots=[0])

        assert count > 0
        assert os.path.getsize(path) > 0

        # Count rows in file (minus header)
        with open(path) as f:
            file_rows = sum(1 for _ in f) - 1  # subtract header
        assert file_rows == count

    def test_test_csv_and_labels(self, tmpdir):
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        csv_path, label_path, _ = writer.write_test(
            gen, num_cycles=20, anomaly_ratio=0.5, seed=42
        )

        # Check CSV has data
        assert os.path.getsize(csv_path) > 0

        # Check labels
        with open(label_path) as f:
            reader = csv.DictReader(f)
            labels = list(reader)
        assert len(labels) == 20

        anomaly_types = {l["anomaly_type"] for l in labels}
        assert "normal" in anomaly_types
        # At least some anomaly types present
        assert len(anomaly_types) > 1

    def test_test_label_fields(self, tmpdir):
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        _, label_path, _ = writer.write_test(
            gen, num_cycles=5, anomaly_ratio=1.0, seed=42
        )

        with open(label_path) as f:
            reader = csv.DictReader(f)
            for label in reader:
                assert "cycle_id" in label
                assert label["cycle_id"].isdigit()
                assert "robot_code" in label
                assert "start_time" in label
                assert "program_number" in label
                assert "is_anomaly" in label
                assert "anomaly_type" in label
                assert label["is_anomaly"] in ("0", "1")
                assert label["anomaly_type"] in (
                    "spike", "thermal_drift", "flatline", "phase_shift",
                    "wire_stutter", "arc_blow", "gas_blockage",
                    "sticky_tc", "cable_wear", "overshoot_degradation",
                    "normal",
                )

    def test_chunked_writing(self, tmpdir):
        """Test with very small chunk size to force multiple flushes."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir, chunk_size=100)
        path, count = writer.write_train(gen, num_days=1, robots=[0])

        with open(path) as f:
            file_rows = sum(1 for _ in f) - 1
        assert file_rows == count

    def test_output_directory_created(self, tmpdir):
        """Writer should create output directory if it doesn't exist."""
        nested_dir = os.path.join(tmpdir, "nested", "output")
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=nested_dir)
        path, _ = writer.write_train(gen, num_days=1, robots=[0])
        assert os.path.exists(path)


class TestThresholdCalibration:
    """Sprint 20 T4: Threshold calibration dataset."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_threshold_calibration_has_both_normal_and_anomaly(self, tmpdir):
        """Threshold calibration CSV should contain both normal and anomaly cycles."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        csv_path, label_path, _ = writer.write_threshold_calibration(
            gen, num_cycles=100, anomaly_ratio=0.5, seed=42
        )

        assert os.path.getsize(csv_path) > 0
        assert os.path.getsize(label_path) > 0

        with open(label_path) as f:
            reader = csv.DictReader(f)
            labels = list(reader)

        assert len(labels) == 100

        anomaly_types = {l["anomaly_type"] for l in labels}
        assert "normal" in anomaly_types, "Should have normal cycles"
        # At least some anomaly types present
        has_anomaly = any(l["is_anomaly"] == "1" for l in labels)
        assert has_anomaly, "Should have anomalous cycles"

    def test_threshold_calibration_filenames(self, tmpdir):
        """Verify correct filenames for threshold calibration."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        csv_path, label_path, _ = writer.write_threshold_calibration(
            gen, num_cycles=20, anomaly_ratio=0.5, seed=42
        )

        assert "threshold_calibration_welding_robot.csv" in csv_path
        assert "threshold_calibration_labels.csv" in label_path
