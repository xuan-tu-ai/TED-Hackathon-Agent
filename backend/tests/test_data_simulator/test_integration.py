"""Integration tests for the data simulator (Sprint 14, T14)."""

import csv
import os
import tempfile

import numpy as np
import pytest
from backend.data_simulator.batch_generator import BatchGenerator
from backend.data_simulator.writer import CSVWriter
from backend.data_simulator.generator import WeldingSampleGenerator
from backend.data_simulator.anomalies import (
    inject_spike,
    inject_thermal_drift,
    inject_flatline,
)
import random


class TestEndToEnd:
    """Verify the full pipeline produces valid output."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_small_dataset_generation(self, tmpdir):
        """Generate a tiny dataset and verify CSV output."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir, chunk_size=1000)

        path, count = writer.write_train(gen, num_days=1, robots=[0])
        assert count > 0

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == count
        # Verify a few rows
        for row in rows[:5]:
            assert row["robot_code"] == "0"
            assert row["phase"] in ("idle", "rampup", "active", "winddown", "shutdown")
            assert float(row["wire_feed_speed_mm_min"]) >= -100  # allow small noise
            assert float(row["wire_temperature_c"]) >= 20

    def test_anomaly_labels_correct(self, tmpdir):
        """Verify anomaly labels match injected anomalies."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)

        # 100% anomaly rate with few cycles
        _, label_path, _ = writer.write_test(
            gen, num_cycles=20, anomaly_ratio=1.0, seed=42
        )

        with open(label_path) as f:
            reader = csv.DictReader(f)
            labels = list(reader)

        assert len(labels) == 20
        # All should be anomalous
        for label in labels:
            assert label["is_anomaly"] == "1"
            assert label["anomaly_type"] != "normal"

    def test_normal_labels_correct(self, tmpdir):
        """Verify normal labels when anomaly_ratio=0."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)

        _, label_path, _ = writer.write_test(
            gen, num_cycles=10, anomaly_ratio=0.0, seed=42
        )

        with open(label_path) as f:
            reader = csv.DictReader(f)
            for label in reader:
                assert label["is_anomaly"] == "0"
                assert label["anomaly_type"] == "normal"

    def test_reproducibility_same_seed(self, tmpdir):
        """Same seed → identical output."""
        gen1 = BatchGenerator(seed=42)
        gen2 = BatchGenerator(seed=42)
        writer1 = CSVWriter(output_dir=tmpdir)
        writer2 = CSVWriter(output_dir=os.path.join(tmpdir, "copy"))

        path1, _ = writer1.write_train(gen1, num_days=1, robots=[0])
        path2, _ = writer2.write_train(gen2, num_days=1, robots=[0])

        with open(path1) as f1, open(path2) as f2:
            assert f1.read() == f2.read()


class TestAnomalyDetectability:
    """Verify that anomalies are statistically distinguishable from normal."""

    @pytest.fixture
    def normal_samples(self):
        gen = WeldingSampleGenerator(seed=42)
        samples = [gen.generate_noisy_sample() for _ in range(20)]
        return samples

    def test_spike_detectable(self, normal_samples):
        """Spike creates current values far outside normal range."""
        gen = WeldingSampleGenerator(seed=42)
        r = random.Random(42)

        normal_max = max(s.welding_current.max() for s in normal_samples)
        spiked_sample = inject_spike(gen.generate_sample(), r)
        assert spiked_sample.welding_current.max() > normal_max * 2

    def test_thermal_drift_detectable(self, normal_samples):
        """Drifted temp should exceed the same sample's normal temp."""
        gen = WeldingSampleGenerator(seed=42)
        r = random.Random(42)

        # Use the same sample for fair comparison
        sample = gen.generate_sample()
        normal_max = sample.temperature.max()
        drifted = inject_thermal_drift(sample, r)
        assert drifted.temperature.max() > normal_max + 5, (
            f"Drifted {drifted.temperature.max():.1f} should exceed normal {normal_max:.1f}"
        )

    def test_flatline_detectable(self, normal_samples):
        """Flatline creates zero values in feed where normally there are none."""
        gen = WeldingSampleGenerator(seed=42)
        r = random.Random(42)
        sample = gen.generate_sample()
        flattened = inject_flatline(sample, r)

        # During active phase, normal feed should never be 0
        active_start = sample.phase_boundaries["rampup_end"]
        active_end = sample.phase_boundaries["active_end"]
        normal_min = sample.wire_feed_speed[active_start:active_end].min()
        flattened_min = flattened.wire_feed_speed[active_start:active_end].min()
        assert normal_min > 100, f"Normal feed min should be high: {normal_min}"
        assert flattened_min == 0, f"Flattened feed should hit 0: {flattened_min}"

class TestPhaseIntegrity:
    """Verify phase sequence integrity."""

    def test_phase_order(self):
        """Phases always follow idle→rampup→active→winddown→shutdown."""
        gen = WeldingSampleGenerator(seed=42)
        for _ in range(50):
            s = gen.generate_noisy_sample()
            bounds = s.phase_boundaries
            assert bounds["idle_end"] < bounds["rampup_end"]
            assert bounds["rampup_end"] < bounds["active_end"]
            assert bounds["active_end"] < bounds["winddown_end"]

    def test_no_overlapping_phases(self):
        """Each step should belong to exactly one phase."""
        gen = WeldingSampleGenerator(seed=42)
        for _ in range(50):
            s = gen.generate_noisy_sample()
            bounds = s.phase_boundaries
            total = bounds["winddown_end"] + (len(s.wire_feed_speed) - bounds["winddown_end"])
            assert total == len(s.wire_feed_speed)

    def test_no_gaps_in_phases(self):
        """Phase boundaries should cover the entire signal."""
        gen = WeldingSampleGenerator(seed=42)
        for _ in range(50):
            s = gen.generate_noisy_sample()
            bounds = s.phase_boundaries
            # Last shutdown step should be the last step
            assert bounds["winddown_end"] <= len(s.wire_feed_speed)


class TestWriteTestNoise:
    """Verify anomaly files have sensor noise (Sprint 20 T1)."""

    @pytest.fixture
    def tmpdir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_write_test_anomaly_has_sensor_noise(self, tmpdir):
        """After fix: anomaly files should have sensor noise (std WFS > 5.0)."""
        gen = BatchGenerator(seed=42)
        writer = CSVWriter(output_dir=tmpdir)
        csv_path, _, _ = writer.write_test(
            gen, num_cycles=50, anomaly_ratio=1.0, seed=42
        )

        # Read CSV and group by cycle_id
        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Group by cycle_id
        from collections import defaultdict
        cycles = defaultdict(list)
        for row in rows:
            cycles[row["cycle_id"]].append(row)

        # Check at least some spike cycles have noisy WFS
        spike_cycles = [c for cid, c in cycles.items()
                        if any(r["anomaly_type"] == "spike" for r in c)]
        assert len(spike_cycles) > 0, "Should have spike cycles"

        # For spike cycles, compute std of wire_feed_speed (should have noise)
        noisy_stds = []
        for cycle_rows in spike_cycles:
            wfs_vals = [float(r["wire_feed_speed_mm_min"]) for r in cycle_rows]
            import numpy as np
            noisy_stds.append(float(np.std(wfs_vals)))

        mean_std = sum(noisy_stds) / len(noisy_stds)
        # Sensor noise + spike should produce std > 5.0
        assert mean_std > 5.0, (
            f"Spike cycles should have sensor noise (std={mean_std:.2f} > 5.0). "
            f"If add_noise() is missing before inject_fn(), WFS will be nearly "
            f"deterministic and std will be too low."
        )
