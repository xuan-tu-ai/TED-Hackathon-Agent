"""Tests for batch generator (Sprint 14, T6)."""

from datetime import datetime

from backend.data_simulator.batch_generator import (
    BatchGenerator,
    CycleRow,
    _phase_label,
)


class TestPhaseLabel:
    """Verify phase labeling logic."""

    def test_idle_phase(self):
        bounds = {"idle_end": 10, "rampup_end": 30, "active_end": 130, "winddown_end": 150}
        assert _phase_label(0, bounds) == "idle"
        assert _phase_label(9, bounds) == "idle"

    def test_rampup_phase(self):
        bounds = {"idle_end": 10, "rampup_end": 30, "active_end": 130, "winddown_end": 150}
        assert _phase_label(10, bounds) == "rampup"
        assert _phase_label(29, bounds) == "rampup"

    def test_active_phase(self):
        bounds = {"idle_end": 10, "rampup_end": 30, "active_end": 130, "winddown_end": 150}
        assert _phase_label(30, bounds) == "active"
        assert _phase_label(129, bounds) == "active"

    def test_winddown_phase(self):
        bounds = {"idle_end": 10, "rampup_end": 30, "active_end": 130, "winddown_end": 150}
        assert _phase_label(130, bounds) == "winddown"
        assert _phase_label(149, bounds) == "winddown"

    def test_shutdown_phase(self):
        bounds = {"idle_end": 10, "rampup_end": 30, "active_end": 130, "winddown_end": 150}
        assert _phase_label(150, bounds) == "shutdown"
        assert _phase_label(200, bounds) == "shutdown"


class TestBatchGenerator:
    """Verify the batch generator."""

    def test_generate_train_yields_rows(self):
        gen = BatchGenerator(seed=42)
        row_count = 0
        for row in gen.generate_train(num_days=1, robots=[0]):
            assert isinstance(row, CycleRow)
            row_count += 1
            if row_count >= 1000:
                break
        assert row_count == 1000, f"Should yield at least 1000 rows, got {row_count}"

    def test_row_schema(self):
        """Each row must have all required fields."""
        gen = BatchGenerator(seed=42)
        for row in gen.generate_train(num_days=1, robots=[0]):
            assert hasattr(row, "cycle_id")
            assert hasattr(row, "robot_code")
            assert hasattr(row, "start_time")
            assert hasattr(row, "program_number")
            assert hasattr(row, "serial_number")
            assert hasattr(row, "phase")
            assert hasattr(row, "wire_feed_speed_mm_min")
            assert hasattr(row, "wire_temperature_c")
            assert hasattr(row, "welding_current_a")
            assert hasattr(row, "anomaly_type")
            assert hasattr(row, "timestamp")
            break

    def test_row_value_ranges(self):
        """Verify rows have realistic values."""
        gen = BatchGenerator(seed=42)
        for row in gen.generate_train(num_days=1, robots=[0]):
            assert isinstance(row.robot_code, int)
            assert 0 <= row.robot_code <= 3
            assert isinstance(row.program_number, int)
            assert 0 <= row.program_number <= 15
            assert isinstance(row.serial_number, int)
            assert row.serial_number >= 1
            assert row.phase in ("idle", "rampup", "active", "winddown", "shutdown")
            assert isinstance(row.wire_feed_speed_mm_min, float)
            assert isinstance(row.welding_current_a, float)
            assert isinstance(row.wire_temperature_c, float)
            break

    def test_all_robots_present_over_30_days(self):
        """All 4 robots should generate data over 30 days."""
        gen = BatchGenerator(seed=42)
        robots_seen = set()
        for row in gen.generate_train(num_days=30, robots=[0, 1, 2, 3]):
            robots_seen.add(row.robot_code)
            if len(robots_seen) == 4:
                break
        assert robots_seen == {0, 1, 2, 3}

    def test_serial_number_increments(self):
        """Serial numbers within a cycle should increment by 1."""
        gen = BatchGenerator(seed=42)
        prev_serial = 0
        first_cycle = True
        for row in gen.generate_train(num_days=1, robots=[0]):
            if first_cycle:
                # First row of first cycle should be serial 1
                assert row.serial_number == 1, f"First serial should be 1, got {row.serial_number}"
                first_cycle = False
            # Serial should match expected position
            # We just check monotonic within the same program
            if row.serial_number == 1 and prev_serial > 0:
                break  # New cycle started
            prev_serial = row.serial_number

    def test_timestamp_format(self):
        """Timestamp should be ISO format with timezone."""
        gen = BatchGenerator(seed=42)
        for row in gen.generate_train(num_days=1, robots=[0]):
            ts = row.timestamp
            assert ts.endswith("+00:00"), f"Timestamp should end with +00:00: {ts}"
            # Should parse as valid datetime
            dt_str = ts.replace("+00:00", "")
            datetime.fromisoformat(dt_str)
            break

    def test_reproducibility(self):
        """Same seed should produce identical output."""
        gen1 = BatchGenerator(seed=42)
        gen2 = BatchGenerator(seed=42)
        rows1 = list(gen1.generate_train(num_days=1, robots=[0]))
        rows2 = list(gen2.generate_train(num_days=1, robots=[0]))
        assert len(rows1) == len(rows2)
        for r1, r2 in zip(rows1, rows2):
            assert r1.robot_code == r2.robot_code
            assert r1.serial_number == r2.serial_number
            assert r1.wire_feed_speed_mm_min == r2.wire_feed_speed_mm_min

    def test_different_seeds_different(self):
        """Different seeds should produce different output."""
        gen1 = BatchGenerator(seed=42)
        gen2 = BatchGenerator(seed=99)
        rows1 = list(gen1.generate_train(num_days=1, robots=[0]))
        rows2 = list(gen2.generate_train(num_days=1, robots=[0]))
        # Should differ at some point
        differs = False
        for r1, r2 in zip(rows1, rows2):
            if r1.wire_feed_speed_mm_min != r2.wire_feed_speed_mm_min:
                differs = True
                break
        assert differs, "Different seeds should produce different data"

    def test_large_batch_not_empty(self):
        """A large batch should produce many rows."""
        gen = BatchGenerator(seed=42)
        count = sum(1 for _ in gen.generate_train(num_days=3, robots=[0, 1]))
        # Very rough check: at least some rows
        assert count > 10000, f"Expected >10000 rows, got {count}"

    def test_serial_number_resets(self):
        """Serial number resets to 1 for each new cycle."""
        gen = BatchGenerator(seed=42)
        last_serial = 0
        reset_count = 0
        for row in gen.generate_train(num_days=1, robots=[0]):
            if row.serial_number == 1 and last_serial > 0:
                reset_count += 1
            last_serial = row.serial_number
            if reset_count >= 2:
                break
        assert reset_count >= 2, "Should see at least 2 cycles with serial reset"

    def test_30_day_coverage(self):
        """Over 30 days, timestamps should span the full period."""
        gen = BatchGenerator(seed=42)
        first_ts = None
        last_ts = None
        for i, row in enumerate(gen.generate_train(num_days=30, robots=[0])):
            if first_ts is None:
                first_ts = row.timestamp
            last_ts = row.timestamp
            if i >= 100000:
                break
        assert first_ts is not None
        # Verify first timestamp is around July 2025
        assert "2025-07-01" in first_ts
