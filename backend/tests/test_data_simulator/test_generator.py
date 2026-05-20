"""Tests for welding sample generator (Sprint 14, T5)."""

import numpy as np
import pytest
from backend.data_simulator.generator import WeldingSampleGenerator, WeldingSample, WeldingSampleNoisy


class TestWeldingSampleGenerator:
    """Verify the main sample generator."""

    def test_generate_clean_sample(self):
        gen = WeldingSampleGenerator(seed=42)
        s = gen.generate_sample(robot_id=0, program_id=0)
        assert isinstance(s, WeldingSample)
        assert s.robot_id == 0
        assert s.program_id == 0
        assert len(s.wire_feed_speed) > 0
        assert len(s.wire_feed_speed) == len(s.welding_current) == len(s.temperature)

    def test_generate_noisy_sample(self):
        gen = WeldingSampleGenerator(seed=42)
        s = gen.generate_noisy_sample(robot_id=1, program_id=5)
        assert isinstance(s, WeldingSampleNoisy)
        assert s.robot_id == 1
        assert len(s.wire_feed_speed) > 0

    def test_random_robot_and_program(self):
        gen = WeldingSampleGenerator(seed=42)
        s = gen.generate_sample()
        assert 0 <= s.robot_id <= 3
        assert 0 <= s.program_id <= 15

    def test_all_robot_ids(self):
        """All 4 robot IDs should produce valid samples."""
        gen = WeldingSampleGenerator(seed=42)
        for robot_id in range(4):
            s = gen.generate_noisy_sample(robot_id=robot_id)
            assert s.robot_id == robot_id
            assert np.all(np.isfinite(s.wire_feed_speed))
            assert np.all(np.isfinite(s.welding_current))
            assert np.all(np.isfinite(s.temperature))

    def test_all_program_ids(self):
        """All 16 program IDs should produce valid samples."""
        gen = WeldingSampleGenerator(seed=42)
        for prog_id in range(16):
            s = gen.generate_noisy_sample(program_id=prog_id)
            assert s.program_id == prog_id

    def test_phase_boundaries_present(self):
        gen = WeldingSampleGenerator(seed=42)
        s = gen.generate_sample()
        for key in ("idle_end", "rampup_end", "active_end", "winddown_end"):
            assert key in s.phase_boundaries

    def test_sensor_value_ranges(self):
        """Verify sensor values are in realistic ranges."""
        gen = WeldingSampleGenerator(seed=42)
        for _ in range(50):
            s = gen.generate_noisy_sample()
            # Wire feed speed should have some positive values
            assert s.wire_feed_speed.max() > 100, "Feed speed too low"
            # Welding current during active should be significant
            active_start = s.phase_boundaries["rampup_end"]
            active_end = s.phase_boundaries["active_end"]
            active_current = s.welding_current[active_start:active_end]
            if len(active_current) > 10:
                assert active_current.mean() > 5, f"Active current too low: {active_current.mean():.0f}"
            # Temperature should be reasonable
            assert s.temperature.min() >= 20, f"Temp too low: {s.temperature.min():.0f}"
            assert s.temperature.max() <= 300, f"Temp too high: {s.temperature.max():.0f}"

    def test_noise_adds_variation(self):
        """Noisy sample should differ from clean sample."""
        gen = WeldingSampleGenerator(seed=42)
        clean = gen.generate_sample(robot_id=0, program_id=0)
        noisy = gen.add_noise(clean)
        # Noise should change values (not identical)
        assert not np.allclose(clean.wire_feed_speed, noisy.wire_feed_speed)
        assert not np.allclose(clean.welding_current, noisy.welding_current)
        assert not np.allclose(clean.temperature, noisy.temperature)

    def test_reproducibility(self):
        """Same seed should produce identical samples."""
        gen1 = WeldingSampleGenerator(seed=42)
        gen2 = WeldingSampleGenerator(seed=42)
        s1 = gen1.generate_noisy_sample(robot_id=0, program_id=0)
        s2 = gen2.generate_noisy_sample(robot_id=0, program_id=0)
        assert np.allclose(s1.wire_feed_speed, s2.wire_feed_speed)
        assert np.allclose(s1.welding_current, s2.welding_current)
        assert np.allclose(s1.temperature, s2.temperature)

    def test_different_seeds_different(self):
        """Different seeds should produce different samples."""
        gen1 = WeldingSampleGenerator(seed=42)
        gen2 = WeldingSampleGenerator(seed=99)
        s1 = gen1.generate_noisy_sample(robot_id=0, program_id=0)
        s2 = gen2.generate_noisy_sample(robot_id=0, program_id=0)
        # At least some difference (samples may have different lengths)
        assert (
            len(s1.wire_feed_speed) != len(s2.wire_feed_speed)
            or not np.allclose(s1.wire_feed_speed, s2.wire_feed_speed)
        )

    def test_robot_3_hotter(self):
        """Robot 3 temperature should be higher than robot 0 for same profile."""
        gen = WeldingSampleGenerator(seed=42)
        s0 = gen.generate_sample(robot_id=0, program_id=0)
        s3 = gen.generate_sample(robot_id=3, program_id=0)
        # Robot 3 has baseline offset of +5°C
        assert s3.temperature.mean() > s0.temperature.mean()

    def test_add_noise_preserves_phase_boundaries(self):
        gen = WeldingSampleGenerator(seed=42)
        clean = gen.generate_sample()
        noisy = gen.add_noise(clean)
        assert noisy.phase_boundaries == clean.phase_boundaries

    def test_phase_transition_smoothness(self):
        """Phase transitions should be smooth (no large jumps in clean signal)."""
        gen = WeldingSampleGenerator(seed=42)
        s = gen.generate_sample(robot_id=0, program_id=5)
        # Check that max step change in clean wire_feed is reasonable
        diffs = np.abs(np.diff(s.wire_feed_speed))
        max_step = diffs.max()
        # Max step should be bounded by rampup slope
        assert max_step < 200, f"Max step change too large: {max_step:.0f}"
