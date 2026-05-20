"""Tests for program profile system (Sprint 14, T4)."""

import random

import numpy as np
import pytest
from backend.data_simulator.profiles import (
    PROGRAM_PROFILES,
    PhaseRanges,
    ProfileConfig,
    SampledProfile,
    sample_profile,
    sample_random_profile,
    build_phase_envelope,
    get_phase_boundaries,
)


class TestProgramProfiles:
    """Verify all 16 program profiles are well-defined."""

    def test_all_profiles_defined(self):
        assert len(PROGRAM_PROFILES) == 16

    def test_unique_ids(self):
        ids = [p.profile_id for p in PROGRAM_PROFILES]
        assert len(set(ids)) == 16
        assert sorted(ids) == list(range(16))

    def test_all_have_names(self):
        for p in PROGRAM_PROFILES:
            assert len(p.name) > 0, f"Profile {p.profile_id} has no name"

    def test_all_have_valid_ranges(self):
        for p in PROGRAM_PROFILES:
            for phase_name, (lo, hi) in [
                ("idle", p.phase_ranges.idle),
                ("rampup", p.phase_ranges.rampup),
                ("active", p.phase_ranges.active),
                ("winddown", p.phase_ranges.winddown),
                ("shutdown", p.phase_ranges.shutdown),
            ]:
                assert 0 < lo <= hi, (
                    f"Profile {p.profile_id} {phase_name}: invalid range ({lo}, {hi})"
                )

    def test_all_have_valid_feed_speed(self):
        for p in PROGRAM_PROFILES:
            lo, hi = p.target_feed_speed_range
            assert 100 < lo <= hi <= 500, (
                f"Profile {p.profile_id}: invalid feed speed range ({lo}, {hi})"
            )


class TestSampleProfile:
    """Verify profile sampling produces valid concrete profiles."""

    def test_sample_concrete_profile(self):
        profile = sample_profile(PROGRAM_PROFILES[0])
        assert isinstance(profile, SampledProfile)
        assert profile.profile_id == 0
        assert profile.total_steps > 0

    def test_all_phases_positive(self):
        profile = sample_profile(PROGRAM_PROFILES[4])
        assert profile.idle_steps > 0
        assert profile.rampup_steps > 0
        assert profile.active_steps > 0
        assert profile.winddown_steps > 0
        assert profile.shutdown_steps > 0

    def test_total_steps_correct(self):
        profile = sample_profile(PROGRAM_PROFILES[0])
        expected = (
            profile.idle_steps
            + profile.rampup_steps
            + profile.active_steps
            + profile.winddown_steps
            + profile.shutdown_steps
        )
        assert profile.total_steps == expected

    def test_target_feed_in_range(self):
        for config in PROGRAM_PROFILES:
            profile = sample_profile(config)
            lo, hi = config.target_feed_speed_range
            assert lo <= profile.target_feed_speed <= hi, (
                f"Profile {config.profile_id}: feed {profile.target_feed_speed:.0f} "
                f"not in [{lo}, {hi}]"
            )

    def test_target_current_derived(self):
        profile = sample_profile(PROGRAM_PROFILES[0])
        expected = profile.target_feed_speed * 0.04
        assert abs(profile.target_current - expected) < 0.01

    def test_random_profile(self):
        rng = random.Random(42)
        p1 = sample_random_profile(rng)
        rng = random.Random(42)
        p2 = sample_random_profile(rng)
        assert p1.profile_id == p2.profile_id
        assert p1.total_steps == p2.total_steps

    def test_multiple_samples_vary(self):
        """Sampling from the same profile should produce different durations."""
        profiles = [sample_profile(PROGRAM_PROFILES[0]) for _ in range(20)]
        # At least some should differ
        total_steps_set = {p.total_steps for p in profiles}
        assert len(total_steps_set) > 1, "All samples identical"


class TestBuildPhaseEnvelope:
    """Verify phase envelope construction."""

    @pytest.fixture
    def sample_profile_obj(self):
        return SampledProfile(
            profile_id=0,
            name="test",
            idle_steps=10,
            rampup_steps=20,
            active_steps=100,
            winddown_steps=20,
            shutdown_steps=10,
            target_feed_speed=3000.0,
            target_current=120.0,
            total_steps=160,
        )

    def test_envelope_length(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        assert len(env) == sample_profile_obj.total_steps

    def test_idle_phase_zero(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        idle_end = sample_profile_obj.idle_steps
        assert np.all(env[:idle_end] == 0), "Idle phase should be 0"

    def test_active_phase_at_target(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        idle_end = sample_profile_obj.idle_steps
        rampup_end = idle_end + sample_profile_obj.rampup_steps
        active_end = rampup_end + sample_profile_obj.active_steps
        active_section = env[rampup_end:active_end]
        assert np.allclose(active_section, 3000.0, atol=1.0), (
            f"Active phase should be at target. Got range {active_section.min()}-{active_section.max()}"
        )

    def test_rampup_monotonic(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        idle_end = sample_profile_obj.idle_steps
        rampup_end = idle_end + sample_profile_obj.rampup_steps
        ramp = env[idle_end:rampup_end]
        assert np.all(np.diff(ramp) >= 0), "Rampup should be monotonically increasing"

    def test_winddown_monotonic(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        idle_end = sample_profile_obj.idle_steps
        rampup_end = idle_end + sample_profile_obj.rampup_steps
        active_end = rampup_end + sample_profile_obj.active_steps
        winddown_end = active_end + sample_profile_obj.winddown_steps
        wind = env[active_end:winddown_end]
        assert np.all(np.diff(wind) <= 0), "Winddown should be monotonically decreasing"

    def test_shutdown_phase_zero(self, sample_profile_obj):
        env = build_phase_envelope(sample_profile_obj)
        winddown_end = (sample_profile_obj.idle_steps + sample_profile_obj.rampup_steps
                        + sample_profile_obj.active_steps + sample_profile_obj.winddown_steps)
        assert np.all(env[winddown_end:] == 0), "Shutdown phase should be 0"


class TestGetPhaseBoundaries:
    def test_boundary_order(self):
        profile = SampledProfile(
            profile_id=0, name="test",
            idle_steps=10, rampup_steps=20, active_steps=100,
            winddown_steps=20, shutdown_steps=10,
            target_feed_speed=3000.0, target_current=120.0, total_steps=160,
        )
        bounds = get_phase_boundaries(profile)
        assert bounds["idle_end"] < bounds["rampup_end"]
        assert bounds["rampup_end"] < bounds["active_end"]
        assert bounds["active_end"] < bounds["winddown_end"]
