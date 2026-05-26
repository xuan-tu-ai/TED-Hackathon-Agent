"""
Central configuration for the welding robot data simulator.

All tunable parameters are collected here for reproducibility and
easy experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from typing import Any


@dataclass
class PhysicsConfig:
    """Configuration for physics engine components."""
    current_damping_ratio: float = 0.55
    current_natural_freq: float = 8.0
    current_delay_steps: int = 2       # ~250ms at 8Hz (was 1 at 4Hz)
    current_dc_gain: float = 1.0
    target_current_ratio: float = 0.04

    thermal_alpha_heat: float = 0.018  # ~57 steps at 8Hz (was 30 steps at 4Hz)
    thermal_alpha_cool: float = 0.006  # ~166 steps at 8Hz (was 80 steps at 4Hz)
    thermal_ambient_temp: float = 25.0
    thermal_current_to_temp_gain: float = 0.65
    robot_baseline_offsets: dict[int, float] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 5.0})


@dataclass
class NoiseConfig:
    """Configuration for noise generators."""
    s0_vibration_amplitude: float = 0.008
    s0_vibration_freq: float = 0.025    # adjusted for 8Hz (was 0.05 at 4Hz)
    s0_gaussian_std: float = 0.005
    s1_arc_noise_std: float = 0.05
    s2_rw_step_std: float = 0.1
    s2_glitch_probability: float = 0.0001
    s2_glitch_magnitude: float = 5.0
    ar1_phi: float = 0.85
    local_window_size: int = 40


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection."""
    spike_magnitude_multiplier: float = 4.0
    spike_duration: int = 2
    drift_start_fraction: float = 0.6
    drift_rate_c_per_step: float = 0.4  # halved for 8Hz (was 0.8 at 4Hz) — keeps ~3.2°C/s
    flatline_duration_min: int = 40    # Sprint 18b: clearly detectable in 512-step windows
    flatline_duration_max: int = 60
    phase_shift_delay_steps: int = 15


@dataclass
class OutputConfig:
    """Configuration for CSV output."""
    sampling_rate_hz: int = 8  # Sprint 18b: 8Hz → 512 steps = 64s
    output_dir: str = "data/generated"
    chunk_size: int = 10000  # rows per CSV write chunk


@dataclass
class SimulatorConfig:
    """Top-level configuration combining all sub-configs."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    num_days: int = 30
    robots: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulatorConfig:
        physics = PhysicsConfig(**data.get("physics", {}))
        noise = NoiseConfig(**data.get("noise", {}))
        anomaly = AnomalyConfig(**data.get("anomaly", {}))
        output = OutputConfig(**data.get("output", {}))
        return cls(
            physics=physics,
            noise=noise,
            anomaly=anomaly,
            output=output,
            seed=data.get("seed", 42),
            num_days=data.get("num_days", 30),
            robots=data.get("robots", [0, 1, 2, 3]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> SimulatorConfig:
        return cls.from_dict(json.loads(json_str))
