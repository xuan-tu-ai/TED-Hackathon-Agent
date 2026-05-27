"""
Batch generator that schedules welding cycles across a 30-day period.

Manages per-robot daily schedules, cycle spacing, and timestamp generation
for both train and test datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from typing import Iterator

import numpy as np

from .generator import WeldingSampleGenerator, WeldingSampleNoisy
from .physics.noise import NoiseConfig, NoiseGenerator


# Sampling parameters
STEPS_PER_SECOND = 8       # 8 Hz (125ms per step) — Sprint 18b: 512 steps = 64s
SECONDS_PER_STEP = 0.125

# Daily operating hours per robot (min, max)
DAILY_OPERATING_HOURS = (16, 20)

# Gap between cycles: fixed 6-second sleep per robot
# Each cycle = 512 steps / 8 Hz = 64s  →  cycle + sleep = 64 + 6 = 70s
CYCLE_GAP_SECONDS = 6


@dataclass
class CycleRow:
    """A single row of the output CSV.

    Each row represents one timestep (250ms) of one welding cycle.
    Rows belonging to the same cycle share the same ``cycle_id``.
    """
    cycle_id: int                      # unique cycle identifier
    robot_code: int
    start_time: str
    program_number: int
    serial_number: int
    phase: str
    wire_feed_speed_mm_min: float
    wire_temperature_c: float
    welding_current_a: float
    timestamp: str
    anomaly_type: str = "none"          # anomaly type (or "none" for normal)


def _phase_label(
    step_idx: int,
    phase_boundaries: dict[str, int],
) -> str:
    """Determine the phase label for a given step index."""
    if step_idx < phase_boundaries["idle_end"]:
        return "idle"
    elif step_idx < phase_boundaries["rampup_end"]:
        return "rampup"
    elif step_idx < phase_boundaries["active_end"]:
        return "active"
    elif step_idx < phase_boundaries["winddown_end"]:
        return "winddown"
    else:
        return "shutdown"


class BatchGenerator:
    """Generates 30 days of welding data across multiple robots.

    Usage:
        gen = BatchGenerator(seed=42)
        for row in gen.generate_train():
            # row is a CycleRow dict
            pass

        for row in gen.generate_test(num_anomaly_samples=500):
            pass
    """

    def __init__(
        self,
        seed: int = 42,
        sample_generator: WeldingSampleGenerator | None = None,
        training_mode: bool = False,
    ):
        self._rng = random.Random(seed)
        self._training_mode = training_mode
        if training_mode:
            # Sprint 18: Use reduced-noise config for convergent training data
            train_noise_config = NoiseConfig().for_training()
            train_noise_gen = NoiseGenerator(config=train_noise_config, seed=seed)
            self.sample_gen = sample_generator or WeldingSampleGenerator(
                seed=seed, noise_generator=train_noise_gen
            )
        else:
            self.sample_gen = sample_generator or WeldingSampleGenerator(seed=seed)
        self._cycle_counter: int = 0  # auto-incrementing cycle ID

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.sample_gen.reseed(seed)
        self._cycle_counter = 0

    def _sample_to_rows(
        self,
        sample: WeldingSampleNoisy,
        start_datetime: datetime,
        cycle_id: int,
        anomaly_type: str = "none",
    ) -> list[CycleRow]:
        """Convert a WeldingSampleNoisy to a list of CycleRow dicts.

        Each row corresponds to one timestep (250ms).

        Args:
            sample: The noisy welding sample.
            start_datetime: Start time of the cycle.
            cycle_id: Unique cycle identifier.
            anomaly_type: Anomaly type string ("none" for normal).

        Returns:
            List of CycleRow instances.
        """
        n = len(sample.wire_feed_speed)
        rows: list[CycleRow] = []

        start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")

        for step_idx in range(n):
            ts = start_datetime + timedelta(seconds=step_idx * SECONDS_PER_STEP)
            phase = _phase_label(step_idx, sample.phase_boundaries)

            rows.append(CycleRow(
                cycle_id=cycle_id,
                robot_code=sample.robot_id,
                start_time=start_time_str,
                program_number=sample.program_id,
                serial_number=step_idx + 1,
                phase=phase,
                wire_feed_speed_mm_min=round(float(sample.wire_feed_speed[step_idx]), 1),
                wire_temperature_c=round(float(sample.temperature[step_idx]), 1),
                welding_current_a=round(float(sample.welding_current[step_idx]), 1),
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
                anomaly_type=anomaly_type,
            ))

        return rows

    def _generate_cycles_for_robot(
        self,
        robot_id: int,
        start_date: datetime,
        num_days: int,
        is_test: bool = False,
        anomaly_fn=None,
        deterministic: bool = False,
    ) -> Iterator[list[CycleRow]]:
        """Generate cycles for one robot over multiple days.

        Each day the robot operates for a random number of hours (16-20),
        during which cycles are scheduled back-to-back with small gaps.

        Args:
            robot_id: Robot identifier.
            start_date: The starting date.
            num_days: Number of days to generate.
            is_test: Whether this is test data (affects anomaly injection).
            anomaly_fn: Optional function that takes (sample, rng) and returns
                        modified sample with anomaly injected.
            deterministic: If True and in training mode, same (robot, program)
                pairs produce the same base envelope every time (Sprint 18).

        Yields:
            Lists of CycleRow for each cycle.
        """
        current_time = start_date

        for _ in range(num_days):
            # Start each day at a random hour between 06:00 and 08:00
            day_start_hour = self._rng.randint(6, 8)
            day_start = current_time.replace(
                hour=day_start_hour, minute=0, second=0, microsecond=0
            )

            # Random daily operating duration
            op_hours = self._rng.uniform(*DAILY_OPERATING_HOURS)
            day_end = day_start + timedelta(hours=op_hours)

            while day_start < day_end:
                # Sprint 18: pick program_id and use deterministic generation
                # for convergent training data
                program_id = self._rng.randint(0, 15)
                sample = self.sample_gen.generate_noisy_sample(
                    robot_id=robot_id,
                    program_id=program_id,
                    deterministic=deterministic,
                )

                self._cycle_counter += 1
                rows = self._sample_to_rows(
                    sample, day_start, cycle_id=self._cycle_counter,
                )

                # Advance time by the cycle duration + fixed 6s sleep
                # Each cycle = 512 steps / 8 Hz = 64s → 64+6=70s per cycle (incremental)
                cycle_duration = len(sample.wire_feed_speed) * SECONDS_PER_STEP
                gap = timedelta(seconds=CYCLE_GAP_SECONDS)
                day_start += timedelta(seconds=cycle_duration) + gap

                yield rows

            # Move to next day
            current_time += timedelta(days=1)

    def generate_train(
        self,
        num_days: int = 30,
        robots: list[int] | None = None,
        start_date: datetime | None = None,
    ) -> Iterator[CycleRow]:
        """Generate 100% normal training data for the specified period.

        Uses deterministic generation + reduced noise (Sprint 18) for
        convergent normal data: same (robot, program) pairs produce
        nearly identical sensor traces across cycles.

        Args:
            num_days: Number of days to generate (default: 30).
            robots: List of robot IDs. If None, uses [0, 1, 2, 3].
            start_date: First day of data generation. If None, defaults to
                2025-07-01. Each robot runs 16-20 hours/day with cycles
                spaced 1-5 min apart. Robots are processed sequentially.

        Yields:
            CycleRow instances (one per timestep).
        """
        if robots is None:
            robots = [0, 1, 2, 3]

        if start_date is None:
            start_date = datetime(2025, 7, 1, 0, 0, 0)

        for robot_id in robots:
            for rows in self._generate_cycles_for_robot(
                robot_id, start_date, num_days, deterministic=True,
            ):
                for row in rows:
                    yield row

    def _estimate_cycles_per_robot_per_day(self, robot_id: int) -> int:
        """Estimate how many cycles one robot produces per day."""
        # Generate a few samples to estimate average cycle length
        total_steps = 0
        n_samples = 10
        for _ in range(n_samples):
            sample = self.sample_gen.generate_noisy_sample(robot_id=robot_id)
            total_steps += len(sample.wire_feed_speed)
        avg_steps = total_steps / n_samples
        avg_cycle_secs = avg_steps * SECONDS_PER_STEP
        avg_gap_secs = CYCLE_GAP_SECONDS
        avg_op_secs = sum(DAILY_OPERATING_HOURS) / 2 * 3600
        return max(1, int(avg_op_secs / (avg_cycle_secs + avg_gap_secs)))
