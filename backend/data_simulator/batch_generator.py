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

from backend.data_simulator.generator import WeldingSampleGenerator, WeldingSampleNoisy


# Sampling parameters
STEPS_PER_SECOND = 4       # 4 Hz (250ms per step)
SECONDS_PER_STEP = 0.25

# Daily operating hours per robot (min, max)
DAILY_OPERATING_HOURS = (16, 20)

# Gap between cycles (minutes)
CYCLE_GAP_MINUTES = (1, 5)


@dataclass
class CycleRow:
    """A single row of the output CSV."""
    robot_code: int
    start_time: str
    program_number: int
    infer_recorded_date: str
    serial_number: int
    phase: str
    wire_feed_speed_mm_min: float
    wire_temperature_c: float
    welding_current_a: float
    timestamp: str


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
    ):
        self._rng = random.Random(seed)
        self.sample_gen = sample_generator or WeldingSampleGenerator(seed=seed)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.sample_gen.reseed(seed)

    def _sample_to_rows(
        self,
        sample: WeldingSampleNoisy,
        start_datetime: datetime,
    ) -> list[CycleRow]:
        """Convert a WeldingSampleNoisy to a list of CycleRow dicts.

        Each row corresponds to one timestep (250ms).
        """
        n = len(sample.wire_feed_speed)
        rows: list[CycleRow] = []

        start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Base start_time for the cycle (the day begins here)

        for step_idx in range(n):
            ts = start_datetime + timedelta(seconds=step_idx * SECONDS_PER_STEP)
            infer_recorded = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            phase = _phase_label(step_idx, sample.phase_boundaries)

            rows.append(CycleRow(
                robot_code=sample.robot_id,
                start_time=start_time_str,
                program_number=sample.program_id,
                infer_recorded_date=infer_recorded,
                serial_number=step_idx + 1,
                phase=phase,
                wire_feed_speed_mm_min=round(float(sample.wire_feed_speed[step_idx]), 1),
                wire_temperature_c=round(float(sample.temperature[step_idx]), 1),
                welding_current_a=round(float(sample.welding_current[step_idx]), 1),
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
            ))

        return rows

    def _generate_cycles_for_robot(
        self,
        robot_id: int,
        start_date: datetime,
        num_days: int,
        is_test: bool = False,
        anomaly_fn=None,
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
                # Generate one cycle
                sample = self.sample_gen.generate_noisy_sample(
                    robot_id=robot_id,
                )

                rows = self._sample_to_rows(sample, day_start)

                # Advance time by the cycle duration + a gap
                cycle_duration = len(sample.wire_feed_speed) * SECONDS_PER_STEP
                gap_minutes = self._rng.uniform(*CYCLE_GAP_MINUTES)
                gap = timedelta(minutes=gap_minutes)
                day_start += timedelta(seconds=cycle_duration) + gap

                yield rows

            # Move to next day
            current_time += timedelta(days=1)

    def generate_train(
        self,
        num_days: int = 30,
        robots: list[int] | None = None,
    ) -> Iterator[CycleRow]:
        """Generate 100% normal training data for the specified period.

        Args:
            num_days: Number of days to generate (default: 30).
            robots: List of robot IDs. If None, uses [0, 1, 2, 3].

        Yields:
            CycleRow instances (one per timestep).
        """
        if robots is None:
            robots = [0, 1, 2, 3]

        start_date = datetime(2025, 7, 1, 0, 0, 0)

        for robot_id in robots:
            for rows in self._generate_cycles_for_robot(
                robot_id, start_date, num_days
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
        avg_gap_secs = (CYCLE_GAP_MINUTES[0] + CYCLE_GAP_MINUTES[1]) / 2 * 60
        avg_op_secs = sum(DAILY_OPERATING_HOURS) / 2 * 3600
        return max(1, int(avg_op_secs / (avg_cycle_secs + avg_gap_secs)))
