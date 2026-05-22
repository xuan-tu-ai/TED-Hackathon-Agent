"""
CSV writer for welding robot data simulator.

Writes batched CycleRow data to CSV files matching the legacy
mock_welding_robot_data.csv schema. Handles large datasets via
chunked writing without loading everything into memory.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
import random

from backend.data_simulator.batch_generator import BatchGenerator, CycleRow, _phase_label, SECONDS_PER_STEP
from backend.data_simulator.anomalies import (
    inject_spike,
    inject_thermal_drift,
    inject_flatline,
    inject_phase_shift,
)
from backend.data_simulator.config import SimulatorConfig


# CSV column order matching legacy schema
CSV_HEADER = [
    "robot_code",
    "start_time",
    "program_number",
    "infer_recorded_date",
    "serial_number",
    "phase",
    "wire_feed_speed_mm_min",
    "wire_temperature_c",
    "welding_current_a",
    "timestamp",
]

# Label CSV header
LABEL_HEADER = [
    "robot_code",
    "start_time",
    "program_number",
    "is_anomaly",
    "anomaly_type",
]


def _write_rows_to_csv(rows: list[CycleRow], filepath: str) -> None:
    """Write a batch of CycleRow objects to a CSV file (appends if exists)."""
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([
                row.robot_code,
                row.start_time,
                row.program_number,
                row.infer_recorded_date,
                row.serial_number,
                row.phase,
                row.wire_feed_speed_mm_min,
                row.wire_temperature_c,
                row.welding_current_a,
                row.timestamp,
            ])


def _write_labels_to_csv(labels: list[dict], filepath: str) -> None:
    """Write anomaly labels to a CSV file (appends if exists)."""
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LABEL_HEADER)
        for label in labels:
            writer.writerow([
                label["robot_code"],
                label["start_time"],
                label["program_number"],
                label["is_anomaly"],
                label["anomaly_type"],
            ])


class CSVWriter:
    """Writes generated welding data to CSV files.

    Usage:
        writer = CSVWriter(output_dir="data/generated")
        writer.write_train(batch_gen, num_days=30)
        writer.write_test(batch_gen, num_cycles=2000, anomaly_ratio=0.5)
    """

    def __init__(self, output_dir: str | None = None, chunk_size: int = 10000):
        self.output_dir = output_dir or os.getenv("GENERATED_DATA_DIR", "data/generated")
        self.chunk_size = chunk_size

    def write_train(
        self,
        batch_gen: BatchGenerator,
        num_days: int = 30,
        robots: list[int] | None = None,
    ) -> tuple[str, int]:
        """Write the training dataset (100% normal cycles).

        Args:
            batch_gen: BatchGenerator instance.
            num_days: Number of days to generate.
            robots: List of robot IDs.

        Returns:
            (filepath, row_count) tuple.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, "train_welding_robot.csv")

        # Remove existing file to start fresh
        if os.path.exists(filepath):
            os.remove(filepath)

        chunk: list[CycleRow] = []
        row_count = 0

        for row in batch_gen.generate_train(num_days=num_days, robots=robots):
            chunk.append(row)
            row_count += 1
            if len(chunk) >= self.chunk_size:
                _write_rows_to_csv(chunk, filepath)
                chunk = []

        # Write remaining rows
        if chunk:
            _write_rows_to_csv(chunk, filepath)

        return filepath, row_count

    def write_test(
        self,
        batch_gen: BatchGenerator,
        num_cycles: int = 2000,
        anomaly_ratio: float = 0.5,
        robots: list[int] | None = None,
        seed: int = 42,
    ) -> tuple[str, str, int]:
        """Write the test dataset with both normal and anomalous cycles.

        Args:
            batch_gen: BatchGenerator instance.
            num_cycles: Total number of cycles to generate.
            anomaly_ratio: Fraction of cycles that are anomalous.
            robots: List of robot IDs.
            seed: Random seed for anomaly selection.

        Returns:
            (csv_filepath, labels_filepath, row_count) tuple.
        """
        if robots is None:
            robots = [0, 1, 2, 3]

        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "test_welding_robot.csv")
        labels_path = os.path.join(self.output_dir, "test_labels.csv")

        # Remove existing files
        for p in [csv_path, labels_path]:
            if os.path.exists(p):
                os.remove(p)

        rng = random.Random(seed)
        anomaly_types = ["spike", "thermal_drift", "flatline", "phase_shift"]
        anomaly_fns = {
            "spike": inject_spike,
            "thermal_drift": inject_thermal_drift,
            "flatline": inject_flatline,
            "phase_shift": inject_phase_shift,
        }

        chunk: list[CycleRow] = []
        labels: list[dict] = []
        row_count = 0
        cycles_written = 0

        while cycles_written < num_cycles:
            robot_id = rng.choice(robots)
            clean_sample = batch_gen.sample_gen.generate_sample(robot_id=robot_id)

            # Decide if this cycle is anomalous
            is_anomaly = rng.random() < anomaly_ratio and cycles_written < num_cycles

            if is_anomaly:
                anomaly_type = rng.choice(anomaly_types)
                inject_fn = anomaly_fns[anomaly_type]
                noisy_sample = inject_fn(clean_sample, rng)
            else:
                anomaly_type = "normal"
                noisy_sample = batch_gen.sample_gen.add_noise(clean_sample)

            # Generate start time (for labeling)
            # We use a simplified timestamp since we're not scheduling over days
            base_date = datetime(2025, 7, 1)
            start_dt = base_date.replace(hour=6, minute=0, second=0)
            start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

            # Convert to CycleRows
            n = len(noisy_sample.wire_feed_speed)

            for step_idx in range(n):
                ts = start_dt + timedelta(seconds=step_idx * SECONDS_PER_STEP)
                phase = _phase_label(step_idx, noisy_sample.phase_boundaries)

                chunk.append(CycleRow(
                    robot_code=noisy_sample.robot_id,
                    start_time=start_time_str,
                    program_number=noisy_sample.program_id,
                    infer_recorded_date=ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    serial_number=step_idx + 1,
                    phase=phase,
                    wire_feed_speed_mm_min=round(float(noisy_sample.wire_feed_speed[step_idx]), 1),
                    wire_temperature_c=round(float(noisy_sample.temperature[step_idx]), 1),
                    welding_current_a=round(float(noisy_sample.welding_current[step_idx]), 1),
                    timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
                ))

                row_count += 1
                if len(chunk) >= self.chunk_size:
                    _write_rows_to_csv(chunk, csv_path)
                    chunk = []

            # Record label
            labels.append({
                "robot_code": noisy_sample.robot_id,
                "start_time": start_time_str,
                "program_number": noisy_sample.program_id,
                "is_anomaly": 1 if is_anomaly else 0,
                "anomaly_type": anomaly_type,
            })

            cycles_written += 1
            if cycles_written % 100 == 0:
                _write_labels_to_csv(labels, labels_path)
                labels = []

        # Write remaining chunks and labels
        if chunk:
            _write_rows_to_csv(chunk, csv_path)
        if labels:
            _write_labels_to_csv(labels, labels_path)

        return csv_path, labels_path, row_count
