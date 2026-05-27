# Welding Robot Data Simulator

Physics-based data generator for welding robot sensor time-series. Produces realistic multivariate data (wire_feed_speed, welding_current, temperature) with LTI dynamics, colored noise, and 4 anomaly types.

## Architecture

```
scripts/data_simulator/
├── __init__.py
├── __main__.py           # python -m scripts.data_simulator
├── cli.py                # CLI entry point with argparse
├── config.py             # Central SimulatorConfig dataclass
├── writer.py             # Chunked CSV writer
├── generator.py          # WeldingSampleGenerator (compositor)
├── profiles.py           # 16 welding program profiles
├── batch_generator.py    # 30-day scheduling across 4 robots
├── physics/
│   ├── __init__.py
│   ├── current_dynamics.py   # LTI 2nd-order underdamped system
│   ├── thermal_dynamics.py   # 1st-order exponential charge/discharge
│   └── noise.py              # 3 sensor-specific colored noise models
└── anomalies/
    ├── __init__.py
    ├── spike.py              # Cross-sensor short circuit
    ├── thermal_drift.py      # Cooling failure / thermal runaway
    ├── flatline.py           # Sensor signal loss
    └── phase_shift.py        # Power supply delay
```

## Physics Model

### Current Dynamics (S1)
Second-order underdamped LTI system via `scipy.signal.TransferFunction`:
- Natural frequency ωₙ = 8 rad/s
- Damping ratio ζ = 0.55 → ~12% overshoot at arc ignition
- 1-step transport delay
- Steady-state: ~165A at 4000 mm/min feed rate

### Thermal Dynamics (S2)
First-order exponential model with separate heating/cooling constants:
- **Heating:** α_heat = 0.035 (~28 steps to 63% of equilibrium)
- **Cooling:** α_cool = 0.012 (~83 steps to 63%, ~3× slower)
- Robot 3 baseline: +5°C (weaker cooling system)

### Colored Noise
| Sensor | Noise Type | Parameters |
|--------|-----------|------------|
| S0: wire_feed_speed | Mechanical vibration (sinusoidal + Gaussian) | 0.8% amplitude, 0.05 freq |
| S1: welding_current | Arc white noise (Gaussian) | 5% of signal amplitude |
| S2: temperature | Thermal random walk + rare glitches | 0.1°C/step, 1:10K glitch prob |

## Program Profiles

16 profiles (IDs 0-15) with varying:
- **Duration:** 40–600 active steps
- **Feed speed:** 2000–5000 mm/min
- **Ramp aggressiveness:** 0.8–2.0 (linear to aggressive)
- **Categories:** Short (0-3), Medium (4-7), Long (8-11), Specialty (12-15)

## Anomaly Types

| ID | Name | Sensor Signature | Physical Cause |
|----|------|-----------------|----------------|
| 1 | Spike | 4× spike in S1 (welding_current), S0 normal | Wire stick / short circuit |
| 2 | Thermal Drift | Linear temp rise in S2 after 60% active phase | Cooling failure |
| 3 | Flatline | S0 drops to 0 for 20 steps, S1 decays | Sensor signal loss |
| 4 | Phase Shift | S1 delayed 15 steps behind S0 | Power supply delay |

## Usage

### CLI

```bash
# Generate full 30-day train + test dataset
uv run python scripts/simulate.py

# Generate only train data
uv run python scripts/simulate.py --train --no-test

# Custom configuration
uv run python scripts/simulate.py \
    --num-days 14 \
    --test-cycles 500 \
    --anomaly-ratio 0.3 \
    --robots 0 1 2 \
    --seed 42 \
    --output-dir ./my_data
```

### Python API

```python
from backend.data_simulator.batch_generator import BatchGenerator
from backend.data_simulator.writer import CSVWriter

# Generate 30 days of data for all 4 robots
gen = BatchGenerator(seed=42)
writer = CSVWriter(output_dir="data/generated")
path, count = writer.write_train(gen)
print(f"Generated {count:,} rows to {path}")

# Generate test data with anomalies
csv_path, label_path, test_count = writer.write_test(
    gen, num_cycles=2000, anomaly_ratio=0.5
)
print(f"Test: {test_count:,} rows, labels at {label_path}")
```

### Generating Individual Samples

```python
from backend.data_simulator.generator import WeldingSampleGenerator

gen = WeldingSampleGenerator(seed=42)
sample = gen.generate_noisy_sample(robot_id=0, program_id=5)
# sample.wire_feed_speed, sample.welding_current, sample.temperature
```

## Output Format

CSV schema matches `data/mock_welding_robot_data.csv`:
- `robot_code`: Robot ID (0-3)
- `start_time`: Cycle start datetime
- `program_number`: Program ID (0-15)
- `infer_recorded_date`: Timestep datetime with ms
- `serial_number`: Timestep index (1-based, resets per cycle)
- `phase`: idle/rampup/active/winddown/shutdown
- `wire_feed_speed_mm_min`: S0 sensor
- `wire_temperature_c`: S2 sensor  
- `welding_current_a`: S1 sensor
- `timestamp`: ISO 8601 with timezone

Test labels CSV:
- `robot_code, start_time, program_number, is_anomaly, anomaly_type`

## Tests

```bash
# Run all data simulator tests
uv run python -m pytest backend/tests/test_data_simulator/ -v

# Run specific test file
uv run python -m pytest backend/tests/test_data_simulator/test_current_dynamics.py -v
```

See `docs/data/GENERATION-GUIDE.md` for the original specification.
