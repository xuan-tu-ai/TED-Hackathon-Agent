# constants.py

Central configuration file defining sensor lists, statistical feature suffixes, and window sizes used throughout the anomaly detection pipeline. Provides NUM_SENSORS, NUM_STATS, WINDOW_SIZE, and other constants that coordinate data dimensions across training and inference.

## Source Code

```python
"""  
This file concluded some of constatns used in module
"""

from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()
sc = spark.sparkContext

SENSORS = [
    'air_motor_rotation_speed',
    'electro_pneumatic_output',
    'high_pressure_current',
    'high_pressure_voltage',
    'paint_pressure',
    'paint_flow_rate'
]
""" 
Sensors with orders saved in slicing table.
"""

STATS_SUFFIXES = [
    'mean', 'stddev', 
    'kurtosis', 'skewness',
    'range', 'rms',
    'slope', 'crest_factor',
    'max', 'min'
]
"""Suffixes of statistics features."""

STATS_COLS = [stat for sensor in SENSORS for stat in [f"{sensor}_{suffix}" for suffix in STATS_SUFFIXES]]


FILE_STATS_SUFFIXES = ['mean', 'min', 'max', 'std', 'kurtosis', 'slope']

FILE_STATS_COLS = [f"{sensor}_file_{suffix}" for sensor in SENSORS for suffix in FILE_STATS_SUFFIXES]

NUM_SENSORS = len(SENSORS)
NUM_STATS = len(STATS_SUFFIXES)
NUM_FILE_STATS = len(FILE_STATS_SUFFIXES)


WINDOW_SIZE = 32
"""Size of each window splitted from a raw signal."""

FLATTENED_SIZE = WINDOW_SIZE * NUM_SENSORS
"""Size of flattened signal saved in DB."""

NORMAL_WINDOW_SIZE = 300

RANDOM_SEED = 42

# This is not actual num robot, program
MAX_ROBOTS = 64
MAX_PROGRAMS = 512
```