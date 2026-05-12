"""
This file concluded some of constants used in the module
"""

# Data config

SENSORS = [
    "air_motor_rotation_speed",
    "electro_pneumatic_output",
    "high_pressure_current",
    "high_pressure_voltage",
    "paint_pressure",
    "paint_flow_rate"
]
"""
Sensors with orders saved in slicing table.
"""

STATS_SUFFIXES = [
    "mean", "stddev", "kurtosis", "skewness", "range",
    "rms", "slope", "crest_factor", "max", "min"
]
"""
Suffixes of statistics of each sensor
"""

STATS_COLS = [f"{sensor}_{suffix}" for sensor in SENSORS for suffix in STATS_SUFFIXES]
"""
Columns of statistics of each sensor
"""

FILE_STATS_SUFFIXES = [
    "mean", "min", "max", "std", "kurtosis", "slope"
]

FILE_STATS_COLS = [f"{sensor}_file_{suffix}" for sensor in SENSORS for suffix in FILE_STATS_SUFFIXES]

NUM_SENSORS = len(SENSORS)
NUM_STATS = len(STATS_SUFFIXES)
NUM_FILE_STATS = len(FILE_STATS_SUFFIXES)

WINDOW_SIZE = 32
"""
Size of each window splitted from a raw signal.
"""

FLATTENED_SIZE = WINDOW_SIZE * NUM_SENSORS
"""
Size of flattened signal saved in db.
"""

NORMAL_WINDOW_SIZE = 300

RANDOM_SEED = 42

# This is not actual num robot, program
MAX_ROBOTS = 64
MAX_PROGRAMS = 512
