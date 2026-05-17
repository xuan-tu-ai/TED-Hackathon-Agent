"""Constants used throughout the anomaly detection module.

All sensor names, statistical feature columns, and window / data-layout
dimensions are centralised here to avoid magic values spread across the
codebase.  Only one import path should be used:
``app.anomaly_detection.constants``.
"""

#: Names of the six pneumatic / electrical sensors monitored per robot.
#: Each sensor produces a univariate time-series at 100 Hz.
SENSORS = [
    "air_motor_rotation_speed",
    "electro_pneumatic_output",
    "high_pressure_current",
    "high_pressure_voltage",
    "paint_pressure",
    "paint_flow_rate",
]

#: Statistical reduction functions applied to each sensor window in the
#: time-domain.  These form the ``STATS_COLS`` feature matrix (60 columns =
#: 6 sensors × 10 suffixes).
STATS_SUFFIXES = [
    "mean", "stddev", "kurtosis", "skewness", "range",
    "rms", "slope", "crest_factor", "max", "min",
]

#: All 60 window-level statistical feature columns, generated as the Cartesian
#: product of :data:`SENSORS` and :data:`STATS_SUFFIXES`.
#: Shape ``(N, 60)`` where ``N`` = number of windows.
STATS_COLS = [f"{sensor}_{suffix}" for sensor in SENSORS for suffix in STATS_SUFFIXES]

#: Statistical reduction functions applied **per file** (one value per file,
#: not per window).  Fewer suffixes than window-level because file-level
#: aggregation is coarser.
FILE_STATS_SUFFIXES = [
    "mean", "min", "max", "std", "kurtosis", "slope",
]

#: All 36 file-level statistical feature columns, generated as the Cartesian
#: product of :data:`SENSORS` and :data:`FILE_STATS_SUFFIXES`.
#: Shape ``(N_files, 36)``.
FILE_STATS_COLS = [f"{sensor}_file_{suffix}" for sensor in SENSORS for suffix in FILE_STATS_SUFFIXES]

#: Number of sensors (6).
NUM_SENSORS = len(SENSORS)

#: Number of window-level stat suffixes (10).
NUM_STATS = len(STATS_SUFFIXES)

#: Number of file-level stat suffixes (6).
NUM_FILE_STATS = len(FILE_STATS_SUFFIXES)

#: Default sliding-window length for short-window models (V1–V6).
#: A window covers 32 time-steps ≈ 0.32 s at 100 Hz.
WINDOW_SIZE = 32

#: Flattened input dimension when 32 time-steps × 6 sensors is viewed as
#: a 1-D vector (192).
FLATTENED_SIZE = WINDOW_SIZE * NUM_SENSORS

#: Context window length for macro-window models (GLC-Flow V7, MaskFormer).
#: 300 steps ≈ 3 s at 100 Hz.
NORMAL_WINDOW_SIZE = 300

#: Global random seed for NumPy and PyTorch.
RANDOM_SEED = 42

#: Maximum number of robots in the plant (used for context embedding tables).
MAX_ROBOTS = 64

#: Maximum number of distinct paint programs (used for context embedding tables).
MAX_PROGRAMS = 512
