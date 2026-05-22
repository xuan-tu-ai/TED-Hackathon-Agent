"""Constants used throughout the anomaly detection module.

All sensor names, statistical feature columns, and window / data-layout
dimensions are centralised here to avoid magic values spread across the
codebase.  Only one import path should be used:
``app.anomaly_detection.constants``.
"""

#: Names of the three welding-robot sensor channels monitored per robot.
#: Each sensor produces a univariate time-series sampled every 250 ms.
SENSORS = [
    "wire_feed_speed",
    "wire_temperature",
    "welding_current",
]

#: Statistical reduction functions applied to each sensor window in the
#: time-domain.  These form the ``STATS_COLS`` feature matrix (30 columns =
#: 3 sensors × 10 suffixes).
STATS_SUFFIXES = [
    "mean", "stddev", "kurtosis", "skewness", "range",
    "rms", "slope", "crest_factor", "max", "min",
]

#: All 30 window-level statistical feature columns, generated as the Cartesian
#: product of :data:`SENSORS` and :data:`STATS_SUFFIXES`.
#: Shape ``(N, 30)`` where ``N`` = number of windows.
STATS_COLS = [f"{sensor}_{suffix}" for sensor in SENSORS for suffix in STATS_SUFFIXES]

#: Statistical reduction functions applied **per file** (one value per file,
#: not per window).  Fewer suffixes than window-level because file-level
#: aggregation is coarser.
FILE_STATS_SUFFIXES = [
    "mean", "min", "max", "std", "kurtosis", "slope",
]

#: All 18 file-level statistical feature columns, generated as the Cartesian
#: product of :data:`SENSORS` and :data:`FILE_STATS_SUFFIXES`.
#: Shape ``(N_files, 18)``.
FILE_STATS_COLS = [f"{sensor}_file_{suffix}" for sensor in SENSORS for suffix in FILE_STATS_SUFFIXES]

#: Number of sensors (3).
NUM_SENSORS = len(SENSORS)

#: Number of window-level stat suffixes (10).
NUM_STATS = len(STATS_SUFFIXES)

#: Number of file-level stat suffixes (6).
NUM_FILE_STATS = len(FILE_STATS_SUFFIXES)

#: Default patch size for MaskFormer local patching.
#: A patch covers 32 time-steps ≈ 8 s at 250 ms sampling.
WINDOW_SIZE = 32

#: Flattened input dimension when 32 time-steps × 3 sensors is viewed as
#: a 1-D vector (96).
FLATTENED_SIZE = WINDOW_SIZE * NUM_SENSORS

#: Macro-window length for MaskFormer models.
#: 240 time-steps ≈ 60 s at 250 ms sampling.
NORMAL_WINDOW_SIZE = 240

#: Global random seed for NumPy and PyTorch.
RANDOM_SEED = 42

#: Maximum number of robots in the plant (used for context embedding tables).
MAX_ROBOTS = 32

#: Maximum number of distinct paint programs (used for context embedding tables).
MAX_PROGRAMS = 256
