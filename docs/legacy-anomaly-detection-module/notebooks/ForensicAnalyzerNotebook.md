# Forensic Analyzer Notebook

Performs forensic pattern detection for robot anomalies by analyzing reconstruction errors across multiple symptom dimensions. Uses the ForensicEngine with configurable symptom detectors (impulse, zero_gravity, volatility, static_bias, temporal_lag, gradient_inertia) and weighted pattern scoring for anomaly classification.

## Source Code
```python
# %% [markdown]
# # Forensic Analyzer - Pattern Detection for Robot Anomalies

# %% [markdown]
# ## Setup * imports

# %%
import os 
import sys
import random
import numpy as np
import pandas as pd
import json

sys.path.append('../../')

# %%
from src.anomaly_detection_dl.utils import set_reproducibility
import src.anomaly_detection_dl.constants as constants

set_reproducibility(constants.RANDOM_SEED)

# %%
from src.anomaly_detection_dl.forensic_analyzer.engine import ForensicEngine

# %% [markdown]
# ## Configuration * Engine Initialization

# %%
config = {
    "roi_settings": {
        "window_size": 5,
        "top_k_percentile": 95.0
    },
    "phases": {
        "startup_pct": 0.1,
        "shutdown_pct": 0.1,
        "transition_pct": 0.05
    },
    "symptoms": {
        "impulse": {
            "max_steps_for_surge": 3,
            "decay_steps": 12,
            "calibration": {"enable": True, "raw_min": 0.0, "raw_max": 1.0},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        },
        "zero_gravity": {
            "noise_floor_sigma": 0.1,
            "calibration": {"enable": True, "raw_min": 0.0, "raw_max": 1.0},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        },
        "volatility": {
            "sensitive_factor": 1.0,
            "calibration": {"enable": True, "raw_min": 1.0, "raw_max": 10.0},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        },
        "static_bias": {
            "min_error_threshold": 0.1,
            "stability_threshold": 0.2,
            "calibration": {"enable": True, "raw_min": -1.0, "raw_max": 1.0},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        },
        "temporal_lag": {
            "max_shift": 100,
            "calibration": {"enable": True, "raw_min": 0.0, "raw_max": 0.5},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        },
        "gradient_inertia": {
            "eps": 0.1,
            "sensitivity": 1.0,
            "calibration": {"enable": True, "raw_min": 0.0, "raw_max": 1.5},
            "clip": {"enable": True, "min_val": 0.0, "max_val": 1.0},
        }
    },
    "patterns": {
        "DROPOUT": {
            "score_type": "weighted_sum",
            "weights": {
                "impulse": -0.3,
                "zero_gravity": 1.0,
                "volatility": 0.0,
                "static_bias": -0.2,
                "temporal_lag": 0.0,
                "gradient_inertia": 0.0
            }
        },
        "SURGE": {
            "score_type": "weighted_sum",
            "weights": {
                "impulse": 0.8,
                "zero_gravity": 0.0,
                "volatility": -0.2,
                "static_bias": -0.1,
                "temporal_lag": 0.0,
                "gradient_inertia": 0.2
            }
        },
        "DEVIATION": {
            "score_type": "weighted_sum",
            "weights": {
                "impulse": 0.0,
                "zero_gravity": -0.6,
                "volatility": -0.5,
                "static_bias": 1.0,
                "temporal_lag": 0.0,
                "gradient_inertia": 0.0
            }
        },
        "INSTABILITY": {
            "score_type": "weighted_sum",
            "weights": {
                "impulse": 0.0,
                "zero_gravity": 0.0,
                "volatility": 0.8,
                "static_bias": -0.4,
                "temporal_lag": 0.0,
                "gradient_inertia": 0.2
            }
        },
        "TRANSITION": {
            "score_type": "weighted_sum",
            "weights": {
                "impulse": 0.0,
                "zero_gravity": 0.0,
                "volatility": -0.1,
                "static_bias": 0.0,
                "temporal_lag": 0.2,
                "gradient_inertia": 0.8
            }
        }
    }
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)

# %%
engine = ForensicEngine.from_config_file('config.json')

# %% [markdown]
# ## Load data and execution

# %%
abnormal_res_df = constants.spark.table("...").cache()

# %%
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry # noqa: F401

first_serial_df = (
    constants.spark.table("bronze table")
    .groupBy('robot_code', 'program_number', 'start_time')
    .agg(F.min('serial_number').alias('first_serial'))
    .select('robot_code', 'program_number', 'start_time', 'first_serial')
)

first_serial_lookup = {(row['robot_code'], row['program_number'], row['start_time']): row['first_serial'] for row in first_serial_df.collect()}

# %%
from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext

rows = abnormal_res_df.collect()

all_outputs = []
skipped = 0

for row in rows:
    key = (row['robot_code'], row['program_number'], row['start_time'])
    first_serial = first_serial_lookup.get(key)

    for feat_id, sensor in enumerate(constants.SENSORS):
        actual_val = row[f"actual_raw_{sensor}"]
        recon_val = row[f"recon_raw_{sensor}"]

        if not actual_val or not recon_val:
            skipped += 1
            continue

        actual = np.array(actual_val, dtype=np.float32)
        recon = np.array(recon_val, dtype=np.float32)
        err_steps_val = row[f"err_steps_{sensor}"]
        err_steps = list(err_steps_val) if err_steps_val else []

        metadata = {
            "robot_code": row['robot_code'],
            "program_number": row['program_number'],
            "start_time": row['start_time'],
            "sensor_name": sensor,
            "feature_id": feat_id,
            "first_serial": first_serial
        }

        ctx = ForensicContext(
            actual=actual,
            recon=recon,
            err_steps=err_steps,
            metadata=metadata
        )

        output = engine.analyze(ctx)
        all_outputs.append(output)

```