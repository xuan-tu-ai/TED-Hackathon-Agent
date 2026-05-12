import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, ArrayType, BooleanType
)
from pyspark import Broadcast

import app.anomaly_detection_v6.constants as constants

class ThresholdApplier:
    """
    Applies configurable thresholds to a pre-result table (with score maps)
    to generate err_steps, is_file_anomaly, and trigger_source per sensor.
    """
    def __init__(self, threshold_config_br: Broadcast[dict], quantile: float):
        self.threshold_config = threshold_config_br.value
        self.quantile = quantile
        self._init_schema(self.quantile)

    def _init_schema(self, quantile: float):
        pq = f"p{quantile * 100:.0f}"
        self.schema = StructType([
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            
            # Anomaly flags
            StructField("is_file_anomaly", BooleanType()),
            StructField("is_shape_triggered", BooleanType()),
            StructField("is_win_triggered", BooleanType()),
            
            # Per-sensor error steps
            *[StructField(f"err_steps_{s}", ArrayType(IntegerType())) for s in constants.SENSORS],
            
            # Per-sensor trigger source: "shape", "win", "both", "none"
            *[StructField(f"trigger_source_{s}", StringType()) for s in constants.SENSORS],
            
            # Pass through scores
            StructField(f"{pq}_shape_score", FloatType()),
            StructField(f"{pq}_win_score", FloatType()),
            *[StructField(f"{pq}_shape_{s}", FloatType()) for s in constants.SENSORS],
            *[StructField(f"{pq}_win_{s}", FloatType()) for s in constants.SENSORS],
            
            # Pass through raw signals for visualization
            *[StructField(f"actual_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            *[StructField(f"recon_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS]
        ])

    def make_apply_fn(self):
        """Returns a Pandas UDF function to apply thresholds to pre-results."""
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            results = []

            for _, row in pdf.iterrows():
                valid_start = row['valid_start_idx']
                valid_end = row['valid_end_idx']
                min_s = row['min_serial']
                total_steps = row['total_steps']

                # Handle empty/short files
                if total_steps == 0:
                    res = {
                        'robot_code': row['robot_code'],
                        'program_number': row['program_number'],
                        'start_time': row['start_time'],
                        'is_file_anomaly': True,
                        'is_shape_triggered': True,
                        'is_win_triggered': True,
                        f'{pq}_shape_score': row[f'{pq}_shape_score'],
                        f'{pq}_win_score': row[f'{pq}_win_score'],
                    }
                    for s in constants.SENSORS:
                        res[f'err_steps_{s}'] = []
                        res[f'trigger_source_{s}'] = 'both'
                        res[f'{pq}_shape_{s}'] = row[f'{pq}_shape_{s}']
                        res[f'{pq}_win_{s}'] = row[f'{pq}_win_{s}']
                        res[f'actual_raw_{s}'] = row[f'actual_raw_{s}']
                        res[f'recon_raw_{s}'] = row[f'recon_raw_{s}']
                    results.append(res)
                    continue

                # Build valid mask
                valid_mask = np.zeros(total_steps, dtype=bool)
                valid_mask[valid_start:valid_end] = True

                sensor_err_steps = {}
                sensor_trigger_sources = {}
                file_shape_triggered = False
                file_win_triggered = False

                for s in constants.SENSORS:
                    shape_map = np.array(row[f'shape_map_{s}'])
                    win_map = np.array(row[f'win_map_{s}'])

                    shape_thresh = thresh_dict[s]['shape']
                    win_thresh = thresh_dict[s]['win']

                    # Per-step anomaly detection by threshold type
                    shape_anom = (shape_map >= shape_thresh) & valid_mask
                    win_anom = (win_map >= win_thresh) & valid_mask

                    # Combined anomaly (either shape or win)
                    is_step_anom = shape_anom | win_anom
                    err_steps = (np.where(is_step_anom)[0] + min_s).tolist()
                    sensor_err_steps[s] = err_steps

                    # Determine trigger source for this sensor
                    has_shape = np.any(shape_anom)
                    has_win = np.any(win_anom)

                    if has_shape and has_win:
                        trigger_source = 'both'
                    elif has_shape:
                        trigger_source = 'shape'
                    elif has_win:
                        trigger_source = 'win'
                    else:
                        trigger_source = 'none'

                    sensor_trigger_sources[s] = trigger_source

                    if has_shape:
                        file_shape_triggered = True
                    if has_win:
                        file_win_triggered = True

                # Build result row
                res = {
                    'robot_code': row['robot_code'],
                    'program_number': row['program_number'],
                    'start_time': row['start_time'],
                    'is_file_anomaly': any(len(sensor_err_steps[s]) > 0 for s in constants.SENSORS),
                    'is_shape_triggered': file_shape_triggered,
                    'is_win_triggered': file_win_triggered,
                    f'{pq}_shape_score': row[f'{pq}_shape_score'],
                    f'{pq}_win_score': row[f'{pq}_win_score'],
                }

                for s in constants.SENSORS:
                    res[f'err_steps_{s}'] = sensor_err_steps[s]
                    res[f'trigger_source_{s}'] = sensor_trigger_sources[s]
                    res[f'{pq}_shape_{s}'] = row[f'{pq}_shape_{s}']
                    res[f'{pq}_win_{s}'] = row[f'{pq}_win_{s}']
                    res[f'actual_raw_{s}'] = row[f'actual_raw_{s}']
                    res[f'recon_raw_{s}'] = row[f'recon_raw_{s}']

                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds