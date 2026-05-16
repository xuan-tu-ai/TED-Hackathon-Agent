import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
    ArrayType, BooleanType,
)
from pyspark import Broadcast

import app.anomaly_detection.constants as constants


class MaskFormerApplier:
    """Apply per-sensor heatmap thresholds to MaskFormerPreResult output.

    Designed for Databricks: takes broadcasted threshold config, returns a Pandas UDF.
    Compatible with MaskFormerPreResult output schema.

    threshold_config: {"global_threshold": float} — applied to p{q}_score.
    Optionally per-sensor: {"sensors": {"sensor_name": float}} for sensor-level anomalies.
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
            StructField("is_file_anomaly", BooleanType()),
            StructField(f"{pq}_score", FloatType()),
            *[
                StructField(f"is_anomaly_{s}", BooleanType())
                for s in constants.SENSORS
            ],
            *[
                StructField(f"heatmap_{s}", ArrayType(FloatType()))
                for s in constants.SENSORS
            ],
        ])

    def make_apply_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            global_thresh = thresh_dict.get("global_threshold", 0.5)
            sensor_thresholds = thresh_dict.get("sensors", {})
            results = []

            for _, row in pdf.iterrows():
                score = row[f"{pq}_score"]
                is_file_anomaly = bool(score > global_thresh)

                res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                    "is_file_anomaly": is_file_anomaly,
                    f"{pq}_score": float(score),
                }

                valid_start = int(row.get("valid_start_idx", 0))
                valid_end = int(row.get("valid_end_idx", 0)) or None

                for s in constants.SENSORS:
                    heatmap = row.get(f"heatmap_{s}", [])
                    res[f"heatmap_{s}"] = heatmap

                    if heatmap and valid_end:
                        sensor_thresh = sensor_thresholds.get(s, global_thresh)
                        valid_region = np.array(heatmap[valid_start:valid_end], dtype=np.float32)
                        res[f"is_anomaly_{s}"] = bool(valid_region.max() > sensor_thresh)
                    else:
                        res[f"is_anomaly_{s}"] = is_file_anomaly

                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds
