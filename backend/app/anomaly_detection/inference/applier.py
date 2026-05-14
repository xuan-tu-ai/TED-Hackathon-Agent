import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, BooleanType,
)
from pyspark import Broadcast

import app.anomaly_detection.constants as constants


class ResultApplier:
    """Universal threshold applier for pre-result score DataFrames.

    score_type="single" (V7): applies a global threshold to {pq_score}.
    score_type="multi"  (V6): applies per-sensor shape/win thresholds.
    """

    def __init__(
        self,
        threshold_config_br: Broadcast[dict],
        quantile: float,
        score_type: str = "multi",
    ):
        self.threshold_config = threshold_config_br.value
        self.quantile = quantile
        self.score_type = score_type
        self._init_schema(quantile, score_type)

    def _init_schema(self, quantile: float, score_type: str):
        pq = f"p{quantile * 100:.0f}"
        base = [
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            StructField("is_file_anomaly", BooleanType()),
        ]
        if score_type == "single":
            self.schema = StructType(base + [
                StructField(f"{pq}_score", FloatType()),
            ])
        else:
            self.schema = StructType(base + [
                StructField("is_shape_triggered", BooleanType()),
                StructField("is_win_triggered", BooleanType()),
                StructField(f"{pq}_shape_score", FloatType()),
                StructField(f"{pq}_win_score", FloatType()),
                *[StructField(f"{pq}_shape_{s}", FloatType()) for s in constants.SENSORS],
                *[StructField(f"{pq}_win_{s}", FloatType()) for s in constants.SENSORS],
            ])

    def make_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile
        score_type = self.score_type

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            results = []

            for _, row in pdf.iterrows():
                base_res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                }

                if score_type == "single":
                    score = row[f"{pq}_score"]
                    global_thresh = thresh_dict.get("global_threshold", 5.0)
                    base_res["is_file_anomaly"] = bool(score > global_thresh)
                    base_res[f"{pq}_score"] = float(score)
                else:
                    file_shape_triggered = False
                    file_win_triggered = False
                    for s in constants.SENSORS:
                        shape_score = row[f"{pq}_shape_{s}"]
                        win_score = row[f"{pq}_win_{s}"]
                        shape_thresh = thresh_dict[s]["shape"]
                        win_thresh = thresh_dict[s]["win"]
                        if shape_score >= shape_thresh:
                            file_shape_triggered = True
                        if win_score >= win_thresh:
                            file_win_triggered = True

                    base_res["is_file_anomaly"] = bool(file_shape_triggered or file_win_triggered)
                    base_res["is_shape_triggered"] = bool(file_shape_triggered)
                    base_res["is_win_triggered"] = bool(file_win_triggered)
                    base_res[f"{pq}_shape_score"] = float(row[f"{pq}_shape_score"])
                    base_res[f"{pq}_win_score"] = float(row[f"{pq}_win_score"])
                    for s in constants.SENSORS:
                        base_res[f"{pq}_shape_{s}"] = float(row[f"{pq}_shape_{s}"])
                        base_res[f"{pq}_win_{s}"] = float(row[f"{pq}_win_{s}"])

                results.append(base_res)

            return pd.DataFrame(results)

        return apply_thresholds
