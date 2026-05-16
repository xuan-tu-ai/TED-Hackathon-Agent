import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
)
from pyspark import Broadcast


class FlowMatchingApplier:
    """Apply global threshold to flow-matching NLL anomaly scores.

    Designed for Databricks: takes broadcasted threshold config, returns a Pandas UDF.
    Compatible with FlowMatchingPreResult output.
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
            StructField("is_file_anomaly", IntegerType()),
            StructField(f"{pq}_score", FloatType()),
            StructField("is_anomaly_per_sensor", IntegerType()),
        ])

    def make_apply_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            results = []

            for _, row in pdf.iterrows():
                total_steps = row["total_steps"]

                if total_steps == 0:
                    res = {
                        "robot_code": row["robot_code"],
                        "program_number": row["program_number"],
                        "start_time": row["start_time"],
                        "is_file_anomaly": 1,
                        f"{pq}_score": row[f"{pq}_score"],
                        "is_anomaly_per_sensor": 0,
                    }
                    results.append(res)
                    continue

                score = row[f"{pq}_score"]
                is_file_anomaly = 1 if score > thresh_dict.get("global_threshold", 5.0) else 0

                res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                    "is_file_anomaly": is_file_anomaly,
                    f"{pq}_score": score,
                    "is_anomaly_per_sensor": is_file_anomaly,
                }
                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds
