import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

import app.anomaly_detection.constants as constants


def _row_to_dict(row: Any) -> dict:
    if hasattr(row, "asDict"):
        return row.asDict()
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return row
    raise TypeError("result_row must be a dict-like row")


def _sanitize_start_time(value: Any) -> str:
    return str(value).replace(":", "_").replace(" ", "_").replace("/", "_")


def analyze_trigger_sources(
    spark, result_table: str, sensors: Sequence[str] | None = None
) -> dict:
    sensors = list(sensors) if sensors is not None else list(constants.SENSORS)
    df = spark.table(result_table).filter(F.col("is_file_anomaly") == True)
    total_anomalies = df.count()

    if total_anomalies == 0:
        print(f"No anomalies found in {result_table}")
        return {"total_anomalies": 0, "file_stats": {}, "sensor_stats": {}}

    file_stats_row = df.select(
        F.sum(
            F.when(F.col("is_shape_triggered") & F.col("is_win_triggered"), 1).otherwise(0)
        ).alias("both"),
        F.sum(
            F.when(F.col("is_shape_triggered") & ~F.col("is_win_triggered"), 1).otherwise(0)
        ).alias("shape_only"),
        F.sum(
            F.when(~F.col("is_shape_triggered") & F.col("is_win_triggered"), 1).otherwise(0)
        ).alias("win_only"),
    ).collect()[0]

    file_stats = {
        "both": int(file_stats_row["both"]),
        "shape_only": int(file_stats_row["shape_only"]),
        "win_only": int(file_stats_row["win_only"]),
    }

    print("=" * 60)
    print(f"TRIGGER SOURCE ANALYSIS: {result_table}")
    print(f"Total anomalous files: {total_anomalies}")
    print("File-level trigger breakdown:")
    print(
        f"  Both shape & win: {file_stats['both']} ({file_stats['both'] / total_anomalies * 100:.1f}%)"
    )
    print(
        f"  Shape only:       {file_stats['shape_only']} ({file_stats['shape_only'] / total_anomalies * 100:.1f}%)"
    )
    print(
        f"  Win only:         {file_stats['win_only']} ({file_stats['win_only'] / total_anomalies * 100:.1f}%)"
    )

    sensor_stats = {}
    print("Per-sensor trigger breakdown:")
    for s in sensors:
        sensor_row = df.select(
            F.sum(F.when(F.col(f"trigger_source_{s}") == "both", 1).otherwise(0)).alias("both"),
            F.sum(F.when(F.col(f"trigger_source_{s}") == "shape", 1).otherwise(0)).alias("shape"),
            F.sum(F.when(F.col(f"trigger_source_{s}") == "win", 1).otherwise(0)).alias("win"),
            F.sum(F.when(F.col(f"trigger_source_{s}") == "none", 1).otherwise(0)).alias("none"),
        ).collect()[0]

        sensor_stats[s] = {
            "both": int(sensor_row["both"]),
            "shape": int(sensor_row["shape"]),
            "win": int(sensor_row["win"]),
            "none": int(sensor_row["none"]),
        }

        triggered = sensor_stats[s]["both"] + sensor_stats[s]["shape"] + sensor_stats[s]["win"]
        print(
            f"  {s}: triggered={triggered} | "
            f"both={sensor_stats[s]['both']}, shape={sensor_stats[s]['shape']}, "
            f"win={sensor_stats[s]['win']}, none={sensor_stats[s]['none']}"
        )

    return {
        "total_anomalies": total_anomalies,
        "file_stats": file_stats,
        "sensor_stats": sensor_stats,
    }


def visualize_feature_level_report_vq(
    spark,
    result_row: Any,
    gold_table_name: str,
    feature_ids: Sequence[int] | None = None,
    save_dir: str | None = None,
    k_skip: int = 1,
) -> None:
    row = _row_to_dict(result_row)
    feature_ids = list(feature_ids) if feature_ids is not None else list(range(constants.NUM_SENSORS))
    if not feature_ids:
        print("No feature ids provided")
        return

    sensor_names = [constants.SENSORS[i] for i in feature_ids]

    file_data = (
        spark.table(gold_table_name)
        .where(
            (F.col("robot_code") == row["robot_code"])
            & (F.col("program_number") == row["program_number"])
            & (F.col("start_time") == row["start_time"])
        )
        .orderBy("window_start_serial")
        .toPandas()
    )

    if len(file_data) <= 2 * k_skip:
        print("File too short to visualize with current K_SKIP")
        return

    serials = file_data["window_start_serial"].values.astype(int)
    min_s = int(serials.min())

    W = constants.WINDOW_SIZE
    C_all = constants.NUM_SENSORS

    raw_all = np.array(
        [([0.0 if x is None else x for x in r]) for r in file_data["raw_values_array"]],
        dtype=np.float32,
    ).reshape(-1, C_all, W)[:, feature_ids, :]

    total_steps = int(serials.max() - min_s + W)
    actual_from_gold = np.zeros((len(feature_ids), total_steps), dtype=np.float32)

    for i in range(len(file_data)):
        t_off = int(serials[i] - min_s)
        actual_from_gold[:, t_off : t_off + W] = raw_all[i]

    actual_from_res = []
    recon_from_res = []
    err_steps = []

    for s in sensor_names:
        actual_from_res.append(np.array(row.get(f"actual_raw_{s}", []), dtype=np.float32))
        recon_from_res.append(np.array(row.get(f"recon_raw_{s}", []), dtype=np.float32))
        err_steps.append(row.get(f"err_steps_{s}", []))

    for idx, s in enumerate(sensor_names):
        if len(actual_from_res[idx]) == 0:
            print(f"Missing actual_raw_{s} in result_row")
        if len(actual_from_res[idx]) != total_steps:
            print(
                f"Length mismatch {s}: result={len(actual_from_res[idx])}, gold={total_steps}"
            )

    fig, axes = plt.subplots(
        len(sensor_names), 1, figsize=(24, 3 * len(sensor_names)), sharex=True
    )
    if len(sensor_names) == 1:
        axes = [axes]

    time_axis = np.arange(total_steps) + min_s

    for i, s in enumerate(sensor_names):
        ax = axes[i]
        actual_sig = (
            actual_from_res[i] if len(actual_from_res[i]) == total_steps else actual_from_gold[i]
        )
        recon_sig = recon_from_res[i]

        if recon_sig.size == 0:
            recon_sig = np.full_like(actual_sig, np.nan)

        ax.plot(
            time_axis[: len(actual_sig)],
            actual_sig,
            label="Actual",
            color="#1f77b4",
            linewidth=1.5,
        )
        ax.plot(
            time_axis[: len(recon_sig)],
            recon_sig,
            label="AI Baseline (VQ)",
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.2,
        )

        err_idx = [step - min_s for step in err_steps[i] if min_s <= step < min_s + total_steps]
        if err_idx:
            ax.scatter(
                time_axis[err_idx],
                actual_sig[err_idx],
                s=20,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                label="Fault Detected",
            )

        ax.set_title(s.upper(), fontweight="bold")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_start_time = _sanitize_start_time(row.get("start_time"))
        filename = f"{row['robot_code']}_{row['program_number']}_{safe_start_time}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()

    plt.close(fig)