"""Validate Sprint 18b data: convergence, anomaly types, flatline, 512-step."""
import pandas as pd
import numpy as np

train_df = pd.read_csv("data/generated_sprint18b/train_welding_robot.csv")
test_df = pd.read_csv("data/generated_sprint18b/test_welding_robot.csv")
labels = pd.read_csv("data/generated_sprint18b/test_labels.csv")

# 1. Step count
steps = train_df.groupby("cycle_id").size()
print(f"STEP COUNT: all = {steps.min()} (fixed 512)  |  unique = {steps.nunique()}")

# 2. Anomaly types
anom = labels[labels["is_anomaly"] == 1]
print(f"ANOMALY TYPES: {anom['anomaly_type'].nunique()}")
print(anom["anomaly_type"].value_counts().to_string())

# 3. Flatline duration
flat_ids = labels[labels["anomaly_type"] == "flatline"]["cycle_id"].values[:5]
for cid in flat_ids:
    feed = test_df[test_df["cycle_id"] == cid]["wire_feed_speed_mm_min"].values
    zero = (feed < 0.1).astype(int)
    changes = np.diff(np.concatenate(([0], zero, [0])))
    runs = np.where(changes == -1)[0] - np.where(changes == 1)[0]
    print(f"  Flatline {cid}: {runs.max()} steps")

# 4. Convergence
robot, prog = 0, 5
cycles = train_df[
    (train_df["robot_code"] == robot) & (train_df["program_number"] == prog)
]["cycle_id"].unique()[:5]
base = train_df[train_df["cycle_id"] == cycles[0]].set_index("serial_number")
print(f"CONVERGENCE (R{robot}P{prog}):")
for col, name in [
    ("wire_feed_speed_mm_min", "Feed"),
    ("welding_current_a", "Current"),
    ("wire_temperature_c", "Temp"),
]:
    devs = []
    for c in cycles[1:]:
        other = train_df[train_df["cycle_id"] == c].set_index("serial_number")
        dev = np.abs(base[col].values - other[col].values).max()
        devs.append(dev)
    pct = max(devs) / base[col].max() * 100
    print(f"  {name}: max dev={max(devs):.2f} ({pct:.2f}%)")

# 5. Noise comparison (active-phase residuals)
def active_residual_std(df, cid):
    vals = df[df["cycle_id"] == cid]["welding_current_a"].values
    s, e = int(len(vals) * 0.2), int(len(vals) * 0.8)
    return np.std(np.diff(vals[s:e])) / 1.414

train_noises = [active_residual_std(train_df, c) for c in train_df["cycle_id"].unique()[:50]]
norm_cycles = labels[labels["anomaly_type"] == "normal"]["cycle_id"].values[:50]
test_noises = [active_residual_std(test_df, c) for c in norm_cycles]
print(f"NOISE (active-phase residual std):")
print(f"  Train: {np.mean(train_noises):.3f} A  (reduced noise)")
print(f"  Test:  {np.mean(test_noises):.3f} A  (full noise)")
print(f"  Ratio: {np.mean(test_noises)/max(np.mean(train_noises), 0.001):.2f}x")

# 6. Robot diversity
r0_cycles = train_df[train_df["robot_code"] == 0]["cycle_id"].unique()
r3_cycles = train_df[train_df["robot_code"] == 3]["cycle_id"].unique()
r0_temp = train_df[train_df["cycle_id"] == r0_cycles[0]]["wire_temperature_c"].mean()
r3_temp = train_df[train_df["cycle_id"] == r3_cycles[0]]["wire_temperature_c"].mean()
print(f"ROBOT DIVERSITY: R0 temp={r0_temp:.1f}C, R3 temp={r3_temp:.1f}C (diff={r3_temp - r0_temp:.1f}C)")

print(f"\n{'='*50}")
print(f"ALL VALIDATIONS COMPLETE")
print(f"{'='*50}")
