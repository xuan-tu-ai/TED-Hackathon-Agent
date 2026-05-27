# Sprint 20 Plan

## Sprint Goal
Nâng cấp pipeline anomaly detection lên V9: sửa data simulator (noise fix + durations), thiết kế 4-dataset strategy, xây dựng `GLCMaeV9` thay thế GlobalRevIN + PerPatchRevIN bằng LogScaler + `GlobalSeqNorm` (lớp mới trong `normalization.py`), bỏ tail window, và nâng aggregation lên p99.

---

## Context & Motivation

| # | Vấn đề | Impact |
|---|--------|--------|
| 1 | `write_test()` thiếu `add_noise()` trước inject → anomaly files không có noise | WFS score bị đảo ngược |
| 2 | Spike duration 2 steps (250ms), cable_wear dropout 1–3 steps quá ngắn | p99 bỏ sót |
| 3 | Val set inference = normal-only → threshold calibrate lệch | FPR cao (normal recall = 0.36) |
| 4 | `GlobalRevIN` xóa thermal drift (ROC-AUC = 0.056) | thermal_drift không detectable |
| 5 | `PerPatchRevIN` hấp thụ spike vào local context (3σ → 25σ nếu bỏ) | Spike bị dilute |
| 6 | Tail window thứ 6 có 64 bước zero-padding → model học zeros = normal | Flatline khó detect |
| 7 | p95 trên 256 steps: top 13 bước → transient ngắn bị chìm | cable_wear recall = 0.10 |

---

## Thiết kế 4-Dataset Strategy

| Dataset | Nội dung | Dùng cho |
|---------|----------|----------|
| `train_welding_robot.csv` | Normal only, reduced noise, deterministic | Training model |
| `val_train_welding_robot.csv` | Normal only, full noise (7 ngày) | Validate loss trong training loop |
| `threshold_calibration_welding_robot.csv` | Normal + 5 anomaly types, labelled | Chọn threshold thủ công (plot distribution per type) |
| `test_welding_robot.csv` | Normal + 5 anomaly types, labelled | Evaluation cuối |

---

## Thiết kế Normalization V9

**V8 hiện tại:**
```
Global stream: raw → GlobalRevIN [B,C,T] → Patchify → CNN_global
Local stream:  raw → Patchify → PerPatchRevIN [B,C,P,L] → CNN_local
```

**V9 thay thế — lớp mới `GlobalSeqNorm` trong `normalization.py`:**
```
Shared pre-norm: raw [B,C,T] → LogScaler → GlobalSeqNorm → x_norm [B,C,T]

Global stream: x_norm → Patchify → CNN_global
Local stream:  x_norm → Patchify → CNN_local   ← bỏ PerPatchRevIN
```

**`GlobalSeqNorm`** (tên gợi ý, tạo trong `normalization.py`):
- Tương tự `GlobalStatsStandardizer` nhưng được đặt tên rõ ràng cho dùng ở sequence-level input [B, C, T], phân biệt với `GlobalStatsStandardizer` dùng cho statistical features [B, C, K] ở V1-V6.
- Cài đặt: `nn.BatchNorm1d(num_sensors, affine=False, track_running_stats=True)` — giống hệt logic cũ.

**Cơ chế tích lũy và lưu trữ stats:**
- Trong `.train()` mode: mỗi batch cập nhật `running_mean` / `running_var` qua EMA tự động
- Stats lưu trong `model.state_dict()` tự động (`glob_seq_norm.bn.running_mean`, v.v.)
- **Training notebook**: không cần extra "fit" call — stats tích lũy tự nhiên trong training loop
- **Inference**: `model.load_state_dict(ckpt)` → `model.eval()` → BatchNorm dùng saved running stats

---

## Atomic Tasks

Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1 — Data Simulator Fixes

- [ ] **T1: `writer.py` — restore `add_noise()` trước anomaly injection**
  `write_test()`: thêm `noisy_base = batch_gen.sample_gen.add_noise(clean_sample)`, gọi `inject_fn(noisy_base, rng)`. Thêm lại test `test_write_test_anomaly_has_sensor_noise` trong `test_integration.py` (std WFS của spike files > 5.0).

- [ ] **T2: `inject_spike()` — tăng default `spike_duration` 2 → 5 steps**
  File: `backend/data_simulator/anomalies/spike.py`. 5 steps @ 8Hz = 625ms. Update test kiểm tra spike spans ≥ 5 steps.

- [ ] **T3: `inject_cable_wear()` — tăng default `dropout_length_range` (1,3) → (3,8)**
  File: `backend/data_simulator/anomalies/cable_wear.py`. 3–8 steps = 375ms–1s per dropout. Update test.

- [ ] **T4: `CSVWriter` — thêm method `write_threshold_calibration()`**
  File: `backend/data_simulator/writer.py`. Output: `threshold_calibration_welding_robot.csv` + `threshold_calibration_labels.csv`. Logic tương tự `write_test()` với filename riêng và configurable `anomaly_ratio`. Test: `test_threshold_calibration_has_both_normal_and_anomaly`.

- [ ] **T5: Tạo `Sprint20-Generate-Data.ipynb`**
  File: `backend/notebooks/Sprint20-Generate-Data.ipynb`. Sinh đủ 4 datasets (train / val_train / threshold_calibration / test) với seed cố định, có cell giải thích vai trò từng dataset.

---

### Phase 2 — GLCMaeV9 Model

- [ ] **T6: Thêm `GlobalSeqNorm` vào `normalization.py`**
  File: `backend/app/anomaly_detection/layers/normalization.py` (thêm vào file hiện có, không tạo file mới).
  - Class `GlobalSeqNorm(nn.Module)`: wraps `nn.BatchNorm1d(num_sensors, affine=False, track_running_stats=True)`.
  - Nhận `[B, C, T]`, trả về `[B, C, T]` (normalized theo running stats per channel C).
  - Export trong `layers/__init__.py`.
  - Test: `test_global_seq_norm_shape`, `test_global_seq_norm_accumulates_in_train_mode`, `test_global_seq_norm_uses_running_stats_in_eval`.

- [ ] **T7: Tạo `glc_mae_v9.py`**
  File mới: `backend/app/anomaly_detection/models/glc_mae_v9.py`. Dựa trên `glc_mae_v8.py`:
  - Import `LogScaler`, `GlobalSeqNorm` thay `GlobalRevIN`, `PerPatchRevIN`.
  - `__init__`: thay `self.global_revin` + `self.local_revin` bằng `self.log_scaler = LogScaler()` và `self.seq_norm = GlobalSeqNorm(num_sensors)`.
  - `_tokenize_base`: `x_norm = self.seq_norm(self.log_scaler(raw))`. Cả global và local stream nhận `x_norm` (bỏ PerPatchRevIN ở local stream).
  - Reconstruction target: `flatten(patchify(x_norm))`.
  - `get_metadata()`: thêm `"normalizer": "log+global_seq_norm"`.
  - Test: `test_glc_mae_v9_forward_shape`, `test_v9_stats_in_state_dict`, `test_v9_eval_mode_deterministic`.

- [ ] **T8: Đăng ký `GLCMaeV9` vào `ReconstructionModels` enum + `create_model()` factory**
  File: `backend/app/anomaly_detection/models/__init__.py`. Thêm `GLC_MAE_V9`. Test: `test_create_model_v9`.

---

### Phase 3 — Training Notebook V9

- [ ] **T9: Tạo `Training-GLCReconV9.ipynb`**
  File: `backend/notebooks/Training-GLCReconV9.ipynb`. Dựa trên `Training-GLCReconV8.ipynb`:
  - Dùng `GLCMaeV9`.
  - Dùng `val_train_welding_robot.csv` (normal, full noise) cho validation loop.
  - **Bỏ tail window**: windowing chỉ dùng `range(0, T - seq_len + 1, stride)` — 5 windows cho T=512, seq_len=256, stride=64.
  - Comment rõ: stats (`running_mean`/`running_var`) tích lũy tự động trong training loop, được lưu qua `model.state_dict()`.

---

### Phase 4 — Inference Pipeline V9

- [ ] **T10: Tạo `recon_v9/local.py`**
  Thư mục mới: `backend/app/anomaly_detection/inference/recon_v9/`.
  Dựa trên `recon_v8/local.py`:
  - Quantile default = **0.99**, score keys `p99_score_*`.
  - **Bỏ tail window** trong windowing logic bên trong class hoặc notebook helper.
  - `model.eval()` sau `load_state_dict()` → `GlobalSeqNorm` tự dùng saved running stats.
  - k_skip=1 default.
  - Test: `test_reconf9_score_keys_p99`, `test_reconf9_no_tail_window`, `test_reconf9_thermal_drift_score_higher_than_normal`.

- [ ] **T11: Tạo `Inference-GLCReconV9.ipynb`**
  File: `backend/notebooks/Inference-GLCReconV9.ipynb`. Dựa trên `Inference-GLCReconV8.ipynb`:
  - `STRIDE = 64`, bỏ tail window trong `build_pre_result_records`.
  - `Reconf9LocalPreResult` (p99, k_skip=1).
  - Score keys: `p99_score_wire_feed_speed`, `p99_score_wire_temperature`, `p99_score_welding_current`.
  - **Threshold calibration flow**: dùng `threshold_calibration_welding_robot.csv` (normal + anomaly, labelled) → plot distribution per sensor × per anomaly type → chọn threshold → lưu vào `THRESHOLD_CONFIG` → `threshold_config.json`.
  - Evaluation: dùng `test_welding_robot.csv`.

---

### Phase 5 — Tests & Cleanup

- [ ] **T12: Tests `test_glc_mae_v9.py`**
  - `test_v9_thermal_drift_raises_score`: sau khi stats accumulate, thermal_drift file có score cao hơn normal (GlobalSeqNorm giữ drift signal).
  - `test_v9_spike_not_diluted`: spike file `p99_score_welding_current` >> normal (không bị PerPatchRevIN hấp thụ).
  - `test_v9_checkpoint_stats_roundtrip`: save/load `state_dict()` giữ nguyên `running_mean`/`running_var`.

---

## Notes / Blockers

- `LogScaler` và `GlobalStatsStandardizer` đã có trong `normalization.py`. `GlobalSeqNorm` là lớp mới, tên rõ ràng hơn cho dùng ở sequence input level, thêm vào cùng file.
- `GlobalSeqNorm` nhận `[B, C, T]` — BatchNorm1d xử lý đúng (normalize per channel C trên toàn B×T).
- Tail window bị xóa ở cả training notebook lẫn inference notebook/class. Đây là change breaking với V8 checkpoint — V9 cần retrain từ đầu.
- Sau khi retrain V9, cần regenerate toàn bộ 4 datasets (T1–T5) trước khi train.
