# MaskFormer Post-Sprint-15 Diagnosis & Fixes

**Date:** 2026-05-20  
**Branch:** dev/ai  
**Context:** After Sprint 15 (T1–T17) was implemented and training ran for 30 epochs, evaluation showed AUROC ≈ 0.51 (near-random). This document covers the full diagnostic session, every change made, and why some things were deliberately left unchanged.

---

## 1. Observed Symptoms (Gate C Training Run)

| Signal | Value | Status |
|--------|-------|--------|
| recon_loss | 0.3247 → 0.2787 (30 epochs) | Improving but slow (var ratio 0.156) |
| assoc_loss_min | 0.28 → 0.62 | Saturated at JS ceiling (~log 2) |
| assoc_loss_max | 0.00 → 0.62 | Identical to min (see §3) |
| sigma_mean | 5.00 → 5.54 | Slow monotonic growth |
| grad_ratio | ~0.4 | Available but not displayed prominently |
| AUROC | 0.5122 | Near-random — primary problem signal |

---

## 2. Root Cause Analysis

### 2.1 AUROC = 0.5 — Four independent causes

**Cause 1: Synthetic anomaly too weak (primary)**  
The eval cell injected `np.random.normal(0, 3.0)` onto raw sensor values.  
- wire_feed_speed: global std ≈ hundreds mm/min → std=3 is ~1% of signal scale  
- PerPatchRevIN normalizes each patch to zero-mean/unit-std → a tiny offset anomaly is completely erased  
- Result: "anomaly" windows are indistinguishable from normal after normalization  

**Cause 2: Per-sample min-max normalization in scoring (bug)**  
`compute_anomaly_score` called `compute_combined_heatmap`, which applied:
```python
mn = x.amin(dim=-1, keepdim=True)  # per-sample minimum
mx = x.amax(dim=-1, keepdim=True)  # per-sample maximum
```
This normalizes every file's score to [0, 1] independently. A slightly-anomalous file and a strongly-anomalous file get the same normalized score distribution → AUROC = 0.5 by construction.

**Cause 3: JS saturation (lambda_end=2.0)**  
In Phase A, loss = recon - λ·assoc. With λ=2.0, the gradient pushing A_series to maximize discrepancy was ~50-100× stronger than the recon gradient. Result: A_series collapsed to a near-uniform distribution, JS(A_prior, A_series) → log(2) ≈ 0.693 for all tokens. The assoc signal became noise (all tokens scored equally).

**Cause 4: Recon underfitting (variance ratio 0.156)**  
The model predicted patches with only 15.6% of the target variance. Partially due to cause 3 (λ too large distorted the shared QKV parameters). This reduced the recon signal's discriminative power.

### 2.2 assoc_min ≈ assoc_max — NOT a bug

A common confusion: Phase A uses `detach_mode="prior"` and Phase B uses `detach_mode="series"`. The `detach` only affects gradient routing, NOT the forward pass values. Both phases compute the exact same JS divergence values, differing only by one small optimizer step between them. At 4 decimal places, they are identical. This is correct behavior — it is NOT a sign that the minimax game isn't playing.

### 2.3 Phase B sign (+λ) — verified correct, NOT changed

The issue diagnosis doc claimed Phase B should use `-λ`. This was verified incorrect by reading the official `thuml/Anomaly-Transformer` source code. Phase B uses `loss_b = +λ·assoc` (maximize discrepancy by adjusting sigma/prior). The current code is correct.

### 2.4 PerPatchRevIN amplitude suppression — known limitation, documented

PerPatchRevIN normalizes each patch `[B, N, patch_size]` to zero-mean/unit-std. This erases:
- Absolute amplitude changes (sensor at 1000 vs 1010)
- Offset shifts (entire patch shifted up by constant)

What survives:
- Shape distortions within a patch (non-uniform perturbations)
- High-frequency oscillations (different frequency content after normalization)
- Flatline patterns (near-constant patch looks anomalous vs normally-varying)

This is an architectural constraint, not a bug. The correct eval strategy is to inject anomaly types that produce shape distortions.

---

## 3. Changes Made

### 3.1 Fix 1: compute_anomaly_score — absolute raw recon mean

**File:** `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py`

**Change:**
- Removed delegation to `compute_combined_heatmap` (which used per-sample normalization)
- Added `_compute_token_scores(batch, fast)` → `(recon_score [B,N], assoc_score [B,N])` raw, no normalization
- `compute_anomaly_score` now returns `recon_score.mean(dim=-1)` — absolute mean MSE per file
- Refactored `compute_combined_heatmap` to call `_compute_token_scores` internally
- Removed dead code: `mask_none` variable that was created but never used

**Why this change:**  
Cross-file AUROC requires scores that are comparable across files. Per-sample normalization by definition collapses every file to the same [0,1] range regardless of how anomalous it actually is. The absolute recon mean is a real signal: a file that the model reconstructs poorly is, on average, more anomalous.

**Why `compute_combined_heatmap` kept separate:**  
The per-sample-normalized combined heatmap is correct for *localization* (showing where in a file the anomaly is). The normalization makes hot spots visible even in slightly-anomalous files. Changing it would break visualization quality. The key distinction: classification uses absolute scores; localization uses relative highlighting.

### 3.2 Fix 2: Inference pipeline alignment

**File:** `backend/app/anomaly_detection/inference/maskformer/local.py`

**Change:**
- `score_batch()` now calls `compute_anomaly_score()` for the `p95_score` field (file classification score)
- Calls `compute_combined_heatmap()` separately for the heatmap fields (localization)
- Accumulates per-window absolute scores in `window_scores: list[float]`
- Final `p95_score` = mean of absolute scores across valid windows

**Why this change:**  
The training eval (using `compute_anomaly_score`) and the inference pipeline (`local.py` using per-sample-normalized P95) were measuring different quantities. A model with AUROC=0.7 in training eval might score very differently in inference. Aligning them ensures that training eval AUROC predicts inference classification performance.

**Why mean (not max) across windows:**  
A single-window outlier (caused by a noisy inference pass) would dominate a max aggregation. Mean is more stable for multi-window files, especially when k_skip trims the edge windows.

**Why heatmap kept per-sample-normalized:**  
Heatmaps are for human visualization. A doctor looking at a heatmap wants to see where in the file the anomaly concentrated, regardless of whether the anomaly is mild or severe. Per-sample normalization achieves this naturally.

### 3.3 Fix 3: Lower lambda_end to prevent JS saturation

**File:** `backend/notebooks/Training-MaskFormer.ipynb` (cell 14)

**Change:** `lambda_end=2.0` → `lambda_end=0.2`

**Why this change:**  
JS divergence is bounded in [0, log(2) ≈ 0.693]. With λ=2.0, Phase A gradient on QKV params from the assoc term was `-2.0 × ∇_QKV assoc`. This was ~50-100× stronger than the recon gradient, pushing A_series to uniform distribution (JS ceiling). With λ=0.2, the assoc gradient is ~5-10× recon gradient (still meaningful but not overwhelming).

**Why 0.2 specifically:**  
Based on the observed grad_ratio ≈ 0.4 during training. With λ=0.02 (warmup value), grad_ratio was 0.4 → A_series had natural structure. Ramping to λ=2.0 destroyed this. λ=0.2 gives 10× scale while keeping the adversarial signal below the JS ceiling. The exact value should be validated by watching `discrepancy_std` (should rise as tokens get more differentiated).

**Why not lambda_end=0.02 (keep flat):**  
Without any lambda increase, Phase B (sigma updates) has no pressure to widen the Gaussian prior, and the adversarial game never engages. A small ramp allows the game to play without saturation.

### 3.4 Fix 4: Per-sensor-σ anomaly injection

**File:** `backend/notebooks/Training-MaskFormer.ipynb` (cell 18)

**Change:** Replaced `np.random.normal(0, 3.0)` with three anomaly types scaled by per-sensor global std:

```python
sensor_stds = windows_arr[train_idx].std(axis=(0, 2))  # [C]

# Type 0: Spike — 8σ impulse over quarter-patch
# Type 1: High-frequency oscillation — 8 cycles per anomaly window
# Type 2: Flatline — near-constant segment
```

**Why per-sensor-σ scaling:**  
Each sensor has a different scale. Wire feed speed (std ≈ hundreds mm/min) needs a much larger absolute spike than temperature (std ≈ tens °C) for the anomaly to be visible. Fixed std=3.0 was invisible for most sensors.

**Why these 3 anomaly types:**  
PerPatchRevIN erases amplitude/offset shifts. The 3 types are chosen to create **shape distortions** that survive normalization:
- Spike: occupies only a fraction of a patch → local std increases dramatically → spike appears as 10-25σ deviation after normalization
- High-frequency oscillation: changes frequency content of the patch → normalized shape looks different from normal
- Flatline: patch becomes near-constant → after normalization it's a very low-variance pattern, anomalous vs normally-varying patches

**Why anom_len = PATCH_SIZE * 2:**  
A single patch-sized anomaly might straddle two patches and be partly normalized out. Two patch widths ensures at least one full patch is clearly anomalous.

### 3.5 Fix 5: 3-way AUROC decomposition cell

**File:** `backend/notebooks/Training-MaskFormer.ipynb` (new cell after cell 18)

**Change:** Added cell computing and plotting AUROC separately for:
- `recon_raw`: absolute mean recon error per file
- `assoc_raw`: mean negative discrepancy per file
- `combined`: globally-normalized product (not per-sample)

**Why this change:**  
Single combined AUROC hides which signal actually contributes. If `auroc_recon=0.75` and `auroc_assoc=0.51`, the assoc signal is noise and lambda should be reduced further. If both are `>0.6`, the combined signal is worth using. This decomposition guides the next tuning iteration without requiring an intuition leap.

**Why "globally-normalized" in combined:**  
Both recon and assoc have different absolute scales. To combine them fairly, normalize by the min/max computed across ALL files (not per-sample). This preserves relative differences across files while making the scales comparable.

### 3.6 Fix 6: discrepancy_std in train_step + epoch logging

**File:** `backend/app/anomaly_detection/training/maskformer_trainer.py`

**Change:**
- Added `discrepancy_std = out_a["discrepancies"].std(dim=-1).mean().detach()` to `train_step`
- Added `discrepancy_std` to history dict
- Batch bar: shows `dstd` (discrepancy std) and `gr` (grad_ratio) every step
- Epoch printout: shows `discrepancy_std`, `grad_ratio`, and `⚠ JS_SAT` warning when assoc > 0.6

**Why discrepancy_std matters:**  
`assoc_loss_min` is the mean JS divergence across all tokens and layers — it tells you the average level. But you want to know if some tokens score much higher than others (= the model is discriminating spatially). `std(discrepancy across tokens)` measures this: a high std means different tokens have very different JS scores, which is the actual anomaly signal. If std → 0, all tokens score the same → assoc is spatial noise regardless of its mean value.

**Why JS saturation warning at > 0.6:**  
log(2) ≈ 0.693. When mean assoc > 0.6, most of the dynamic range is used up and the signal is near-uniform. This threshold gives a warning with ~10% headroom before the actual ceiling.

### 3.7 Fix 7: JS upper-bound test

**File:** `backend/tests/test_anomaly_attention.py`

**Change:** Added `test_discrepancy_bounded_by_log2` — asserts every per-token JS divergence ≤ log(2) + 1e-5.

**Why:** Sprint 15 T6 specified "add test: JS in [0, log(2)]" but only the lower bound (≥ 0) was implemented. The upper bound is the mathematically guaranteed property of JS divergence and validates the implementation is correct.

### 3.8 Fix 8: FlowMatchingTrainer no-arg construction removed

**File:** `backend/app/anomaly_detection/training/flow_matching_trainer.py`

**Change:** `else` branch of `__init__` now raises `ValueError` instead of calling `GLCFlowV7Config()` with no args.

**Why:** `num_sensors` and `seq_len` were removed from `GLCFlowV7Config` defaults in a previous session (data-specific params must be explicitly provided — they come from real measurement setup). The no-arg construction was silently broken. A clear error message is better than a cryptic `TypeError` from the dataclass.

### 3.9 Fix 9: Pre-existing test failures fixed

**Files:** `backend/tests/test_train.py`, `backend/tests/test_local_inference.py`

**Change:** Added `_BASE_META = {"num_sensors": 6, "seq_len": 240}` helper and passed required params to all model construction calls that previously used no-arg defaults.

**Why:** Same root cause as Fix 8 — defaults removed in previous session broke these tests. These were pre-existing failures (not introduced in this session) but fixing them was needed to get a clean test suite.

---

## 4. Things Deliberately NOT Changed

### 4.1 Phase B sign: `loss_b = +λ·assoc`

**Why not changed:** Verified correct against official `thuml/Anomaly-Transformer` code. The "maximize" phase trains sigma to widen the Gaussian prior to better match the series distribution — which means increasing assoc loss. `+λ·assoc` with gradient ascent on sigma is the correct formulation. The issue diagnosis doc was wrong.

### 4.2 PerPatchRevIN itself

**Why not changed:** PerPatchRevIN is a key architectural component — it makes the model invariant to per-patch amplitude/offset, which is necessary to generalize across different operating conditions (different robots, programs, phases). Removing it would make the model sensitive to absolute sensor levels rather than shape anomalies. The correct fix is to use shape-distortion anomaly types in eval (Fix 4), not to remove normalization.

### 4.3 `compute_combined_heatmap` per-sample normalization

**Why not changed:** Kept for visualization/localization purposes. The fix was to remove it from the classification path (`compute_anomaly_score`), not from the heatmap visualization. Per-sample normalization correctly highlights hot spots for human inspection regardless of overall anomaly severity.

### 4.4 warmup_epochs=5

**Why not changed:** Five epochs of Phase-A-only training before adversarial competition gives the encoder/decoder a baseline reconstruction quality before sigma starts adapting. Too few warmup epochs → sigma adapts before recon is stable → adversarial gradient competes with a still-learning encoder. This value was kept from Sprint 15 design.

### 4.5 sigma_max=50.0

**Why not changed:** The sigma clamp prevents the Gaussian prior from becoming uniform (sigma → ∞ → uniform → no discrepancy signal). With JS saturation now addressed by lower lambda, sigma runaway is less of a concern, but the safety clamp remains appropriate.

### 4.6 CNN decoder

**Why not changed:** V6-style depthwise-separable Conv1d decoder provides temporal structure across patches. No evidence it was causing problems — recon_loss did decrease. The underfitting (var ratio 0.156) is primarily caused by JS saturation distorting QKV gradients, not decoder capacity.

---

## 5. Expected Behavior After These Fixes

Running a fresh 60-epoch training with the updated config should show:

| Signal | Expected |
|--------|----------|
| recon_loss | Clear downward trend; < 0.20 by epoch 30 (warmup helps) |
| assoc_loss | Stays < 0.5 (not saturated); some variation across epochs |
| discrepancy_std | Rising trend from epoch 5+ — tokens getting differentiated |
| grad_ratio | < 5 throughout ramp; stable |
| sigma_mean | May decrease or oscillate (prior-chase engaging) |
| AUROC (recon_raw) | > 0.65 with stronger anomaly injection |
| AUROC (assoc_raw) | > 0.55 if game is playing (may need more epochs) |

If `discrepancy_std` stays near 0: lambda is still too large, lower `lambda_end` to 0.05.  
If `auroc_recon < 0.6`: model still underfitting, check recon_loss trend; may need more warmup.

---

## 6. File Map

| File | Change type | Section |
|------|-------------|---------|
| `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py` | Bug fix (scoring) | §3.1 |
| `backend/app/anomaly_detection/inference/maskformer/local.py` | Bug fix (alignment) | §3.2 |
| `backend/app/anomaly_detection/training/maskformer_trainer.py` | Feature (logging) | §3.6 |
| `backend/app/anomaly_detection/training/flow_matching_trainer.py` | Bug fix (no-arg) | §3.8 |
| `backend/tests/test_anomaly_attention.py` | Test (upper bound) | §3.7 |
| `backend/tests/test_maskformer_trainer.py` | Test fix (discrepancy_std key) | §3.6 |
| `backend/tests/test_train.py` | Test fix (required params) | §3.9 |
| `backend/tests/test_local_inference.py` | Test fix (required params) | §3.9 |
| `backend/notebooks/Training-MaskFormer.ipynb` cell 14 | Config fix (lambda) | §3.3 |
| `backend/notebooks/Training-MaskFormer.ipynb` cell 18 | Fix (anomaly injection) | §3.4 |
| `backend/notebooks/Training-MaskFormer.ipynb` new cell | Feature (AUROC decomp) | §3.5 |
