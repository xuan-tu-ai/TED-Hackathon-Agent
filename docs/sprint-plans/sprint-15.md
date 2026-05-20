# Sprint 15 Plan

## Sprint Goal
Fix MaskFormer training: flat reconstruction, broken minimax game, and λ sensitivity — by normalizing discrepancy scale (JS divergence), adding CNN decoder, phase-A warmup, gradient-norm instrumentation, and λ ramp with AUROC validation.

## Motivation
Analysis of the 30-epoch training run (`artifacts/maskformer_checkpoint/training_extra.json`) revealed:
- **recon_loss** flat at 0.33 → 0.32 — model predicts near-zero (lazy reconstruction)
- **assoc_min ≈ assoc_max** (~28 both) — adversarial game not engaging because λ=0.02 dwarfs both phases
- **assoc scale ~28** dominated by eps-floor artifacts in symmetric KL (`log(1e-8)` ≈ −18 per element), making λ impossible to reason about
- **Phase B sign stays `+λ`** (verified: prior-chase game per official Anomaly-Transformer)
- Training data is normal-only (Phase 8 = data hygiene is correct as-is)

Full diagnosis: `artifacts/issues/001-maskformer-training-diagnosis.md`

## Design Strategy
- **Architecture first, tune later.** JS divergence (Phase 2) eliminates the eps-floor pathology → discrepancy becomes O(0.1–0.5). CNN decoder (Phase 3) gives cross-token temporal structure. Only then do we warm up (Phase 4) and ramp λ (Phase 6).
- **Measure, don't guess.** Gradient-norm logging (Phase 5) lets the data set λ, not analogy.
- **Verify with AUROC.** Per-epoch AUROC/AUPRC (Phase 1) is the only health signal that matters.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Observability — No model changes, immediate diagnostic value

- [x] **T1: Add recon_loss subplot to training curves**
  - In `Training-MaskFormer.ipynb`, Section 6: add a standalone `recon_loss` vs epoch plot with trend line.
  - Add annotation: "Must trend down — this is the actual health signal."
  - File: `backend/notebooks/Training-MaskFormer.ipynb`

- [x] **T2: Add per-epoch AUROC/AUPRC via MaskFormerEvaluator**
  - Add a cell after training loop that loads held-out normal+anomaly data (even if mock anomalies for now).
  - Call `MaskFormerEvaluator.evaluate()` after each epoch during training, or after training with per-epoch snapshots.
  - Plot AUROC + AUPRC vs epoch.
  - File: `backend/notebooks/Training-MaskFormer.ipynb`

- [x] **T3: Stop treating total loss as health signal**
  - Add markdown cell explaining: "Total loss = 0.32 − λ·assoc is a composite; recon_loss and AUROC are the real metrics."
  - Remove total loss from the primary training curve panel.
  - File: `backend/notebooks/Training-MaskFormer.ipynb`

---

### Phase 2: Fix Discrepancy Scale — JS Divergence replaces Symmetric KL

- [x] **T4: Replace symmetric KL with Jensen-Shannon in `_symmetric_kl`**
  - In `AnomalyAttentionBlock._symmetric_kl()`, replace the computation:
    ```python
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    discrepancy = 0.5 * (kl_pm + kl_qm).mean(dim=1)
    ```
  - JS bounded in [0, log(2)] ≈ [0, 0.69] — no eps-floor pathology, no magical λ values.
  - Rename method to `_jensen_shannon` (keep old name as alias during transition).
  - File: `backend/app/anomaly_detection/layers/anomaly_attention.py`

- [x] **T5: Verify AssociationDiscrepancyLoss still works with JS output**
  - Confirm `AssociationDiscrepancyLoss.forward()` just calls `.mean()` — no change needed.
  - Update docstring to reflect JS divergence instead of symmetric KL.
  - File: `backend/app/anomaly_detection/criterions/association_discrepancy.py`

- [x] **T6: Update tests for JS divergence**
  - Update `test_maskformer_trainer.py` — expected assoc values will be O(0.1–0.5) instead of O(27).
  - Add test: JS divergence is in [0, log(2)] for any pair of normalized distributions.
  - Update `test_anomaly_attention.py` if it exists.
  - Files: `backend/tests/test_maskformer_trainer.py`, `backend/tests/test_anomaly_attention.py`

---

### Phase 3: V6-Style CNN Decoder — Temporal structure in reconstruction

- [x] **T7: Implement `MaskFormerCNNDecoder`**
  - Per-token decode: `Linear(D,D) → GELU → Linear(D, patch_size)` (preserve existing).
  - Temporal refinement: reshape `[B, N, PS]` → `[B, C, P*PS]` → `Conv1d(C→4C, k=3, groups=C) → SiLU → Conv1d(4C→C, k=1)` → reshape back.
  - Keep existing `PerTokenDecoder` as option; add `CNNDecoder` class.
  - File: `backend/app/anomaly_detection/layers/reconstruction_decoder.py`

- [x] **T8: Wire CNN decoder into `GLCAnomalyMaskFormer`**
  - In `GLCAnomalyMaskFormer.__init__()`, accept `use_cnn_decoder` flag.
  - Pass `num_sensors`, `num_patches` to decoder constructor.
  - In `_forward_impl()`, decoder call stays identical (same input/output signature).
  - File: `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py`

- [x] **T9: Test CNN decoder output shapes and forward pass**
  - Test: output shape `[B, N, patch_size]` unchanged.
  - Test: forward pass with random tokens, no NaN/Inf.
  - Test: CNN refinement changes output from per-token-only baseline.
  - File: `backend/tests/test_maskformer_trainer.py` (or new test file)

---

### Phase 4: Phase-A-Only Warmup — Build reconstruction before adversarial

- [x] **T10: Add `warmup_epochs` and `warmup_lambda` params to `MaskFormerTrainerRunner`**
  - `warmup_epochs: int = 0` — number of epochs with λ=0 (Phase A only, no Phase B).
  - `warmup_lambda: float = 0.0` — λ during warmup (default 0).
  - Store on self; pass to `MaskFormerTrainer`.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

- [x] **T11: Implement warmup mode in `MaskFormerTrainer.train_step()`**
  - If `current_epoch < warmup_epochs`: skip Phase B entirely; only run Phase A with λ=0.
  - Return `assoc_loss_max=0.0` during warmup for logging consistency.
  - After warmup: resume full minimax with the normal λ.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

- [x] **T12: Test warmup behavior**
  - Test: during warmup epochs, sigma does not change (no Phase B → no sigma optimizer step).
  - Test: recon_loss decreases during warmup.
  - Test: after warmup, Phase B resumes and sigma changes.
  - File: `backend/tests/test_maskformer_trainer.py`

---

### Phase 5: Gradient-Norm Logging — Measure, don't guess

- [x] **T13: Add gradient-norm ratio logging to `MaskFormerTrainer.train_step()`**
  - After Phase A backward (before `optimizer_main.step()`), compute:
    - `||∇_QKV recon||` — gradient norm of recon loss on Q/K/V params
    - `||∇_QKV assoc||` — gradient norm of −λ·assoc term on Q/K/V params
  - Store ratio = `||∇_QKV assoc|| / (||∇_QKV recon|| + eps)` in step output.
  - Exclude decoder and sigma params from norm computation.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

- [x] **T14: Log grad-norm ratio in training history and notebook**
  - Add `grad_ratio` key to history dict in `MaskFormerTrainerRunner.train()`.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

---

### Phase 6: Lambda Ramp — Let data set λ

- [x] **T15: Add lambda ramp schedule to `MaskFormerTrainerRunner`**
  - `lambda_end: float` and `lambda_ramp_epochs: int` added to Runner.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

- [x] **T16: Implement ramp logic in `train()`**
  - Before each epoch, compute `current_lambda = linear_interpolate(start, end, ramp_progress)`.
  - Pass to `self.trainer.lambda_assoc = current_lambda`.
  - Log current λ in epoch printout.
  - File: `backend/app/anomaly_detection/training/maskformer_trainer.py`

- [x] **T17: Test ramp behavior**
  - Test: λ reaches target after ramp_epochs.
  - Test: lambda_assoc on trainer is updated each epoch.
  - Test: ramp starts after warmup.
  - File: `backend/tests/test_maskformer_trainer.py`

---

### Phase 7: Tune Mask Ratio — Validated by AUROC, not assumed

- [ ] **T18: Add mask_ratio grid search to training notebook**
  - Define mask_ratios = [0.15, 0.30, 0.50].
  - Train short runs (5–10 epochs) with each ratio, record AUROC.
  - Plot AUROC vs mask_ratio.
  - File: `backend/notebooks/Training-MaskFormer.ipynb`

- [ ] **T19: Select optimal mask_ratio**
  - Pick ratio with highest AUROC on held-out data.
  - Rationale: for high-frequency welding signals, lower masking may reduce MSE blur.
  - Document selection in notebook markdown cell.

- [ ] **T20: Apply selected mask_ratio to final training run**
  - Set `mask_ratio` in `GLCAnomalyMaskFormerConfig`.
  - Run full training with all previous fixes (JS + CNN + warmup + ramp).
  - Save final checkpoint and training_extra.json.
  - File: `backend/notebooks/Training-MaskFormer.ipynb`

---

## Validation Gates — When to Retrain

We do **not** retrain after every phase. There are 3 natural stop-points where the
architecture changes are substantial enough to warrant a training run + validation
before continuing. Phase 1 (observability) requires no retraining — it uses the
existing training log. Phase 5 (gradient-norm logging) is pure instrumentation and
can be tested at any gate without a dedicated retrain.

```
Phase 1 ────► Phase 2 ────► [Gate A] ────► Phase 3 ────► Phase 4 ────► [Gate B]
(no retrain)              (retrain)                                     (retrain)
                                │                                           │
                          Phase 5 (instrumentation can be added here or at B)
                                │                                           │
                                ▼                                           ▼
                          Phase 6 ────► [Gate C] ────► Phase 7
                                       (retrain)     (grid search retrains)
```

### Gate A: After Phase 2 (JS Divergence) — *Sanity check the assoc scale*

**What changed:** Symmetric KL → Jensen-Shannon in `_symmetric_kl`.
**Retrain:** Short run, 10–15 epochs with existing config (λ=0.02, no CNN, no warmup).
**What to verify:**

| Signal | Expected | If failing |
|--------|----------|------------|
| `assoc_loss` scale | O(0.1–0.5) instead of O(28) | JS implementation has a bug — debug `_jensen_shannon` |
| `assoc_loss` stability | No NaN/Inf, no spikes | eps constant too small, check numerical edge cases |
| `recon_loss` trend | Still flat (~0.32) — expected, JS doesn't fix recon alone | If recon gets *worse*, JS is distorting representation |
| `sigma_mean` | Still monotonic growth (expected, λ still 0.02) | Not a concern at this gate |

**Decision:** If assoc scale is O(0.1–0.5) and numerically stable → **proceed to Phase 3**.
If assoc scale is still O(10+) → JS is not working, fix before moving on.

> 💡 **Why retrain here?** JS is a model architecture change — the forward pass computes
> different values. We need to confirm the new divergence behaves correctly before
> stacking CNN decoder + warmup on top. A 10-epoch run is cheap (< 2 minutes).


### Gate B: After Phase 4 (JS + CNN Decoder + Warmup) — *Architecture health check*

**What changed:** JS divergence + V6-style CNN decoder + 5 epochs Phase-A-only warmup.
**Retrain:** Full run, 5 warmup epochs + 25 minimax epochs (30 total), λ=0.02 (not ramped yet).
**What to verify:**

| Signal | Expected | If failing |
|--------|----------|------------|
| `recon_loss` during warmup | Drops from ~0.33 toward < 0.15 | Encoder/decoder can't learn even without adversarial competition — investigate encoder capacity or decoder architecture |
| `recon_loss` after warmup | Continues dropping or stabilizes (does not regress) | Minimax is destroying reconstruction — enable grad-norm logging (Phase 5) to measure the competition, consider smaller λ or slower ramp |
| Reconstructed waveform | Shows shape, not flat line near 0 | CNN decoder may not be wired correctly, or token reshape is wrong |
| `assoc_min` vs `assoc_max` | Small gap emerging (0.01–0.05 difference) | λ=0.02 may still be too small for JS-scaled discrepancy; proceed to Phase 6 ramp |
| `sigma_mean` | May decrease or oscillate (JS + prior-chase should engage) | Still monotonic growth means game still not playing — Phase 6 ramp will help |
| AUROC on held-out | > 0.5 (better than random) | Model is not learning anything useful — check data pipeline |

**Decision:** If `recon_loss` drops during warmup AND reconstructed waveforms show shape → **architecture is healthy → proceed to Phase 6**.
If recon flat even during warmup → CNN decoder or encoder is broken → fix before proceeding.
If recon drops during warmup but regresses after → minimax competition is too strong → do Phase 5 (grad-norm logging) before Phase 6 to measure the ratio.

> 💡 **Why retrain here?** This is the first point where all architectural fixes are in
> place. If recon+assoc+AUROC all trend correctly here, the model is fundamentally
> sound and only needs tuning (Phases 6–7). If something is wrong, we catch it before
> spending time on λ ramp logic and grid search.


### Gate C: After Phase 6 (Full Pipeline with Tuned λ) — *Final validation*

**What changed:** Grad-norm logging + λ ramp schedule (data-driven λ selection).
**Retrain:** Full run with warmup + λ ramp (0 → tuned_value), monitored by grad ratio.
**What to verify:**

| Signal | Expected | If failing |
|--------|----------|------------|
| `grad_ratio` (`||∇_QKV assoc|| / ||∇_QKV recon||`) | ≤ 10× throughout ramp | Ratio is stable and recon continues improving | Back off λ to the last stable value |
| `recon_loss` after ramp | Stable or still decreasing | λ is too aggressive — back off |
| `assoc_min` vs `assoc_max` | Clear gap (≥ 0.1 difference) | Adversarial game is playing, discrepancy is a meaningful signal | |
| AUROC | Trending up, ≥ 0.6 | Model is not separating normal/anomaly — check data labels or anomaly injection |

**Decision:** If all metrics are healthy → **proceed to Phase 7** (mask_ratio grid search).
If grad_ratio spikes → back off λ and re-evaluate.

> 💡 **Why retrain here?** This is the first training run where λ is data-driven rather
> than hardcoded. We need to confirm the ramp works and λ converges to a stable value
> that balances the minimax game. Phase 7 (mask_ratio grid search) then runs multiple
> short retrains from this stable configuration.


## Notes / Blockers
- Phase 1 (observability) requires **no retraining** — can be done immediately on the existing `training_extra.json`.
- Phase 2 (JS divergence) changes the scale of `assoc_loss` — all existing training checkpoints using symmetric KL will report different assoc values. This is expected and intentional.
- Phase B sign stays `+λ` — verified against official Anomaly-Transformer code. The prior-chase game is correct.
- Training data is normal-only (confirmed). Phase 8 (data hygiene) from the issue doc is already satisfied.
- If JS divergence alone fixes the assoc scale and reconstruction starts learning without the CNN decoder, Phase 3 can be deferred — but Gate B will catch this.
- Phase 5 (gradient-norm logging) can be implemented anytime after Phase 2 and tested at Gate B or Gate C. It does not require its own retrain.
