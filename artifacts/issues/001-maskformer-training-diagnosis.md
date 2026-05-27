# Issue: MaskFormer Trainer — Three Critical Training Problems

**Date:** 2026-05-20
**Source:** Training run on welding robot data (30 epochs, lambda_assoc=0.02, d_model=128, 4 transformer layers)
**Training log:** `artifacts/maskformer_checkpoint/training_extra.json`

---

## Problem 1: Recon Loss Near-Zero-Decrease — Model Reconstructs Lazy (≈0)

### Symptoms
- `recon_loss` goes from 0.334 → 0.319 (only 4.5% decrease over 30 epochs)
- Reconstructed signal (red dashed) is essentially a flat line near 0
- Target signal (blue solid) has significant variation
- Masked-token predict variance / target variance ratio ≈ 0 → collapsed

### Root Cause Analysis

**1. No effective adversarial signal.**
With Phase B sign broken (see Problem 2), the transformer layers never receive pressure to build meaningful representations that distinguish normal patterns. The model converges to the simplest strategy: predict near-zero for masked patches.

**2. Decoder is too shallow.**
`ReconstructionDecoder` is only `Linear(D,D) → GELU → Linear(D, patch_size)` (2 linear layers). For `d_model=32` in tiny configs, this is extremely limited. Even at `d_model=128`, inverting the patch encoding with only 2 linear layers is challenging without rich token representations.

**3. Mask ratio is low (30%).**
70% of tokens are visible at all times — the model can potentially rely on simple interpolation from visible neighbors rather than learning semantic understanding of the data.

### Suggested Fixes
- Fix Phase B sign first (Problem 2) — this should indirectly improve reconstruction
- Increase decoder depth: `Linear(D, 2D) → GELU → Linear(2D, D) → GELU → Linear(D, patch_size)`
- Increase `mask_ratio` from 0.30 to 0.50 to force the model to rely more on context
- Consider pre-training encoder+decoder for reconstruction alone (Phase A only, no minimax) for a few warmup epochs

---

## Problem 2: Assoc Min ≈ Assoc Max — Minimax Game NOT Working

### Symptoms
- `assoc_loss_min` and `assoc_loss_max` are nearly identical across all 30 epochs
- Both start at ~27.5, fluctuate to ~29.3, end at ~28.3
- sigma_mean grows monotonically (5.0 → 5.67) instead of oscillating

### Root Cause: PHASE B GRADIENT SIGN IS WRONG

In `MaskFormerTrainer.train_step()`, Phase B code:

```python
# Phase B (current, WRONG):
loss_b = self.lambda_assoc * assoc_dict_max["loss"]   # POSITIVE
loss_b.backward()
self.optimizer_sigma.step()                            # Adam MINIMIZES
```

**Adam does:** `param = param - lr * grad`

With a **positive** `loss_b`, sigma moves to **minimize discrepancy** — the opposite of what we want.

**Why this is wrong:**
- Phase B intent: MAXIMIZE association discrepancy (push A_prior away from A_series so normal vs anomalous patterns separate)
- Current behavior: Adam performs gradient DESCENT on `+lambda * discrepancy` → MINIMIZES discrepancy
- Both phases now push sigma in the SAME direction (reducing discrepancy) → `assoc_min ≈ assoc_max`

**Evidence from the test comment** at `test_maskformer_trainer.py` line 133:
```
"Phase B loss must be positive after sign fix (was -lambda, now +lambda)"
```
The "fix" went from `-lambda` to `+lambda` — but that's the WRONG direction. The original `-lambda` was correct for maximization with standard Adam.

### Correct Fix

```python
# Phase B (fixed):
loss_b = -self.lambda_assoc * assoc_dict_max["loss"]   # NEGATIVE
loss_b.backward()
self.optimizer_sigma.step()                             # Adam minimizes (-loss) = maximizes (+loss)
```

**Math verification:**
- `loss_b = -λ * discrepancy`
- `loss_b.backward()` → `∇_σ(-λ * D) = -λ * ∇_σ(D)`
- `optimizer_sigma.step()` → `σ = σ - lr * (-λ * ∇_σ(D))` = `σ + lr * λ * ∇_σ(D)`
- This INCREASES discrepancy ✓

### File to Fix
- `backend/app/anomaly_detection/training/maskformer_trainer.py` line ~148
- `backend/tests/test_maskformer_trainer.py` — `test_phase_b_loss_is_positive` tests the wrong value (loss magnitude, not gradient direction)

### Expected Behavior After Fix
- sigma should DECREASE (making A_prior more local/narrow, further from broad A_series)
- `assoc_max` should INCREASE (larger KL divergence)
- `assoc_min` should DECREASE (Phase A successfully pushes A_series toward prior)
- The gap `assoc_max - assoc_min` should grow over epochs

---

## Problem 3: When Assoc Min Decreases, Loss Increases — Is This Correct?

### The Formula
```
Phase A loss = recon + orth - lambda_assoc * assoc_min
```

### Analysis: This is EXPECTED behavior

When `assoc_min` decreases, `-lambda * assoc_min` becomes **less negative** (moves toward 0), so loss increases. This is not a bug — it's the natural trade-off of the minimax formulation.

**Healthy training dynamic:**
- Recon loss drops faster than the penalty from declining assoc
- Net Phase A loss still decreases

**Current training dynamic (broken):**
- Recon barely moves (0.334 → 0.319)
- Assoc doesn't really decline (27.5 → 28.3)
- The formula is dominated by the `-lambda*assoc` term

The real issue isn't the formula — it's that **neither recon nor assoc are changing meaningfully**, which traces back to Problem 2 (broken adversarial signal).

### Conclusion on Problem 3
No change needed to the loss formula. Fix Problem 2 first; Problem 3 resolves naturally when the minimax game works.

---

## Summary: Required Fixes (Priority Order)

| Priority | Fix | File |
|----------|-----|------|
| 🔴 P0 | **Flip Phase B sign:** `loss_b = -self.lambda_assoc * assoc_dict_max["loss"]` | `maskformer_trainer.py` train_step() |
| 🔴 P0 | **Fix misleading test:** `test_phase_b_loss_is_positive` tests loss value, not gradient direction | `test_maskformer_trainer.py` |
| 🟡 P1 | **Increase decoder capacity:** Add one more hidden layer to `ReconstructionDecoder` | `reconstruction_decoder.py` |
| 🟡 P1 | **Increase mask_ratio:** 0.30 → 0.50 to force contextual learning | `GLCAnomalyMaskFormerConfig` |
| 🟢 P2 | **Pre-training warmup:** Run 5 epochs of Phase-A-only training before starting minimax | `MaskFormerTrainerRunner.train()` |

The **Phase B sign flip** is the critical fix — once the adversarial game works correctly, the model will learn richer representations, and reconstruction should improve as a side effect.

---

## Addendum: Can a CNN Decoder (V6-family) Help the Reconstruction Problem?

### Architecture Comparison

| Aspect | V6 `decoder_shape` | MaskFormer `ReconstructionDecoder` |
|--------|-------------------|-----------------------------------|
| Input | `[B, C, D]` (sensor × hidden) | `[B, N, D]` (N=C×P tokens) |
| Layer 1 | `Linear(D, W)` → [B, C, W] | `Linear(D, D) → GELU` |
| Layer 2 | `Conv1d(C, 4C, k=3, groups=C)` — **depthwise temporal conv** | `Linear(D, patch_size)` |
| Layer 3 | `Conv1d(4C, C, k=1)` — **cross-sensor mixing** | *(none)* |
| Cross-token interaction | ✅ Conv kernel slides across time | ❌ Each token decoded independently |
| Symmetry with encoder | ✅ Mirror of ShapeEncoder | ❌ Not symmetric to PatchEncoder |

The V6 encoder/decoder are **symmetric mirrors**:
- **Encoder:** `depthwise_conv(C→4C, k=3, groups=C) → SiLU → pointwise_conv(4C→C, k=1) → LayerNorm(W) → Linear(W, D)`
- **Decoder:** `Linear(D, W) → SiLU → depthwise_conv(C→4C, k=3, groups=C) → SiLU → pointwise_conv(4C→C, k=1)`

### Would a CNN Decoder Help?

**Short answer: Yes, but it is a secondary fix — not the root cause.**

#### ✅ Why It Would Help

1. **Temporal smoothness bias.** A `Conv1d(k=3)` inherently models that neighboring time steps are related. The current flat-line reconstruction shows the model isn't even doing simple linear interpolation between visible tokens. A depthwise conv would trivially learn that.

2. **Proven on this data.** The V6 decoder has already been validated on the exact same welding robot data. The depthwise-separable conv pattern (`groups=C` → `k=1`) is a strong inductive bias for multi-sensor time series.

3. **Cross-token interaction in the decoder.** The MaskFormer decoder currently decodes each token completely independently. A CNN decoder with a kernel > 1 would let neighboring patch tokens influence each other's reconstruction.

#### ⚠️ Caveats

1. **Root cause is the broken minimax game (Problem 2).** The transformer layers produce poor representations because Phase B pushes sigma the wrong way. Fix the sign first — the shallow decoder may become adequate once the transformer produces meaningful token embeddings.

2. **Architecture mismatch.** The MaskFormer uses patch tokens `[B, N, D]` where N = C × P = sensors × patches. The V6 decoder operates on `[B, C, D]` (single window, no patches). You'd need to reshape tokens into `[B, C, P, D]` and apply Conv1d along the patch dimension, which would lose cross-sensor interaction in the decoder.

3. **Scale difference.** V6 decodes a single window of W=32 steps. MaskFormer decodes P=14 patches × 32 steps each = 448 timesteps per sensor — much larger scope.

### Recommended Approach

```
Priority:  Fix Phase B sign  →  retrain  →  if recon still poor, add CNN decoder
   P0                           P0             P1
```

### Proposed V6-Inspired CNN Decoder

```python
class MaskFormerCNNDecoder(nn.Module):
    """V6-style CNN decoder with per-token decode + temporal refinement."""

    def __init__(self, num_sensors: int, num_patches: int, d_model: int, patch_size: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_patches = num_patches
        self.patch_size = patch_size

        # Per-token decode (keep existing)
        self.token_decode = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),  # [B, N, patch_size]
        )

        # Temporal refinement across patches (V6-style)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens to patch predictions with temporal refinement.

        Args:
            tokens: [B, N, D] where N = C × P

        Returns:
            patches_pred: [B, N, patch_size]
        """
        B, N, D = tokens.shape
        C, P, PS = self.num_sensors, self.num_patches, self.patch_size

        patches = self.token_decode(tokens)                  # [B, N, PS]
        patches = patches.reshape(B, C, P, PS)               # [B, C, P, PS]
        # Reshape patches to continuous signal per sensor
        signal = patches.reshape(B, C, P * PS)               # [B, C, P*PS]
        refined = self.temporal_refine(signal)                # [B, C, P*PS]
        patches_out = refined.reshape(B, C, P, PS)           # [B, C, P, PS]
        return patches_out.reshape(B, N, PS)                  # [B, N, PS]
```

### Updated Summary Table

| Priority | Fix | File |
|----------|-----|------|
| 🔴 P0 | **Flip Phase B sign:** `loss_b = -self.lambda_assoc * assoc_dict_max["loss"]` | `maskformer_trainer.py` train_step() |
| 🔴 P0 | **Fix misleading test:** `test_phase_b_loss_is_positive` tests loss value, not gradient direction | `test_maskformer_trainer.py` |
| 🟡 P1 | **V6-style CNN decoder:** Add depthwise-separable Conv1d temporal refinement after per-token decode | `reconstruction_decoder.py` (replace or add) |
| 🟡 P1 | **Increase mask_ratio:** 0.30 → 0.50 to force contextual learning | `GLCAnomalyMaskFormerConfig` |
| 🟢 P2 | **Pre-training warmup:** Run 5 epochs of Phase-A-only training before starting minimax | `MaskFormerTrainerRunner.train()` |
