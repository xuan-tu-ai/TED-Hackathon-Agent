# Sprint 12 Plan

## Sprint Goal
Fix all valid issues identified in `artifacts/issues/verification_260518.md`: vectorize GPU bottlenecks, improve numerical stability, refactor RevIN to pure-Tensor API, replace 1D PE with 2D axial PE, and fix the Minimax Phase B sign bug that causes training divergence.

**Out of scope (reviewer claims rejected):**
- `_prior_association` stays as `kernel / sum` (paper-correct; do NOT change to softmax)
- `loss_a` Phase A sign stays as `-λ * L_assoc` (paper-correct; do NOT change to `+`)

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

- [ ] **T1** — `sensor_block_masking.py`: Replace `for b in range(B)` with vectorized `scatter_(dim=1, index=masked_time_ids, value=True)` + `expand` over sensors.

- [ ] **T2** — `anomaly_attention.py` `_symmetric_kl`: Change `torch.log((p + eps) / (q + eps))` → `torch.log(p + eps) - torch.log(q + eps)` for numerical stability. Matches official `my_kl_loss`.

- [ ] **T3** — `revin.py` `GlobalRevIN`: Remove `RevINStat` import + list accumulation + for-loop over channels. Replace with vectorized `x.mean(dim=-1, keepdim=True)` / `x.std(dim=-1, keepdim=True)` returning `tuple[Tensor, tuple[Tensor, Tensor]]` (stats shape `[B, C, 1]`).

- [ ] **T4** — `revin.py` `PerPatchRevIN`: Remove nested `list[list[RevINStat]]` + double for-loop. Replace with `x.amin(dim=-1, keepdim=True)` / `x.amax(dim=-1, keepdim=True)` returning `tuple[Tensor, tuple[Tensor, Tensor]]` (stats shape `[B, C, P, 1]`). Also fix `.abs()` inconsistency in `inverse`.

- [ ] **T5** — Update all callers of `GlobalRevIN.forward` / `PerPatchRevIN.forward` / `.inverse` in `glc_anomaly_maskformer.py` (and any other model files) to use the new `tuple[Tensor, Tensor]` stats API instead of `list[RevINStat]`.

- [ ] **T6** — `patchify.py`: Delete `flatten_to_tokens` helper function (creates untrained `nn.Linear` on every call — weights are never learned). Verify no callers exist outside tests before deleting.

- [ ] **T7** — `encoders.py` `PositionalEmbedding`: Replace single `nn.Embedding(num_tokens, d_model)` + 1D index with 2D axial PE: `sensor_embedding = nn.Embedding(num_sensors, d_model)` + `patch_embedding = nn.Embedding(num_patches, d_model)`. Update constructor signature `(num_tokens=84)` → `(num_sensors=6, num_patches=14, d_model=128)`.

- [ ] **T8** — Update `PositionalEmbedding` instantiation site(s) in `glc_anomaly_maskformer.py` (and any other callers) to pass `num_sensors` / `num_patches` instead of `num_tokens`.

- [ ] **T9** — `maskformer_trainer.py` line 144: Fix Phase B sign — change `loss_b = -self.lambda_assoc * assoc_dict_max["loss"]` → `loss_b = self.lambda_assoc * assoc_dict_max["loss"]`. This is the critical Minimax convergence fix.

- [ ] **T10** — Update / add tests in `tests/` for all changed modules: `SensorBlockMasking` (vectorize equivalence), `AnomalyAttentionBlock` (KL value unchanged), `GlobalRevIN` / `PerPatchRevIN` (forward/inverse round-trip), `PositionalEmbedding` (shape + same-time-different-sensor share patch_emb), `MaskFormerTrainer` (Phase B loss sign positive).

## Notes / Blockers
- `RevINStat` is defined in `app/anomaly_detection/schemas.py`. After T3/T4, check if any other code (inference, applier, etc.) still imports it — if not, the dataclass can be removed from schemas.
- `PositionalEmbedding` constructor change (T7) is a breaking API change. Grep all usages before implementing T8.
- Phase B fix (T9) changes training behavior fundamentally — run a short training sanity check after applying (loss_b should now be positive and sigma should converge).
