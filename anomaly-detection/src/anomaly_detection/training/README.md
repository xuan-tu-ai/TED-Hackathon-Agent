# Training — Anomaly Detection

## Trainer Hierarchy

| Trainer | Model Family | Stages | When to use |
|---|---|---|---|
| `SingleStageTrainer` | V1–V6 | 1 (generic) | Legacy models with a single criterion + DataLoader. Model-agnostic. |
| `FlowMatchingTrainer` | GLC-Flow V7 | 2 (encoder → CNF) | Flow-Matching paradigm with RevIN encoder and CNF backbone. |
| `MaskFormerTrainerRunner` | GLCAnomalyMaskFormer | 1 (minimax) | Transformer-based maskformer with association-discrepancy minimax. |

### `SingleStageTrainer`

- Works with any `BaseAnomalyDetectionModel` + `BaseAnomalyDetectionLoss`.
- MLflow logging and manual checkpointing are optional.
- Single training loop with configurable `train_steps`/`val_steps` per epoch.
- File: `single_stage_trainer.py`

### `FlowMatchingTrainer` (V7)

**Stage 1 — Encoder training** (`train_stage1`)
- CNF backbone is **frozen**.
- Loss: global reconstruction + patch (local) reconstruction + orthogonality regularisation.
- Updates: all model parameters **except** CNF.

**Stage 2 — CNF / Flow Matching** (`train_stage2`)
- Encoder (global & local projections) is **frozen**.
- Loss: MSE between predicted vector field `v_pred` and ground-truth `z1 - z0`.
- Updates: CNF parameters only.

**Evaluation**
- Anomaly score = negative log-likelihood of the CNF on `z_pooled`.
- Threshold fitted via `MADThreshold` on normal scores.
- Metrics: AUROC, AUPRC.

Sub-classes: `Stage1Trainer`, `Stage2Trainer`, `Evaluator`, `EvalResult`.
File: `flow_matching_trainer.py`

### `MaskFormerTrainerRunner` (MaskFormer)

Single-stage minimax training (no Stage 1 / Stage 2 split):

**Phase A (minimise)**
- `detach_mode="prior"` — sigma is frozen; Q/K/V get gradients.
- Loss: `L_recon - lambda_assoc * L_assoc + lambda_orth * L_orth`.
- Updates: all parameters **except** sigma.

**Phase B (maximise)**
- `detach_mode="series"` — Q/K/V are frozen; sigma gets gradients.
- Loss: `+lambda_assoc * L_assoc_max` (positive gradient → maximise discrepancy).
- Updates: sigma parameters only.

Two separate Adam optimisers (`optimizer_main`, `optimizer_sigma`) cleanly separate parameter groups.

Sub-classes: `MaskFormerTrainer`, `MaskFormerEvaluator`.
File: `maskformer_trainer.py`

## Dataset Contracts

| Dataset | File | Keys in output dict |
|---|---|---|
| `RawWindowDataset` | `datasets/raw.py` | `input` → `Tensor[C, seq_len]` |
| `RawWindowMosaicDataset` | `datasets/raw.py` | `input` → `Tensor[C, seq_len]` |
| `StatWindowDataset` | `datasets/stat.py` | `raw_window`, `stats`, `rel_pos`, `robot_idx`, `program_idx`, (opt.) `file_stats` |
| `StatWindowMosaicDataset` | `datasets/stat.py` | `raw_window`, `stats`, `rel_pos`, `robot_idx`, `program_idx`, (opt.) `file_stats` |

See `datasets/README.md` for full details.

## How to Add a New Trainer

1. Create a new file under `training/` (e.g. `my_trainer.py`).
2. Implement a class with at least:
   - `__init__(self, model, ...)` — configure optimiser(s), loss(es).
   - `train_step(self, x: dict) -> dict[str, Tensor]` — single-step forward/backward.
   - (optional) `evaluate(self, normal_files, anomaly_files) -> EvalResult`.
3. Export the class in `training/__init__.py`.

## How to Add a New Dataset

1. Create a new file under `training/datasets/` (e.g. `my_dataset.py`).
2. Subclass `torch.utils.data.Dataset` (or `_StreamingDataset` for Mosaic).
3. Implement `__len__` and `__getitem__` returning a `dict[str, Tensor]`.
4. Export the class in `training/datasets/__init__.py`.
