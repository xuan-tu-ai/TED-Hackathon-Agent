# Anomaly Detection — Welding Robot Model Zoo

Self-contained anomaly detection sub-project for welding robots. Contains a deep-learning **model zoo** spanning 6 architecture families (V1–V9), a **physics-based data simulator**, **training pipelines**, **inference pipelines**, and **Databricks notebooks** for end-to-end deployment.

Current production model: **GLCMaeV9** — Masked Autoencoder with LogScaler + GlobalSeqNorm and per-sensor p99 scoring.

---

## Model Zoo

| Model | Paradigm | Description |
|---|---|---|
| **V1 VAE** | Reconstruction | Simple convolutional auto-encoder for short windows |
| **V5 VQ** | VQ-VAE | Vector-quantised variational auto-encoder |
| **V5 VQ File** | VQ-VAE | Per-file VQ variant with file-level codebook |
| **V6 VQ / V6 RVQ** | VQ-VAE | Residual VQ extensions over V5 |
| **GLC-Flow V7** | Flow-Matching | Continuous Normalising Flow with dual-stream encoder + transformer backbone + CNF likelihood head |
| **GLC-Recon V8** | Reconstruction | Full reconstruction with dual-stream CNN encoders, standard transformer, per-token MSE loss |
| **GLCMaeV8** | Masked Autoencoder | Masked prediction with dual-head (global + local) + LearnableMaskToken |
| **GLCMaeV9** | Masked Autoencoder | ✨ **Current production** — LogScaler + GlobalSeqNorm + per-sensor p99 scoring |

---

## Structure

```
anomaly-detection/
├── src/anomaly_detection/     # Core Python package
│   ├── constants.py           # Sensor names, stat suffixes, window sizes
│   ├── schemas.py             # Config dataclasses (model configs, RevIN stats)
│   ├── utils.py               # Helpers (seeding, param counting, device mgmt)
│   │
│   ├── layers/                # Reusable nn.Module building blocks
│   │   ├── preprocessing.py   # LogScaler, GlobalSeqNorm, StandardScaler
│   │   ├── revin.py           # GlobalRevIN, PerPatchRevIN
│   │   ├── patchify.py        # Patchify, FlattenPatches
│   │   ├── encoders.py        # PatchEncoderGlobal, PatchEncoderLocal
│   │   ├── transformer.py     # TransformerEncoder, CLSToken, DualPooling
│   │   ├── backbone.py        # Shared Transformer backbone
│   │   ├── bottleneck.py      # VQ / RVQ bottleneck
│   │   ├── cnf.py             # VectorFieldMLP, CNF (continuous normalising flow)
│   │   ├── mask_token.py      # LearnableMaskToken
│   │   ├── reconstruction_decoder.py
│   │   ├── context_embedding.py   # RobotEmbedding, ProgramEmbedding
│   │   ├── sensor_block_masking.py
│   │   └── scorer.py              # MADThreshold, AnomalyScorer, PatchMasker
│   │
│   ├── models/                # Concrete architectures (extend BaseModel)
│   │   ├── base.py            # BaseAnomalyDetectionModel (abstract)
│   │   ├── factory.py         # create_model(), SupportedAnomalyDetectionModels
│   │   ├── v1_vae.py          # V1 Convolutional VAE
│   │   ├── v5_vq.py / v5_vq_file.py / v6_vq.py / v6_rvq.py  # V5-V6 VQ family
│   │   ├── glc_flow_v7.py     # GLC-Flow V7 (Flow-Matching)
│   │   ├── glc_recon_v8.py    # GLC-Recon V8
│   │   ├── glc_mae_v8.py      # GLCMaeV8 (Masked Autoencoder)
│   │   └── glc_mae_v9.py      # GLCMaeV9 ✨ (current production)
│   │
│   ├── criterions/            # Loss functions
│   │   ├── flow_matching.py   # FlowMatchingLoss (CNF likelihood)
│   │   ├── dual_head.py       # DualHeadLoss (global + local)
│   │   ├── masked_reconstruction.py
│   │   └── association_discrepancy.py
│   │
│   ├── training/              # Trainers + Datasets
│   │   ├── datasets/
│   │   │   ├── raw.py         # RawWindowDataset (window-level)
│   │   │   └── stat.py        # StatWindowDataset (window + file stats)
│   │   ├── single_stage_trainer.py   # V1–V6 trainer
│   │   └── flow_matching_trainer.py  # Two-stage GLC-Flow V7 trainer
│   │
│   └── inference/             # Post-training inference pipelines
│       ├── reconstruction/    # V1–V6: recon error + percentile threshold
│       ├── flow_matching/     # GLCFlowV7: NLL + MAD threshold
│       ├── recon_v8/          # GLCReconV8: per-patch MSE + multi-score MAD
│       └── recon_v9/          # GLCMaeV9: masked prediction + per-sensor p99
│
├── notebooks/                 # 11 Jupyter notebooks
│   ├── Training-*.ipynb       # V8/V9 training on Databricks
│   ├── Inference-*.ipynb      # V8/V9 inference with threshold calibration
│   ├── Load-Inference-Json-To-Silver.ipynb  # → silver_layer.perception_results
│   ├── Calculate-L1-L2-Gold.ipynb           # Gold-layer pre-computation
│   ├── Seeding-*.ipynb        # PM cards & maintenance event seeding
│   └── Sprint*-Generate-Data.ipynb          # Test data generation
│
├── checkpoints/               # Trained model weights (gitignored)
├── data/                      # Generated training/test CSV data
├── pyproject.toml             # Dependencies (PyTorch, NumPy, Pandas, PySpark, …)
└── uv.lock                    # Locked dependencies
```

---

## Hardware Requirements

Training GLCMaeV9 requires:

| Component | Requirement |
|---|---|
| **GPU** | 24 GB+ VRAM (A10G, A100, H100 recommended) |
| **RAM** | 32 GB+ |
| **Storage** | 50 GB+ for datasets + checkpoints |
| **Platform** | Databricks ML cluster (GPU) recommended |

---

## Setup

```bash
cd anomaly-detection
uv sync                             # install dependencies
uv run pytest tests/ -v             # run all tests
```

---

## Notebook Pipeline (End-to-End)

The recommended workflow for training and deploying GLCMaeV9 on Databricks:

```
1. Sprint20-Generate-Data.ipynb        ─── Generate training/test datasets
2. Training-GLCReconV9.ipynb           ─── Train GLCMaeV9 on Databricks GPU
3. Inference-GLCReconV9.ipynb          ─── Run inference → inference_results_silver.json
4. Load-Inference-Json-To-Silver.ipynb ─── Load JSON into silver_layer.perception_results
5. Calculate-L1-L2-Gold.ipynb          ─── Pre-compute Gold-layer L1/L2/SensorStats
6. Seeding-PM-Cards-Bronze-Gold.ipynb  ─── Seed PM cards
7. Seeding-Maintenance-Events-Bronze-Silver.ipynb  ─── Seed maintenance events
```

---

## Data Simulator

The `data_simulator/` module generates physics-based welding robot telemetry:

| Generator | Output |
|---|---|
| `batch_generator.py` | `BatchGenerator.generate_train()` / `generate_test()` → `CycleRow` objects |
| `profiles.py` | Robot-specific profiles with configurable anomaly patterns |
| `writer.py` | Chunked CSV writer (10,000 rows/chunk) |

**Cycle specifications:**
- **Sampling rate:** 8 Hz (125 ms per step)
- **Steps per cycle:** 512 (fixed)
- **Cycle duration:** 64 seconds (512 × 0.125 s)
- **Daily operation:** 16–20 hours/robot
- **Gap between cycles:** 1–5 minutes
- **3 sensors:** `wire_feed_speed`, `wire_temperature`, `welding_current`
- **7 anomaly patterns:** `spike`, `cable_wear`, `flatline`, `thermal_drift`, `highfreq`, `burst`, `transition`

---

## GLCMaeV9 Architecture (Production)

```
Input: [B, 3 sensors, 512 steps]
        │
        ▼
  LogScaler + GlobalSeqNorm (preprocessing.py)
        │
        ▼
  Patchify → Dual-stream Encoders (PatchEncoderGlobal + PatchEncoderLocal)
        │
        ▼
  Masking (LearnableMaskToken + random masking strategy)
        │
        ▼
  Transformer Backbone (shared encoder)
        │
        ▼
  Dual Head:
  ├── Global Head → global reconstruction loss
  └── Local Head → per-patch masked prediction loss
        │
        ▼
  Per-sensor p99 score aggregation
  └── { p99_score_wire_feed_speed, p99_score_wire_temperature, p99_score_welding_current }
```

---

## Common Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run a specific test
uv run pytest tests/test_batch_generator.py -v

# Verify with type checking
uv run mypy src/ --ignore-missing-imports
```
