# Datasets — Training

## Purpose

This subpackage provides PyTorch `Dataset` implementations used by the
anomaly detection trainers. Two families exist:

1. **Raw-only** — windows of raw sensor values, no statistical context.
2. **Stat-context** — raw sensor windows plus per-sensor statistics and metadata.

Each family has an **in-memory** variant (for local/notebook use) and a
**streaming** variant (for large-scale training via MosaicML StreamingDataset).

## Raw-only Datasets (`raw.py`)

Used by raw-only models (GLCFlowV7, seq\_len=240).

### `RawWindowDataset`

In-memory `torch.utils.data.Dataset`.

- **Constructor:** `RawWindowDataset(windows: np.ndarray)` — shape `[N, C, seq_len]`.
- **Output:**
  ```python
  {"input": Tensor[C, seq_len]}
  ```

### `RawWindowMosaicDataset`

Streaming variant backed by MDS shards.

- **Constructor:** `RawWindowMosaicDataset(remote_dir, local_dir, batch_size, num_workers, shuffle=True)`.
- **Output:**
  ```python
  {"input": Tensor[C, seq_len]}
  ```
- Reads from an MDS column named `"raw_window"`.

## Stat-context Datasets (`stat.py`)

Used by stat-context models (V1–V6, window\_size=32).

### `StatWindowDataset`

In-memory `torch.utils.data.Dataset`.

- **Constructor:** `StatWindowDataset(windows, stats, rel_pos=None, robot_indices=None, program_indices=None, file_stats=None)`.
- **Output (all tensors):**
  ```python
  {
    "raw_window":   Tensor[C, window_size],   # required
    "stats":        Tensor[C, num_stats],      # required
    "rel_pos":      Tensor[1],                # optional
    "robot_idx":    Tensor[],                 # optional (long)
    "program_idx":  Tensor[],                 # optional (long)
    "file_stats":   Tensor[C, num_file_stats],# optional
  }
  ```

### `StatWindowMosaicDataset`

Streaming variant backed by MDS shards.

- **Constructor:** `StatWindowMosaicDataset(remote_dir, local_dir, batch_size, num_workers, shuffle=True)`.
- Output keys: `raw_window`, `stats`, `rel_pos`, `robot_idx`, `program_idx`, (opt.) `file_stats`.

### `write_sdf_to_mds`

Converts a PySpark DataFrame into MDS shards.

- Reads columns:`raw_values_array`, `*_stat_*`, `window_start_serial`, `robot_code`, `program_number`, (opt. `*_file_stat_*`).
- Reshapes arrays into `[num_sensors, window_size]` / `[num_sensors, num_stats]`.
- Writes compressed (`zstd` + SHA1) MDS shards.

## Contract Summary

| Method | Input | Output keys |
|---|---|---|
| `RawWindowDataset.__getitem__` | `idx: int` | `input` |
| `RawWindowMosaicDataset.__getitem__` | `idx: int` | `input` |
| `StatWindowDataset.__getitem__` | `idx: int` | `raw_window`, `stats`, `rel_pos?`, `robot_idx?`, `program_idx?`, `file_stats?` |
| `StatWindowMosaicDataset.__getitem__` | `idx: int` | `raw_window`, `stats`, `rel_pos`, `robot_idx`, `program_idx`, `file_stats?` |
| `write_sdf_to_mds` | Spark DF → MDS shards | `raw_window`, `stats`, `rel_pos`, `robot_idx`, `program_idx`, `file_stats?` |

## Adding a New Dataset

1. Create `training/datasets/my_dataset.py`.
2. Subclass `torch.utils.data.Dataset` (in-memory) or `streaming.StreamingDataset` (Mosaic).
3. Implement `__len__` and `__getitem__` → return `dict[str, Tensor]`.
4. Export in `training/datasets/__init__.py`.
