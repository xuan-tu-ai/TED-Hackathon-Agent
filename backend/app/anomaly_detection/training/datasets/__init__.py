"""Dataset implementations for the anomaly detection training pipeline.

Provides two families of datasets:

  **Raw-only datasets** (raw.py):
    - ``RawWindowDataset``        — in-memory, outputs ``{input: Tensor[C, seq_len]}``.
    - ``RawWindowMosaicDataset``  — streaming MosaicML counterpart.

  **Stat-context datasets** (stat.py):
    - ``StatWindowDataset``       — in-memory, outputs raw+stats+metadata.
    - ``StatWindowMosaicDataset`` — streaming MosaicML counterpart.
    - ``write_sdf_to_mds``        — helper to serialise a Spark DataFrame to MDS shards.
"""

from .raw import RawWindowDataset, RawWindowMosaicDataset
from .stat import StatWindowDataset, StatWindowMosaicDataset, write_sdf_to_mds
