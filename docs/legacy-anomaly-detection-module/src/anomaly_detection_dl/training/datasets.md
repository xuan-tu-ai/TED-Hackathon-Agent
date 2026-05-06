# datasets.py

Streaming dataset implementation for robot anomaly detection that loads preprocessed Moses data from MDS format. Wraps StreamingDataset and transforms stored arrays into PyTorch tensors with proper dtype conversions for model input.

## Source Code

```python
import torch
from streaming import MDSWriter, StreamingDataset, StreamingDataLoader

class RobotMosaicDataset(StreamingDataset):
    def __init__(self, remote: str, local: str, batch_size: int,
                 num_workers: int = 4, shuffle: bool = True):
        super().__init__(local=local, remote=remote, batch_size=batch_size,
                         predownload=num_workers*2, shuffle_block_size=batch_size*32)
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return {
            'raw_window': torch.from_numpy(sample['raw_window'].copy()),
            'stats': torch.from_numpy(sample['stats'].copy()),
            'rel_pos': torch.tensor([sample['rel_pos']], dtype=torch.float32),
            'robot_idx': torch.tensor(sample['robot_idx'], dtype=torch.long),
            'program_idx': torch.tensor(sample['program_idx'], dtype=torch.long),
        }
```