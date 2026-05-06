# feature_extraction.py

Neural network modules for encoding different input features including ShapeEncoder (1D convolutions over raw window), StatProjector (MLP for statistical features), RelativePositionalEncoding (linear encoding of relative position), and ContextualEmbedding (lookup embeddings for robot and program IDs).

## Source Code

```python
import torch
import torch.nn as nn

class ShapeEncoder(nn.Module):
    """ 
    Use to encode the normalized raw signal [B, C, W] into [B, C, D]
    """
    def __init__(self, c_in: int, window_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.c_in = c_in
        self.window_size = window_size
        self.d_model = d_model
        self.dropout = dropout

        self.local_extractor = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_in * 4, kernel_size=3, padding=1, groups=c_in),
            nn.SiLU(),
            nn.Conv1d(in_channels=c_in * 4, out_channels=c_in, kernel_size=1),
        )

        self.norm_temporal = nn.LayerNorm(window_size)
        self.temporal_projector = nn.Linear(window_size, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.local_extractor(x)
        out = self.norm_temporal(out)
        out = self.temporal_projector(out)
        out = self.norm_final(out)
        out = self.dropout(out)
        return out
    
class StatProjector(nn.Module):
    def __init__(self, c_in: int, k_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.c_in = c_in
        self.k_in = k_in
        self.d_model = d_model
        self.dropout = dropout

        self.projector = nn.Sequential(
            nn.Linear(k_in, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.projector(x)
        return self.dropout(out)
    
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)
    
class ContextualEmbedding(nn.Module):
    def __init__(self, num_robots: int, num_programs: int, d_model: int):
        super().__init__()
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model

        self.robot_embedding = nn.Embedding(num_robots, d_model // 2)
        self.program_embedding = nn.Embedding(num_programs, d_model // 2)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, robot_ids, program_ids):
        robot_embs = self.robot_embedding(robot_ids)
        program_embs = self.program_embedding(program_ids)

        context = torch.cat([robot_embs, program_embs], dim=-1)
        return self.projector(context).unsqueeze(1)
```