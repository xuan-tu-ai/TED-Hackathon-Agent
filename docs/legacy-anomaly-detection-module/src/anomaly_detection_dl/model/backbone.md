# backbone.py

MLP-Mixer architecture components including MixerBlock (performs token mixing on sensor dimension and channel mixing on model dimension) and MLPMixerBackbone (stacks multiple mixer blocks with final layer normalization).

## Source Code

```python
import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    def __init__(self, c_in: int, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        # Token mixing - C dimension
        self.norm1 = nn.LayerNorm(d_model)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(c_in, c_in * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(c_in * expansion_factor, c_in),
            nn.Dropout(dropout),
        )

        # Channel mixing - D dimension
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x 
        out = self.norm1(x)
        out = out.transpose(1, 2) # [B, D, C]
        out = self.sensor_mlp(out)
        out = out.transpose(1, 2) # [B, C, D]
        out = out + residual

        residual = out
        out = self.norm2(out)
        out = self.channel_mlp(out)
        out = out + residual

        return out
    
class MLPMixerBackbone(nn.Module):
    def __init__(self, c_in: int, d_model: int, num_blocks: int = 4, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.blocks = nn.ModuleList([
            MixerBlock(c_in, d_model, expansion_factor, dropout)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return self.final_norm(x)
```