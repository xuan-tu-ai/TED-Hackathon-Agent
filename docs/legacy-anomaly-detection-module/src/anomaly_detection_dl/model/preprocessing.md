# preprocessing.py

Input normalization layers including RevIN (reversible instance normalization for raw signals), LogScaler (signed logarithm for handling sparse statistics), and GlobalStatStandardizer (batch normalization over sensor dimension for z-score normalization).

## Source Code

```python
import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reverse Instance-Normalization Layer.
    Use for raw signal [B, C, W].
    """
    def __init__(self, num_features: int, eps: float = 1e-3):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        # Calculate through W dimension
        mean = torch.mean(x, dim=-1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()

        x_norm = (x - mean) / stdev
        return x_norm, (mean, stdev)
    
    def denormalize(self, x_norm, stats):
        mean, stdev = stats
        x_denorm = x_norm * stdev + mean
        return x_denorm
    
class LogScaler(nn.Module):
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
class GlobalStatStandardizer(nn.Module):
    """ 
    Global Stat Standardizer Layer. 
    Use to normalize stats feature: [B, C, K] after Log Scaler.
    """
    def __init__(self, c_in: int):
        super().__init__()
        self.c_in = c_in
        self.standardizer = nn.BatchNorm1d(num_features=c_in, affine=False, track_running_stats=True)

    def forward(self, x):
        return self.standardizer(x)
    
```