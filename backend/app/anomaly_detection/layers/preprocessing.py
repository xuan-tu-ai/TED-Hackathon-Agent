import torch
import torch.nn as nn

class RevIN(nn.Module):
    """ 
    Reverse Instance Normalization (RevIN) layer for time series data.
    Use for raw signal [B, C, W] where B is batch size, C is number of sensors, W is window size.
    """

    def __init__(self, num_sensors: int, eps: float = 1e-5):
        super().__init__()
        self.num_sensors = num_sensors
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean = torch.mean(x, dim=-1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()

        x_norm = (x - mean) / stdev

        return x_norm, (mean, stdev)
    
    def denormalize(self, x_norm: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, stdev = stats
        x_denorm = x_norm * stdev + mean
        return x_denorm
    
class LogScaler(nn.Module):
    """ 
    Logarit Scaler Layer.
    Use to normalize stats feature: [B, C, K] where B is batch size, C is number of sensors, K is number of stats features.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
class GlobalStatsStandardizer(nn.Module):
    """ 
    Use to standardize global stats features after log scaling: [B, C, K] where B is batch size, C is number of sensors, K is number of stats features.
    """

    def __init__(self, num_sensors: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.standarizer = nn.BatchNorm1d(num_features=num_sensors, affine=False, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.standarizer(x)