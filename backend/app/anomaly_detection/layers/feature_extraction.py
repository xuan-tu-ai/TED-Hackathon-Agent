import torch
import torch.nn as nn

class ShapeEncoder(nn.Module):
    """ 
    Encode [B,C,W] to [B,C,D] where B is batch size, C is number of sensors, W is window size, D is d_model.
    """

    def __init__(self, num_sensors: int, window_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.d_model = d_model

        self.local_extractor = nn.Sequential(
            nn.Conv1d(in_channels=num_sensors, out_channels=num_sensors * 4, kernel_size=3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(in_channels=num_sensors * 4, out_channels=num_sensors, kernel_size=1),
        )

        self.norm_temporal = nn.LayerNorm(window_size)
        self.temporal_projector = nn.Linear(window_size, d_model)

        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.local_extractor(x)
        out = self.norm_temporal(out)
        out = self.temporal_projector(out)
        out = self.norm_final(out)
        out = self.dropout(out)
        return out
    
class StatProjector(nn.Module):
    """ 
    Project global stats features [B,C,K] to [B,C,D] where B is batch size, C is number of sensors, K is number of stats features, D is d_model.
    """

    def __init__(self, num_sensors: int, num_stats_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.d_model = d_model

        self.projector = nn.Sequential(
            nn.Linear(num_stats_features, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projector(x)
        out = self.dropout(out)
        return out
    
class RelativePositionEncoder(nn.Module):
    """ 
    Encode relative position of a window
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        return out.unsqueeze(1)
    
class ContextualEmbedding(nn.Module):
    """ 
    Encode ID of robot (Robot code, program number)
    """

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
            nn.LayerNorm(d_model),
        )

    def forward(self, robot_ids: torch.Tensor, program_ids: torch.Tensor) -> torch.Tensor:
        robot_emb = self.robot_embedding(robot_ids)
        program_emb = self.program_embedding(program_ids)
        combined_emb = torch.cat([robot_emb, program_emb], dim=-1)
        out = self.projector(combined_emb)
        return out