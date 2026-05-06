# anomaly_model.py

Neural network models for robot anomaly detection including LegacyRobotAnomalyModelVQ (legacy version), RobotAnomalyModelVQ (single-level VQ), and RobotAnomalyModelRVQ (residual VQ with multiple levels). Models combine shape encoding, statistical projection, positional encoding, and contextual embeddings through an MLP mixer backbone with vector quantization bottleneck.

## Source Code

```python
import torch
import torch.nn as nn

from .preprocessing import RevIN, LogScaler, GlobalStatStandardizer
from .feature_extraction import StatProjector, RelativePositionalEncoding, ContextualEmbedding, ShapeEncoder
from .backbone import MLPMixerBackbone
from .bottleneck import LegacyFactorizedVectorQuantizerEMA, LegacyVQMultiTaskVAEHead, FactorizedVectorQuantizerEMA, ResidualFactorizedVectorQuantizerEMA

class LegacyRobotAnomalyModelVQ(nn.Module):
    def __init__(self, c_in: int, window_size: int,
                 k_in: int, num_robots: int, num_programs: int,
                 d_model: int = 128, codebook_size: int = 256,
                 num_mixer_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = 'legacy_robot_anomaly_model_vq'
        self.c_in = c_in
        self.window_size = window_size
        self.k_in = k_in
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_mixer_blocks = num_mixer_blocks
        self.dropout = dropout

        # Input Preprocessing
        self.revin = RevIN(num_features=c_in)
        self.log_scaler = LogScaler()
        self.stat_standardizer = GlobalStatStandardizer(c_in)
        
        assert d_model % 2 == 0, "d_model must be divisible by 2 for contextual embedding."

        # Feature Extraction
        self.shape_encoder = ShapeEncoder(c_in=c_in, window_size=window_size, d_model=d_model//2, dropout=dropout)
        self.stat_projector = StatProjector(c_in=c_in, k_in=k_in, d_model=d_model//2, dropout=dropout)
        self.pos_encoder = RelativePositionalEncoding(d_model=d_model)
        self.context_embedding = ContextualEmbedding(num_robots=num_robots, num_programs=num_programs, d_model=d_model)

        # Backbone
        self.backbone = MLPMixerBackbone(c_in=c_in, d_model=d_model, num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout)

        # Bottleneck
        self.vq_head = LegacyVQMultiTaskVAEHead(c_in=c_in, d_model=d_model, k_in=k_in, window_size=window_size, codebook_size=codebook_size)

    def get_metadata(self):
        return {
            'model_name': self.name,
            'c_in': self.c_in,
            'window_size': self.window_size,
            'k_in': self.k_in,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'd_model': self.d_model,
            'codebook_size': self.codebook_size,
            'num_mixer_blocks': self.num_mixer_blocks,
            'dropout': self.dropout
        }
    
    def forward(self, raw_window, stats, rel_pos, robot_idx, program_idx):
        s_log = self.log_scaler(stats)
        s_zscore = self.stat_standardizer(s_log)
        f_phys = self.stat_projector(s_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos)
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_phys], dim=-1) + f_pos + f_context
        
        mixer_out = self.backbone(combined)

        recon_norm, pred_stats_z, vq_loss, indices = self.vq_head(mixer_out)
        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return recon_raw, recon_norm, x_norm, pred_stats_z, s_zscore, vq_loss, indices
    
class RobotAnomalyModelVQ(nn.Module):
    def __init__(self, c_in: int, window_size: int,
                 k_win: int, num_robots: int, num_programs: int,
                 d_model: int = 128, codebook_size: int = 256,
                 num_mixer_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = 'robot_anomaly_model_vq'
        self.c_in = c_in
        self.window_size = window_size
        self.k_win = k_win
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_mixer_blocks = num_mixer_blocks
        self.dropout = dropout

        # Input Preprocessing
        self.revin = RevIN(num_features=c_in)
        self.log_scaler = LogScaler()
        self.win_stat_standardizer = GlobalStatStandardizer(c_in)

        # Feature Extraction
        self.shape_encoder = ShapeEncoder(c_in=c_in, window_size=window_size, d_model=d_model, dropout=dropout)
        self.win_stat_projector = StatProjector(c_in=c_in, k_in=k_win, d_model=d_model, dropout=dropout)
        self.pos_encoder = RelativePositionalEncoding(d_model=d_model)
        self.context_embedding = ContextualEmbedding(num_robots=num_robots, num_programs=num_programs, d_model=d_model)

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

        self.mixer_backbone = MLPMixerBackbone(c_in=c_in, d_model=d_model, num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout)

        self.quantizer = FactorizedVectorQuantizerEMA(
            num_sensors=c_in, codebook_size=codebook_size, embedding_dim=d_model
        )

        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(in_channels=c_in, out_channels=c_in * 4, kernel_size=3, padding=1, groups=c_in),
            nn.SiLU(),
            nn.Conv1d(in_channels=c_in * 4, out_channels=c_in, kernel_size=1),
        )

        self.decoder_win_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, k_win),
        )

    def get_metadata(self):
        return {
            'model_name': self.name,
            'c_in': self.c_in,
            'window_size': self.window_size,
            'k_win': self.k_win,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'd_model': self.d_model,
            'codebook_size': self.codebook_size,
            'num_mixer_blocks': self.num_mixer_blocks,
            'dropout': self.dropout
        }
    
    def forward(self, raw_window, win_stats, rel_pos, robot_idx, program_idx):

        s_win_log = self.log_scaler(win_stats)
        s_win_zscore = self.win_stat_standardizer(s_win_log)
        f_win_phys = self.win_stat_projector(s_win_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos)
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_win_phys], dim=-1)
        z_e = self.fusion_layer(combined) + f_pos + f_context

        z_q, commitment_loss, indices = self.quantizer(z_e)
        z_mixed = self.mixer_backbone(z_q)

        recon_norm = self.decoder_shape(z_mixed)
        pred_win_stats_z = self.decoder_win_stats(z_mixed).view(-1, self.c_in, self.k_win)

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, commitment_loss, indices

class RobotAnomalyModelRVQ(nn.Module):
    def __init__(self, c_in: int, window_size: int,
                 k_win: int, num_robots: int, num_programs: int,
                 d_model: int = 128, codebook_size: int = 256,
                 num_vq_levels: int = 2,
                 num_mixer_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = 'robot_anomaly_model_rvq'
        self.c_in = c_in
        self.window_size = window_size
        self.k_win = k_win
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_vq_levels = num_vq_levels
        self.num_mixer_blocks = num_mixer_blocks
        self.dropout = dropout

        # Input Preprocessing
        self.revin = RevIN(num_features=c_in)
        self.log_scaler = LogScaler()
        self.win_stat_standardizer = GlobalStatStandardizer(c_in)

        # Feature Extraction
        self.shape_encoder = ShapeEncoder(c_in=c_in, window_size=window_size, d_model=d_model, dropout=dropout)
        self.win_stat_projector = StatProjector(c_in=c_in, k_in=k_win, d_model=d_model, dropout=dropout)
        self.pos_encoder = RelativePositionalEncoding(d_model=d_model)
        self.context_embedding = ContextualEmbedding(num_robots=num_robots, num_programs=num_programs, d_model=d_model)

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

        self.mixer_backbone = MLPMixerBackbone(c_in=c_in, d_model=d_model, num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout)

        self.quantizer = ResidualFactorizedVectorQuantizerEMA(
            num_sensors=c_in, codebook_size=codebook_size, embedding_dim=d_model, num_levels=num_vq_levels
        )

        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(in_channels=c_in, out_channels=c_in * 4, kernel_size=3, padding=1, groups=c_in),
            nn.SiLU(),
            nn.Conv1d(in_channels=c_in * 4, out_channels=c_in, kernel_size=1),
        )

        self.decoder_win_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, k_win),
        )

    def forward(self, raw_window, stats, rel_pos, robot_idx, program_idx):

        s_win_log = self.log_scaler(stats)
        s_win_zscore = self.win_stat_standardizer(s_win_log)
        f_win_phys = self.win_stat_projector(s_win_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos)
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_win_phys], dim=-1)
        z_e = self.fusion_layer(combined) + f_pos + f_context

        z_q, commitment_loss, indices = self.quantizer(z_e)
        z_mixed = self.mixer_backbone(z_q)

        recon_norm = self.decoder_shape(z_mixed)
        pred_win_stats_z = self.decoder_win_stats(z_mixed).view(-1, self.c_in, self.k_win)

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, commitment_loss, indices
```