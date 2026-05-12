import torch
import torch.nn as nn
import torch.nn.functional as F

class MultitaskVAEHead(nn.Module):

    def __init__(self, num_sensors: int, num_stats_features: int, window_size: int, d_model: int, latent_dim: int = 32):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.window_size = window_size
        self.d_model = d_model

        flatten_dim = num_sensors * d_model

        self.share_encoder = nn.Sequential(
            nn.Linear(flatten_dim, flatten_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(flatten_dim // 2),
            nn.Linear(flatten_dim // 2, latent_dim * 2),
        )

        # Decoder shape
        self.shape_decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim // 2),
            nn.SiLU(),
            nn.Linear(flatten_dim // 2, num_sensors * window_size),
        )

        # Decoder stats
        self.stats_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, num_sensors * num_stats_features),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        hidden = self.share_encoder(x.view(B, -1))

        mu, logvar = torch.chunk(hidden, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        recon_norm = self.shape_decoder(z).view(B, self.num_sensors, self.window_size)
        recon_stats = self.stats_decoder(z).view(B, self.num_sensors, self.num_stats_features)

        return recon_norm, recon_stats, mu, logvar
    
class FactorizedVectorQuantizerEMA(nn.Module):

    def __init__(self, num_sensors: int, codebook_size: int, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer("codebooks", torch.empty(num_sensors, codebook_size, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_sensors, codebook_size))
        self.register_buffer("embed_avg", self.codebooks.clone())
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.long))

        self._codebook_usage = [0.0] * num_sensors
        self.register_buffer("_step", torch.tensor(0))

    def _init_from_data(self, z_e: torch.Tensor):
        print("VQ-VAE Initializing codebooks from data...")
        B, C, D = z_e.shape

        for c in range(self.num_sensors):
            z_e_c = z_e[:, c, :]

            if B >= self.codebook_size:
                indices = torch.randperm(B)[:self.codebook_size]
                samples = z_e_c[indices]
            else:
                repeats = self.codebook_size // B + 1
                samples = z_e_c.repeat(repeats, 1)[:self.codebook_size]
                samples = samples + 0.01 * torch.randn_like(samples)

            self.codebooks[c].data.copy_(samples)
            self.cluster_size[c].data.fill_(1.0)
            self.embed_avg[c].data.copy_(samples)

        self.is_initialized.fill_(1)
        print("VQ-VAE Codebooks initialized.")

    def _random_restart(self, c: int, z_e_c: torch.Tensor, encoding_indices: torch.Tensor):
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]

        if len(dead_codes) == 0:
            return
        
        num_to_restart = min(len(dead_codes), z_e_c.shape[0])
        replace_indices = torch.randperm(z_e_c.shape[0])[:num_to_restart]

        self.codebooks[c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]
        self.cluster_size[c].data[dead_codes[:num_to_restart]] = 1.0
        self.embed_avg[c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]

    def _log_codebook_usage(self, indice_list: list[torch.Tensor]):
        usage_per_sensor = []
        for c, indices in enumerate(indice_list):
            num_used = indices.bincount(minlength=self.codebook_size).gt(0).sum()
            usage_per_sensor.append((num_used.float() / self.codebook_size).item())

        self._codebook_usage = usage_per_sensor
        self._step += 1

        if self._step % 500 != 0:
            return
        
        avg_usage = sum(usage_per_sensor) / len(usage_per_sensor)
        if avg_usage < 0.5:
            dead = [c for c, u in enumerate(usage_per_sensor) if u < 0.3]
            print(f"[VQ WARNING] Codebook collapse! Avg usage: {avg_usage:.2f}, Dead codebooks: {dead}")

        else:
            print(f"[VQ INFO] Codebook usage: {avg_usage:.1%}")
    
    def forward(self, z_e: torch.Tensor):
        z_e_norm = F.normalize(z_e, p=2, dim=-1)
        if self.training and self.is_initialized.item() == 0:
            self._init_from_data(z_e_norm)

        B, C, D = z_e.shape
        z_q = torch.zeros_like(z_e)
        commitment_loss = 0.0
        indices_list = []

        for c in range(self.num_sensors):
            z_e_c = z_e_norm[:, c, :]

            # Compute distance
            cb_c = F.normalize(self.codebooks[c], p=2, dim=-1)
            dist = 2 - 2 * torch.matmul(z_e_c, cb_c.t())

            encoding_indices = torch.argmin(dist, dim=1)
            indices_list.append(encoding_indices)

            z_q_c = cb_c[encoding_indices]
            z_q[:, c, :] = z_q_c

            if self.training:

                # EMA Update
                encodings_one_hot = F.one_hot(encoding_indices, self.codebook_size).type(z_e.dtype)
                cluster_size_batch = torch.sum(encodings_one_hot, dim=0)
                embed_sum_batch = torch.matmul(encodings_one_hot.t(), z_e_c)
                
                self.cluster_size[c].data.mul_(self.decay).add_(cluster_size_batch, alpha=1 - self.decay)
                self.embed_avg[c].data.mul_(self.decay).add_(embed_sum_batch, alpha=1 - self.decay)

                n = torch.sum(self.cluster_size[c].data)
                self.cluster_size[c].data = (
                    (self.cluster_size[c].data + self.eps) / (n + self.codebook_size * self.eps) * n
                )

                embed_normalized = self.embed_avg[c] / self.cluster_size[c].data.unsqueeze(1)
                self.codebooks[c].data.copy_(embed_normalized)
                self._random_restart(c, z_e_c, encoding_indices)
            
            e_loss = F.mse_loss(z_q_c.detach(), z_e_c)
            commitment_loss += self.commitment_cost * e_loss

        if self.training:
            self._log_codebook_usage(indices_list)

        z_q = z_e_norm + (z_q - z_e_norm).detach()
        return z_q, commitment_loss / self.num_sensors, torch.stack(indices_list, dim=1)
    
class ResidualFactorizedVectorQuantizerEMA(nn.Module):
    """Residual Factorized Vector Quantizer with EMA for multi-level quantization.
    
    Args:
        num_sensors (int): Number of sensors (channels).
        codebook_size (int): Size of the codebook.
        num_levels (int, optional): Number of quantization levels. Defaults to 2.
        embedding_dim (int, optional): Embedding dimension. Defaults to 128.
        commitment_cost (float, optional): Commitment loss coefficient. Defaults to 0.25.
        decay (float, optional): EMA decay rate. Defaults to 0.99.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-5.
    """
    
    def __init__(self, num_sensors: int, codebook_size: int, num_levels=2, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        self.register_buffer("codebooks", torch.empty(num_levels, num_sensors, codebook_size, embedding_dim))
        # self.codebooks.normal_()
        
        self.register_buffer("cluster_size", torch.zeros(num_levels, num_sensors, codebook_size))
        self.register_buffer("embed_avg", self.codebooks.clone())
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.long))
        
        self._codebook_usage = [[0.0] * num_sensors for _ in range(num_levels)]
        
        self.register_buffer("_step", torch.tensor(0))
        
    def _init_from_data(self, z_e):
        """Initialize codebooks for all levels from data samples.
        
        Args:
            z_e (Tensor): Encoder output tensor [B, C, D].
        """
        B, C, D = z_e.shape
        print(f"VQ-VAE Initializing Codebooks with data-driven samples")
        
        residual = z_e.clone()
        
        for l in range(self.num_levels):
            for c in range(self.num_sensors):
                z_e_c = residual[:, c, :] # [B, D]
                
                if B >= self.codebook_size:
                    indices = torch.randperm(B)[:self.codebook_size]
                    samples = z_e_c[indices]
                else:
                    repeats = (self.codebook_size // B + 1)
                    samples = z_e_c.repeat(repeats, 1)[:self.codebook_size]
                    samples = samples + 0.01 * torch.randn_like(samples)
                    
                self.codebooks[l, c].data.copy_(samples)
                self.cluster_size[l, c].data.fill_(1.0)
                self.embed_avg[l, c].data.copy_(samples)
                
        self.is_initialized.fill_(1)
        print(f"RVQ-VAE Codebooks initialized")
        
    def forward(self, z_e: torch.Tensor):
        """Forward pass for residual factorized vector quantization.
        
        Args:
            z_e (Tensor): Encoder output tensor [B, C, D].
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Quantized tensor [B, C, D]
                - Commitment loss (scalar)
                - Indices tensor [num_levels, B, C]
        """
        z_e_norm = F.normalize(z_e, p=2, dim=-1)
        if self.training and self.is_initialized.item() == 0:
            self._init_from_data(z_e_norm)
            
        B, C, D = z_e.shape
        residual = z_e_norm
        z_q_total = torch.zeros_like(z_e_norm)
        commitment_loss = 0
        all_levels_indices = []
        
        for l in range(self.num_levels):
            z_q_level = torch.zeros_like(z_e_norm)
            level_indices = []
            
            for c in range(self.num_sensors):
                z_e_c = residual[:, c, :]
                cb_lc = F.normalize(self.codebooks[l, c], p=2, dim=-1)
                
                dists = (2 - 2 * torch.matmul(z_e_c, cb_lc.t()))
                idx = torch.argmin(dists, dim=1)
                level_indices.append(idx)
                
                z_q_lc = cb_lc[idx]
                z_q_level[:, c, :] = z_q_lc
                
                if self.training:
                    one_hot = F.one_hot(idx, self.codebook_size).type(z_e_c.dtype)
                    self.cluster_size[l, c].data.mul_(self.decay).add_(one_hot.sum(0), alpha=1-self.decay)
                    self.embed_avg[l, c].data.mul_(self.decay).add_(one_hot.t() @ z_e_c, alpha=1-self.decay)
                    
                    # Laplace smoothing
                    n = self.cluster_size[l, c].sum()
                    self.cluster_size[l, c].data = (self.cluster_size[l, c].data + self.eps) / (n + self.codebook_size * self.eps) * n
                    self.codebooks[l, c].data.copy_(self.embed_avg[l, c] / self.cluster_size[l, c].unsqueeze(1))
                    
                    self._random_restart(l, c, z_e_c, idx)
                    
            z_q_total += z_q_level
            commitment_loss += F.mse_loss(z_q_level.detach(), residual)
            
            residual = residual - z_q_level.detach()
            all_levels_indices.append(torch.stack(level_indices, dim=1))
            
        if self.training:
            self._log_codebook_usage(all_levels_indices)
            
        z_q_total = z_e_norm + (z_q_total - z_e_norm).detach()
        
        return z_q_total, (commitment_loss / (self.num_levels * self.num_sensors)), torch.stack(all_levels_indices, dim=0)

    def _random_restart(self, l: int, c: int, z_e_c: torch.Tensor, encoding_indices: torch.Tensor):
        """Revive unused codes in the current batch for a given level and sensor.
        
        Args:
            l (int): Level index.
            c (int): Sensor index.
            z_e_c (Tensor): Encoder output for sensor c, shape [B, D].
            encoding_indices (Tensor): Indices of nearest codes, shape [B].
        """
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]
        
        if len(dead_codes) == 0:
            return
            
        num_to_restart = min(len(dead_codes), z_e_c.shape[0])
        replace_indices = torch.randperm(z_e_c.shape[0])[:num_to_restart]
        
        self.codebooks[l, c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]
        self.cluster_size[l, c].data[dead_codes[:num_to_restart]] = 1.0
        self.embed_avg[l, c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]

    def _log_codebook_usage(self, all_levels_indices_list: list):
        """Compute percentage of codes used for each sensor codebook in all levels in the current batch.
        
        Args:
            all_levels_indices_list (list): List of Tensor[B, C], length num_levels.
        """
        self._step += 1
        for l, level_indices in enumerate(all_levels_indices_list):
            for c in range(self.num_sensors):
                sensor_indices = level_indices[:, c]
                num_used = sensor_indices.bincount(minlength=self.codebook_size).gt(0).sum()
                self._codebook_usage[l][c] = (num_used.float() / self.codebook_size).item()
                
        if self._step % 500 == 0:
            avg_usage = sum([sum(l_usage) for l_usage in self._codebook_usage]) / (self.num_levels * self.num_sensors)
            print(f"[INFO] Global Codebook Usage: {avg_usage:.2%}")

class LegacyFactorizedVectorQuantizerEMA(nn.Module):
    """Legacy Factorized Vector Quantizer with EMA for VQ-VAE.
    
    Args:
        num_sensors (int): Number of sensors (channels).
        codebook_size (int): Size of the codebook.
        embedding_dim (int, optional): Embedding dimension. Defaults to 128.
        commitment_cost (float, optional): Commitment loss coefficient. Defaults to 0.25.
        decay (float, optional): EMA decay rate. Defaults to 0.99.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-5.
    """
    
    def __init__(self, num_sensors: int, codebook_size: int, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        self.register_buffer("codebooks", torch.empty(num_sensors, codebook_size, embedding_dim))
        self.codebooks.normal_()
        
        self.register_buffer("cluster_size", torch.zeros(num_sensors, codebook_size))
        self.register_buffer("embed_avg", self.codebooks.clone())
        
        self._codebook_usage = [0.0] * num_sensors
        
        self.register_buffer("_step", torch.tensor(0))
        
    def forward(self, z_e):
        """Forward pass for legacy factorized vector quantization.
        
        Args:
            z_e (Tensor): Encoder output tensor [B, C, D].
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Quantized tensor [B, C, D]
                - Commitment loss (scalar)
                - Indices tensor [B, C]
        """
        B, C, D = z_e.shape
        z_q = torch.zeros_like(z_e)
        commitment_loss = 0
        indices_list = []
        
        for c in range(self.num_sensors):
            z_e_c = z_e[:, c, :] # [B, D]
            
            # 1. Compute distance
            cb_c = self.codebooks[c]
            distances = (torch.sum(z_e_c ** 2, dim=1, keepdim=True)
                       + torch.sum(cb_c ** 2, dim=1)
                       - 2 * torch.matmul(z_e_c, cb_c.t())) # [B, codebook_size]
                       
            encoding_indices = torch.argmin(distances, dim=1)
            indices_list.append(encoding_indices)
            
            z_q_c = cb_c[encoding_indices]
            z_q[:, c, :] = z_q_c
            
            if self.training:
                # 2. EMA update (unchanged)
                encodings_one_hot = F.one_hot(encoding_indices, self.codebook_size).type(z_e_c.dtype)
                cluster_size_batch = torch.sum(encodings_one_hot, dim=0)
                embed_sum_batch = torch.matmul(encodings_one_hot.t(), z_e_c)
                
                self.cluster_size[c].data.mul_(self.decay).add_(cluster_size_batch, alpha=1 - self.decay)
                self.embed_avg[c].data.mul_(self.decay).add_(embed_sum_batch, alpha=1 - self.decay)
                
                n = torch.sum(self.cluster_size[c].data)
                self.cluster_size[c].data = (
                    (self.cluster_size[c].data + self.eps)
                    / (n + self.codebook_size * self.eps) * n
                )
                
                embed_normalized = self.embed_avg[c] / self.cluster_size[c].unsqueeze(1)
                self.codebooks[c].data.copy_(embed_normalized)
                
                self._random_restart(c, z_e_c, encoding_indices)
                
            e_loss = F.mse_loss(z_q_c.detach(), z_e_c)
            commitment_loss += self.commitment_cost * e_loss
            
        if self.training:
            self._log_codebook_usage(indices_list)
            
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, commitment_loss / self.num_sensors, torch.stack(indices_list, dim=1)
        
    def _random_restart(self, c: int, z_e_c: torch.Tensor, encoding_indices: torch.Tensor):
        """Revive unused codes in the current batch by replacing them with random encoder outputs.
        
        Args:
            c (int): Sensor index.
            z_e_c (Tensor): Encoder output for sensor c, shape [B, D].
            encoding_indices (Tensor): Indices of nearest codes, shape [B].
        """
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]
        
        if len(dead_codes) == 0:
            return
            
        random_indices = torch.randint(0, z_e_c.shape[0], (len(dead_codes),), device=z_e_c.device)
        self.codebooks[c].data[dead_codes] = z_e_c[random_indices].detach()

    def _log_codebook_usage(self, indices_list: list):
        """Compute percentage of codes used for each sensor codebook in the current batch.
        
        Args:
            indices_list (list): List of Tensor[B], length C.
        """
        usage_per_sensor = []
        for c, indices in enumerate(indices_list):
            num_used = indices.bincount(minlength=self.codebook_size).gt(0).sum()
            usage_per_sensor.append((num_used.float() / self.codebook_size).item())
            
        self._codebook_usage = usage_per_sensor
        self._step += 1
        
        if self._step % 500 != 0:
            return
            
        avg_usage = sum(usage_per_sensor) / len(usage_per_sensor)
        if avg_usage < 0.5:
            dead = [c for c, u in enumerate(usage_per_sensor) if u < 0.3]
            print(f"[VQ WARNING] Codebook collapse! Avg usage: {avg_usage:.1%}, Dead sensors: {dead}")

class MultiTaskVQVAEHead(nn.Module):
    """Multi-task VAE head with vector quantization for shape and statistics prediction.
    
    Args:
        num_sensors (int): Number of input channels.
        num_stats_features (int): Number of statistics to predict.
        window_size (int): Size of the window for shape reconstruction.
        d_model (int, optional): Model dimension. Defaults to 128.
        codebook_size (int, optional): Size of the codebook. Defaults to 256.
    """
    
    def __init__(self, num_sensors, num_stats_features, window_size, d_model, codebook_size=256):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.window_size = window_size
        self.d_model = d_model
        self.codebook_size = codebook_size
        
        self.quantizer = LegacyFactorizedVectorQuantizerEMA(
            num_sensors=num_sensors, codebook_size=codebook_size, embedding_dim=d_model
        )
        
        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1)
        )
        
        self.decoder_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_stats_features)
        )
        
    def forward(self, x):
        """Forward pass for the VQ multi-task VAE head.
        
        Args:
            x (Tensor): Input tensor [B, C, D].
            
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - Reconstructed shape [B, C, window_size]
                - Predicted statistics [B, C, num_stats_features]
                - Quantization loss (scalar)
                - Indices tensor [B, C]
        """
        # x: [B, C, D]
        z_q, quant_loss, indices = self.quantizer(x)
        
        recon_norm = self.decoder_shape(z_q).view(-1, self.num_sensors, self.window_size) # [B, C, W]
        pred_stats_z = self.decoder_stats(z_q).view(-1, self.num_sensors, self.num_stats_features)
        
        return recon_norm, pred_stats_z, quant_loss, indices
