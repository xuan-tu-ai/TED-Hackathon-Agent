"""Context embeddings for robot identity and paint program (MaskFormer V7).

Provides learnable embeddings that encode robot-specific baseline characteristics
and car-model-specific paint cycle patterns.  The :class:`ContextEmbedding`
composite module returns a ``[B, 1, D]`` tensor broadcast-ready to add to all
84 tokens — representing a uniform contextual shift per file.

The design supports optional context (null → zero contribution) for datasets
that do not yet provide robot/program metadata.
"""

import torch
import torch.nn as nn
from typing import Optional


class RobotEmbedding(nn.Module):
    """Learnable embedding for robot identity.

    Encodes robot-specific baseline characteristics. Each of the 47 paint-shop
    robots has a distinct embedding that shifts all 84 tokens uniformly.

    Args:
        num_robots: Number of distinct robots (default 47).
        d_model:    Model dimension (default 128).
    """

    def __init__(self, num_robots: int = 47, d_model: int = 128) -> None:
        super().__init__()
        self.num_robots = num_robots
        self.d_model = d_model
        self.embedding = nn.Embedding(num_robots, d_model)

    def forward(self, robot_idx: torch.Tensor) -> torch.Tensor:
        """Look up robot embedding.

        Args:
            robot_idx: LongTensor [B] of robot IDs

        Returns:
            emb: [B, D]
        """
        return self.embedding(robot_idx)

    def extra_repr(self) -> str:
        return f"num_robots={self.num_robots}, d_model={self.d_model}"


class ProgramEmbedding(nn.Module):
    """Learnable embedding for paint program identity.

    Encodes car-model-specific paint cycle patterns. Each program produces
    a distinct pressure/flow profile that shifts all 84 tokens uniformly.

    Args:
        num_programs: Number of distinct programs (default 16).
        d_model:      Model dimension (default 128).
    """

    def __init__(self, num_programs: int = 16, d_model: int = 128) -> None:
        super().__init__()
        self.num_programs = num_programs
        self.d_model = d_model
        self.embedding = nn.Embedding(num_programs, d_model)

    def forward(self, program_idx: torch.Tensor) -> torch.Tensor:
        """Look up program embedding.

        Args:
            program_idx: LongTensor [B] of program IDs

        Returns:
            emb: [B, D]
        """
        return self.embedding(program_idx)

    def extra_repr(self) -> str:
        return f"num_programs={self.num_programs}, d_model={self.d_model}"


class ContextEmbedding(nn.Module):
    """Combined robot + program context embedding, broadcast-ready.

    Returns [B, 1, D] that can be added to all 84 tokens uniformly.
    Supports optional context: if robot_idx or program_idx is None, the
    corresponding embedding contribution is zero (for use with datasets
    that do not provide context metadata, e.g. RawWindowDataset).

    Args:
        num_robots:   Number of distinct robots (default 47).
        num_programs: Number of distinct programs (default 16).
        d_model:      Model dimension (default 128).
    """

    def __init__(
        self,
        num_robots: int = 47,
        num_programs: int = 16,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model
        self.robot_emb = RobotEmbedding(num_robots=num_robots, d_model=d_model)
        self.program_emb = ProgramEmbedding(num_programs=num_programs, d_model=d_model)

    def forward(
        self,
        robot_idx: Optional[torch.Tensor],
        program_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined context embedding broadcast-ready for all tokens.

        Args:
            robot_idx:   LongTensor [B] of robot IDs, or None (→ zero contribution).
            program_idx: LongTensor [B] of program IDs, or None (→ zero contribution).

        Returns:
            context: FloatTensor [B, 1, D] — add to tokens [B, N, D] to broadcast
        """
        # Determine batch size and device from whichever index is available
        if robot_idx is not None:
            B = robot_idx.shape[0]
            device = robot_idx.device
        elif program_idx is not None:
            B = program_idx.shape[0]
            device = program_idx.device
        else:
            raise ValueError("At least one of robot_idx or program_idx must be provided.")

        robot_part = (
            self.robot_emb(robot_idx)    # [B, D]
            if robot_idx is not None
            else torch.zeros(B, self.d_model, device=device)
        )
        program_part = (
            self.program_emb(program_idx)  # [B, D]
            if program_idx is not None
            else torch.zeros(B, self.d_model, device=device)
        )

        context = (robot_part + program_part).unsqueeze(1)  # [B, 1, D]
        return context

    def extra_repr(self) -> str:
        return f"num_robots={self.num_robots}, num_programs={self.num_programs}, d_model={self.d_model}"
