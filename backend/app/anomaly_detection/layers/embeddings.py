"""Embedding layers — context, positional, and metadata embeddings.

Consolidates embedding-related modules previously split across
``context_embedding.py`` and ``feature_extraction.py``:

*   :class:`ContextEmbedding` — unified robot + program context embedding
    (supports both V7 "sum" and V1-V6 "concat" merge modes).
*   :class:`RelativePositionEncoder` — MLP encoder for relative window
    position (V1-V6 legacy).
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


class ContextEmbedding(nn.Module):
    """Combined robot + program context embedding with configurable merge mode.

    Supports two merge strategies controlled by *merge_mode*:

    ``"sum"`` (V7 default)
        Uses a full ``d_model``-dimensional embedding for both robot and
        program, then **sums** them.  Returns ``[B, 1, D]`` broadcast-ready
        to add to all tokens.

        If ``robot_idx`` or ``program_idx`` is ``None``, the corresponding
        contribution is zero (supports optional context).

    ``"concat"`` (V1-V6 legacy)
        Uses ``d_model // 2``-dimensional embeddings for each, **concatenates**
        them, then projects back to ``d_model`` via ``Linear + SiLU + LayerNorm``.

    Args:
        num_robots:   Number of distinct robot identities.
        num_programs: Number of distinct program identities.
        d_model:      Model dimension.
        merge_mode:   ``"sum"`` (V7, default) or ``"concat"`` (V1-V6 legacy).
    """

    def __init__(
        self,
        num_robots: int,
        num_programs: int,
        d_model: int,
        merge_mode: Literal["sum", "concat"] = "sum",
    ) -> None:
        super().__init__()
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model
        self.merge_mode = merge_mode

        if merge_mode == "sum":
            # V7: full-D embeddings, independent tables
            self.robot_emb = nn.Embedding(num_robots, d_model)
            self.program_emb = nn.Embedding(num_programs, d_model)
        else:
            # V1-V6: half-D embeddings, concat + project
            half = d_model // 2
            self.robot_emb = nn.Embedding(num_robots, half)
            self.program_emb = nn.Embedding(num_programs, half)
            self.projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )

    def forward(
        self,
        robot_ids: Optional[torch.Tensor],
        program_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined context embedding.

        Args:
            robot_ids:   LongTensor ``[B]`` of robot IDs, or ``None``.
            program_ids: LongTensor ``[B]`` of program IDs, or ``None``.

        Returns:
            context: FloatTensor ``[B, 1, D]`` (merge_mode="sum") or
                     ``[B, 1, D]`` (merge_mode="concat", unsqueezed).
        """
        if self.merge_mode == "sum":
            return self._forward_sum(robot_ids, program_ids)
        else:
            return self._forward_concat(robot_ids, program_ids)

    def _forward_sum(
        self,
        robot_ids: Optional[torch.Tensor],
        program_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """V7-style: sum full-D embeddings."""
        if robot_ids is not None:
            B, device = robot_ids.shape[0], robot_ids.device
        elif program_ids is not None:
            B, device = program_ids.shape[0], program_ids.device
        else:
            raise ValueError("At least one of robot_ids or program_ids must be provided.")

        robot_part = (
            self.robot_emb(robot_ids)  # [B, D]
            if robot_ids is not None
            else torch.zeros(B, self.d_model, device=device)
        )
        program_part = (
            self.program_emb(program_ids)  # [B, D]
            if program_ids is not None
            else torch.zeros(B, self.d_model, device=device)
        )
        return (robot_part + program_part).unsqueeze(1)  # [B, 1, D]

    def _forward_concat(
        self,
        robot_ids: Optional[torch.Tensor],
        program_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """V1-V6-style: concat half-D embeddings and project."""
        robot_ids = torch.zeros(1, dtype=torch.long) if robot_ids is None else robot_ids
        program_ids = torch.zeros(1, dtype=torch.long) if program_ids is None else program_ids

        r = self.robot_emb(robot_ids)      # [B, D/2]
        p = self.program_emb(program_ids)  # [B, D/2]
        x = torch.cat([r, p], dim=-1)      # [B, D]
        x = self.projector(x)              # [B, D]
        return x.unsqueeze(1)              # [B, 1, D]

    def extra_repr(self) -> str:
        return (
            f"num_robots={self.num_robots}, num_programs={self.num_programs}, "
            f"d_model={self.d_model}, merge_mode={self.merge_mode!r}"
        )


class RelativePositionEncoder(nn.Module):
    """Encodes the relative position of a window within a file.

    Given a scalar position (normalised to ``[-1, 1]``), produces a dense
    embedding via a small MLP (Linear -> SiLU -> Linear + LayerNorm).

    Args:
        d_model: Output feature dimension (D).
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
        """Encode relative position.

        Args:
            x: Position scalar ``[B, 1]`` in range ``[-1, 1]``.

        Returns:
            Position embedding ``[B, 1, D]``.
        """
        out = self.encoder(x)
        return out.unsqueeze(1)
