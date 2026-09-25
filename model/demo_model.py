#!/usr/bin/env python
"""A compact multi-label model for the runnable reference pipeline."""

from __future__ import annotations

import torch
from torch import nn


class ReferenceMLP(nn.Module):
    """A fully connected baseline operating on the 3,387-D feature matrix."""

    def __init__(
        self,
        input_dim: int = 3387,
        hidden_dim: int = 256,
        output_dim: int = 21,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        bottleneck_dim = max(hidden_dim // 2, output_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
