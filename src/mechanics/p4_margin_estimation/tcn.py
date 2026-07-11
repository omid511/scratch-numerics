"""Temporal Convolutional Network backbone for sequence-to-sequence features.

Causal convolutions only — no future leakage.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Conv1d with causal padding (left-only)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Single TCN block: CausalConv → ReLU → Dropout × 2 + residual.

    No normalization — GroupNorm/LayerNorm over the time axis violates
    strict causality (truncating input changes stats for all timesteps).
    Small networks work fine without it.
    """

    def __init__(self, n_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual


class TCNBackbone(nn.Module):
    """Stacked causal TCN blocks.

    Input:  (batch, n_channels, seq_len)
    Output: (batch, hidden_dim, seq_len)

    Receptive field: 61 timesteps with default config (4 layers, kernel=3, dilations 1,2,4,8).
    Each TCNBlock has 2 causal convolutions with residual connection.
    """

    def __init__(
        self,
        n_channels: int = 8,
        hidden_dim: int = 32,
        n_layers: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Receptive field: sum of (kernel-1)*dilation for each conv, +1 for initial
        # With 4 layers, kernel=3, dilations=[1,2,4,8]: RF = 61 timesteps
        self.input_proj = nn.Conv1d(n_channels, hidden_dim, 1)
        blocks = [
            TCNBlock(hidden_dim, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_channels, seq_len) → (batch, hidden_dim, seq_len)"""
        return self.blocks(self.input_proj(x))
