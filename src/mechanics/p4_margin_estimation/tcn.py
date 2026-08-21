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


class ChannelLayerNorm(nn.Module):
    """Normalizes across channels at each time step (preserves causality)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        return x.transpose(1, 2)  # (B, C, T)


class TCNBlock(nn.Module):
    """Single TCN block: CausalConv → ChannelLayerNorm → ReLU → Dropout × 2 + residual."""

    def __init__(self, n_ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(n_ch, n_ch, kernel, dilation)
        self.norm1 = ChannelLayerNorm(n_ch)
        self.conv2 = CausalConv1d(n_ch, n_ch, kernel, dilation)
        self.norm2 = ChannelLayerNorm(n_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = torch.relu(self.norm2(self.conv2(y)))
        y = self.dropout(y)
        return residual + y


class TCNBackbone(nn.Module):
    """Stacked causal TCN blocks.

    Input:  (batch, n_channels, seq_len)
    Output: (batch, hidden_dim, seq_len)

    Receptive field: 511 timesteps with default config (8 layers, kernel=3, dilations 1..128).
    Each TCNBlock has 2 causal convolutions with residual connection.
    """

    def __init__(
        self,
        n_channels: int = 8,
        hidden_dim: int = 32,
        n_layers: int = 8,
        kernel: int = 3,
        dropout: float = 0.1,
        sequence_length: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_channels, hidden_dim, 1)
        self.blocks = nn.Sequential(
            *[TCNBlock(hidden_dim, kernel, dilation=2 ** i, dropout=dropout)
              for i in range(n_layers)]
        )
        # Receptive field: 1 + 2 * (kernel - 1) * sum(2**i for i in range(n_layers))
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(2 ** i for i in range(n_layers))
        if self.receptive_field < sequence_length:
            raise ValueError(
                f"Receptive field {self.receptive_field} < sequence_length {sequence_length}; "
                f"increase n_layers or kernel"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_channels, seq_len) → (batch, hidden_dim, seq_len)"""
        return self.blocks(self.input_proj(x))


class TemporalSummary(nn.Module):
    """Summarize temporal features via early/late means + last timestep."""

    def __init__(self, window: int = 64):
        super().__init__()
        self.window = window

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, T) → (B, 3*C)"""
        window = min(self.window, feat.shape[-1])
        early = feat[:, :, :window].mean(dim=2)
        late = feat[:, :, -window:].mean(dim=2)
        last = feat[:, :, -1]
        return torch.cat([early, late, last], dim=1)
