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

    Receptive field: 1021 timesteps with the default 8 two-convolution blocks.
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
    """Summarize temporal features via early/late means + last timestep.

    The window scales with sequence length (``min(max_window, T // 4)``) so
    short clips keep disjoint early/late views instead of duplicating the
    whole clip in both means.
    """

    def __init__(self, window: int = 64):
        super().__init__()
        self.window = window

    def forward(self, feat: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Summarize only valid prefix samples when lengths are supplied."""
        T = feat.shape[-1]
        if lengths is None:
            window = min(self.window, max(1, T // 4))
            early = feat[:, :, :window].mean(dim=2)
            late = feat[:, :, -window:].mean(dim=2)
            last = feat[:, :, -1]
        else:
            lengths = lengths.to(feat.device)
            if lengths.shape != (len(feat),) or lengths.dtype != torch.long:
                raise ValueError("lengths must be one int64 value per signal")
            if torch.any((lengths < 1) | (lengths > T)):
                raise ValueError("prefix lengths must be within the sequence")
            windows = torch.clamp(lengths // 4, min=1, max=self.window)
            cumulative = torch.nn.functional.pad(feat.cumsum(dim=2), (1, 0))
            def at(values, index):
                return values.gather(2, index[:, None, None].expand(-1, feat.shape[1], 1)).squeeze(2)
            early = at(cumulative, windows) / windows[:, None]
            late = (at(cumulative, lengths) - at(cumulative, lengths - windows)) / windows[:, None]
            last = at(feat, lengths - 1)
        return torch.cat([early, late, last], dim=1)
