"""Quantile regression head on top of the TCN backbone.

Predicts per-timestep quantile estimates of the stability margin.
Uses median-centered parameterization for asymmetric uncertainty.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import TCNBackbone


DEFAULT_QUANTILES = (0.05, 0.50, 0.95)


class QuantileMarginModel(nn.Module):
    """TCN backbone + global-average-pooled quantile head.

    Input:  (batch, n_channels, seq_len)
    Output: (batch, n_quantiles)  — scalar margin prediction per clip

    Quantile parameterization:
        q0.50 = median  (direct)
        q0.05 = median - softplus(lower_raw)
        q0.95 = median + softplus(upper_raw)
    """

    def __init__(
        self,
        n_channels: int = 8,
        hidden_dim: int = 32,
        n_layers: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.tcn = TCNBackbone(n_channels, hidden_dim, n_layers, kernel, dropout)
        n_q = len(quantiles)
        self._n_q = n_q
        if n_q == 3 and quantiles[1] == 0.5:
            # Median-centered parameterization: median + lower_width + upper_width
            self.head = nn.Linear(hidden_dim, 3)
            self._mode = 'median_centered'
        else:
            # General quantile regression: one output per quantile
            self.head = nn.Linear(hidden_dim, n_q)
            self._mode = 'direct'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_quantiles) — scalar margin per clip."""
        feat = self.tcn(x)                # (B, hidden, T)
        feat = feat.mean(dim=2)            # (B, hidden) — global average pooling
        out = self.head(feat)              # (B, n_head)

        if self._mode == 'median_centered':
            median = out[:, 0:1]
            lower_raw = out[:, 1:2]
            upper_raw = out[:, 2:3]
            lower_width = torch.nn.functional.softplus(lower_raw)
            upper_width = torch.nn.functional.softplus(upper_raw)
            q_lo = median - lower_width
            q_med = median
            q_hi = median + upper_width
            return torch.cat([q_lo, q_med, q_hi], dim=1)  # (B, 3)
        else:
            return out


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Pinball (quantile) loss.

    Parameters
    ----------
    pred    : (B, n_quantiles) — scalar predictions per quantile
    target  : (B,) or (B, 1) — the true margin value
    """
    if target.dim() == 1:
        target = target.unsqueeze(1)  # (B, 1)

    losses = []
    for i, tau in enumerate(quantiles):
        err = target - pred[:, i : i + 1]  # (B, 1)
        losses.append(torch.max(tau * err, (tau - 1) * err))
    return torch.stack(losses).mean()
