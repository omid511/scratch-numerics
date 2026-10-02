"""Measured-prefix GRU/TCN with the existing P4 modal/spatial representation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .tcn import TCNBackbone, TemporalSummary
from .quantile_head import SUPPORTED_QUANTILES


class TemporalMarginModel(nn.Module):
    """Ordered quantiles from an unsaturated prefix and standardized side features.

    Signals: (B,16,T), lengths: (B,) strictly positive valid-prefix lengths,
    side: (B,175) = 174 measured features plus log reported dt. Empty prefixes
    are represented by a single zero sample; availability stays in the sidecar.
    No solver modes, velocity, design identity or target enters the model.
    """
    def __init__(self, kind, n_channels=16, n_features=175, sequence_length=512):
        super().__init__()
        self.kind = kind
        self.quantiles = SUPPORTED_QUANTILES
        hidden = 64 if kind == 'gru' else 32
        if kind == 'gru':
            self.backbone = nn.GRU(n_channels, hidden, batch_first=True)
        elif kind == 'tcn':
            self.backbone = TCNBackbone(n_channels, hidden, sequence_length=sequence_length)
        else:
            raise ValueError(f'Unknown temporal model: {kind}')
        self.summary = TemporalSummary(window=64)
        self.side = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * hidden + 32, 32), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(32, 3))
        with torch.no_grad():
            self.head[-1].bias[1:].fill_(float(torch.log(torch.expm1(torch.tensor(.1)))))

    def forward(self, signals, lengths, side):
        if self.kind == 'gru':
            # Dense recurrence avoids packed-sequence scatter allocations on CPU.
            # Prefix-indexed summaries make padded tails irrelevant to outputs.
            states, _ = self.backbone(signals.transpose(1, 2))
            states = states.transpose(1, 2)
        else:
            states = self.backbone(signals)
        summary = self.summary(states, lengths)
        out = self.head(torch.cat((summary, self.side(side)), dim=1))
        median = out[:, :1]
        return torch.cat((median - F.softplus(out[:, 1:2]), median,
                          median + F.softplus(out[:, 2:3])), dim=1)
