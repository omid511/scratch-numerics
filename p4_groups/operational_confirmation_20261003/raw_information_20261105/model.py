"""Fixed raw-only grouped TCN used by the waveform and mask controls."""
import torch
from experiment_raw_tcn import _GroupedTCNBackbone
from mechanics.p4_margin_estimation.tcn import TemporalSummary


class RawOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _GroupedTCNBackbone()
        self.summary = TemporalSummary(window=64)
        self.projection = torch.nn.Sequential(torch.nn.Linear(96, 32), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        self.float()

    def forward(self, raw):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (384, 18):
            raise ValueError('Expected raw [B,384,18]')
        return self.head(self.projection(self.summary(self.backbone(raw.transpose(1, 2)))))[:, 0]
