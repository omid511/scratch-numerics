"""Quantile regression head on top of the TCN backbone.

Predicts per-timestep quantile estimates of the stability margin.
Uses median-centered parameterization for asymmetric uncertainty.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import TCNBackbone, TemporalSummary


SUPPORTED_QUANTILES = (0.05, 0.50, 0.95)


class QuantileMarginModel(nn.Module):
    """TCN backbone + temporal-summary quantile head.

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
        n_layers: int = 8,
        kernel: int = 3,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = SUPPORTED_QUANTILES,
        sequence_length: int = 512,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        if self.quantiles != SUPPORTED_QUANTILES:
            raise ValueError(
                f"Median-centered head supports exactly {SUPPORTED_QUANTILES}; "
                f"got {self.quantiles}"
            )
        self.tcn = TCNBackbone(n_channels, hidden_dim, n_layers, kernel, dropout, sequence_length=sequence_length)
        self.summary = TemporalSummary(window=64)
        self.head = nn.Linear(3 * hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_quantiles) — scalar margin per clip."""
        feat = self.tcn(x)                # (B, hidden, T)
        feat = self.summary(feat)          # (B, 3*hidden)
        out = self.head(feat)              # (B, 3)

        median = out[:, 0:1]
        lower_raw = out[:, 1:2]
        upper_raw = out[:, 2:3]
        lower_width = F.softplus(lower_raw)
        upper_width = F.softplus(upper_raw)
        q_lo = median - lower_width
        q_med = median
        q_hi = median + upper_width
        return torch.cat([q_lo, q_med, q_hi], dim=1)  # (B, 3)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Pinball (quantile) loss with shape validation and safety weights.

    Parameters
    ----------
    pred    : (B, n_quantiles) — scalar predictions per quantile
    target  : (B,) or (B, 1) — the true margin value
    weights : optional per-quantile weights (e.g. (2.0, 1.0, 1.0))
    """
    if pred.ndim != 2:
        raise ValueError(f"pred must be (B, Q), got {tuple(pred.shape)}")

    target = target.reshape(-1, 1)
    if target.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Batch mismatch: pred={pred.shape[0]}, target={target.shape[0]}"
        )

    q = pred.new_tensor(quantiles).reshape(1, -1)
    if q.shape[1] != pred.shape[1]:
        raise ValueError(
            f"Quantile mismatch: pred has {pred.shape[1]} outputs, "
            f"quantiles has {q.shape[1]}"
        )

    error = target - pred
    loss = torch.maximum(q * error, (q - 1.0) * error)

    if weights is not None:
        weight_tensor = pred.new_tensor(weights).reshape(1, -1)
        loss = loss * weight_tensor

    return loss.mean()


# ── Conformalized Quantile Regression (CQR) ────────────────────────────────

def fit_cqr_adjustment(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Fit CQR adjustment on calibration set.

    Returns the nonconformity quantile that widens intervals to achieve
    nominal coverage (1 - alpha).
    """
    scores = np.maximum(lower - target, target - upper)
    n = scores.size
    if n == 0:
        raise ValueError("Calibration set is empty")
    quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


def apply_cqr_adjustment(
    lower: np.ndarray,
    upper: np.ndarray,
    adjustment: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen lower/upper by the fitted CQR adjustment.

    A negative fitted adjustment (over-covering base intervals) shrinks the
    band; crossed ends (shrink larger than half the base width) collapse to
    their midpoint so width floors at zero and intervals never invert.
    """
    lo_adj = np.asarray(lower, dtype=float) - float(adjustment)
    hi_adj = np.asarray(upper, dtype=float) + float(adjustment)
    crossed = lo_adj > hi_adj
    if bool(np.any(crossed)):
        mid = 0.5 * (lo_adj + hi_adj)
        lo_adj = np.where(crossed, mid, lo_adj)
        hi_adj = np.where(crossed, mid, hi_adj)
    return lo_adj, hi_adj


# ── Safety-aware Huber loss ─────────────────────────────────────────────────

def safety_aware_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 0.05,
    false_safe_weight: float = 4.0,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Huber loss with extra penalty for predicting positive margin when true margin <= 0."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    base = F.huber_loss(pred, target, delta=delta, reduction="none")
    unsafe = (target <= 0.0).to(pred.dtype)
    false_safe_penalty = unsafe * F.softplus(pred / temperature) * temperature
    return (base + false_safe_weight * false_safe_penalty).mean()
