"""Torch losses for P2 inverse damage training."""
from __future__ import annotations

import torch


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Quantile (pinball) loss, one entry per quantile level.

    For prediction p, target y and quantile level τ::

        L_τ(p, y) = max(τ·(y − p), (τ − 1)·(y − p))

    Averaged over all non-tau dimensions; returns shape ``(len(tau),)``,
    a per-τ vector suitable for weighting or summing by the caller.
    """
    if tau.ndim != 1:
        raise ValueError("tau must be a 1-D vector of quantile levels")
    error = target - pred                                  # broadcast tau on last axis
    loss = torch.maximum(tau * error, (tau - 1.0) * error)  # (..., n_tau)
    dims = tuple(range(loss.ndim - 1))
    return loss.mean(dim=dims) if dims else loss


def field_total_variation(field: torch.Tensor) -> torch.Tensor:
    """Total-variation penalty on a damage field: mean |∇ field|.

    Uses forward differences along the two trailing spatial dims; returns
    scalar mean absolute gradient magnitude (anisotropic TV).
    """
    if field.dim() < 2:
        raise ValueError("field must have at least 2 spatial dims")
    dy = (field[..., 1:, :] - field[..., :-1, :]).abs()
    dx = (field[..., :, 1:] - field[..., :, :-1]).abs()
    return (dy.sum() + dx.sum()) / (dy.numel() + dx.numel())


def gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with heteroscedastic variance.

        NLL = 0.5 · (log σ² + (y − μ)² / σ²)

    up to the constant ½log(2π). Scalar mean over all elements.
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must share shape")
    return 0.5 * (logvar + (target - mu) ** 2 / torch.exp(logvar)).mean()
