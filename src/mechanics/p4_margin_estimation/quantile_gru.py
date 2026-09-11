"""Quantile GRU for P4 aeroelastic margin estimation.

Phase-1 model slice: a GRU backbone with an ordered 3-quantile head, plus a
same-state paired-consistency variant that pulls pair medians together
without constraining interval widths.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baselines import GRUMarginModel
from .quantile_head import SUPPORTED_QUANTILES, pinball_loss

QUANTILES = SUPPORTED_QUANTILES


class QuantileGRUModel(nn.Module):
    """GRU backbone (final hidden state) + ordered 3-quantile head.

    Input:  (B, n_channels, T) — full channels, DR mask channels included.
    Output: (B, 3) — [q05, q50, q95] with q05 <= q50 <= q95 by construction:
        q50 = median (direct); q05 = median - softplus(w_lo);
        q95 = median + softplus(w_hi).
    """

    def __init__(self, n_channels: int = 16, hidden_dim: int = 86):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.quantiles = SUPPORTED_QUANTILES
        # Reuse the shared GRU backbone (not its point-prediction head).
        self.gru = GRUMarginModel(n_channels, hidden_dim).gru
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)
        _, hidden = self.gru(x)
        out = self.head(hidden[-1])  # (B, 3)
        median = out[:, 0:1]
        q_lo = median - F.softplus(out[:, 1:2])
        q_hi = median + F.softplus(out[:, 2:3])
        return torch.cat([q_lo, median, q_hi], dim=1)


def _valid_clips(clips):
    """Clips with finite margin and finite signals (mirrors _build_tensors)."""
    return [c for c in clips if _is_valid_clip(c)]


def _is_valid_clip(c) -> bool:
    if not np.isfinite(c.margin):
        return False
    try:
        sig = torch.tensor(c.sensor_signals, dtype=torch.float32)
    except Exception:
        return False
    return bool(torch.isfinite(sig).all().item())


def _stack(clips, n_channels: int):
    X_list = [torch.tensor(np.asarray(c.sensor_signals), dtype=torch.float32) for c in clips]
    X = torch.stack(X_list)
    if X.shape[1] != n_channels:
        raise ValueError(f"Expected {n_channels} channels, got {X.shape[1]}")
    return X, torch.tensor([c.margin for c in clips], dtype=torch.float32)


def _checked_splits(clips, val_clips):
    """Explicit caller-provided train/val tensors; design overlap raises."""
    train_valid = _valid_clips(clips)
    if not train_valid:
        raise ValueError("No valid clips (all margins NaN or non-finite signals)")
    if val_clips is None:
        raise ValueError("Explicit val_clips is required (no internal val split)")
    val_valid = _valid_clips(val_clips)
    if not val_valid:
        raise ValueError("No valid val clips (all margins NaN or non-finite signals)")
    overlap = {c.design_id for c in train_valid} & {c.design_id for c in val_valid}
    if overlap:
        raise ValueError(f"Train/val design overlap: {sorted(overlap)[:5]}")
    return train_valid, val_valid


def _run_epochs(model, train_dl, val_dl, epochs, lr, quantile_weights, on_epoch, extra_loss=None):
    """Shared Adam + cosine + best-val-restore loop; extra_loss adds terms."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_dl:
            batch = [t.to(device) for t in batch]
            loss = extra_loss(model, batch) if extra_loss is not None else _pinball_batch(model, batch, quantile_weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += pinball_loss(model(xb), yb, QUANTILES, weights=quantile_weights).item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(epoch, model, history)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _pinball_batch(model, batch, quantile_weights):
    xb, yb = batch
    return pinball_loss(model(xb), yb, QUANTILES, weights=quantile_weights)


def train_quantile_gru(
    clips: list,
    *,
    n_channels: int = 16,
    hidden_dim: int = 86,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
    val_clips: list | None = None,
    quantile_weights: tuple[float, ...] = (2.0, 1.0, 1.0),
    on_epoch=None,
) -> tuple[QuantileGRUModel, dict]:
    """Train the quantile GRU with pinball loss.

    ``val_clips`` is required: callers pass explicit validation clips from
    disjoint designs (overlap raises ``ValueError``); no val split is ever
    carved from the training clips. Returns (model, history).
    """
    torch.manual_seed(seed)
    train_valid, val_valid = _checked_splits(clips, val_clips)
    X, y = _stack(train_valid, n_channels)
    Xv, yv = _stack(val_valid, n_channels)
    train_dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)
    model = QuantileGRUModel(n_channels, hidden_dim).to(device)
    history = _run_epochs(model, train_dl, val_dl, epochs, lr, quantile_weights, on_epoch)
    return model, history


def build_state_pairs(clips) -> list:
    """Group clips by (design_id, velocity) with exact float equality.

    Emits disjoint pairs per group (members zipped pairwise in encounter
    order; an odd leftover is dropped). Singleton groups are dropped. Pair
    members share nominal design + velocity but differ in excitation or
    perturbation realization.
    """
    groups: dict = {}
    for c in clips:
        v = getattr(c, "velocity", None)
        groups.setdefault((c.design_id, None if v is None else float(v)), []).append(c)
    return [
        (members[i], members[i + 1])
        for members in groups.values()
        for i in range(0, len(members) - 1, 2)
    ]


def train_quantile_gru_consistency(
    clips: list,
    *,
    n_channels: int = 16,
    hidden_dim: int = 86,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
    val_clips: list | None = None,
    quantile_weights: tuple[float, ...] = (2.0, 1.0, 1.0),
    pairs: list | None = None,
    lambda_cons: float = 0.1,
    on_epoch=None,
) -> tuple[QuantileGRUModel, dict]:
    """Train with supervised pinball on both pair members plus a median-gap penalty.

    Loss per pair batch = mean pinball over both members
    + ``lambda_cons`` * mean|median_a - median_b|. Interval widths are never
    penalized. Validation uses plain pinball (no consistency term).

    Batching draws from the pair list (``pairs``, or ``build_state_pairs``
    over the valid training clips when None): each epoch visits every pair
    once, so each paired clip contributes one supervised term per epoch —
    the same per-example, per-epoch supervised budget as
    :func:`train_quantile_gru` when every clip pairs up (unpaired odd
    leftovers and singletons are unseen here). ``batch_size`` counts pairs
    per batch. Returns (model, history).
    """
    torch.manual_seed(seed)
    train_valid, val_valid = _checked_splits(clips, val_clips)
    if pairs is None:
        pairs = build_state_pairs(train_valid)
    else:
        keep = {id(c) for c in train_valid}
        pairs = [(a, b) for (a, b) in pairs if id(a) in keep and id(b) in keep]
    if not pairs:
        raise ValueError("No valid same-state pairs for consistency training")
    Xa, ya = _stack([a for (a, _) in pairs], n_channels)
    Xb, yb = _stack([b for (_, b) in pairs], n_channels)
    Xv, yv = _stack(val_valid, n_channels)
    pair_dl = DataLoader(TensorDataset(Xa, Xb, ya, yb), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)
    model = QuantileGRUModel(n_channels, hidden_dim).to(device)

    def _pair_loss(model, batch):
        xa, xb, ya_, yb_ = batch
        pa, pb = model(xa), model(xb)
        sup = 0.5 * (
            pinball_loss(pa, ya_, QUANTILES, weights=quantile_weights)
            + pinball_loss(pb, yb_, QUANTILES, weights=quantile_weights)
        )
        cons = torch.mean(torch.abs(pa[:, 1] - pb[:, 1]))
        return sup + lambda_cons * cons

    history = _run_epochs(model, pair_dl, val_dl, epochs, lr, quantile_weights, on_epoch, extra_loss=_pair_loss)
    return model, history
