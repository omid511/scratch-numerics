"""Training loop and coverage evaluation for the margin model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .tcn import TCNBackbone
from .quantile_head import QuantileMarginModel, pinball_loss, SUPPORTED_QUANTILES


def safety_aware_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.05,
    false_safe_weight: float = 4.0,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Huber loss that penalizes predicting positive margin on unsafe samples."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    base = F.huber_loss(pred, target, delta=delta, reduction="none")

    unsafe = (target <= 0.0).to(pred.dtype)
    false_safe_penalty = unsafe * F.softplus(pred / temperature) * temperature

    return (base + false_safe_weight * false_safe_penalty).mean()


class HuberMarginModel(nn.Module):
    """TCN backbone + single-output Huber regression."""

    def __init__(self, n_channels=8, hidden_dim=32, n_layers=4, kernel=3, dropout=0.1):
        super().__init__()
        self.tcn = TCNBackbone(n_channels, hidden_dim, n_layers, kernel, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.tcn(x)           # (B, hidden, T)
        feat = feat.mean(dim=2)      # (B, hidden)
        return self.head(feat)       # (B, 1)


def _build_tensors(clips):
    """Build (X, y) tensors from clips, filtering NaN."""
    X_list, y_list = [], []
    for c in clips:
        if np.isnan(c.margin):
            continue
        sig = torch.tensor(c.sensor_signals, dtype=torch.float32)
        if torch.isnan(sig).any():
            continue
        X_list.append(sig)
        y_list.append(c.margin)
    if not X_list:
        raise ValueError("No valid clips (all margins NaN)")
    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    return X, y


def _grouped_3way_split(clips, val_split, test_split, seed):
    if val_split <= 0.0 or test_split <= 0.0:
        raise ValueError("val_split and test_split must both be positive")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be less than 1")

    group_ids = np.asarray([c.design_id for c in clips])
    unique_groups = np.asarray(sorted(set(group_ids.tolist())))

    if unique_groups.size < 3:
        raise ValueError(
            "At least three independent design groups are required for splitting"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_test = max(1, int(round(unique_groups.size * test_split)))
    n_val = max(1, int(round(unique_groups.size * val_split)))

    if n_test + n_val >= unique_groups.size:
        raise ValueError("Not enough design groups")

    test_groups = set(unique_groups[:n_test].tolist())
    val_groups = set(unique_groups[n_test:n_test + n_val].tolist())
    train_groups = set(unique_groups[n_test + n_val:].tolist())

    train_idx = np.flatnonzero(np.fromiter((g in train_groups for g in group_ids), dtype=bool))
    val_idx = np.flatnonzero(np.fromiter((g in val_groups for g in group_ids), dtype=bool))
    test_idx = np.flatnonzero(np.fromiter((g in test_groups for g in group_ids), dtype=bool))

    return train_idx, val_idx, test_idx


def _split(clips, X, y, val_split, test_split, seed):
    """Dispatch to grouped 3-way split."""
    valid_clips = [c for c in clips if not np.isnan(c.margin) and not torch.isnan(torch.tensor(c.sensor_signals, dtype=torch.float32)).any()]
    train_idx, val_idx, test_idx = _grouped_3way_split(valid_clips, val_split, test_split, seed)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Split failed: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train(
    clips: list,
    n_channels: int = 8,
    hidden_dim: int = 32,
    n_layers: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    device: str = "cpu",
    seed: int = 0,
    velocities: list[float] | None = None,
    train_clips: list | None = None,
    val_clips: list | None = None,
    test_clips: list | None = None,
) -> tuple[QuantileMarginModel, dict, list, list]:
    """Train margin model on transient clips.

    Each clip has .sensor_signals (n_sensors, T) and .margin (float).
    Target margin is broadcast as constant across time.

    Returns (model, history, test_clips, test_vels).
    """
    torch.manual_seed(seed)

    if train_clips is not None and val_clips is not None and test_clips is not None:
        X_train, y_train = _build_tensors(train_clips)
        X_val, y_val = _build_tensors(val_clips)
        X_test, y_test = _build_tensors(test_clips)
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_vels = [c.velocity for c in test_clips]
    else:
        X, y = _build_tensors(clips)
        train_idx, val_idx, test_idx = _split(clips, X, y, val_split, test_split, seed)
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        val_ds = TensorDataset(X[val_idx], y[val_idx])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        X_test = X[test_idx]
        test_clips = [clips[i] for i in test_idx]
        test_vels = [velocities[i] for i in test_idx] if velocities else []

    seq_len = X_train.shape[-1] if train_clips is not None else X.shape[-1]
    model = QuantileMarginModel(n_channels, hidden_dim, n_layers, sequence_length=seq_len).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

    quantiles = model.quantiles

    SAFETY_WEIGHTS = (2.0, 1.0, 1.0)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)  # (B, n_q) — scalar per quantile
            target = yb       # (B,)
            loss = pinball_loss(pred, target, quantiles, weights=SAFETY_WEIGHTS)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                target = yb
                val_loss += pinball_loss(pred, target, quantiles, weights=SAFETY_WEIGHTS).item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, test_clips, test_vels


def train_huber(
    clips: list,
    n_channels: int = 8,
    hidden_dim: int = 32,
    n_layers: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    device: str = "cpu",
    seed: int = 0,
    velocities: list[float] | None = None,
    train_clips: list | None = None,
    val_clips: list | None = None,
    test_clips: list | None = None,
) -> tuple[HuberMarginModel, dict, list, list]:
    """Train Huber point-prediction baseline.

    Returns (model, history, test_clips, test_vels).
    """
    torch.manual_seed(seed)

    if train_clips is not None and val_clips is not None and test_clips is not None:
        X_train, y_train = _build_tensors(train_clips)
        X_val, y_val = _build_tensors(val_clips)
        X_test, y_test = _build_tensors(test_clips)
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_vels = [c.velocity for c in test_clips]
    else:
        X, y = _build_tensors(clips)
        train_idx, val_idx, test_idx = _split(clips, X, y, val_split, test_split, seed)
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        val_ds = TensorDataset(X[val_idx], y[val_idx])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        X_test = X[test_idx]
        test_clips = [clips[i] for i in test_idx]
        test_vels = [velocities[i] for i in test_idx] if velocities else []

    model = HuberMarginModel(n_channels, hidden_dim, n_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    criterion = nn.HuberLoss(delta=0.1)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)  # (B,)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += criterion(pred, yb).item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, test_clips, test_vels


def train_median(
    clips: list,
    n_channels: int = 8,
    hidden_dim: int = 32,
    n_layers: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    device: str = "cpu",
    seed: int = 0,
    velocities: list[float] | None = None,
    train_clips: list | None = None,
    val_clips: list | None = None,
    test_clips: list | None = None,
) -> tuple[QuantileMarginModel, dict, list, list]:
    """Train median-only quantile model (single quantile).

    Returns (model, history, test_clips, test_vels).
    """
    torch.manual_seed(seed)

    if train_clips is not None and val_clips is not None and test_clips is not None:
        X_train, y_train = _build_tensors(train_clips)
        X_val, y_val = _build_tensors(val_clips)
        X_test, y_test = _build_tensors(test_clips)
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_vels = [c.velocity for c in test_clips]
    else:
        X, y = _build_tensors(clips)
        train_idx, val_idx, test_idx = _split(clips, X, y, val_split, test_split, seed)
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        val_ds = TensorDataset(X[val_idx], y[val_idx])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        X_test = X[test_idx]
        test_clips = [clips[i] for i in test_idx]
        test_vels = [velocities[i] for i in test_idx] if velocities else []

    seq_len = X_train.shape[-1] if train_clips is not None else X.shape[-1]
    model = QuantileMarginModel(n_channels, hidden_dim, n_layers, sequence_length=seq_len).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)  # (B, 3)
            target = yb       # (B,)
            loss = pinball_loss(pred, target, model.quantiles)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                target = yb
                val_loss += pinball_loss(pred, target, model.quantiles).item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, test_clips, test_vels


def evaluate_coverage(
    model: QuantileMarginModel,
    clips: list,
    device: str = "cpu",
    velocities: list[float] | None = None,
) -> dict:
    """Compute coverage stats on a clip list.

    Returns dict with coverage per quantile, mean interval width, MAE.
    Optionally logs per-velocity diagnostics when velocities is provided.
    """
    model.eval()
    quantiles = model.quantiles

    X_list, y_list, valid_vels = [], [], []
    for i, c in enumerate(clips):
        if np.isnan(c.margin):
            continue
        sig = torch.tensor(c.sensor_signals, dtype=torch.float32)
        if torch.isnan(sig).any():
            continue
        X_list.append(sig)
        y_list.append(c.margin)
        if velocities is not None and i < len(velocities):
            valid_vels.append(velocities[i])

    if not X_list:
        return {}

    X = torch.stack(X_list).to(device)
    y_true = torch.tensor(y_list, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(X)  # (B, n_q) — scalar per quantile

    pred_last = pred  # (B, n_q) — already scalar, no timestep dimension

    # Coverage: P(y <= q_tau) per quantile
    coverage = {}
    for i, tau in enumerate(quantiles):
        if i == 0:
            lo = pred_last[:, i]
        elif i == len(quantiles) - 1:
            hi = pred_last[:, i]
            in_interval = (y_true >= lo) & (y_true <= hi)
            coverage[f"interval_{quantiles[0]:.2f}_{tau:.2f}"] = in_interval.float().mean().item()
        coverage[f"q{tau:.2f}"] = (y_true <= pred_last[:, i]).float().mean().item()

    # MAE (median)
    med_idx = len(quantiles) // 2
    mae = (pred_last[:, med_idx] - y_true).abs().mean().item()

    # Mean interval width (only when we have a low/high pair)
    if len(quantiles) >= 2:
        width = (hi - lo).mean().item()
    else:
        width = float("nan")

    result = {
        "coverage": coverage,
        "mae": mae,
        "mean_interval_width": width,
    }

    # Per-velocity diagnostics
    if valid_vels:
        unique_vels = sorted(set(valid_vels))
        per_vel = {}
        for v in unique_vels:
            mask = torch.tensor([vel == v for vel in valid_vels], device=device)
            if mask.sum() == 0:
                continue
            y_v = y_true[mask]
            pred_v = pred_last[mask]
            med_v = (pred_v[:, med_idx] - y_v).abs().mean().item()
            cov_v = {}
            for qi, tau in enumerate(quantiles):
                cov_v[f"q{tau:.2f}"] = (y_v <= pred_v[:, qi]).float().mean().item()
            per_vel[v] = {"mae": med_v, "coverage": cov_v, "n_clips": int(mask.sum().item())}
        result["per_velocity"] = per_vel

    return result
