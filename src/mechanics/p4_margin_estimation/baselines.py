"""Baseline models for P4 aeroelastic margin estimation."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@runtime_checkable
class MarginPredictor(Protocol):
    def predict(self, clips) -> np.ndarray: ...


class ConstantMedianBaseline:
    """Predicts training-set median margin for every clip."""

    def __init__(self):
        self.median_ = None

    def fit(self, clips):
        self.median_ = np.median([c.margin for c in clips if not np.isnan(c.margin)])

    def predict(self, clips):
        return np.full(len(clips), self.median_)


class VelocityOracle:
    """Perfect predictor when velocity and u_crit are known. Upper bound."""

    def predict(self, clips):
        return np.array([
            (c.u_crit - c.velocity) / c.u_crit
            if c.u_crit else np.nan
            for c in clips
        ])


class VelocityLinearBaseline:
    """Linear regression: margin = a * velocity + b."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, clips):
        v = np.array([c.velocity for c in clips])
        m = np.array([c.margin for c in clips])
        A = np.column_stack([v, np.ones_like(v)])
        result = np.linalg.lstsq(A, m, rcond=None)
        self.coef_ = result[0][0]
        self.intercept_ = result[0][1]

    def predict(self, clips):
        v = np.array([c.velocity for c in clips])
        return self.coef_ * v + self.intercept_


class PhysicsFeatureRidge:
    """Ridge regression on physics-inspired envelope features."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = None

    def _extract_features(self, signals):
        """Extract envelope features from (n_sensors, n_t) signals."""
        n_sensors, n_t = signals.shape
        eps = 1e-12
        t = np.arange(n_t)

        # Spatial RMS envelope A(t) = sqrt(mean_s(x_s(t)^2))
        A = np.sqrt(np.mean(signals ** 2, axis=0) + eps)  # (n_t,)

        # Log-envelope
        log_A = np.log(A + eps)

        # Linear + quadratic fit of log(A) over time
        T_mat = np.column_stack([t, np.ones(n_t)])
        coeffs_lin = np.linalg.lstsq(T_mat, log_A, rcond=None)[0]
        slope_lin = coeffs_lin[0]

        T_mat2 = np.column_stack([t ** 2, t, np.ones(n_t)])
        coeffs_quad = np.linalg.lstsq(T_mat2, log_A, rcond=None)[0]
        slope_quad = coeffs_quad[0]

        # Early/late log-energy ratio
        half = n_t // 2
        early_energy = np.mean(log_A[:half])
        late_energy = np.log(np.mean(A[half:] ** 2) + eps)
        ratio_early_late = early_energy - late_energy

        # Per-sensor log-envelope slopes
        sensor_slopes = []
        for s in range(n_sensors):
            A_s = np.sqrt(signals[s] ** 2 + eps)
            log_A_s = np.log(A_s + eps)
            c_s = np.linalg.lstsq(T_mat, log_A_s, rcond=None)[0]
            sensor_slopes.append(c_s[0])
        sensor_slopes = np.array(sensor_slopes)
        median_slope = np.median(sensor_slopes)
        iqr_slope = np.percentile(sensor_slopes, 75) - np.percentile(sensor_slopes, 25)

        # Dominant frequency (FFT of pooled signal)
        pooled = np.mean(signals, axis=0)
        fft_vals = np.fft.rfft(pooled)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_t)
        dom_freq = freqs[np.argmax(fft_mag[1:]) + 1]

        # Spectral bandwidth
        fft_power = fft_mag ** 2
        total_power = np.sum(fft_power) + eps
        weighted_freq = np.sum(freqs * fft_power) / total_power
        bandwidth = np.sqrt(np.sum((freqs - weighted_freq) ** 2 * fft_power) / total_power)

        return np.array([
            slope_lin,
            slope_quad,
            ratio_early_late,
            median_slope,
            iqr_slope,
            dom_freq,
            bandwidth,
        ])

    def fit(self, clips):
        X = np.array([self._extract_features(c.sensor_signals) for c in clips])
        y = np.array([c.margin for c in clips])
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_
        A = X_norm.T @ X_norm + self.alpha * np.eye(X_norm.shape[1])
        self.coef_ = np.linalg.solve(A, X_norm.T @ y)
        self.intercept_ = np.mean(y)

    def predict(self, clips):
        X = np.array([self._extract_features(c.sensor_signals) for c in clips])
        X_norm = (X - self.mean_) / self.std_
        return X_norm @ self.coef_ + self.intercept_


class GrowthRateBaseline:
    """Detect flutter via amplitude growth rate in sensor signals."""

    def __init__(self, growth_threshold=0.0):
        self.growth_threshold = growth_threshold
        self.max_rate_ = None

    def predict_growth_rate(self, clips) -> np.ndarray:
        """Compute mean amplitude growth rate across sensors for each clip."""
        return np.asarray(
            [self._growth_rate(c.sensor_signals) for c in clips],
            dtype=float,
        )

    def _growth_rate(self, signals, fs=1024.0):
        """Compute mean amplitude growth rate across sensors."""
        n_sensors, n_t = signals.shape
        t = np.arange(n_t) / fs
        slopes = []
        for s in range(n_sensors):
            A_s = np.sqrt(signals[s] ** 2 + 1e-12)
            log_A_s = np.log(A_s)
            T_mat = np.column_stack([t, np.ones(n_t)])
            c = np.linalg.lstsq(T_mat, log_A_s, rcond=None)[0]
            slopes.append(c[0])
        return np.mean(slopes)

    def fit(self, clips):
        rates = self.predict_growth_rate(clips)
        self.max_rate_ = np.max(np.abs(rates)) + 1e-8

    def predict_margin(self, clips) -> np.ndarray:
        """Convert growth rate to margin estimate."""
        if self.max_rate_ is None:
            self.fit(clips)
        rates = self.predict_growth_rate(clips)
        margins = 1.0 - np.abs(rates) / self.max_rate_
        return np.clip(margins, 0.0, 1.0)

    def predict(self, clips) -> np.ndarray:
        return self.predict_margin(clips)


class GRUMarginModel(nn.Module):
    """GRU backbone + final hidden state + linear head.
    Parameter count matched to TCN (~25k params)."""

    def __init__(self, n_channels=8, hidden_dim=86, n_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            n_channels, hidden_dim, n_layers,
            batch_first=True, dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C)
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]
        return self.head(final_hidden)


def train_gru(
    clips: list,
    n_channels: int = 8,
    hidden_dim: int = 86,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    device: str = "cpu",
    seed: int = 0,
    velocities: list[float] | None = None,
) -> tuple[GRUMarginModel, dict]:
    """Train GRU baseline with same protocol as TCN."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Build tensors
    X_list, y_list = [], []
    for i, c in enumerate(clips):
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

    # Grouped split (same as train.py)
    valid_clips = [c for c in clips if not np.isnan(c.margin) and not torch.isnan(torch.tensor(c.sensor_signals, dtype=torch.float32)).any()]
    if not valid_clips:
        raise ValueError("No valid clips (all margins NaN or non-finite signals)")
    if len({c.design_id for c in valid_clips}) < 2:
        raise ValueError(
            "Need at least two design groups for a grouped train/val split"
        )
    # Design-grouped validation split (no _grouped_3way_split: it
    # requires test_split > 0, which a train/val-only baseline does not
    # have). Groups, not clips, are shuffled so no design leaks across.
    group_ids = np.asarray([c.design_id for c in valid_clips])
    unique_groups = np.asarray(sorted(set(group_ids.tolist())))
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(round(unique_groups.size * val_split)))
    n_val_groups = min(n_val_groups, max(1, unique_groups.size - 1))
    val_groups = set(unique_groups[:n_val_groups].tolist())
    train_idx = [i for i, c in enumerate(valid_clips)
                 if c.design_id not in val_groups]
    val_idx = [i for i, c in enumerate(valid_clips)
               if c.design_id in val_groups]

    train_dl = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=batch_size, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(X[val_idx], y[val_idx]),
        batch_size=batch_size,
    )

    model = GRUMarginModel(n_channels, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    criterion = nn.HuberLoss(delta=0.1)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
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
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += criterion(pred, yb).item()
                n_val_batches += 1
        history["val_loss"].append(val_loss / max(n_val_batches, 1))

    return model, history
