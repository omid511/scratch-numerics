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


def _split_signals_mask(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split concatenated ``[signals; mask]`` DR inputs.

    Returns ``(physical, mask)`` where ``mask`` is ``None`` for clean inputs.
    Detection: even row count with an exactly-binary second half (0/1).
    Clean 8-channel inputs pass through with ``mask=None``.
    """
    signals = np.asarray(signals)
    n = signals.shape[0]
    if n % 2 == 0 and n >= 2:
        half = n // 2
        second = signals[half:]
        if second.size and bool(np.all((second == 0.0) | (second == 1.0))):
            return signals[:half], second
    return signals, None


def _physical_signals(signals: np.ndarray) -> np.ndarray:
    """Return physical sensor rows, stripping DR validity-mask channels.

    Thin wrapper around :func:`_split_signals_mask` kept for backward
    compatibility. Mask-aware baselines below use the mask half to ignore
    zero-filled invalid samples; signal-only callers that strip the mask
    must compare on CLEAN data only (see class docstrings).
    """
    physical, _ = _split_signals_mask(signals)
    return physical


def _clip_dt(clip, default: float | None = None) -> float | None:
    """Per-clip sampling interval from ``clip.time`` (seconds per sample).

    Returns ``default`` (per-sample units) when the clip carries no time
    grid. Threading the true ``dt`` keeps growth slopes (1/s) and spectral
    frequencies (Hz) in physical units across the adaptive-dt dataset;
    sample-index units would mix timescales between clips.
    """
    t = getattr(clip, "time", None)
    try:
        if t is not None and len(t) >= 2:
            dt = float(t[1] - t[0])
            if np.isfinite(dt) and dt > 0:
                return dt
    except Exception:
        pass
    return default


class ConstantMedianBaseline:
    """Predicts training-set median margin for every clip."""

    def __init__(self):
        self.median_ = None

    def fit(self, clips):
        self.median_ = np.median([c.margin for c in clips if np.isfinite(c.margin)])

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
        pairs = [(c.velocity, c.margin) for c in clips
                 if np.isfinite(c.velocity) and np.isfinite(c.margin)]
        if not pairs:
            raise ValueError("No valid clips (all margins NaN)")
        v = np.array([vv for vv, _ in pairs])
        m = np.array([mm for _, mm in pairs])
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

    def _extract_features(self, signals, dt=None):
        """Extract envelope features from (n_sensors, n_t) signals.

        Mask-aware: concatenated ``[signals; mask]`` DR inputs use the mask
        half to ignore zero-filled invalid samples (dropped channels are
        excluded; burst-invalid timesteps are excluded from envelope fits
        and interpolated for the FFT). Clean inputs use all samples.
        ``dt`` is the per-clip sampling interval in seconds; ``None`` falls
        back to per-sample units (documented, not physical).
        """
        signals, mask = _split_signals_mask(signals)
        n_sensors, n_t = signals.shape
        eps = 1e-12
        dt_use = float(dt) if dt is not None and np.isfinite(dt) and dt > 0 else 1.0
        t = np.arange(n_t) * dt_use

        if mask is not None:
            valid_ch = np.asarray(mask.mean(axis=1) > 0.5).ravel()
            if not valid_ch.any():
                return np.full(7, np.nan)
            signals_eff = signals[valid_ch]
            valid_t = np.asarray(mask[valid_ch].mean(axis=0) > 0.5).ravel()
            if valid_t.sum() < max(8, n_t // 4):
                return np.full(7, np.nan)
        else:
            signals_eff = signals
            valid_t = np.ones(n_t, dtype=bool)

        # Spatial RMS envelope A(t) = sqrt(mean_s(x_s(t)^2)) on valid steps
        A_full = np.sqrt(np.mean(signals_eff ** 2, axis=0) + eps)  # (n_t,)
        A = A_full[valid_t]
        t_fit = t[valid_t]

        # Log-envelope
        log_A = np.log(A + eps)

        # Linear + quadratic fit of log(A) over PHYSICAL time (per-second slopes)
        T_mat = np.column_stack([t_fit, np.ones_like(t_fit)])
        coeffs_lin = np.linalg.lstsq(T_mat, log_A, rcond=None)[0]
        slope_lin = coeffs_lin[0]

        T_mat2 = np.column_stack([t_fit ** 2, t_fit, np.ones_like(t_fit)])
        coeffs_quad = np.linalg.lstsq(T_mat2, log_A, rcond=None)[0]
        slope_quad = coeffs_quad[0]

        # Early/late log-energy ratio: both mean(log A) (~0 for flat envelopes)
        half = len(log_A) // 2
        early_energy = np.mean(log_A[:half])
        late_energy = np.mean(log_A[half:])
        ratio_early_late = early_energy - late_energy

        # Per-sensor log-envelope slopes (valid steps only)
        sensor_slopes = []
        for s in range(signals_eff.shape[0]):
            A_s = np.sqrt(signals_eff[s, valid_t] ** 2 + eps)
            log_A_s = np.log(A_s + eps)
            T_s = np.column_stack([t_fit, np.ones_like(t_fit)])
            c_s = np.linalg.lstsq(T_s, log_A_s, rcond=None)[0]
            sensor_slopes.append(c_s[0])
        sensor_slopes = np.array(sensor_slopes)
        median_slope = np.median(sensor_slopes)
        iqr_slope = np.percentile(sensor_slopes, 75) - np.percentile(sensor_slopes, 25)

        # Dominant frequency (FFT of pooled signal, Hz via dt; gaps interpolated)
        pooled = np.mean(signals_eff, axis=0)
        if mask is not None and not bool(valid_t.all()):
            pooled = pooled.copy()
            pooled[~valid_t] = np.interp(t[~valid_t], t[valid_t], pooled[valid_t])
        fft_vals = np.fft.rfft(pooled)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_t, d=dt_use)
        dom_freq = freqs[np.argmax(fft_mag[1:]) + 1] if len(fft_mag) > 1 else 0.0

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
        pairs = [(self._extract_features(c.sensor_signals, dt=_clip_dt(c)), c.margin) for c in clips
                 if np.isfinite(c.margin)]
        pairs = [(x, y) for x, y in pairs if np.all(np.isfinite(x)) and np.isfinite(y)]
        if not pairs:
            raise ValueError("No valid clips (all margins NaN)")
        X = np.array([x for x, _ in pairs])
        y = np.array([y for _, y in pairs])
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_
        A = X_norm.T @ X_norm + self.alpha * np.eye(X_norm.shape[1])
        self.coef_ = np.linalg.solve(A, X_norm.T @ y)
        self.intercept_ = np.mean(y)

    def predict(self, clips):
        X = np.array([self._extract_features(c.sensor_signals, dt=_clip_dt(c)) for c in clips])
        X_norm = (X - self.mean_) / self.std_
        return X_norm @ self.coef_ + self.intercept_


class GrowthRateBaseline:
    """Detect flutter via amplitude growth rate in sensor signals.

    Mask-aware signal baseline: concatenated ``[signals; mask]`` DR inputs
    use the mask half to ignore zero-filled invalid samples (see
    :func:`_split_signals_mask`). For head-to-head TCN comparisons prefer
    CLEAN clips for both models, or compare the TCN against this
    mask-aware rate; comparing a mask-blind rate against a mask-aware TCN
    on corrupted clips measures mask access, not model quality.
    """

    def __init__(self, growth_threshold=0.0):
        self.growth_threshold = growth_threshold
        self.max_rate_ = None

    def predict_growth_rate(self, clips) -> np.ndarray:
        """Compute mean amplitude growth rate across sensors for each clip."""
        return np.asarray(
            [self._growth_rate(c.sensor_signals, dt=_clip_dt(c)) for c in clips],
            dtype=float,
        )

    def _growth_rate(self, signals, dt=None, fs=None):
        """Compute mean amplitude growth rate (per second) across sensors.

        ``dt`` is the per-clip sampling interval in seconds. The legacy
        ``fs`` (Hz) keyword is accepted as ``dt = 1/fs`` for backward
        compatibility; passing neither falls back to per-sample units.
        Invalid (mask-zeroed) samples are excluded per sensor; fully
        dropped channels are skipped.
        """
        if fs is not None and dt is None:
            dt = 1.0 / float(fs)
        signals, mask = _split_signals_mask(signals)
        n_sensors, n_t = signals.shape
        dt_use = float(dt) if dt is not None and np.isfinite(dt) and dt > 0 else 1.0
        t = np.arange(n_t) * dt_use
        slopes = []
        for s in range(n_sensors):
            if mask is not None:
                valid = np.asarray(mask[s] > 0.5).ravel()
                if valid.sum() < max(8, n_t // 8):
                    continue
                t_s = t[valid]
                sig_s = signals[s][valid]
            else:
                t_s = t
                sig_s = signals[s]
            A_s = np.sqrt(sig_s ** 2 + 1e-12)
            log_A_s = np.log(A_s)
            T_mat = np.column_stack([t_s, np.ones_like(t_s)])
            c = np.linalg.lstsq(T_mat, log_A_s, rcond=None)[0]
            slopes.append(c[0])
        if not slopes:
            return float("nan")
        return float(np.mean(slopes))

    def fit(self, clips):
        rates = self.predict_growth_rate(clips)
        finite = rates[np.isfinite(rates)]
        if finite.size == 0:
            raise ValueError("No valid clips (all growth rates NaN)")
        self.max_rate_ = np.max(np.abs(finite)) + 1e-8

    def predict_margin(self, clips) -> np.ndarray:
        """Convert growth rate to margin estimate (signed, unclipped).

        Negative slope (decaying, stable) maps to positive margin;
        positive slope (growing, unstable) maps to negative margin.
        No [0, 1] clip so supercritical margins stay representable.
        """
        if self.max_rate_ is None:
            raise RuntimeError(
                "GrowthRateBaseline must be fit on train clips before predict; "
                "call fit() first (fitting on test clips leaks test scale)"
            )
        rates = self.predict_growth_rate(clips)
        return -rates / self.max_rate_

    def predict(self, clips) -> np.ndarray:
        return self.predict_margin(clips)


class GRUMarginModel(nn.Module):
    """GRU backbone + final hidden state + linear head.
    Parameter count matched to TCN (~25k params)."""

    def __init__(self, n_channels=8, hidden_dim=86, n_layers=1, dropout=0.0):
        super().__init__()
        # NOTE: nn.GRU applies dropout only between layers, so with
        # n_layers=1 any nonzero dropout is a silent no-op.
        self.gru = nn.GRU(
            n_channels, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
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
        if not np.isfinite(c.margin):
            continue
        sig = torch.tensor(c.sensor_signals, dtype=torch.float32)
        if not bool(torch.isfinite(sig).all()):
            continue
        X_list.append(sig)
        y_list.append(c.margin)

    if not X_list:
        raise ValueError("No valid clips (all margins NaN)")

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)

    # Grouped split (same as train.py)
    valid_clips = [c for c in clips if np.isfinite(c.margin) and bool(torch.isfinite(torch.tensor(c.sensor_signals, dtype=torch.float32)).all())]
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
    best_val_loss = float("inf")
    best_state = None

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
        avg_val_loss = val_loss / max(n_val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
