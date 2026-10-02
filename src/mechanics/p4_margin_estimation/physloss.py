#!/usr/bin/env python3
"""G5 physics loss members for P4 quantile training (additive only).

G5a: sample weights from clamp_frac + participation entropy.
  w = (1 - clamp_frac) * entropy_weight, entropy from normalized FFT power
  (low entropy = one clean tone = trustworthy slope; high entropy =
  beating soup = distrust). Weights multiply per-sample pinball loss,
  normalized to mean 1 per batch so LR dynamics are unchanged.
G5b: kurtosis sidecar member (registered in sidecar.py MEMBERS already).
G5c: monotonicity-in-V penalty (same-design pairs in batch):
  penalty = mean over pairs (i,j), same design, V_i < V_j, of
  relu(med_i - med_j + margin_gap). Needs design_id + velocity per row.

Loss wiring lives in a G5 trainer variant; existing trainers untouched.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def participation_entropy(signals: np.ndarray) -> float:
    """Normalized spectral entropy in [0,1]; NaN if uncomputable."""
    try:
        s = np.asarray(signals, dtype=np.float64)
    except (TypeError, ValueError):
        return float("nan")
    if s.ndim != 2 or s.shape[1] < 8:
        return float("nan")
    # physical rows only (mask halves excluded by caller shape)
    n = s.shape[0]
    if n % 2 == 0 and n >= 2:
        second = s[n // 2:]
        if second.size and bool(np.all((second == 0.0) | (second == 1.0))):
            s = s[:n // 2]
    pooled = s.mean(axis=0)
    power = np.abs(np.fft.rfft(pooled)) ** 2
    tot = float(power.sum())
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    p = power / tot
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    hmax = float(np.log(len(p))) if len(p) > 1 else 1.0
    return h / hmax if hmax > 0 else float("nan")


def sample_weight(clip: Any, entropy_floor: float = 0.2) -> float:
    """Per-clip pinball weight in (0,1]. NaN inputs -> 1.0 (never drop)."""
    try:
        clamp = float(getattr(clip, "clamp_frac", 0.0))
    except (TypeError, ValueError):
        clamp = 0.0
    if not np.isfinite(clamp):
        clamp = 0.0
    clamp = min(max(clamp, 0.0), 1.0)
    ent = participation_entropy(np.asarray(getattr(clip, "sensor_signals", None)))
    if not np.isfinite(ent):
        ent_w = 1.0
    else:
        # low entropy (clean tone) -> 1; high entropy (soup) -> floor
        ent_w = float(entropy_floor + (1.0 - entropy_floor) * (1.0 - min(max(ent, 0.0), 1.0)))
    return float((1.0 - clamp) * ent_w)


def batch_weights(clips: list[Any]) -> np.ndarray:
    """Per-row weights normalized to mean 1 (LR-neutral)."""
    w = np.array([sample_weight(c) for c in clips], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    m = float(w.mean())
    return (w / m).astype(np.float32) if m > 0 else np.ones_like(w, dtype=np.float32)
