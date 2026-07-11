"""Spatial encoder: Δ(x,y) → latent z via CNN or PCA fallback."""
from __future__ import annotations
import numpy as np
import torch


class CorrectionEncoder:
    """Encode correction fields to latent codes.

    Uses PCA (truncated SVD) with torch backend.
    Input: (n_samples, grid_ny, grid_nx) → Output: (n_samples, d_z)
    """

    def __init__(self, d_z: int = 16, grid_size: tuple[int, int] = (64, 64)):
        self.d_z = d_z
        self.grid_size = grid_size
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self._fitted = False

    def fit(self, fields: np.ndarray) -> None:
        n = fields.shape[0]
        flat = fields.reshape(n, -1)
        self.mean_ = flat.mean(axis=0)
        centered = flat - self.mean_

        k = min(self.d_z, min(centered.shape) - 1)
        k = max(k, 1)

        centered_t = torch.as_tensor(centered, dtype=torch.float64)
        _, _, Vt = torch.linalg.svd(centered_t, full_matrices=False)
        Vt_np = Vt.numpy()
        self.components_ = Vt_np[:k]
        self.d_z = k
        self._fitted = True

    def encode(self, fields: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        single = fields.ndim == 2
        if single:
            fields = fields[np.newaxis]

        n = fields.shape[0]
        flat = (fields.reshape(n, -1) - self.mean_)
        z = flat @ self.components_.T
        return z[0] if single else z

    def decode(self, z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        single = z.ndim == 1
        if single:
            z = z[np.newaxis]

        flat = z @ self.components_ + self.mean_
        fields = flat.reshape(-1, *self.grid_size)
        return fields[0] if single else fields
