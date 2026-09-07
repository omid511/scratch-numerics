"""Spatial encoder: \u0394(x,y) \u2192 latent z via CNN or PCA fallback."""
from __future__ import annotations
import warnings
import numpy as np
import torch


class CorrectionEncoder:
    """Encode correction fields to latent codes.

    Uses PCA (truncated SVD) with torch backend.
    Input: (n_samples, grid_ny, grid_nx) → Output: (n_samples, d_z)

    Grid convention: ``grid_size`` is ``(ny, nx)`` matching field arrays.
    """

    def __init__(self, d_z: int = 16, grid_size: tuple[int, int] = (64, 64)):
        self.d_z = d_z
        self.grid_size = grid_size
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self._fitted = False

    def _check_grid(self, fields: np.ndarray) -> None:
        if fields.ndim not in (2, 3):
            raise ValueError(f"fields must be 2D or 3D, got shape {fields.shape}")
        grid = fields.shape[-2:] if fields.ndim == 3 else fields.shape
        if tuple(grid) != tuple(self.grid_size):
            raise ValueError(
                f"field grid {tuple(grid)} != encoder grid_size {tuple(self.grid_size)} "
                "(ny, nx); reshape would silently transpose data"
            )

    def fit(self, fields: np.ndarray) -> None:
        if fields.ndim != 3:
            raise ValueError(f"fit expects (n, ny, nx), got shape {fields.shape}")
        self._check_grid(fields)
        if not np.all(np.isfinite(fields)):
            raise ValueError("fit fields contain non-finite values")
        n = fields.shape[0]
        flat = fields.reshape(n, -1)
        self.mean_ = flat.mean(axis=0)
        centered = flat - self.mean_

        k = min(self.d_z, min(centered.shape) - 1)
        k = max(k, 1)
        if k != self.d_z:
            warnings.warn(
                f"requested d_z={self.d_z} exceeds the rank budget "
                f"(n_samples={n}, n_features={flat.shape[1]}); using d_z={k}",
                UserWarning,
                stacklevel=2,
            )

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
        self._check_grid(fields)
        if not np.all(np.isfinite(fields)):
            raise ValueError("encode fields contain non-finite values")
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        n = fields.shape[0]
        flat = (fields.reshape(n, -1) - self.mean_)
        z = flat @ self.components_.T
        return z[0] if single else z

    def decode(self, z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        single = z.ndim == 1
        if single:
            z = z[np.newaxis]
        if z.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"latent dim {z.shape[1]} != fitted components {self.components_.shape[0]}"
            )
        if not np.all(np.isfinite(np.asarray(z))):
            raise ValueError("latent codes contain non-finite values")

        flat = z @ self.components_ + self.mean_
        fields = flat.reshape(-1, *self.grid_size)
        return fields[0] if single else fields
