"""Decoder: latent z → Δ̂(x,y) with boundary enforcement b(x,y) = x(1-x)y(1-y)."""
from __future__ import annotations
import numpy as np
from .data import boundary_envelope


class CorrectionDecoder:
    """Decode latent codes to correction fields with boundary enforcement.

    Uses PCA reconstruction (mirrors encoder) for numpy-only operation.
    Boundary envelope b(x,y) = x(1-x)y(1-y) applied automatically.
    """

    def __init__(self, encoder):
        """Wrap an encoder for decoding with boundary enforcement.

        Args:
            encoder: fitted CorrectionEncoder instance
        """
        self.encoder = encoder
        self._grid_size = encoder.grid_size

        # Precompute boundary envelope
        gx = np.linspace(0, 1, self._grid_size[0])
        gy = np.linspace(0, 1, self._grid_size[1])
        X, Y = np.meshgrid(gx, gy)
        self._b = boundary_envelope(X, Y)  # (grid_ny, grid_nx)

    def decode(self, z: np.ndarray, apply_boundary: bool = True) -> np.ndarray:
        """Decode latent codes to correction fields.

        Args:
            z: (n_samples, d_z) or (d_z,)
            apply_boundary: whether to apply boundary envelope

        Returns: (n_samples, grid_ny, grid_nx) or (grid_ny, grid_nx)
        """
        raw = self.encoder.decode(z)
        if apply_boundary:
            raw = raw * self._b[np.newaxis, :, :] if raw.ndim == 3 else raw * self._b
        return raw

    @property
    def boundary_envelope(self) -> np.ndarray:
        """The precomputed boundary envelope b(x,y)."""
        return self._b
