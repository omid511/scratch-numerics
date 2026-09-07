"""Decoder: latent z \u2192 \u0394\u0302(x,y) with boundary enforcement (normalized)."""
from __future__ import annotations
import numpy as np
from .data import boundary_envelope


# Analytic peak of b(x,y)=x(1-x)y(1-y) on [0,1]^2 is 1/16 at (0.5, 0.5).
_ENVELOPE_NORM = 16.0


class CorrectionDecoder:
    """Decode latent codes to correction fields with boundary enforcement.

    Uses PCA reconstruction (mirrors encoder) for numpy-only operation.
    Boundary envelope b_norm(x,y) = 16*x(1-x)y(1-y) applied automatically
    (peak 1 at domain center, exactly 0 on all edges).

    Grid convention: ``grid_size`` is ``(ny, nx)`` matching field arrays
    ``(n, ny, nx)`` / ``(ny, nx)``.
    """

    def __init__(self, encoder):
        """Wrap an encoder for decoding with boundary enforcement.

        Args:
            encoder: fitted CorrectionEncoder instance
        """
        self.encoder = encoder
        self._grid_size = encoder.grid_size

        # Precompute normalized boundary envelope with (ny, nx) orientation.
        ny, nx = self._grid_size
        gx = np.linspace(0, 1, nx)
        gy = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(gx, gy)
        self._b = _ENVELOPE_NORM * boundary_envelope(X, Y)  # (ny, nx)

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
        """The precomputed normalized boundary envelope b_norm(x,y)."""
        return self._b
