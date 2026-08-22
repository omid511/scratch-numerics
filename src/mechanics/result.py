"""SolverResult dataclass — complete output from a single solver run."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverResult:
    """Complete output from a single solver run."""
    frequencies: np.ndarray          # (n_modes,) natural frequencies in Hz
    mode_shapes: np.ndarray          # (n_modes, grid_ny, grid_nx) transverse DOF
    grid_x: np.ndarray               # (grid_nx,)
    grid_y: np.ndarray               # (grid_ny,)
    eigenvalues: np.ndarray | None = None    # (n_modes,) complex eigenvalues
    left_eigenvectors: np.ndarray | None = None
    stable: np.ndarray | None = None         # (n_modes,) bool stability flags
    coefficients: np.ndarray | None = None   # (n_modes, 5*M*N) raw coefficients
    L1: float = 0.0
    L2: float = 0.0
    M: int = 0
    N: int = 0
    lambda_cr: float | None = None


@dataclass
class AeroelasticResult:
    """Extended result for aeroelastic (flutter) analysis."""
    frequencies: np.ndarray
    mode_shapes: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    eigenvalues: np.ndarray
    left_eigenvectors: np.ndarray | None = None
    stable: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    L1: float = 0.0
    L2: float = 0.0
    M: int = 0
    N: int = 0
    lambda_cr: float | None = None
    velocity: float | None = None
    mach_number: float | None = None
    mode_shapes_complex: np.ndarray | None = None
    lambda_value: float | None = None
