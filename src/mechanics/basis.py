"""Basis functions and precomputed integrals for the FSDT solver."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


def legendre_and_derivative(order: int, x: np.ndarray):
    """Evaluate Legendre polynomials and first derivatives via recurrence."""
    n = order + 1
    vals = np.zeros((n, len(x)))
    dvals = np.zeros((n, len(x)))
    vals[0] = 1.0
    if n > 1:
        vals[1] = x
        dvals[1] = 1.0
    for k in range(2, n):
        vals[k] = ((2*k-1) * x * vals[k-1] - (k-1) * vals[k-2]) / k
        dvals[k] = (2*k-1) * vals[k-1] + dvals[k-2]
    return vals, dvals


def trig_and_derivative(order: int, x: np.ndarray):
    """Evaluate trigonometric basis and derivatives on [0, 1]."""
    n = 2 * order + 1
    vals = np.zeros((n, len(x)))
    dvals = np.zeros((n, len(x)))
    vals[0] = 1.0
    for k in range(1, order + 1):
        vals[2*k-1] = np.sin(k * np.pi * x)
        vals[2*k] = np.cos(k * np.pi * x)
        dvals[2*k-1] = k * np.pi * np.cos(k * np.pi * x)
        dvals[2*k] = -k * np.pi * np.sin(k * np.pi * x)
    return vals, dvals


@dataclass
class BasisIntegrals:
    """Precomputed integral matrices for one axis. Material-free."""
    int_00: np.ndarray  # (M, M): int phi_i * phi_j
    int_01: np.ndarray  # (M, M): int phi_i * phi_j'
    int_10: np.ndarray  # (M, M): int phi_i' * phi_j
    int_11: np.ndarray  # (M, M): int phi_i' * phi_j'

    def get(self, order_i: int, order_j: int) -> np.ndarray:
        if order_i == 0 and order_j == 0:
            return self.int_00
        if order_i == 0 and order_j == 1:
            return self.int_01
        if order_i == 1 and order_j == 0:
            return self.int_10
        return self.int_11


def precompute_integrals(M: int, a: float, basis_type: str = "legendre",
                         n_quad: int | None = None) -> BasisIntegrals:
    """Precompute basis integral matrices for one axis.

    Args:
        M: Number of basis functions
        a: Domain length
        basis_type: "legendre" or "trigonometric"
        n_quad: Number of quadrature points (default: 2*M)
    """
    if n_quad is None:
        n_quad = 2 * M

    xi, w = leggauss(n_quad)
    x = a * (xi + 1) / 2
    w_scaled = w * a / 2

    if basis_type == "legendre":
        xi_eval = 2 * x / a - 1
        vals, dvals = legendre_and_derivative(M - 1, xi_eval)
        dvals = dvals * (2.0 / a)
    elif basis_type == "trigonometric":
        x_norm = x / a
        vals, dvals = trig_and_derivative(M - 1, x_norm)
        dvals = dvals / a
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    int_00 = (vals * w_scaled) @ vals.T
    int_01 = (vals * w_scaled) @ dvals.T
    int_10 = (dvals * w_scaled) @ vals.T
    int_11 = (dvals * w_scaled) @ dvals.T

    # Symmetrize where applicable
    int_00[:] = (int_00 + int_00.T) / 2
    int_11[:] = (int_11 + int_11.T) / 2

    return BasisIntegrals(int_00=int_00, int_01=int_01,
                          int_10=int_10, int_11=int_11)


def expand_basis(int_x: np.ndarray, int_y: np.ndarray,
                 M: int, N: int) -> np.ndarray:
    """Outer product basis matrix: vv[i,j] = int_x[ix,jx] * int_y[iy,jy].

    i = ix*N + iy, j = jx*N + jy
    """
    # Use actual matrix dimensions if they differ from M, N
    M_eff = int_x.shape[0]
    N_eff = int_y.shape[0]
    MN = M_eff * N_eff
    ix = np.arange(MN) // N_eff
    iy = np.arange(MN) % N_eff
    return int_x[np.ix_(ix, ix)] * int_y[np.ix_(iy, iy)]
