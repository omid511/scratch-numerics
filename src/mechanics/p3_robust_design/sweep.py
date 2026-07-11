"""Design parameter sweep with boundary stiffness variation.

Runs the FSDT solver at many design points, each with varying boundary
spring stiffnesses, and records the critical flutter lambda (λ_cr).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..laminate import Material, Laminate
from ..solver import FSDTSolver


@dataclass
class SweepResult:
    """Output from a design sweep."""
    design_params: np.ndarray       # (n, d) log10 stiffness values
    boundary_stiffnesses: np.ndarray # (n, n_edges) actual stiffness values
    flutter_lambda: np.ndarray       # (n,) critical λ_cr per design point
    mode_shapes: list | None = None  # optional mode shapes at each point
    n_success: int = 0
    n_total: int = 0


def _make_solver(
    laminate: Laminate,
    L1: float = 0.3,
    L2: float = 0.3,
    M: int = 8,
    N: int = 8,
    k_stiffness: float = 1e12,
) -> FSDTSolver:
    """Create a base solver with clamped-clamped-clamped-clamped BCs."""
    solver = FSDTSolver(L1=L1, L2=L2, M=M, N=N, laminate=laminate,
                        k_stiffness=k_stiffness)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    return solver


def generate_design_sweep(
    n_samples: int,
    laminate: Laminate,
    stiffness_bounds: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
    M: int = 8,
    N: int = 8,
) -> SweepResult:
    """Generate design points via LHS and solve flutter boundary at each.

    Args:
        n_samples: Number of design points.
        laminate: Laminate layup for the plate.
        stiffness_bounds: Dict of edge_name -> (log10_k_min, log10_k_max).
            Defaults to [8, 13] for all 4 edges (clamped -> elastic).
        seed: Random seed for reproducibility.
        M, N: Basis function counts.

    Returns:
        SweepResult with design parameters and flutter lambdas.
    """
    # Default bounds: 4 edges (left, right, top, bottom), log10 stiffness
    edge_names = ["left", "right", "top", "bottom"]
    if stiffness_bounds is None:
        stiffness_bounds = {e: (8.0, 13.0) for e in edge_names}

    bounds = np.array([stiffness_bounds[e] for e in edge_names])
    lb, ub = bounds[:, 0], bounds[:, 1]

    # LHS in log10 space
    lhs = LatinHypercube(d=len(edge_names), seed=seed)
    samples_log10 = lhs.random(n=n_samples)  # (n, 4) in [0, 1]
    samples_log10 = lb + samples_log10 * (ub - lb)

    # Actual stiffnesses
    stiffnesses = 10.0 ** samples_log10

    flutter_lambdas = np.full(n_samples, np.nan)
    n_success = 0

    for i in range(n_samples):
        solver = _make_solver(laminate, M=M, N=N)
        # Set each edge to elastic with the sampled stiffness
        from ..boundary import build_boundary_springs
        spring_list = build_boundary_springs(
            solver.L1, solver.L2,
            {"type": "elastic", "k": [float(stiffnesses[i, 0])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[i, 1])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[i, 2])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[i, 3])] * 5},
            k_stiffness=1e12,
        )
        solver.set_boundary_springs(spring_list)

        try:
            lam_cr = solver.find_flutter_boundary(
                lambda_lower=10.0, lambda_upper=1000.0, tol=1.0, n_modes=8,
            )
            if lam_cr is not None:
                flutter_lambdas[i] = lam_cr
                n_success += 1
        except Exception:
            pass  # ponytail: skip failed points, track ratio

    return SweepResult(
        design_params=samples_log10,
        boundary_stiffnesses=stiffnesses,
        flutter_lambda=flutter_lambdas,
        n_success=n_success,
        n_total=n_samples,
    )
