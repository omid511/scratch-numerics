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
from ..piston_theory import AIR_DENSITY, SOUND_SPEED


@dataclass
class SweepResult:
    """Output from a design sweep.

    Scalar-only by design: only the critical λ_cr per point is stored.
    Mode shapes are intentionally not kept (use the stored design point
    with a direct solver call when shapes are needed).
    """
    design_params: np.ndarray       # (n, d) log10 stiffness values
    boundary_stiffnesses: np.ndarray # (n, n_edges) actual stiffness values
    flutter_lambda: np.ndarray       # (n,) critical λ_cr per design point
    n_success: int = 0
    n_total: int = 0
    failure_reasons: dict[str, int] | None = None  # category -> count


def _make_solver(
    laminate: Laminate,
    L1: float = 0.3,
    L2: float = 0.3,
    M: int = 8,
    N: int = 8,
    k_stiffness: float = 1e12,
) -> FSDTSolver:
    """Create a base solver with no boundary springs.

    The caller must set boundaries (e.g. per-design elastic springs via
    ``set_boundary_springs``); a spring-free solver is intentionally left
    unconfigured here rather than given clamped BCs that the sweep would
    immediately overwrite.
    """
    solver = FSDTSolver(L1=L1, L2=L2, M=M, N=N, laminate=laminate,
                        k_stiffness=k_stiffness)
    return solver


def _classify_no_boundary(solver: FSDTSolver) -> str:
    """Categorize why find_flutter_boundary returned None for this design.

    Uses only public API: D11 from ``solver.laminate.ABD()``, the bracket
    floor formula mirrored from ``find_flutter_boundary`` (lambda at M=2
    with the 0.9 safety margin), and a ``solve_complex_modal`` stability
    probe at that floor. NOTE: the probe goes through the modal filter
    path (no transverse-participation floor), while ``find_flutter_boundary``
    uses the spectral-abscissa path, so near-marginal cases can disagree;
    this is a diagnostic label, not a stability verdict. The 1e-4 threshold
    matches ``find_flutter_boundary``'s default ``stability_tol``.
    """
    from ..piston_theory import velocity_from_lambda

    rho, c_sound = AIR_DENSITY, SOUND_SPEED
    ABBD, _ = solver.laminate.ABD()
    D11 = ABBD[3, 3]
    V_min = 2.0 * c_sound
    lam_floor = 0.9 * rho * V_min**2 * solver.L1**3 / (D11 * np.sqrt(3.0))
    vel = velocity_from_lambda(lam_floor, rho, solver.L1, D11, c_sound)
    result = solver.solve_complex_modal(vel, n_modes=8)
    if len(result.eigenvalues) > 0 and bool(
        np.max(np.real(result.eigenvalues)) >= 1e-4
    ):
        return "unstable_at_lambda_lower"
    return "no_crossing_in_bracket"


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
        SweepResult with design parameters and flutter lambdas. Failed
        points are not silently dropped: each failure is counted in
        ``SweepResult.failure_reasons`` (dict category -> count) with
        categories 'unstable_at_lambda_lower' (system already unstable at
        the per-design Mach-floor bracket, so no crossing exists),
        'no_crossing_in_bracket' (stable throughout), or
        'solver_error:<ExcType>' for solver exceptions.
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
    failure_reasons: dict[str, int] = {}

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
                lambda_lower=None, lambda_upper=1000.0, tol=1.0, n_modes=8,
            )
            if lam_cr is not None:
                flutter_lambdas[i] = lam_cr
                n_success += 1
            else:
                reason = _classify_no_boundary(solver)
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        except Exception as exc:
            key = f"solver_error:{type(exc).__name__}"
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

    return SweepResult(
        design_params=samples_log10,
        boundary_stiffnesses=stiffnesses,
        flutter_lambda=flutter_lambdas,
        n_success=n_success,
        n_total=n_samples,
        failure_reasons=failure_reasons,
    )
