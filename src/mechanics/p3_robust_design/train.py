"""Training pipeline: fit GP, active learning loop, brute-force MC reference.

End-to-end workflow for Proposal 3 surrogate development.
"""
from __future__ import annotations
import numpy as np
from scipy.stats.qmc import LatinHypercube

from .gp_surrogate import GPSurrogate
from .active_learning import select_batch
from .sweep import generate_design_sweep, SweepResult
from ..laminate import Laminate


def fit_gp(
    sweep: SweepResult,
    use_gradients: bool = False,
    objective_fn=None,
) -> GPSurrogate:
    """Fit a GP surrogate to sweep results.

    Args:
        sweep: SweepResult from generate_design_sweep.
        use_gradients: Whether to use gradient-enhanced GP.
        objective_fn: Optional function for gradient computation.

    Returns:
        Fitted GPSurrogate.
    """
    # Filter to successful points
    valid = ~np.isnan(sweep.flutter_lambda)
    X = sweep.design_params[valid]
    y = sweep.flutter_lambda[valid]

    gp = GPSurrogate(use_gradients=use_gradients)
    gp.fit(X, y, objective_fn=objective_fn)
    return gp


def active_learning_loop(
    gp: GPSurrogate,
    laminate: Laminate,
    lambda_target: float,
    n_initial: int = 50,
    n_iterations: int = 5,
    batch_size: int = 5,
    stiffness_bounds: dict[str, tuple[float, float]] | None = None,
    L1: float = 0.3,
    L2: float = 0.3,
    M: int = 8,
    N: int = 8,
    seed: int = 42,
) -> tuple[GPSurrogate, list[dict]]:
    """Run active learning loop: fit → acquire → solve → refit.

    Args:
        gp: Initial fitted GP.
        laminate: Laminate for solver.
        lambda_target: Safety threshold.
        n_initial: Number of initial LHS points for candidate pool.
        n_iterations: Max active learning iterations.
        batch_size: Points per iteration.
        stiffness_bounds: Design space bounds.
        seed: Random seed.

    Returns:
        Updated GP and list of iteration stats.
    """
    edge_names = ["left", "right", "top", "bottom"]
    if stiffness_bounds is None:
        stiffness_bounds = {e: (8.0, 13.0) for e in edge_names}

    bounds = np.array([stiffness_bounds[e] for e in edge_names])
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Generate candidate pool
    lhs = LatinHypercube(d=4, seed=seed + 1)
    cand_log10 = lhs.random(n=n_initial)
    cand_log10 = lb + cand_log10 * (ub - lb)

    history = []
    X_train, y_train = gp.get_training_data()
    X_train = X_train.copy()
    y_train = y_train.copy()

    for iteration in range(n_iterations):
        # Select batch
        selected = select_batch(cand_log10, gp, lambda_target, batch_size)

        # Solve at selected points
        new_X = []
        new_y = []
        for x_log10 in selected:
            stiffnesses = 10.0 ** x_log10
            from ..solver import FSDTSolver
            solver = FSDTSolver(L1=L1, L2=L2, M=M, N=N, laminate=laminate,
                                k_stiffness=1e12)
            from ..boundary import build_boundary_springs
            springs = build_boundary_springs(
                L1, L2,
                {"type": "elastic", "k": [float(stiffnesses[0])] * 5},
                {"type": "elastic", "k": [float(stiffnesses[1])] * 5},
                {"type": "elastic", "k": [float(stiffnesses[2])] * 5},
                {"type": "elastic", "k": [float(stiffnesses[3])] * 5},
                k_stiffness=1e12,
            )
            solver.set_boundary_springs(springs)
            try:
                lam_cr = solver.find_flutter_boundary(
                    lambda_lower=10.0, lambda_upper=500.0, tol=1.0, n_modes=8,
                )
                if lam_cr is not None:
                    new_X.append(x_log10)
                    new_y.append(lam_cr)
            except Exception:
                pass

        if not new_X:
            history.append({"iteration": iteration, "n_added": 0})
            continue

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # Refit GP with all data
        X_all = np.vstack([X_train, new_X])
        y_all = np.concatenate([y_train, new_y])
        gp_new = GPSurrogate(use_gradients=gp.use_gradients)
        gp_new.fit(X_all, y_all)

        # Update
        X_train = X_all
        y_train = y_all
        gp = gp_new

        # Remove selected from candidates by index
        selected_indices = [
            int(np.argmin(np.linalg.norm(cand_log10 - sel, axis=1)))
            for sel in selected
        ]
        mask = np.ones(len(cand_log10), dtype=bool)
        mask[selected_indices] = False
        cand_log10 = cand_log10[mask]

        history.append({
            "iteration": iteration,
            "n_added": len(new_X),
            "total_train": len(X_train),
        })

    return gp, history


def brute_force_mc(
    laminate: Laminate,
    n_samples: int = 1000,
    lambda_target: float = 100.0,
    L1: float = 0.3,
    L2: float = 0.3,
    M: int = 8,
    N: int = 8,
    seed: int = 42,
) -> dict:
    """Brute-force Monte Carlo reference for failure probability.

    Samples random boundary stiffnesses, computes λ_cr, estimates P_f.

    Returns:
        dict with 'failure_probability', 'mean_lambda', 'std_lambda',
        'n_success', 'n_total'.
    """
    edge_names = ["left", "right", "top", "bottom"]
    bounds = np.array([(8.0, 13.0)] * 4)
    lb, ub = bounds[:, 0], bounds[:, 1]

    lhs = LatinHypercube(d=4, seed=seed)
    samples_log10 = lhs.random(n=n_samples)
    samples_log10 = lb + samples_log10 * (ub - lb)

    lambdas = []
    for i in range(n_samples):
        stiffnesses = 10.0 ** samples_log10[i]
        from ..solver import FSDTSolver
        solver = FSDTSolver(L1=L1, L2=L2, M=M, N=N, laminate=laminate,
                            k_stiffness=1e12)
        from ..boundary import build_boundary_springs
        springs = build_boundary_springs(
            L1, L2,
            {"type": "elastic", "k": [float(stiffnesses[0])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[1])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[2])] * 5},
            {"type": "elastic", "k": [float(stiffnesses[3])] * 5},
            k_stiffness=1e12,
        )
        solver.set_boundary_springs(springs)
        try:
            lam_cr = solver.find_flutter_boundary(
                lambda_lower=10.0, lambda_upper=500.0, tol=1.0, n_modes=8,
            )
            if lam_cr is not None:
                lambdas.append(lam_cr)
        except Exception:
            pass

    lambdas = np.array(lambdas)
    n_success = len(lambdas)
    n_fail = int(np.sum(lambdas < lambda_target)) if n_success > 0 else 0

    return {
        "failure_probability": n_fail / n_success if n_success > 0 else 0.0,
        "mean_lambda": float(np.mean(lambdas)) if n_success > 0 else 0.0,
        "std_lambda": float(np.std(lambdas)) if n_success > 0 else 0.0,
        "n_success": n_success,
        "n_total": n_samples,
    }


def evaluate_gp_accuracy(
    gp: GPSurrogate,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate GP prediction accuracy on test set.

    Returns:
        dict with 'rmse', 'mae', 'max_error', 'r_squared'.
    """
    mean, _ = gp.predict(X_test)
    residuals = y_test - mean
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    max_err = float(np.max(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)
    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_err,
        "r_squared": float(r_squared),
    }
