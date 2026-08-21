"""LF data generation and synthetic HF correction fields."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleConfig:
    """Configuration for LF dataset generation."""
    L1: float = 0.3
    L2: float = 0.3
    M: int = 15
    N: int = 15
    grid: tuple[int, int] = (64, 64)
    n_modes: int = 6
    face_thickness_range: tuple[float, float] = (0.001, 0.005)
    core_thickness_range: tuple[float, float] = (0.005, 0.02)
    cell_angle_range: tuple[float, float] = (np.pi / 12, np.pi / 4)
    seed: int = 42


def boundary_envelope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """b(x,y) = x(1-x)y(1-y) on normalized [0,1]^2 domain.

    Guarantees correction vanishes at domain edges.
    """
    return x * (1.0 - x) * y * (1.0 - y)


def _lhs_sample(n: int, ranges: list[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling in [0,1]^d mapped to ranges."""
    d = len(ranges)
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            samples[i, j] = (perm[i] + rng.uniform()) / n
    for j, (lo, hi) in enumerate(ranges):
        samples[:, j] = lo + samples[:, j] * (hi - lo)
    return samples


def generate_lf_dataset(
    n_samples: int,
    config: SampleConfig | None = None,
) -> dict[str, Any]:
    """Generate LF dataset from FSDT solver.

    Runs the solver across sampled design parameters, storing frequencies
    and mode shapes.

    Returns dict with:
        frequencies: (n_samples, n_modes)
        mode_shapes: (n_samples, n_modes, grid_ny, grid_nx)
        params: (n_samples, 3) — [face_thickness, core_thickness, cell_angle]
        grid_x: (grid_nx,)
        grid_y: (grid_ny,)
    """
    if config is None:
        config = SampleConfig()

    from mechanics.solver import FSDTSolver
    from mechanics.laminate import Material, Laminate

    rng = np.random.default_rng(config.seed)

    param_ranges = [
        config.face_thickness_range,
        config.core_thickness_range,
        config.cell_angle_range,
    ]
    params = _lhs_sample(n_samples, param_ranges, rng)

    grid_nx, grid_ny = config.grid
    all_freqs = np.zeros((n_samples, config.n_modes))
    all_shapes = np.zeros((n_samples, config.n_modes, grid_ny, grid_nx))
    grid_x = None
    grid_y = None

    for i in range(n_samples):
        tf, tc, theta = params[i]
        # Face sheets: quasi-isotropic glass/epoxy
        face_mat = Material(E1=39e9, E2=8.6e9, G23=3.0e9, G13=3.0e9, G12=3.0e9, nu12=0.3, rho=1800)
        # Core: use honeycomb homogenization
        from mechanics.honeycomb import honeycomb_properties
        core_props = honeycomb_properties(
            Ec=1.0e6, Gc=0.5e6, rho_c=50, tc=tc, l1c=0.005, l2c=0.005, theta_c=theta,
        )
        core_mat = Material(**core_props)

        h_total = 2 * tf + tc
        z = [
            -h_total / 2,
            -h_total / 2 + tf,
            h_total / 2 - tf,
            h_total / 2,
        ]
        laminate = Laminate(
            materials=[face_mat, core_mat, face_mat],
            angles=[0.0, 0.0, 0.0],
            z=z,
        )

        solver = FSDTSolver(
            L1=config.L1, L2=config.L2,
            M=config.M, N=config.N,
            laminate=laminate,
            grid=config.grid,
        )
        solver.set_boundary(
            left={"type": "simply_supported"},
            right={"type": "simply_supported"},
            top={"type": "simply_supported"},
            bottom={"type": "simply_supported"},
        )
        result = solver.solve_modal(n_modes=config.n_modes)
        all_freqs[i] = result.frequencies
        all_shapes[i] = result.mode_shapes
        if grid_x is None:
            grid_x = result.grid_x
            grid_y = result.grid_y

    if grid_x is None:
        grid_x = np.linspace(0, config.L1, config.grid[0])
        grid_y = np.linspace(0, config.L2, config.grid[1])

    return {
        "frequencies": all_freqs,
        "mode_shapes": all_shapes,
        "params": params,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "config": config,
    }


def create_synthetic_hf(
    lf_dataset: dict[str, Any],
    seed: int = 123,
) -> dict[str, Any]:
    """Create synthetic HF data by perturbing LF mode shapes.

    Perturbation: κ(x,y) = 5/6 * (1 + 0.04 * sin(πx/L1) * sin(πy/L2))
    This mimics spatial variation in shear correction that FSDT misses.
    """
    rng = np.random.default_rng(seed)
    config = lf_dataset["config"]
    grid_x = lf_dataset["grid_x"]
    grid_y = lf_dataset["grid_y"]

    # Normalized coordinates
    xn = (grid_x - grid_x[0]) / (grid_x[-1] - grid_x[0])
    yn = (grid_y - grid_y[0]) / (grid_y[-1] - grid_y[0])
    Xn, Yn = np.meshgrid(xn, yn)

    # Spatial perturbation field: κ(x,y)/κ_FSDT - 1
    delta_kappa = 0.04 * np.sin(np.pi * Xn) * np.sin(np.pi * Yn)

    n_samples = lf_dataset["mode_shapes"].shape[0]
    n_modes = lf_dataset["mode_shapes"].shape[1]
    hf_shapes = np.zeros_like(lf_dataset["mode_shapes"])

    for i in range(n_samples):
        for m in range(n_modes):
            lf_mode = lf_dataset["mode_shapes"][i, m]
            # HF = LF * (1 + spatial_perturbation + small noise)
            noise = rng.normal(0, 0.002, lf_mode.shape)
            hf_shapes[i, m] = lf_mode * (1.0 + delta_kappa + noise)

    # Frequencies shift slightly too
    hf_freqs = lf_dataset["frequencies"] * (1.0 + rng.normal(0, 0.01, lf_dataset["frequencies"].shape))

    return {
        "frequencies": hf_freqs,
        "mode_shapes": hf_shapes,
        "params": lf_dataset["params"],
        "grid_x": grid_x,
        "grid_y": grid_y,
        "config": config,
    }


def extract_correction_fields(
    lf: dict[str, Any],
    hf: dict[str, Any],
    mode_idx: int = 0,
) -> np.ndarray:
    """Compute correction fields Δ(x,y) = hf_mode / lf_mode - 1.

    Returns: (n_samples, grid_ny, grid_nx)
    """
    lf_shapes = lf["mode_shapes"][:, mode_idx]
    hf_shapes = hf["mode_shapes"][:, mode_idx]
    # Sign-aware normalization: hf/lf - 1 preserves relative sign
    lf_abs = np.abs(lf_shapes)
    mask = lf_abs < 1e-12
    result = np.zeros_like(lf_shapes)
    result[~mask] = hf_shapes[~mask] / lf_shapes[~mask] - 1.0
    return result
