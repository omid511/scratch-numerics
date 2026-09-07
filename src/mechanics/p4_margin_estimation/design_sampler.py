"""Design variation sampler for P4 data generation.

Samples diverse plate configurations using Sobol quasi-random sequences
for space-filling coverage of the design space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver, compute_structural_damping


# Nominal material properties
NOMINAL_FACE = Material(
    E1=70e9, E2=70e9, G23=26.32e9, G13=26.32e9, G12=26.32e9,
    nu12=0.33, rho=2710.0,
)
NOMINAL_CORE = Material(
    E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7,
    nu12=0.98, rho=278.15,
)


@dataclass
class DesignSample:
    """One sampled plate configuration."""
    design_id: str
    L1: float
    L2: float
    face_thickness: float
    core_thickness: float
    face_E_mult: float
    face_rho_mult: float
    core_G_mult: float
    core_rho_mult: float
    zeta: float
    rho_air: float
    c_air: float


def _log_uniform(rng: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Transform [0,1] samples to log-uniform in [lo, hi]."""
    return lo * (hi / lo) ** rng


def sample_designs(n_designs: int, seed: int = 0) -> list[DesignSample]:
    """Sample n_designs using Sobol sequence for space-filling coverage.

    11 Sobol dimensions: L1, L2_ratio, face_thick, core_thick, face_E,
    face_rho, core_G, core_rho, zeta, rho_air, c_air.
    Uses random_base2 (powers of two preserve Sobol balance; a non-power-of-2
    n_designs takes a prefix, which stays space-filling but is not perfectly balanced).
    """
    if n_designs <= 0:
        raise ValueError("n_designs must be positive")
    sampler = Sobol(d=11, seed=seed, scramble=True)
    exponent = math.ceil(math.log2(max(n_designs, 2)))
    unit = sampler.random_base2(m=exponent)[:n_designs]

    L1 = unit[:, 0] * 0.4 + 0.8                          # [0.8, 1.2]
    L2_ratio = unit[:, 1] * 0.58 + 0.75                   # [0.75, 1.33]
    L2 = L1 * L2_ratio
    face_thick = _log_uniform(unit[:, 2], 0.6e-3, 1.6e-3)   # log [0.6, 1.6] mm
    core_thick = _log_uniform(unit[:, 3], 5e-3, 12e-3)      # log [5, 12] mm
    face_E = unit[:, 4] * 0.2 + 0.9                        # [0.9, 1.1]
    face_rho = unit[:, 5] * 0.1 + 0.95                     # [0.95, 1.05]
    core_G = unit[:, 6] * 0.4 + 0.8                        # [0.8, 1.2]
    core_rho = unit[:, 7] * 0.2 + 0.9                      # [0.9, 1.1]
    zeta = unit[:, 8] * 0.005                              # [0.0, 0.005]
    rho_air = unit[:, 9] * 0.2 + 1.0                      # [1.0, 1.2]
    c_air = unit[:, 10] * 10 + 335                          # [335, 345]

    designs = []
    for i in range(n_designs):
        designs.append(DesignSample(
            design_id=f"D{i:06d}",
            L1=float(L1[i]),
            L2=float(L2[i]),
            face_thickness=float(face_thick[i]),
            core_thickness=float(core_thick[i]),
            face_E_mult=float(face_E[i]),
            face_rho_mult=float(face_rho[i]),
            core_G_mult=float(core_G[i]),
            core_rho_mult=float(core_rho[i]),
            zeta=float(zeta[i]),
            rho_air=float(rho_air[i]),
            c_air=float(c_air[i]),
        ))
    return designs


def make_solver_from_design(design: DesignSample) -> FSDTSolver:
    """Create an FSDTSolver configured for this design sample.

    Always applies fixed CFCF boundary conditions.
    """
    face = Material(
        E1=NOMINAL_FACE.E1 * design.face_E_mult,
        E2=NOMINAL_FACE.E2 * design.face_E_mult,
        G23=NOMINAL_FACE.G23 * design.face_E_mult,
        G13=NOMINAL_FACE.G13 * design.face_E_mult,
        G12=NOMINAL_FACE.G12 * design.face_E_mult,
        nu12=NOMINAL_FACE.nu12,
        rho=NOMINAL_FACE.rho * design.face_rho_mult,
    )
    core = Material(
        E1=NOMINAL_CORE.E1,
        E2=NOMINAL_CORE.E2,
        G23=NOMINAL_CORE.G23 * design.core_G_mult,
        G13=NOMINAL_CORE.G13 * design.core_G_mult,
        G12=NOMINAL_CORE.G12 * design.core_G_mult,
        nu12=NOMINAL_CORE.nu12,
        rho=NOMINAL_CORE.rho * design.core_rho_mult,
    )

    tf, tc = design.face_thickness, design.core_thickness
    half = tc / 2
    lam = Laminate(
        materials=[face, core, face],
        angles=[0.0, 0.0, 0.0],
        z=[-half - tf, -half, half, half + tf],
    )

    solver = FSDTSolver(
        L1=design.L1, L2=design.L2,
        M=6, N=6,
        laminate=lam,
        grid=(32, 32),
        # NOTE: do NOT set penalty_factor here. _base_matrices scales springs to
        # penalty_factor * max(|diag(K_struct)|), which makes the second-order
        # pencil so ill-conditioned that dense eig returns backward errors
        # O(0.1-1) for most of the spectrum and spectral_abscissa's validity
        # gate rejects every eigenvalue (flutter evaluation then fails for all
        # designs). Absolute k_stiffness springs keep min backward error
        # <= ~4e-7 while still enforcing effectively rigid clamps.
    )
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "free"},
        top={"type": "clamped"}, bottom={"type": "free"},
    )
    return solver


def compute_design_u_crit(solver: FSDTSolver, design: DesignSample, v_lower: float | None = None) -> float | None:
    """Compute flutter velocity for a design, accounting for structural damping."""
    kwargs: dict = dict(
        rho=design.rho_air,
        c_sound=design.c_air,
        zeta=design.zeta,
        v_lower=v_lower,
        n_scan=40,
        velocity_tol=2.0,
    )
    return solver.find_flutter_velocity(**kwargs)
