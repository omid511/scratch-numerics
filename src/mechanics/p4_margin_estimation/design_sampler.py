"""Design variation sampler for P4 data generation.

Samples diverse plate configurations using Sobol quasi-random sequences
for space-filling coverage of the design space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.stats.qmc import Sobol

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver


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
    left_k: float
    top_k: float
    zeta: float
    rho_air: float
    c_air: float


def _log_uniform(rng: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Transform [0,1] samples to log-uniform in [lo, hi]."""
    return lo * (hi / lo) ** rng


def sample_designs(n_designs: int, seed: int = 0) -> list[DesignSample]:
    """Sample n_designs using Sobol sequence for space-filling coverage."""
    sampler = Sobol(d=13, seed=seed, scramble=True)
    unit = sampler.random(n_designs)

    L1 = unit[:, 0] * 0.4 + 0.8                          # [0.8, 1.2]
    L2_ratio = unit[:, 1] * 0.58 + 0.75                   # [0.75, 1.33]
    L2 = L1 * L2_ratio
    face_thick = _log_uniform(unit[:, 2], 0.6e-3, 1.6e-3)   # log [0.6, 1.6] mm → Mach 2-8
    core_thick = _log_uniform(unit[:, 3], 5e-3, 12e-3)      # log [5, 12] mm
    face_E = unit[:, 4] * 0.2 + 0.9                        # [0.9, 1.1]
    face_rho = unit[:, 5] * 0.1 + 0.95                     # [0.95, 1.05]
    core_G = unit[:, 6] * 0.4 + 0.8                        # [0.8, 1.2]
    core_rho = unit[:, 7] * 0.2 + 0.9                      # [0.9, 1.1]
    left_k = _log_uniform(unit[:, 8], 1e12, 1e14)           # log [1e12, 1e14] (penalty stiffness)
    top_k = _log_uniform(unit[:, 9], 1e12, 1e14)            # log [1e12, 1e14] (elastic top)
    zeta = unit[:, 10] * 0.005                              # [0.0, 0.005] (reduced)
    rho_air = unit[:, 11] * 0.2 + 1.0                      # [1.0, 1.2]
    c_air = unit[:, 12] * 10 + 335                          # [335, 345]

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
            left_k=float(left_k[i]),
            top_k=float(top_k[i]),
            zeta=float(zeta[i]),
            rho_air=float(rho_air[i]),
            c_air=float(c_air[i]),
        ))
    return designs


def make_solver_from_design(design: DesignSample) -> FSDTSolver:
    """Create an FSDTSolver configured for this design sample."""
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
        k_stiffness=1e14,  # high penalty for clamped BC (always)
    )
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "free"},
        top={"type": "clamped"}, bottom={"type": "free"},
    )
    solver._design_zeta = design.zeta  # ponytail: stash zeta for flutter
    return solver


def _find_flutter_with_damping(
    solver: FSDTSolver,
    rho: float,
    c_sound: float,
    zeta: float,
    v_lower: float | None = None,
    v_upper: float = 10000.0,
    n_scan: int = 40,
    tol: float = 2.0,
    n_modes: int = 10,
    stability_tol: float = 1e-4,
) -> float | None:
    """Find flutter velocity with structural Rayleigh damping.

    Uses coarse scan then bisection. Precomputes structural damping once.
    """
    if v_lower is None:
        v_lower = max(300.0, c_sound * 1.01)

    M_mat, K_base, _ = solver._base_matrices()
    size = M_mat.shape[0]
    MN_eff = size // 5

    # Precompute structural damping once
    I_mat = np.eye(size)
    M_lu = linalg.lu_factor(M_mat)

    if zeta > 0:
        KM = K_base @ M_mat
        sqrt_KM = linalg.sqrtm(KM).real
        C_struct = 2.0 * zeta * sqrt_KM
    else:
        C_struct = np.zeros((size, size))

    def _max_re(vel: float) -> float:
        """Max real part of physical eigenvalues. Minimal filtering — just frequency."""
        K_air, C_air = solver.assemble_aerodynamic(vel, 0.0, rho, c_sound)
        K_total = K_base + K_air
        C_total = C_air + C_struct

        A = np.zeros((2 * size, 2 * size))
        A[:size, size:] = I_mat
        A[size:, :size] = -linalg.lu_solve(M_lu, K_total)
        A[size:, size:] = -linalg.lu_solve(M_lu, C_total)

        eigvals = linalg.eigvals(A)
        eigvals = eigvals[np.isfinite(eigvals)]
        freqs = np.abs(eigvals.imag)

        # Only keep modes with physical frequencies
        phys = (freqs > 10) & (freqs < 100000)
        if not phys.any():
            return -1.0
        return float(eigvals.real[phys].max())

    # Coarse scan
    velocities = np.linspace(v_lower, v_upper, n_scan)
    alpha = np.array([_max_re(v) for v in velocities])

    # Find first stable->unstable crossing
    trans_idx = None
    for i in range(len(alpha) - 1):
        if alpha[i] < stability_tol and alpha[i + 1] >= stability_tol:
            trans_idx = i
            break

    if trans_idx is None:
        return None

    lo, hi = float(velocities[trans_idx]), float(velocities[trans_idx + 1])
    for _ in range(30):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        mid_re = _max_re(mid)
        if mid_re < stability_tol:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def compute_design_u_crit(solver: FSDTSolver, design: DesignSample) -> float | None:
    """Compute flutter velocity for a design, accounting for structural damping."""
    zeta = getattr(solver, "_design_zeta", design.zeta)
    return _find_flutter_with_damping(
        solver, design.rho_air, design.c_air, zeta,
    )
