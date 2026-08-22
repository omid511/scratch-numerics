"""Physics gates for FSDTSolver spatial damage fields (Proposal 2 enabler).

Ones-field equivalence (quadrature vs analytic integrals), legendre basis:
8.7e-15 relative Frobenius — machine precision. For the trigonometric
basis the QUADRATURE path is exact (~5e-15) while the pre-existing
analytic integral tables under-resolve high-mode trig products
(~3e-4 relative); an all-ones field therefore shifts trig-basis
frequencies ~0.015% vs the no-field state. The trig gate below uses that
honest tolerance; see RevP2CodeB/SciVerif notes in the P2 review round.
"""
from __future__ import annotations

import numpy as np

from mechanics.laminate import Laminate, Material
from mechanics.solver import FSDTSolver


def _isotropic_laminate() -> Laminate:
    E, nu, rho = 70e9, 0.33, 2700.0
    G = E / (2 * (1 + nu))
    return Laminate(
        [Material(E, E, G, G, G, nu, rho)] * 3,
        [0.0, 0.0, 0.0],
        [-0.006, -0.002, 0.002, 0.006],
    )


def _solver() -> FSDTSolver:
    solver = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6,
                        laminate=_isotropic_laminate(), grid=(16, 16))
    solver.set_boundary(
        left={"type": "simply_supported"},
        right={"type": "simply_supported"},
        top={"type": "simply_supported"},
        bottom={"type": "simply_supported"},
    )
    return solver


def _patch_field(n: int = 8, iy0: int | None = None, ix0: int | None = None,
                 size: int = 3, depth: float = 0.5) -> np.ndarray:
    field = np.ones((n, n))
    iy0 = (n - size) // 2 if iy0 is None else iy0
    ix0 = (n - size) // 2 if ix0 is None else ix0
    field[iy0:iy0 + size, ix0:ix0 + size] = depth
    return field


def test_ones_field_matches_analytic_path():
    solver = _solver()
    k_ref = solver.assemble_stiffness()
    solver.set_damage_field(np.ones((8, 8)))
    k_dmg = solver._assemble_stiffness_quadrature()
    err = np.linalg.norm(k_dmg - k_ref) / np.linalg.norm(k_ref)
    # Measured 8.7e-15 on this fixture; gate is a regression tripwire.
    assert err < 0.01
    assert err < 1e-10


def test_clearing_damage_restores_reference():
    solver = _solver()
    k_ref = solver.assemble_stiffness().copy()
    solver.set_damage_field(_patch_field(depth=0.4))
    k_dmg = solver.assemble_stiffness()
    assert np.linalg.norm(k_dmg - k_ref) > 0
    solver.set_damage_field(None)
    assert np.allclose(solver.assemble_stiffness(), k_ref)


def test_center_patch_decreases_frequencies():
    f_pristine = _solver().solve_modal(n_modes=4).frequencies
    solver = _solver()
    solver.set_damage_field(_patch_field(n=8, depth=0.5))  # 3x3/8x8 = 14%
    f_damaged = solver.solve_modal(n_modes=4).frequencies
    assert np.all(f_damaged < f_pristine[: len(f_damaged)])


def test_deeper_damage_monotonicity():
    # Field stores stiffness RETENTION; damage severity = 1 - retention.
    f1 = {}
    for severity in (0.5, 0.9):
        s = _solver()
        s.set_damage_field(_patch_field(n=8, depth=1.0 - severity))
        f1[severity] = s.solve_modal(n_modes=2).frequencies[0]
    # More severe damage -> lower first frequency.
    assert f1[0.9] < f1[0.5]


def test_edge_patch_locality():
    n, size = 8, 3
    s_center = _solver()
    s_center.set_damage_field(_patch_field(n=n, size=size, depth=0.4))
    fc = s_center.solve_modal(n_modes=2).frequencies[0]

    s_edge = _solver()
    s_edge.set_damage_field(_patch_field(n=n, iy0=0, ix0=0,
                                         size=size, depth=0.4))
    fe = s_edge.solve_modal(n_modes=2).frequencies[0]

    # Center damage removes stiffness where transverse mode amplitude is
    # largest; same-size corner damage perturbs f1 less or equal.
    assert fe >= fc


def test_trig_basis_ones_field_tolerance():
    """Trig analytic tables under-resolve high-mode products; the
    quadrature path is the accurate side. All-ones field may shift
    frequencies ~0.015% — bounded here, not machine precision."""
    E, nu, rho = 70e9, 0.33, 2700.0
    G = E / (2 * (1 + nu))
    lam = Laminate([Material(E, E, G, G, G, nu, rho)] * 3,
                   [0.0, 0.0, 0.0], [-0.006, -0.002, 0.002, 0.006])
    s = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam,
                   basis_type="trigonometric", grid=(16, 16))
    s.set_boundary(left={"type": "simply_supported"},
                   right={"type": "simply_supported"},
                   top={"type": "simply_supported"},
                   bottom={"type": "simply_supported"})
    f_ref = s.solve_modal(n_modes=4).frequencies.copy()
    s.set_damage_field(np.ones((8, 8)))
    f_ones = s.solve_modal(n_modes=4).frequencies
    rel = np.abs(f_ones - f_ref) / f_ref
    assert np.all(rel < 5e-4), f"trig ones-field drift {rel.max():.2e}"
