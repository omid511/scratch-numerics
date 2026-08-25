"""Physics gates for the exact adjoint identifiability atlas.

Covers four properties of ``mechanics.sensitivity_adjoint``:

1. Hellmann-Feynman: analytic d(omega)/d(f_c) agrees with central finite
   differences of the forward modal solve.
2. Ones-field equivalence: the quadrature assembly partitions -- the sum of
   per-cell stiffness matrices reproduces the full stiffness matrix.
3. Atlas output contract: ranks, indices, and basis shapes are consistent.
4. Determinism: identical inputs give bit-identical atlases.
"""
from __future__ import annotations

import numpy as np

from mechanics.laminate import Laminate, Material
from mechanics.solver import FSDTSolver
from mechanics.sensitivity_adjoint import (
    frequency_sensitivity,
    identifiability_atlas,
    per_cell_stiffness,
)

FIELD_SHAPE = (4, 4)


def _isotropic_laminate() -> Laminate:
    E, nu, rho = 70e9, 0.33, 2700.0
    G = E / (2 * (1 + nu))
    return Laminate(
        [Material(E, E, G, G, G, nu, rho)] * 3,
        [0.0, 0.0, 0.0],
        [-0.006, -0.002, 0.002, 0.006],
    )


def _solver() -> FSDTSolver:
    # Non-square plate breaks (m, n) <-> (n, m) modal degeneracy, so every
    # frequency derivative below is well defined per mode.
    solver = FSDTSolver(L1=0.3, L2=0.22, M=4, N=4,
                        laminate=_isotropic_laminate(), grid=(8, 8))
    solver.set_boundary(
        left={"type": "simply_supported"},
        right={"type": "simply_supported"},
        top={"type": "simply_supported"},
        bottom={"type": "simply_supported"},
    )
    return solver


def _damage_field() -> np.ndarray:
    # Non-uniform retentions strictly inside (0, 1] so +/- perturbations
    # stay valid damage fields.
    return np.linspace(0.70, 0.98, 16).reshape(FIELD_SHAPE)


# ---- 1. Hellmann-Feynman vs central finite differences -------------------


def test_hellmann_feynman_sensitivities_match_finite_differences():
    solver = _solver()
    field = _damage_field()
    solver.set_damage_field(field)

    n_modes = 3
    sens, freqs, centers = frequency_sensitivity(solver, FIELD_SHAPE,
                                                 n_modes=n_modes)
    assert sens.shape == (n_modes, FIELD_SHAPE[0] * FIELD_SHAPE[1])

    # At least 3 spread-out cells: corner, interior, opposite corner.
    probe_cells = [(0, 0), (2, 1), (3, 3)]
    delta = 1.0e-3
    for iy, ix in probe_cells:
        c = iy * FIELD_SHAPE[1] + ix
        omegas = {}
        for tag, df in (("plus", +delta), ("minus", -delta)):
            f = field.copy()
            f[iy, ix] += df
            solver.set_damage_field(f)
            res = solver.solve_modal(n_modes=n_modes)
            omegas[tag] = 2.0 * np.pi * np.asarray(res.frequencies[:n_modes])
        fd = (omegas["plus"] - omegas["minus"]) / (2.0 * delta)
        rel = np.abs(sens[:, c] - fd) / np.abs(fd)
        assert rel.max() < 1.0e-6, (
            f"HF vs FD mismatch at cell {c}: {sens[:, c]} vs {fd}"
        )


# ---- 2. Ones-field equivalence -------------------------------------------


def test_ones_field_per_cell_matrices_sum_to_full_stiffness():
    solver = _solver()
    solver.set_damage_field(np.ones(FIELD_SHAPE))

    cells, centers = per_cell_stiffness(solver, FIELD_SHAPE)
    assert len(cells) == FIELD_SHAPE[0] * FIELD_SHAPE[1]
    assert centers.shape == (len(cells), 2)

    K_full = solver.assemble_stiffness()
    K_sum = np.zeros_like(K_full)
    for Kc in cells:
        assert Kc.shape == K_full.shape
        K_sum += Kc

    rel = np.linalg.norm(K_sum - K_full) / np.linalg.norm(K_full)
    assert rel < 1.0e-12


# ---- 3. Atlas output contract --------------------------------------------


def test_atlas_output_contract():
    solver = _solver()
    solver.set_damage_field(_damage_field())

    atlas = identifiability_atlas(solver, FIELD_SHAPE, n_modes=4)
    n_cells = FIELD_SHAPE[0] * FIELD_SHAPE[1]

    eff = atlas["effective_rank"]
    assert isinstance(eff, int)
    assert 0 < eff <= n_cells
    assert atlas["nullspace_dimension"] == n_cells - eff

    blind = atlas["blind_spot_cell_indices"]
    assert isinstance(blind, list)
    assert all(isinstance(i, int) and 0 <= i < n_cells for i in blind)
    assert len(set(blind)) == len(blind)

    basis = np.asarray(atlas["identifiable_basis"])
    assert basis.shape == (eff, n_cells)

    sv = np.asarray(atlas["singular_values"])
    assert sv.shape == (min(atlas["n_modes"], n_cells),)
    assert np.all(sv >= 0.0)

    cid = np.asarray(atlas["per_cell_identifiability"])
    assert cid.shape == (n_cells,)
    assert np.all(cid >= 0.0)
    # Blind spots are exactly the lowest-identifiability cells.
    assert set(blind) == set(
        np.flatnonzero(cid <= np.quantile(cid, 0.1)).tolist()
    )


# ---- 4. Determinism -------------------------------------------------------


def test_atlas_is_deterministic():
    def build():
        solver = _solver()
        solver.set_damage_field(_damage_field())
        return identifiability_atlas(solver, FIELD_SHAPE, n_modes=4)

    a, b = build(), build()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], f"atlas entry {key!r} differs between runs"
