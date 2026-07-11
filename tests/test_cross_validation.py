"""Cross-validation of new FSDTSolver against original plate-main reference code.

Compares natural frequencies from both implementations across multiple
configurations. Original plate-main is the ground truth.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'plate-main'))

import plate
import honeycomb


def _make_mat(E, nu, G, rho):
    return plate.Material(E, E, G, G, G, nu, rho)


def _honeycomb_core():
    E, nu, rho = 70e9, 0.33, 2710
    G = E / (2 * (1 + nu))
    return plate.Material(*honeycomb.material_property(
        E, G, rho, 0.2e-3, 3e-3, 3e-3, 30 / 180 * np.pi
    ))


def _solve_original(profile, L1, L2, M, N, springs):
    p = plate.Plate(L1, L2, M, N, profile)
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    p.set_basis(basis_x, basis_y)
    for value, dof, x, y in springs:
        p.add_spring(value, dof, x=x, y=y)
    solver = plate.ModalSolver(p)
    result = solver.solve(20)
    return np.array([r.frequency for r in result])


def _solve_new(materials, angles, z, L1, L2, M, N, springs_cfg, k=1e12):
    from mechanics.laminate import Material, Laminate
    from mechanics.solver import FSDTSolver

    lam = Laminate(materials=materials, angles=angles, z=z)
    solver = FSDTSolver(L1=L1, L2=L2, M=M, N=N, laminate=lam,
                        basis_type='legendre', k_stiffness=k)
    solver.set_boundary(**springs_cfg)
    result = solver.solve_modal(n_modes=20)
    return np.real(result.frequencies)


def _compare(freqs_orig, freqs_new, n_modes=6, label=""):
    n = min(n_modes, len(freqs_orig), len(freqs_new))
    max_err = 0.0
    for i in range(n):
        rel_err = abs(freqs_orig[i] - freqs_new[i]) / freqs_orig[i]
        max_err = max(max_err, rel_err)
    return max_err


# --- Config 1: Sandwich plate, CCCC (paper example 2_1) ---

def _config1():
    E, nu, rho = 70e9, 0.33, 2710
    G = E / (2 * (1 + nu))
    face = _make_mat(E, nu, G, rho)
    core = _honeycomb_core()
    z = [-5e-3, -4e-3, 4e-3, 5e-3]
    bc = dict(left={"type": "clamped"}, right={"type": "clamped"},
              top={"type": "clamped"}, bottom={"type": "clamped"})
    springs = [(1e12, d, x, y) for d in range(5) for x, y in
               [(0, None), (0.3, None), (None, 0), (None, 0.3)]]
    return {
        "label": "Sandwich CCCC 0.3x0.3",
        "orig_materials": [face, core, face],
        "orig_angles": [0, 0, 0],
        "orig_z": z,
        "new_materials": [face, core, face],
        "new_angles": [0, 0, 0],
        "new_z": z,
        "L1": 0.3, "L2": 0.3, "M": 15, "N": 15,
        "springs_orig": springs,
        "springs_new": bc,
        "n_modes": 6,
    }


# --- Config 2: Isotropic aluminium, SSSS ---

def _config2():
    E, nu, rho, h = 210e9, 0.3, 7800, 0.002
    G = E / (2 * (1 + nu))
    mat = _make_mat(E, nu, G, rho)
    z = [-h / 2, h / 2]
    bc = dict(left={"type": "simply_supported"}, right={"type": "simply_supported"},
              top={"type": "simply_supported"}, bottom={"type": "simply_supported"})
    springs = []
    for dof in plate.BC_S:
        for x, y in [(0, None), (0.4, None), (None, 0), (None, 0.3)]:
            springs.append((1e12, dof, x, y))
    return {
        "label": "Isotropic SSSS 0.4x0.3",
        "orig_materials": [mat],
        "orig_angles": [0],
        "orig_z": z,
        "new_materials": [mat],
        "new_angles": [0],
        "new_z": z,
        "L1": 0.4, "L2": 0.3, "M": 12, "N": 12,
        "springs_orig": springs,
        "springs_new": bc,
        "n_modes": 6,
    }


# --- Config 3: Aluminium, CCCC, thick plate ---

def _config3():
    E, nu, rho, h = 70e9, 0.33, 2710, 0.003
    G = E / (2 * (1 + nu))
    mat = _make_mat(E, nu, G, rho)
    z = [-h / 2, h / 2]
    bc = dict(left={"type": "clamped"}, right={"type": "clamped"},
              top={"type": "clamped"}, bottom={"type": "clamped"})
    springs = [(1e12, d, x, y) for d in range(5) for x, y in
               [(0, None), (0.2, None), (None, 0), (None, 0.2)]]
    return {
        "label": "Isotropic CCCC 0.2x0.2",
        "orig_materials": [mat],
        "orig_angles": [0],
        "orig_z": z,
        "new_materials": [mat],
        "new_angles": [0],
        "new_z": z,
        "L1": 0.2, "L2": 0.2, "M": 15, "N": 15,
        "springs_orig": springs,
        "springs_new": bc,
        "n_modes": 6,
    }


CONFIGS = [_config1(), _config2(), _config3()]


class TestCrossValidation:
    @pytest.mark.parametrize("cfg", CONFIGS, ids=[c["label"] for c in CONFIGS])
    def test_modal_frequencies_match(self, cfg):
        """Natural frequencies agree within tolerance."""
        from mechanics.laminate import Material, Laminate

        profile = plate.Profile(
            cfg["orig_materials"], cfg["orig_angles"], cfg["orig_z"]
        )
        freqs_orig = _solve_original(
            profile, cfg["L1"], cfg["L2"], cfg["M"], cfg["N"],
            cfg["springs_orig"]
        )

        materials = [
            Material(m._E1, m._E2, m._G23, m._G13, m._G12, m._nu12, m._rho)
            for m in cfg["new_materials"]
        ]
        freqs_new = _solve_new(
            materials, cfg["new_angles"], cfg["new_z"],
            cfg["L1"], cfg["L2"], cfg["M"], cfg["N"],
            cfg["springs_new"]
        )

        max_err = _compare(freqs_orig, freqs_new, cfg["n_modes"], cfg["label"])
        assert max_err < 2e-4, (
            f"{cfg['label']}: max relative error {max_err:.2e} exceeds 2e-4"
        )

    def test_mass_stiffness_matrices_match(self):
        """Mass and stiffness matrices agree element-wise."""
        from mechanics.laminate import Material, Laminate
        from mechanics.solver import FSDTSolver

        E, nu, rho = 70e9, 0.33, 2710
        G = E / (2 * (1 + nu))
        face = _make_mat(E, nu, G, rho)
        core = _honeycomb_core()
        z = [-5e-3, -4e-3, 4e-3, 5e-3]

        profile = plate.Profile([face, core, face], [0, 0, 0], z)
        p = plate.Plate(0.3, 0.3, 10, 10, profile)
        basis_x = plate.legendre_basis(50, interval=[0, 0.3])
        basis_y = plate.legendre_basis(50, interval=[0, 0.3])
        p.set_basis(basis_x, basis_y)
        M_orig = p.get_M()
        K_orig = p.get_K()

        materials = [
            Material(m._E1, m._E2, m._G23, m._G13, m._G12, m._nu12, m._rho)
            for m in [face, core, face]
        ]
        lam = Laminate(materials=materials, angles=[0, 0, 0], z=z)
        solver = FSDTSolver(L1=0.3, L2=0.3, M=10, N=10, laminate=lam,
                            basis_type='legendre', k_stiffness=1e12)
        M_new = solver.assemble_mass()
        K_new = solver.assemble_stiffness()

        rel_err_M = np.max(np.abs(M_orig - M_new)) / np.max(np.abs(M_orig))
        rel_err_K = np.max(np.abs(K_orig - K_new)) / np.max(np.abs(K_orig))

        assert rel_err_M < 1e-10, f"Mass matrix rel error: {rel_err_M:.2e}"
        assert rel_err_K < 1e-10, f"Stiffness matrix rel error: {rel_err_K:.2e}"


# Allow running standalone
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
