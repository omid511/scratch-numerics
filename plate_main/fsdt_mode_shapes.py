#!/usr/bin/env python3
"""
FSDT Mode Shape Generator
==========================
Computes FSDT mode shapes for all 100 LHS samples on 80x80 grid.
Saves to CSV files matching COMSOL mode shape format.

Usage:
    python fsdt_mode_shapes.py
"""

import sys
import os
import csv
import time
import numpy as np

sys.path.insert(0, '.')
import plate
import honeycomb

# ============================================================================
# CONFIGURATION (must match lhs_sampling.py)
# ============================================================================

N_SAMPLES = 100
N_MODES = 10
M_ORDER = 15
N_BASIS = 50
GRID_RES = 80

L1 = 0.3
L2 = 0.3
H_TOTAL = 0.01

E_face = 70e9
rho_face = 2710
nu_face = 0.33
G_face = E_face / (2 * (1 + nu_face))

K_SPRING = 1e12

PARAM_RANGES = {
    'alpha':   [0.1, 0.9],
    'beta':    [0.1, 1.0],
    'theta_c': [0.0, 75.0],
    'eta1':    [0.5, 3.0],
    'eta2':    [0.02, 0.15],
}

LHS_FILE = "D:/plate-main/lhs_samples.csv"
OUTPUT_DIR = "D:/plate-main/fsdt_mode_shapes"


def compute_fsdt_mode_shapes(alpha, beta, theta_c, eta1, eta2):
    """Compute FSDT frequencies and mode shapes on 80x80 grid."""
    t_core = eta2 * 3e-3
    l1 = 3e-3
    l2 = l1 * eta1
    theta = theta_c / 180 * np.pi

    h2 = H_TOTAL * alpha
    h1 = (H_TOTAL - h2) / (1 + beta) * beta
    h3 = H_TOTAL - h1 - h2

    z = np.array([0, h1, h1 + h2, h1 + h2 + h3])
    z -= z[-1] / 2

    al = plate.Material(E_face, E_face, G_face, G_face, G_face,
                        nu_face, rho_face)
    core_props = honeycomb.material_property(
        E_face, G_face, rho_face, t_core, l1, l2, theta)
    core = plate.Material(*core_props)

    profile = plate.Profile([al, core, al], [0, 0, 0], z)

    p = plate.Plate(L1, L2, M_ORDER, M_ORDER, profile)
    basis_x = plate.legendre_basis(N_BASIS, interval=[0, L1])
    basis_y = plate.legendre_basis(N_BASIS, interval=[0, L2])
    p.set_basis(basis_x, basis_y)

    p.add_spring(K_SPRING, plate.BC_C, x=0)
    p.add_spring(K_SPRING, plate.BC_C, x=L1)
    p.add_spring(K_SPRING, plate.BC_C, y=0)
    p.add_spring(K_SPRING, plate.BC_C, y=L2)

    p.assemble()
    solver = plate.ModalSolver(p)
    result = solver.solve(N_MODES * 2)

    freqs = np.array([result[i].frequency for i in range(N_MODES)])

    x_grid = np.linspace(0, L1, GRID_RES)
    y_grid = np.linspace(0, L2, GRID_RES)

    mode_shapes = []
    for i in range(N_MODES):
        frame = result[i]
        w_grid = np.zeros((GRID_RES, GRID_RES))
        for ix, xi in enumerate(x_grid):
            for iy, yi in enumerate(y_grid):
                w_grid[iy, ix] = frame.w(xi, yi)
        max_abs = np.max(np.abs(w_grid))
        if max_abs > 0:
            w_grid = w_grid / max_abs
        mode_shapes.append(w_grid)

    return freqs, mode_shapes


def main():
    print("=" * 70)
    print("FSDT MODE SHAPE GENERATOR")
    print("=" * 70)

    samples = []
    with open(LHS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({k: float(v) for k, v in row.items()})
    print(f"Loaded {len(samples)} LHS samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xv = np.linspace(0, L1, GRID_RES)
    yv = np.linspace(0, L2, GRID_RES)
    Xg, Yg = np.meshgrid(xv, yv)

    t_start = time.time()
    for idx, s in enumerate(samples):
        run_num = idx + 1
        print(f"  Run {run_num}/{len(samples)}: alpha={s['alpha']:.3f} beta={s['beta']:.3f} "
              f"theta_c={s['theta_c']:.1f} eta1={s['eta1']:.3f} eta2={s['eta2']:.4f} ...", end="", flush=True)

        t_run = time.time()
        try:
            freqs, modes = compute_fsdt_mode_shapes(
                s['alpha'], s['beta'], s['theta_c'], s['eta1'], s['eta2'])

            for mode_idx in range(N_MODES):
                out_path = os.path.join(OUTPUT_DIR, f"fsdt_run{run_num}_mode{mode_idx+1}.csv")
                flat = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(GRID_RES**2),
                                        np.zeros(GRID_RES**2), modes[mode_idx].ravel()])
                header = (f"Run {run_num} Mode {mode_idx+1} @ {freqs[mode_idx]:.1f} Hz\n"
                          f"x,y,u,v,w\n{GRID_RES}x{GRID_RES} grid")
                np.savetxt(out_path, flat, delimiter=",", header=header, comments="")

            elapsed = time.time() - t_run
            print(f" OK ({freqs[0]:.1f} Hz, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t_run
            print(f" FAILED: {e} ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nDone! {len(samples)} samples in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
