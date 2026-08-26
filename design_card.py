#!/usr/bin/env python3
"""Design-card CLI: one-command honeycomb sandwich plate analysis.

Usage:
    uv run python design_card.py --L 0.3 --h-face 0.001 --h-core 0.008 \
        --E-face 70e9 --E-core 47.3e6 --bc CFCF [--velocities 700 900 1200]

Generates a design card with: natural frequencies, flutter boundary,
stability margins, and mode-shape summaries.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def build_solver(L, h_face, h_core, E_face, E_core, bc):
    from mechanics.laminate import Laminate, Material
    from mechanics.solver import FSDTSolver

    nu = 0.33
    G_face = E_face / (2 * (1 + nu))
    G_core = E_core / (2 * (1 + nu))
    rho_face, rho_core = 2710.0, 278.0

    face = Material(E_face, E_face, G_face, G_face, G_face, nu, rho_face)
    core = Material(E_core, E_core, G_core, G_core, G_core, nu, rho_core)
    h_total = 2 * h_face + h_core
    z = [-h_total / 2, -h_total / 2 + h_face,
         h_total / 2 - h_face, h_total / 2]
    lam = Laminate([face, core, face], [0, 0, 0], z)

    solver = FSDTSolver(L1=L, L2=L, M=8, N=8, laminate=lam, grid=(32, 32))
    bc_map = {
        "CFCF": ("clamped", "clamped", "free", "free"),
        "SSSS": ("simply_supported", "simply_supported",
                 "simply_supported", "simply_supported"),
        "CCCC": ("clamped", "clamped", "clamped", "clamped"),
    }
    edges = bc_map.get(bc.upper(), bc_map["CFCF"])
    solver.set_boundary(
        left={"type": edges[0]}, right={"type": edges[1]},
        top={"type": edges[2]}, bottom={"type": edges[3]},
    )
    return solver


def main():
    parser = argparse.ArgumentParser(description="Honeycomb sandwich plate design card")
    parser.add_argument("--L", type=float, default=0.3, help="Plate side length (m)")
    parser.add_argument("--h-face", type=float, default=0.001, help="Face thickness (m)")
    parser.add_argument("--h-core", type=float, default=0.008, help="Core thickness (m)")
    parser.add_argument("--E-face", type=float, default=70e9, help="Face Young's modulus (Pa)")
    parser.add_argument("--E-core", type=float, default=47.3e6, help="Core Young's modulus (Pa)")
    parser.add_argument("--bc", type=str, default="CFCF", help="Boundary condition (CFCF/SSSS/CCCC)")
    parser.add_argument("--velocities", type=float, nargs="*", default=[], help="Flow velocities to evaluate (m/s)")
    parser.add_argument("--n-modes", type=int, default=6, help="Number of modes")
    args = parser.parse_args()

    from mechanics.solver import FSDTSolver

    solver = build_solver(args.L, args.h_face, args.h_core,
                          args.E_face, args.E_core, args.bc)

    print("=" * 60)
    print("DESIGN CARD")
    print("=" * 60)
    print(f"  Plate: {args.L} x {args.L} m, BC: {args.bc}")
    print(f"  Face: h={args.h_face*1e3:.2f} mm, E={args.E_face/1e9:.1f} GPa")
    print(f"  Core: h={args.h_core*1e3:.2f} mm, E={args.E_core/1e6:.2f} MPa")

    # Modal analysis
    result = solver.solve_modal(n_modes=args.n_modes)
    print(f"\n  Natural frequencies (Hz):")
    for i, f in enumerate(result.frequencies):
        print(f"    mode {i+1}: {f:.1f}")

    # Flutter boundary
    print(f"\n  Flutter analysis:")
    try:
        u_crit = solver.find_flutter_boundary(n_modes=6)
        if u_crit is not None:
            print(f"    lambda_cr = {u_crit:.1f}")
        else:
            print(f"    lambda_cr = not found in search range")
    except Exception as e:
        print(f"    lambda_cr = ERROR: {e}")

    # Flow velocity evaluation
    if args.velocities:
        print(f"\n  Flow velocity evaluation:")
        from mechanics.piston_theory import non_dimensional_lambda
        D11 = solver._compute_D11()
        rho_air, c_sound = 1.2, 340.0
        for v in args.velocities:
            try:
                lam_val = non_dimensional_lambda(v, rho_air, args.L, D11, c_sound)
                mach = v / c_sound
                print(f"    V={v:7.1f} m/s (M={mach:.2f}) lambda={lam_val:.1f}")
            except Exception as e:
                print(f"    V={v:7.1f} ERROR: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
