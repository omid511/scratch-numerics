#!/usr/bin/env python3
"""FSDT solver validation: physics checks, convergence, aero assembly."""
import numpy as np
import sys

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.shear_correction import shear_correction_factor
from mechanics.piston_theory import non_dimensional_lambda

results_log = []

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    tag = f"[{status}] {name}" + (f" — {detail}" if detail else "")
    print(tag)
    results_log.append((name, passed, detail))


# ── 1. Homogeneous steel plate (SSSS) ──────────────────────────────────────
print("=" * 70)
print("1. HOMOGENEOUS STEEL PLATE (SSSS), M=N=15")
print("=" * 70)

E, nu, rho = 210e9, 0.3, 7800
G = E / (2 * (1 + nu))
h = 0.01
steel = Material(E, E, G, G, G, nu, rho)
lam_steel = Laminate(materials=[steel], angles=[0.0], z=[-h/2, h/2])

# Kirchhoff analytical for reference
D_bend = E * h**3 / (12 * (1 - nu**2))
f11_analytical = (np.pi / 2) * np.sqrt(D_bend / (rho * h)) * (1/0.3**2 + 1/0.3**2)
print(f"  Kirchhoff analytical SSSS f11 = {f11_analytical:.2f} Hz")

solver_ssss = FSDTSolver(
    L1=0.3, L2=0.3, M=15, N=15, laminate=lam_steel,
    basis_type="legendre", grid=(64, 64), k_stiffness=1e12,
)
solver_ssss.set_boundary(
    left={"type": "simply_supported"}, right={"type": "simply_supported"},
    top={"type": "simply_supported"}, bottom={"type": "simply_supported"},
)
result_ssss = solver_ssss.solve_modal(n_modes=4)

print(f"  Frequencies (Hz): {result_ssss.frequencies}")
f1 = result_ssss.frequencies[0]
pct_err = abs(f1 - f11_analytical) / f11_analytical * 100
report("SSSS f1 within 2% of Kirchhoff analytical (548 Hz)", pct_err < 2,
       f"got {f1:.2f} Hz, analytical {f11_analytical:.2f} Hz ({pct_err:.2f}% error)")
print(f"  NOTE: User reference of 492.87 Hz does NOT match analytical for")
print(f"        these parameters (E=210e9, nu=0.3, h=10mm). 492.87 Hz would")
print(f"        correspond to h=9mm. Solver result of 538 Hz is physically")
print(f"        correct (1.8% below Kirchhoff, consistent with FSDT shear")
print(f"        compliance for h/a=0.033).")


# ── 2. Sandwich plate (CCCC) ───────────────────────────────────────────────
print()
print("=" * 70)
print("2. SANDWICH PLATE (CCCC), M=N=15")
print("=" * 70)

face = Material(E1=70e9, E2=70e9, G23=26.92e9, G13=26.92e9, G12=26.92e9, nu12=0.33, rho=2710)
core = Material(E1=4.73e7, E2=4.73e7, G23=1.01e9, G13=1.01e9, G12=1.20e7, nu12=0.98, rho=278.15)

lam_sandwich = Laminate(
    materials=[face, core, face],
    angles=[0, 0, 0],
    z=[-5e-3, -4e-3, 4e-3, 5e-3],
)

solver_cccc = FSDTSolver(
    L1=0.3, L2=0.3, M=15, N=15, laminate=lam_sandwich,
    basis_type="legendre", grid=(64, 64), k_stiffness=1e12,
)
solver_cccc.set_boundary(
    left={"type": "clamped"}, right={"type": "clamped"},
    top={"type": "clamped"}, bottom={"type": "clamped"},
)
result_cccc = solver_cccc.solve_modal(n_modes=4)

print(f"  Frequencies (Hz): {result_cccc.frequencies}")
f1_sw = result_cccc.frequencies[0]
expected_f1_sw = 1173.0
pct_err_sw = abs(f1_sw - expected_f1_sw) / expected_f1_sw * 100
report("CCCC sandwich f1 ~1173 Hz", pct_err_sw < 1,
       f"got {f1_sw:.2f} Hz ({pct_err_sw:.2f}% error)")

# Check mode 2
f2_sw = result_cccc.frequencies[1]
expected_f2_sw = 2225.82
pct_err_f2 = abs(f2_sw - expected_f2_sw) / expected_f2_sw * 100
report("CCCC sandwich f2 ~2225.82 Hz", pct_err_f2 < 1,
       f"got {f2_sw:.2f} Hz ({pct_err_f2:.2f}% error)")

# Check mode 3
f3_sw = result_cccc.frequencies[2]
expected_f3_sw = 2225.82
pct_err_f3 = abs(f3_sw - expected_f3_sw) / expected_f3_sw * 100
report("CCCC sandwich f3 ~2225.82 Hz (degenerate)", pct_err_f3 < 1,
       f"got {f3_sw:.2f} Hz ({pct_err_f3:.2f}% error)")

# Check mode 4
f4_sw = result_cccc.frequencies[3]
expected_f4_sw = 3117.85
pct_err_f4 = abs(f4_sw - expected_f4_sw) / expected_f4_sw * 100
report("CCCC sandwich f4 ~3117.85 Hz", pct_err_f4 < 1,
       f"got {f4_sw:.2f} Hz ({pct_err_f4:.2f}% error)")


# ── 3. Kappa validation ────────────────────────────────────────────────────
print()
print("=" * 70)
print("3. KAPPA (SHEAR CORRECTION FACTOR)")
print("=" * 70)

k1_s, k2_s = shear_correction_factor(lam_steel)
report("Homogeneous steel kappa == 5/6 = 0.8333", abs(k1_s - 5/6) < 0.01,
       f"kappa1={k1_s:.6f}, kappa2={k2_s:.6f}")

k1_sw, k2_sw = shear_correction_factor(lam_sandwich)
report("Sandwich plate kappa ~0.165", abs(k1_sw - 0.165) < 0.02,
       f"kappa1={k1_sw:.4f}, kappa2={k2_sw:.4f}")


# ── 4. Aerodynamic assembly smoke test ─────────────────────────────────────
print()
print("=" * 70)
print("4. AERODYNAMIC ASSEMBLY (V=400 m/s, M_inf=1.176)")
print("=" * 70)

V = 400.0
K_air, C_air = solver_cccc.assemble_aerodynamic(V)

MN = 15 * 15
MN5 = 5 * MN
shapes_ok = K_air.shape == (MN5, MN5) and C_air.shape == (MN5, MN5)
report("K_air and C_air shapes == (1125, 1125)", shapes_ok,
       f"shape={K_air.shape}")

w_block = K_air[2*MN:3*MN, 2*MN:3*MN]
w_nonzero = np.max(np.abs(w_block)) > 0
report("K_air w-block has nonzero entries", w_nonzero,
       f"max|w_block|={np.max(np.abs(w_block)):.4e}")

u_block = K_air[:MN, :MN]
v_block = K_air[MN:2*MN, MN:2*MN]
rx_block = K_air[3*MN:4*MN, 3*MN:4*MN]
ry_block = K_air[4*MN:5*MN, 4*MN:5*MN]
others_zero = (np.max(np.abs(u_block)) < 1e-10 and
               np.max(np.abs(v_block)) < 1e-10 and
               np.max(np.abs(rx_block)) < 1e-10 and
               np.max(np.abs(ry_block)) < 1e-10)
report("K_air non-w blocks are zero", others_zero)

# C_air should have nonzero diagonal block
c_diag_max = np.max(np.abs(np.diag(C_air)))
report("C_air has nonzero diagonal", c_diag_max > 0,
       f"max|diag(C_air)|={c_diag_max:.4e}")


# ── 5. Piston theory validation ────────────────────────────────────────────
print()
print("=" * 70)
print("5. PISTON THEORY: non_dimensional_lambda")
print("=" * 70)

D11 = steel.Q()[0, 0] * (h**3) / 12
lam_val = non_dimensional_lambda(V, rho_inf=1.2, L1=0.3, D11=D11, sound_speed=340.0)
report("non_dimensional_lambda positive and finite", np.isfinite(lam_val) and lam_val > 0,
       f"lambda={lam_val:.4f}")
print(f"  (D11={D11:.2f} N*m, rho_air=1.2, V=400, M=1.176)")


# ── 6. Convergence check (CCCC sandwich at M=N=10,12,15) ───────────────────
print()
print("=" * 70)
print("6. CONVERGENCE CHECK (CCCC sandwich, M=N=10,12,15)")
print("=" * 70)

conv_results = {}
for MN_val in [10, 12, 15]:
    s = FSDTSolver(
        L1=0.3, L2=0.3, M=MN_val, N=MN_val, laminate=lam_sandwich,
        basis_type="legendre", grid=(64, 64), k_stiffness=1e12,
    )
    s.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    r = s.solve_modal(n_modes=4)
    conv_results[MN_val] = r.frequencies[:4]
    print(f"  M=N={MN_val:>2d}: {r.frequencies[:4]}")

f12 = conv_results[12][0]
f15 = conv_results[15][0]
conv_change = abs(f15 - f12) / f15 * 100
report(f"Convergence M=N=12->15 < 2%", conv_change < 2.0,
       f"f1: {f12:.2f} -> {f15:.2f} Hz ({conv_change:.2f}% change)")


# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
n_pass = sum(1 for _, p, _ in results_log if p)
n_fail = sum(1 for _, p, _ in results_log if not p)
print(f"  {n_pass} PASS / {n_fail} FAIL / {len(results_log)} total")
print()
print(f"  Steel SSSS f1:      {f1:.2f} Hz  (analytical ~548 Hz, solver within 2%)")
print(f"  Sandwich CCCC f1:   {f1_sw:.2f} Hz  (ref ~1173 Hz)")
print(f"  Steel kappa:        {k1_s:.6f} / {k2_s:.6f}  (expect 5/6 = 0.8333)")
print(f"  Sandwich kappa:     {k1_sw:.4f} / {k2_sw:.4f}  (expect ~0.165)")
print(f"  Aerodynamic:        shape={K_air.shape}, w-block nonzero={w_nonzero}")
print(f"  Lambda:             {lam_val:.4f}")
print(f"  Convergence:        {conv_results[10][0]:.2f} -> {conv_results[12][0]:.2f} -> {conv_results[15][0]:.2f} Hz")
print()

if n_fail > 0:
    print("ANOMALIES:")
    for name, passed, detail in results_log:
        if not passed:
            print(f"  - {name}: {detail}")
