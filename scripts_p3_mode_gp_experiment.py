"""Phase-5 SP5 gate experiment: ModeDecompositionGP vs grad-enhanced GPSurrogate
on real solver sweep data near mode crossings.

Pipeline (roadmap SP5):
  1. Sample n_designs=20 LHS points over log10(k) in [6, 12] (4 edges).
  2. For each design solve complex modal at an 8-step velocity ladder
     (M=N=6, n_modes=4), track branches with track_modes_across_velocity,
     and record each branch's flutter load lambda_cr (velocity at which the
     tracked damping ratio crosses zero, converted via non_dimensional_lambda).
  3. Split 15 train / 5 test designs.
  4. Train ModeDecompositionGP (per-mode targets, min combination).
  5. Train GPSurrogate on scalar lambda_cr targets (= min across tracked
     branch lambdas, matching the min-combination convention). Gradients:
     no closed-form objective exists for lambda_cr(theta), so the scalar GP
     runs in standard (non-gradient-enhanced) mode — recorded honestly.
  6. Evaluate both models on the test designs: overall RMSE and RMSE on the
     near-crossing subset (top-two branch means gap < tolerance, from the
     mode-GP prediction). SP5 gate: >=15% near-crossing RMSE reduction.

Runs solver sweeps in the background; prints progress throughout.
Nothing is committed; results are printed as a table.
"""
from __future__ import annotations

import sys
import time
import traceback

import numpy as np
from scipy.stats.qmc import LatinHypercube

from mechanics.boundary import build_boundary_springs
from mechanics.laminate import Laminate, Material
from mechanics.piston_theory import AIR_DENSITY, SOUND_SPEED, non_dimensional_lambda, velocity_from_lambda
from mechanics.p3_robust_design.gp_surrogate import GPSurrogate
from mechanics.p3_robust_design.mode_gp import ModeDecompositionGP
from mechanics.p3_robust_design.mode_tracking import track_modes_across_velocity
from mechanics.solver import FSDTSolver

# ── Experiment configuration ──────────────────────────────────────────
N_DESIGNS = 20
N_TRAIN = 15
N_VEL_STEPS = 8          # velocity ladder resolution
N_MODES = 4              # tracked branches
M_BASIS = N_BASIS = 6    # runtime discipline
LAM_LOWER_FRAC = 0.9     # safety margin below the per-design Mach-floor lambda
LAM_UPPER = 1500.0       # upper end of the scanned lambda window
SEED = 42


def make_laminate() -> Laminate:
    """Steel isotropic plate, h = 1 mm (same fixture as tests/test_solver)."""
    E, nu, rho = 210e9, 0.33, 7930.0
    G = E / (2 * (1 + nu))
    mat = Material(E, E, G, G, G, nu, rho)
    h = 1e-3
    return Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])


def build_solver(laminate: Laminate, log10_k: np.ndarray) -> FSDTSolver:
    """Solver with four elastic edges at stiffness 10**log10_k."""
    L1 = L2 = 0.3
    solver = FSDTSolver(L1=L1, L2=L2, M=M_BASIS, N=N_BASIS,
                        laminate=laminate, k_stiffness=1e12)
    k = 10.0 ** np.asarray(log10_k, dtype=float)
    springs = build_boundary_springs(
        L1, L2,
        {"type": "elastic", "k": [float(k[0])] * 5},
        {"type": "elastic", "k": [float(k[1])] * 5},
        {"type": "elastic", "k": [float(k[2])] * 5},
        {"type": "elastic", "k": [float(k[3])] * 5},
        k_stiffness=1e12,
    )
    solver.set_boundary_springs(springs)
    return solver


def solve_one_design(laminate: Laminate, x_log10: np.ndarray) -> dict:
    """Velocity-ladder sweep with mode tracking for one design point.

    Returns dict with per-branch lambda_cr (NaN where the branch does not
    cross zero damping inside the ladder) and the tracked data.
    """
    solver = build_solver(laminate, x_log10)
    D11 = solver._compute_D11()

    # Lambda window: just above the per-design Mach floor up to LAM_UPPER.
    v_floor_mach2 = 2.0 * SOUND_SPEED
    lam_floor = AIR_DENSITY * v_floor_mach2**2 * solver.L1**3 / (D11 * np.sqrt(3.0))
    lam_floor *= LAM_LOWER_FRAC
    lam_grid = np.linspace(lam_floor, LAM_UPPER, N_VEL_STEPS)
    velocities = np.array([
        velocity_from_lambda(lam, AIR_DENSITY, solver.L1, D11, SOUND_SPEED)
        for lam in lam_grid
    ])
    order = np.argsort(velocities)
    velocities = velocities[order]
    lam_sorted = lam_grid[order]

    tracked = track_modes_across_velocity(solver, velocities, n_modes=N_MODES)
    damping = tracked["damping"]            # (n_vel, n_modes), >0 = stable

    lam_branch = np.full(N_MODES, np.nan)
    for j in range(N_MODES):
        d = damping[:, j]
        for i in range(len(d) - 1):
            # stability loss: damping ratio goes positive -> negative
            if d[i] > 0.0 >= d[i + 1]:
                frac = d[i] / (d[i] - d[i + 1])
                v_cross = velocities[i] + frac * (velocities[i + 1] - velocities[i])
                try:
                    lam_branch[j] = non_dimensional_lambda(
                        v_cross, AIR_DENSITY, solver.L1, D11, SOUND_SPEED
                    )
                except ValueError:
                    pass  # subsonic crossing: outside model validity
                break

    return {
        "lam_branch": lam_branch,
        "lam_grid": lam_sorted,
        "velocities": velocities,
        "tracked": tracked,
    }


def main() -> int:
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    laminate = make_laminate()

    # ── 1. LHS designs over log10(k) in [6, 12], 4 edges ──────────────
    lhs = LatinHypercube(d=4, seed=SEED)
    X = 6.0 + lhs.random(n=N_DESIGNS) * 6.0   # (n, 4) log10 stiffness
    edge_names = ["left", "right", "top", "bottom"]
    print(f"[setup] {N_DESIGNS} LHS designs, log10(k) in [6,12], "
          f"M=N={M_BASIS}, n_modes={N_MODES}, ladder={N_VEL_STEPS} steps",
          flush=True)

    # ── 2. Solver sweeps (the expensive part) ─────────────────────────
    lam_branches = np.full((N_DESIGNS, N_MODES), np.nan)
    for i in range(N_DESIGNS):
        t0 = time.perf_counter()
        try:
            res = solve_one_design(laminate, X[i])
            lam_branches[i] = res["lam_branch"]
            ok = int(np.isfinite(res["lam_branch"]).sum())
            print(f"[sweep {i + 1:2d}/{N_DESIGNS}] "
                  f"x={np.round(X[i], 2)} branches_crossed={ok}/{N_MODES} "
                  f"lam_branch={np.round(res['lam_branch'], 1)} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception:
            print(f"[sweep {i + 1:2d}/{N_DESIGNS}] FAILED:\n"
                  f"{traceback.format_exc(limit=2)}", flush=True)

    # ── 3. Assemble dataset. Branches whose damping never crosses zero
    #      inside the lambda window are right-censored at LAM_UPPER: their
    #      true flutter load lies above the window, which is exactly what
    #      the min-combination convention needs (they cannot govern).
    #      Failed designs (solver exception) are dropped.
    failed_rows = ~np.isfinite(lam_branches).any(axis=1)
    censor_mask = ~np.isfinite(lam_branches) & ~failed_rows[:, None]
    n_censored = int(censor_mask.sum())
    lam_branches[censor_mask] = LAM_UPPER
    scalar_target = np.min(lam_branches, axis=1)  # min convention
    usable = ~failed_rows & np.isfinite(scalar_target)
    print(f"\n[data] designs={N_DESIGNS} usable={int(usable.sum())} "
          f"censored_branch_entries={n_censored} "
          f"dropped_designs={int((~usable).sum())}", flush=True)
    Xu = X[usable]
    Yu = lam_branches[usable]
    yu = scalar_target[usable]
    if len(Xu) < N_TRAIN + 3:
        print(f"[abort] only {len(Xu)} usable designs — cannot split "
              f"{N_TRAIN} train / rest test reliably.", flush=True)
        return 1

    perm = rng.permutation(len(Xu))
    train_idx, test_idx = perm[:N_TRAIN], perm[N_TRAIN:]
    Xtr, Ytr, ytr = Xu[train_idx], Yu[train_idx], yu[train_idx]
    Xte, Yte, yte = Xu[test_idx], Yu[test_idx], yu[test_idx]
    print(f"[split] train={len(train_idx)} test={len(test_idx)}", flush=True)

    # ── 4. Mode-decomposition GP (per-mode targets) ───────────────────
    print("\n[train] ModeDecompositionGP (min combination, "
          "hyperparameter optimization on)...", flush=True)
    # Resolve the crossing tolerance explicitly (same rule as the class
    # default: 1% of the training-target range) so it can be reported.
    t0 = time.perf_counter()
    mode_gp = ModeDecompositionGP(combination="min", optimize_hyperparams=True)
    mode_gp.crossing_tol = 0.01 * float(Ytr.max() - Ytr.min())
    mode_gp.fit(Xtr, Ytr)
    print(f"[train] mode-GP done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ── 5. Scalar baseline GP ─────────────────────────────────────────
    # No analytic objective_fn for lambda_cr(theta) exists, so gradient
    # enhancement is unavailable; standard GP is the honest baseline.
    print("[train] GPSurrogate (standard; gradients unavailable for "
          "lambda_cr(theta))...", flush=True)
    t0 = time.perf_counter()
    scalar_gp = GPSurrogate(use_gradients=False)
    scalar_gp.fit(Xtr, ytr)
    print(f"[train] scalar GP done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ── 6. Evaluation ─────────────────────────────────────────────────
    pred_mode = mode_gp.predict(Xte)
    mu_scalar, _ = scalar_gp.predict(Xte)

    err_mode = pred_mode.lam - yte
    err_scalar = mu_scalar - yte
    rmse_mode_all = float(np.sqrt(np.mean(err_mode ** 2)))
    rmse_scalar_all = float(np.sqrt(np.mean(err_scalar ** 2)))

    near = pred_mode.near_crossing
    n_near = int(near.sum())

    def _rmse(err: np.ndarray, mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("nan")

    rmse_mode_near = _rmse(err_mode, near)
    rmse_scalar_near = _rmse(err_scalar, near)

    # Per-mode branch RMSE (each per-mode GP against its tracked truth).
    per_mode_rmse = []
    for j in range(mode_gp.n_modes):
        mu_j, _ = mode_gp.mode_gps[j].predict(Xte)
        per_mode_rmse.append(float(np.sqrt(np.mean((mu_j - Yte[:, j]) ** 2))))
    improvement = ((rmse_scalar_near - rmse_mode_near) / rmse_scalar_near
                   if np.isfinite(rmse_scalar_near) and rmse_scalar_near > 0
                   else float("nan"))


    # ── 7. Report ─────────────────────────────────────────────────────
    if n_near == 0:
        verdict = "SKIP"
    elif np.isfinite(improvement) and improvement >= 0.15:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    print("\n" + "=" * 72)
    print("SP5 GATE: ModeDecompositionGP vs scalar GP near mode crossings")
    print("=" * 72)
    hdr = (f"{'test':>4} {'near?':>5} {'y_true':>10} {'modeGP':>10} "
           f"{'scalarGP':>10}")
    print(hdr)
    for i in range(len(yte)):
        tag = "*" if near[i] else ""
        print(f"{i:>4} {tag:>5} {yte[i]:>10.1f} {pred_mode.lam[i]:>10.1f} "
              f"{mu_scalar[i]:>10.1f}")
    print("-" * 72)
    print(f"per-mode branch RMSE: "
          + ", ".join(f"mode{j}={r:.1f}" for j, r in enumerate(per_mode_rmse)))
    print(f"overall  RMSE   : mode-GP={rmse_mode_all:.2f}  "
          f"scalar-GP={rmse_scalar_all:.2f}")
    print(f"near-xing RMSE  : mode-GP={rmse_mode_near:.2f}  "
          f"scalar-GP={rmse_scalar_near:.2f}  (n={n_near}/{len(yte)}, "
          f"tol={mode_gp.crossing_tol:.2f})")
    print(f"near-xing improvement: {100 * improvement:.1f}%  "
          f"(gate threshold: >= 15%)")
    print(f"SP5 VERDICT: {verdict}"
          + ("" if n_near else "  [SKIP: no near-crossing test points]"))
    print(f"total wall time: {time.perf_counter() - t_start:.0f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
