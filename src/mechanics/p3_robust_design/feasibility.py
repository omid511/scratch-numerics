"""Phase 2 feasibility study: mode identity across the design space.

Roadmap gate (proposal3_roadmap.md, Phase 2): if MAC consistency across the
design parameter space is >= 70%, mode-decomposition surrogates (Phase 5)
proceed; below 70% Phase 5 is skipped and the gradient-enhanced GP fallback
is used. The verdict is recorded in data/proposal3/feasibility_report.json.

Two MAC metrics are measured over an LHS design sweep in elastic edge
stiffness k (all four edges equal per point, log10 k in [6, 12]):

* Cross-design identity (primary): Hungarian-matched MAC between each
  design's reference shapes (6 lowest modes at the base velocity) and the
  first design's shapes. High values mean mode *character* survives across
  the design space, so a fixed mode decomposition is meaningful.
* Per-design velocity-tracking consistency (secondary): median step-MAC of
  ``track_modes_across_velocity`` on a small ladder above the base velocity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..boundary import build_boundary_springs
from ..laminate import Laminate, Material
from ..piston_theory import SOUND_SPEED
from ..solver import FSDTSolver
from .mode_tracking import _optimal_match, track_modes_across_velocity

GATE_THRESHOLD = 0.70
LOG10_K_BOUNDS = (6.0, 12.0)

_SLOW_POINT_SECONDS = 60.0


def _default_laminate() -> Laminate:
    """Sandwich laminate matching the p3 test fixtures."""
    E = 70e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
    return Laminate(
        materials=[face, core, face],
        angles=[0, 0, 0],
        z=[-5e-3, -4e-3, 4e-3, 5e-3],
    )


def _elastic_solver(
    laminate: Laminate, k: float, M: int, N: int
) -> FSDTSolver:
    """Solver with identical elastic stiffness ``k`` on all four edges."""
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=M, N=N, laminate=laminate, k_stiffness=1e12
    )
    springs = build_boundary_springs(
        solver.L1, solver.L2,
        *[{"type": "elastic", "k": [float(k)] * 5}] * 4,
        k_stiffness=1e12,
    )
    solver.set_boundary_springs(springs)
    return solver


def run_phase2_feasibility(
    n_points: int = 50,
    *,
    seed: int = 42,
    laminate: Laminate | None = None,
    velocity: float = 1.0,
    n_modes: int = 6,
    ladder_steps: int = 5,
    ladder_span_mach: float = 0.08,
    M: int = 8,
    N: int = 8,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the Phase-2 mode-identity feasibility study.

    Args:
        n_points: Number of LHS design points.
        seed: LHS seed.
        laminate: Layup; defaults to the sandwich plate from the p3 tests.
        velocity: Base free-stream velocity as a Mach multiplier of the
            sound speed. ``solve_complex_modal`` requires strictly supersonic
            flow, so this must exceed 1.0 (the real study uses 1.05).
        n_modes: Modes tracked per design (reference fingerprint size).
        ladder_steps: Steps in the per-design velocity-tracking ladder.
        ladder_span_mach: Mach span of the ladder above the base velocity.
        M, N: Basis function counts.
        output_path: Where to write the JSON report; defaults to
            ``<repo root>/data/proposal3/feasibility_report.json``.

    Returns:
        The report dict that was written to disk. ``gate_pass`` /
        ``decision`` record the roadmap's >=70% cross-design MAC gate.
    """
    if not velocity > 1.0:
        raise ValueError(
            "velocity is a Mach multiplier and solve_complex_modal requires "
            f"supersonic flow (M > 1); got M={velocity}"
        )
    laminate = _default_laminate() if laminate is None else laminate
    v_base = velocity * SOUND_SPEED
    ladder = np.linspace(v_base, v_base + ladder_span_mach * SOUND_SPEED,
                         ladder_steps)

    # LHS in log10(k), one coordinate, all four edges share k per point.
    lhs = LatinHypercube(d=1, seed=seed)
    u = lhs.random(n=n_points).ravel()
    log10_k = LOG10_K_BOUNDS[0] + u * (LOG10_K_BOUNDS[1] - LOG10_K_BOUNDS[0])

    notes: list[str] = []


    while True:
        report = _run_sweep(
            laminate, log10_k, v_base, ladder, n_modes=n_modes, M=M, N=N,
            velocity_mach=velocity, seed=seed, verbose=verbose,
        )
        slow = report.pop("_slow_point", False)
        if slow and n_modes > 4:
            reduced = max(4, n_modes // 2)
            notes.append(
                f"a single point exceeded {_SLOW_POINT_SECONDS:.0f} s at "
                f"n_modes={n_modes}; restarted with n_modes={reduced}"
            )
            if verbose:
                print(f"[feasibility] {notes[-1]}")
            n_modes = reduced
            continue
        break

    per_design = report["_per_design"]
    ref_index = report["_ref_index"]
    cross = [d["cross_design_macs"] for i, d in enumerate(per_design)
             if i != ref_index]
    tracking = [d["median_step_mac"] for d in per_design]
    median_cross = float(np.median(np.concatenate(cross)))
    median_tracking = float(np.median(tracking))
    gate_pass = bool(median_cross >= GATE_THRESHOLD)

    out = {
        "study": "proposal3_phase2_mode_identity_feasibility",
        "n_points": int(n_points),
        "n_successful_points": len(per_design),
        "gate_threshold": GATE_THRESHOLD,
        "median_cross_design_mac": median_cross,
        "median_velocity_tracking_mac": median_tracking,
        "per_design": per_design,
        "gate_pass": gate_pass,
        "decision": "phase5_proceeds" if gate_pass else "phase5_skipped",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": seed,
            "laminate": "quasi-isotropic-aluminum-face honeycomb-core "
                        "sandwich (p3 test fixture)",
            "velocity_mach_multiplier": float(velocity),
            "base_velocity_mps": float(v_base),
            "velocity_ladder_mps": ladder.tolist(),
            "log10_k_bounds": list(LOG10_K_BOUNDS),
            "all_four_edges_same_k": True,
            "M": M,
            "N": N,
            "n_modes": n_modes,
            "ladder_steps": ladder_steps,
            "reference_point_index": ref_index,
        },
        "notes": notes,
        "n_failed_points": len(report.get("_failures", [])),
    }
    if report.get("_failures"):
        out["failure_examples"] = report["_failures"][:5]

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    return out



def _run_sweep(
    laminate: Laminate,
    log10_k: np.ndarray,
    v_base: float,
    ladder: np.ndarray,
    *,
    n_modes: int,
    M: int,
    N: int,
    velocity_mach: float,
    seed: int,
    verbose: bool,
) -> dict:
    """Solve each design point; internal helper of run_phase2_feasibility.

    Returns a partial dict with '_per_design', '_ref_index' and '_slow_point'
    keys for the caller to finalize.
    """
    t_start = time.perf_counter()
    per_design: list[dict] = []
    failures: list[str] = []
    ref_shapes = ref_freqs = None
    ref_index = -1
    slow_point = False

    n = len(log10_k)
    for i in range(n):
        k = 10.0 ** log10_k[i]
        t0 = time.perf_counter()
        try:
            solver = _elastic_solver(laminate, k, M, N)
            result = solver.solve_complex_modal(float(ladder[0]),
                                                n_modes=n_modes)
            if len(result.frequencies) < n_modes:
                raise RuntimeError(
                    f"solver returned {len(result.frequencies)} modes"
                )
            freqs = result.frequencies[:n_modes]
            shapes = result.mode_shapes_complex[:n_modes]

            tracker = track_modes_across_velocity(
                _elastic_solver(laminate, k, M, N),
                ladder, n_modes=n_modes,
            )
            step_macs = tracker["mac_values"]
        except Exception as exc:  # keep sweeping; record honestly
            failures.append(f"point {i} (log10_k={log10_k[i]:.3f}): "
                            f"{type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        if elapsed > _SLOW_POINT_SECONDS:
            slow_point = True

        entry = {
            "point_index": i,
            "log10_k": float(log10_k[i]),
            "k": float(k),
            "solve_seconds": float(elapsed),
            "reference_frequencies_hz": freqs.tolist(),
            "median_step_mac": float(np.median(step_macs)),
            "worst_step_mac": float(step_macs.min()),
            "cross_design_macs": None,
        }
        if ref_shapes is None:
            ref_shapes, ref_freqs, ref_index = shapes, freqs, i
            entry["cross_design_macs"] = [1.0] * n_modes  # self-match
        else:
            # _optimal_match returns macs already indexed by the reference
            # (first design) branch order, so no reordering is needed.
            macs, _ = _optimal_match(ref_shapes, shapes, ref_freqs, freqs)
            entry["cross_design_macs"] = macs.tolist()
        per_design.append(entry)

        if verbose and ((i + 1) % 10 == 0 or i + 1 == n):
            done = i + 1
            rate = (time.perf_counter() - t_start) / done
            print(
                f"[feasibility] {done}/{n} points, "
                f"{rate:.1f} s/point avg, "
                f"elapsed {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )

    if ref_shapes is None:
        raise RuntimeError(
            "no design point produced a usable modal solution; "
            f"{len(failures)} failures: {failures[:5]}"
        )
    if failures:
        pass  # recorded via returned list below
    return {
        "_per_design": per_design,
        "_ref_index": ref_index,
        "_slow_point": slow_point,
        "_failures": failures,
    }

