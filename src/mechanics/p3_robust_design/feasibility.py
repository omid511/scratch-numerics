"""Phase 2 feasibility study: mode identity across the design space.

Roadmap gate (proposal3_roadmap.md, Phase 2): if MAC consistency across the
design parameter space is >= 70%, mode-decomposition surrogates (Phase 5)
proceed; below 70% Phase 5 is skipped and the gradient-enhanced GP fallback
is used. The verdict is recorded in data/proposal3/feasibility_report.json.

Scope: the sweep varies ONE scalar stiffness k shared by all four edges
(log10 k in [6, 12]), i.e. the 1D all-edges-equal diagonal — not the full
4D independent-edge design space used by the production sweep. The gate
decision below applies to that diagonal subspace only.

Two MAC metrics are measured over the LHS sweep:

* Cross-design identity (primary): Hungarian-matched MAC between each
  design's shapes (lowest modes at the base velocity) and the medoid
  design's shapes (median-log10_k success; order-independent, unlike a
  first-success reference). High values mean mode *character* survives
  across the design space, so a fixed mode decomposition is meaningful.
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

# Minimum non-reference successes required for a passing gate; fewer
# forces phase5_skipped regardless of the fraction (no vacuous passes).
MIN_GATED_DESIGNS = 3


def _default_report_path() -> Path:
    """Default gate-artifact path: <repo root>/data/proposal3/feasibility_report.json."""
    return (
        Path(__file__).resolve().parents[3]
        / "data" / "proposal3" / "feasibility_report.json"
    )

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
    velocity: float = 1.05,
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
    ref_pos = report["_ref_pos"]
    failures = report.get("_failures", [])
    cross = [d["cross_design_macs"] for i, d in enumerate(per_design)
             if i != ref_pos]
    tracking = [d["median_step_mac"] for d in per_design
                if d["median_step_mac"] is not None]
    median_cross = float(np.median(np.concatenate(cross))) if cross else None
    median_tracking = float(np.median(tracking)) if tracking else None

    # Roadmap gate: fraction of SWEPT (not merely successful) design points
    # with ALL tracked modes above the MAC threshold must be >= 70%.
    # Per-design criterion, not pooled median. Failed points count against
    # the gate (they are not silently dropped); the medoid reference is a
    # self-match of 1.0s, so it is excluded from the fraction (it would
    # otherwise inflate the rate by 1/n).
    gated = [d for i, d in enumerate(per_design) if i != ref_pos]
    n_pass = sum(
        np.min(d["cross_design_macs"]) >= GATE_THRESHOLD for d in gated
    )
    denom = len(gated) + len(failures)
    frac_above = float(n_pass / denom) if denom else 0.0
    gate_pass = bool(frac_above >= 0.70 and len(gated) >= MIN_GATED_DESIGNS)
    if len(gated) < MIN_GATED_DESIGNS:
        notes.append(
            f"only {len(gated)} non-reference successes (< "
            f"MIN_GATED_DESIGNS={MIN_GATED_DESIGNS}); gate force-failed"
        )
    notes.append(
        "design subspace is the 1D all-edges-equal-k diagonal "
        "(all_four_edges_same_k=True); the decision does not cover the "
        "full 4D independent-edge design space"
    )

    out = {
        "study": "proposal3_phase2_mode_identity_feasibility",
        "n_points": int(n_points),
        "n_successful_points": len(per_design),
        "gate_threshold": GATE_THRESHOLD,
        "median_cross_design_mac": median_cross,
        "frac_designs_above_gate": frac_above,
        "n_gated_designs": len(gated),
        "min_gated_designs": MIN_GATED_DESIGNS,
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
            "design_subspace": "1D diagonal (all four edges share one k); "
                               "not the full 4D independent-edge space",
            "reference_selection": "medoid (median-log10_k success; "
                                   "order-independent)",
            "M": M,
            "N": N,
            "n_modes": n_modes,
            "ladder_steps": ladder_steps,
            "reference_point_index": ref_index,
        },
        "notes": notes,
        "n_failed_points": len(failures),
    }
    if failures:
        out["failure_examples"] = failures[:5]

    if output_path is None:
        output_path = _default_report_path()
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

    Returns a partial dict with '_per_design', '_ref_index' (medoid point
    loop-index), '_ref_pos' (medoid position within '_per_design') and
    '_slow_point' keys for the caller to finalize. The cross-design
    reference is the medoid success (closest to the median log10_k), so
    the gate does not depend on LHS point order.
    """
    t_start = time.perf_counter()
    per_design: list[dict] = []
    failures: list[str] = []
    # Success records kept separately so the medoid reference can be
    # chosen after the sweep (order-independent).
    records: list[dict] = []
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
            complex_shapes = getattr(result, "mode_shapes_complex", None)
            shapes = (
                complex_shapes[:n_modes]
                if complex_shapes is not None
                else result.mode_shapes[:n_modes]
            )

            # Reuse the reference solver: tracking only calls
            # solve_complex_modal (read-only), so no second build needed.
            tracker = track_modes_across_velocity(
                solver,
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

        # A single-step ladder has no consecutive pairs: step MACs are
        # undefined (None), never a perfect 1.0.
        records.append({
            "point_index": i,
            "log10_k": float(log10_k[i]),
            "k": float(k),
            "solve_seconds": float(elapsed),
            "reference_frequencies_hz": freqs.tolist(),
            "median_step_mac": (float(np.median(step_macs))
                                if step_macs.size else None),
            "worst_step_mac": (float(step_macs.min())
                               if step_macs.size else None),
            "shapes": shapes,
            "freqs": freqs,
        })

        if verbose and ((i + 1) % 10 == 0 or i + 1 == n):
            done = i + 1
            rate = (time.perf_counter() - t_start) / done
            print(
                f"[feasibility] {done}/{n} points, "
                f"{rate:.1f} s/point avg, "
                f"elapsed {time.perf_counter() - t_start:.0f} s",
                flush=True,
            )

    if not records:
        raise RuntimeError(
            "no design point produced a usable modal solution; "
            f"{len(failures)} failures: {failures[:5]}"
        )

    # Medoid reference: success closest to the median log10_k (ties broken
    # by lowest point index for determinism).
    median_log10 = float(np.median([r["log10_k"] for r in records]))
    medoid_pos = min(
        range(len(records)),
        key=lambda r: (abs(records[r]["log10_k"] - median_log10),
                       records[r]["point_index"]),
    )
    ref_shapes = records[medoid_pos]["shapes"]
    ref_freqs = records[medoid_pos]["freqs"]
    ref_index = records[medoid_pos]["point_index"]
    for r_pos, r in enumerate(records):
        if r_pos == medoid_pos:
            cross = [1.0] * n_modes  # self-match
        else:
            # _optimal_match returns macs already indexed by the reference
            # (medoid) branch order, so no reordering is needed.
            macs, _ = _optimal_match(ref_shapes, r["shapes"],
                                     ref_freqs, r["freqs"])
            cross = macs.tolist()
        per_design.append({
            "point_index": r["point_index"],
            "log10_k": r["log10_k"],
            "k": r["k"],
            "solve_seconds": r["solve_seconds"],
            "reference_frequencies_hz": r["reference_frequencies_hz"],
            "median_step_mac": r["median_step_mac"],
            "worst_step_mac": r["worst_step_mac"],
            "cross_design_macs": cross,
        })
    return {
        "_per_design": per_design,
        "_ref_index": ref_index,
        "_ref_pos": medoid_pos,
        "_slow_point": slow_point,
        "_failures": failures,
    }

