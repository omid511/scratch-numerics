"""End-to-end parameter sweep demo with provenance.

Runs a small sweep over face-sheet thickness for the SSSS sandwich plate,
computes modal frequencies at each design point, and saves results to NPZ
with full provenance metadata.
"""
import json
import hashlib
import subprocess
import time
import numpy as np
from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.config import _get_git_hash


def solve_single(face_thickness: float, core_thickness: float = 0.008,
                 M: int = 10, N: int = 10) -> dict:
    """Solve one design point and return results + metadata."""
    E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)

    h_total = 2 * face_thickness + core_thickness
    z = [-h_total/2, -h_total/2 + face_thickness,
         h_total/2 - face_thickness, h_total/2]
    lam = Laminate(materials=[face, core, face], angles=[0, 0, 0], z=z)

    solver = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, k_stiffness=1e12)
    solver.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                        top={"type": "clamped"}, bottom={"type": "clamped"})

    t0 = time.time()
    result = solver.solve_modal(n_modes=6)
    solve_time = time.time() - t0

    ABBD, _ = lam.ABD()
    D11 = ABBD[3, 3]

    return {
        "face_thickness": face_thickness,
        "core_thickness": core_thickness,
        "D11": D11,
        "frequencies": np.real(result.frequencies),
        "solve_time": solve_time,
        "M": M, "N": N,
    }


def run_sweep():
    """Run sweep over face thickness."""
    face_thicknesses = np.linspace(0.0005, 0.003, 15)
    results = []
    for ft in face_thicknesses:
        r = solve_single(ft)
        results.append(r)
        print(f"  t_face={ft*1e3:.2f}mm: f1={r['frequencies'][0]:.1f} Hz, "
              f"D11={r['D11']:.0f}, time={r['solve_time']:.3f}s")

    # Save with provenance
    output = {
        "results": {k: np.array([r[k] for r in results])
                    for k in ["face_thickness", "D11", "frequencies", "solve_time"]},
        "provenance": {
            "git_hash": _get_git_hash(),
            "solver_version": "0.1.0",
            "schema_version": "1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sweep_params": {"face_thickness": {"min": float(face_thicknesses[0]),
                                                  "max": float(face_thicknesses[-1]),
                                                  "n": len(face_thicknesses)}},
            "plate_params": {"L1": 0.3, "L2": 0.3, "M": 10, "N": 10,
                             "boundary": "clamped", "core_thickness": 0.008},
            "n_samples": len(results),
            "total_solve_time": sum(r["solve_time"] for r in results),
        },
    }
    np.savez("sweep_output.npz", **{f"result_{k}": v for k, v in output["results"].items()},
             provenance=json.dumps(output["provenance"]))
    print(f"\nSaved sweep_output.npz with {len(results)} samples")
    print(f"Total time: {output['provenance']['total_solve_time']:.2f}s")
    return output


if __name__ == "__main__":
    run_sweep()
