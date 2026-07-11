"""Mode-tracking diagnostics for velocity sweeps."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver


def _make_solver(M=10, N=10):
    """SSSS sandwich plate solver."""
    E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
    lam = Laminate(materials=[face, core, face], angles=[0, 0, 0],
                   z=[-0.005, -0.004, 0.004, 0.005])
    s = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, k_stiffness=1e12)
    s.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                   top={"type": "clamped"}, bottom={"type": "clamped"})
    return s


def mac(a, b):
    """Modal Assurance Criterion."""
    a, b = a.flatten(), b.flatten()
    return abs(np.dot(a, b))**2 / (np.dot(a, a) * np.dot(b, b) + 1e-30)


def _optimal_match(prev, curr):
    """Hungarian-optimal mode assignment from previous modes to current modes."""
    n = len(curr)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = 1.0 - mac(curr[i], prev[j])
    rows, cols = linear_sum_assignment(cost)
    assignment = np.empty(n, dtype=int)
    macs = np.empty(n)
    for curr_i, prev_i in zip(rows, cols):
        assignment[prev_i] = curr_i
        macs[prev_i] = 1.0 - cost[curr_i, prev_i]
    return macs, assignment


def test_optimal_match_returns_prev_to_current_assignment():
    prev = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    curr = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
    macs, assignment = _optimal_match(prev, curr)
    np.testing.assert_array_equal(assignment, [1, 0])
    np.testing.assert_allclose(macs, [1.0, 1.0])


def test_eigenvalue_continuity():
    """Eigenvalues should vary smoothly with velocity — no discontinuous jumps.

    Tracks eigenvalues by MAC-optimal mode assignment, then checks that
    assigned frequencies and damping ratios vary smoothly. Mode veering
    (low MAC) is detected but not a failure — eigenvalue continuity is.
    """
    s = _make_solver()
    velocities = np.linspace(400, 1200, 40)
    prev_freqs = None
    prev_damp = None
    prev_modes = None
    veerings = []

    for v in velocities:
        result = s.solve_complex_modal(v, n_modes=4)
        freqs = result.frequencies
        # damping ratio: -Re(lambda) / |lambda|
        damp = np.array([
            -e.real / (abs(e) + 1e-30) for e in result.eigenvalues
        ])

        if prev_modes is not None:
            macs, assignment = _optimal_match(prev_modes, result.mode_shapes)

            # Track assigned eigenvalues across steps
            assigned_freqs = freqs[assignment]
            assigned_damp = damp[assignment]

            # Frequency continuity: relative jump per mode
            for i in range(len(freqs)):
                if prev_freqs[i] > 0:
                    rel_jump = abs(assigned_freqs[i] - prev_freqs[i]) / prev_freqs[i]
                    assert rel_jump < 0.05, (
                        f"Mode {i} at V={v:.0f}: freq jump "
                        f"{prev_freqs[i]:.1f} -> {assigned_freqs[i]:.1f} "
                        f"({rel_jump:.4f})"
                    )

            # Damping continuity: absolute jump
            for i in range(len(damp)):
                damp_jump = abs(assigned_damp[i] - prev_damp[i])
                assert damp_jump < 0.05, (
                    f"Mode {i} at V={v:.0f}: damping jump "
                    f"{prev_damp[i]:.4f} -> {assigned_damp[i]:.4f}"
                )

            # Detect veering (low MAC = mode shape exchange)
            for i, m in enumerate(macs):
                if m < 0.5:
                    veerings.append((v, i, m))

            prev_freqs = assigned_freqs
            prev_damp = assigned_damp
        else:
            prev_freqs = freqs.copy()
            prev_damp = damp.copy()

        prev_modes = result.mode_shapes

    if veerings:
        print(f"Mode veering detected at {len(veerings)} point(s):")
        for v, mode, m in veerings:
            print(f"  V={v:.0f} mode {mode} MAC={m:.3f}")
    else:
        print("No mode veering detected")


def test_flutter_crossing_detected():
    """Velocity sweep should detect a stability transition if one exists."""
    s = _make_solver()
    prev_stable = None
    crossing_vel = None
    for v in np.linspace(345, 2000, 50):
        r = s.solve_complex_modal(float(v), n_modes=4)
        stable = bool(np.all(r.stable))
        if prev_stable is not None and stable != prev_stable:
            crossing_vel = v
            break
        prev_stable = stable

    if crossing_vel is not None:
        r_below = s.solve_complex_modal(crossing_vel - 50, n_modes=4)
        r_above = s.solve_complex_modal(crossing_vel + 50, n_modes=4)
        assert bool(np.all(r_below.stable)) != bool(np.all(r_above.stable)), \
            "Expected stability transition around crossing"
        print(f"Flutter crossing detected at V={crossing_vel:.0f}")
    else:
        print("No flutter crossing in scanned range")
