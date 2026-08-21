"""Torque-free rigid-body attitude propagation.

Conventions: q = [w, x, y, z] rotates body-frame vectors into the inertial
frame; angular velocity and angular momentum are body-frame quantities.
"""

from __future__ import annotations

import numpy as np


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qmatrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def rotation_increment(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = np.linalg.norm(omega) * dt
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    return np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def midpoint_momentum(
    L0: np.ndarray,
    inertia: np.ndarray,
    dt: float,
    tol: float = 1e-11,
    max_iter: int = 12,
) -> np.ndarray:
    """Solve the implicit midpoint update for body angular momentum."""
    L1 = L0.copy()

    def residual(x: np.ndarray) -> np.ndarray:
        mid = 0.5 * (L0 + x)
        omega = np.linalg.solve(inertia, mid)
        return x - L0 - dt * np.cross(omega, mid)

    for _ in range(max_iter):
        r = residual(L1)
        if np.linalg.norm(r) <= tol * np.linalg.norm(L1):
            return L1
        J = np.empty((3, 3))
        eps = 1e-8
        for j in range(3):
            shifted = L1.copy()
            shifted[j] += eps
            J[:, j] = (residual(shifted) - r) / eps
        L1 -= np.linalg.solve(J, r)
    raise RuntimeError("midpoint solve did not converge")


def simulate(
    inertia: np.ndarray,
    omega0: np.ndarray,
    q0: np.ndarray,
    duration: float,
    dt: float,
) -> dict[str, np.ndarray | float]:
    inertia = np.asarray(inertia, dtype=float)
    omega0 = np.asarray(omega0, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    if inertia.shape != (3, 3) or np.any(np.diag(inertia) <= 0):
        raise ValueError("inertia must be positive")
    if omega0.shape != (3,) or q0.shape != (4,):
        raise ValueError("bad initial state shape")

    steps = int(round(duration / dt))
    times = np.arange(steps + 1) * dt
    qs = np.empty((steps + 1, 4))
    momenta = np.empty((steps + 1, 3))
    qs[0] = q0
    momenta[0] = inertia @ omega0

    for k in range(steps):
        L0 = momenta[k]
        L1 = midpoint_momentum(L0, inertia, dt)
        omega_mid = np.linalg.solve(inertia, 0.5 * (L0 + L1))
        dq = rotation_increment(omega_mid, dt)
        qs[k + 1] = qmul(dq, qs[k])
        qs[k + 1] /= np.linalg.norm(qs[k + 1])
        momenta[k + 1] = L1

    omegas = np.linalg.solve(inertia, momenta.T).T
    energy = 0.5 * np.einsum("ij,ij->i", momenta, omegas)
    inertial_momentum = np.array([qmatrix(q) @ L for q, L in zip(qs, momenta)])
    relative_energy_drift = np.max((energy - energy[0]) / energy[0])
    momentum_drift = np.max(
        np.linalg.norm(inertial_momentum - inertial_momentum[0], axis=1)
    )
    return {
        "time": times,
        "quaternion": qs,
        "omega": omegas,
        "body_momentum": momenta,
        "energy": energy,
        "inertial_momentum": inertial_momentum,
        "relative_energy_drift": float(relative_energy_drift),
        "momentum_drift": float(momentum_drift),
    }


def demo() -> None:
    result = simulate(
        np.diag([2.0, 3.0, 5.0]),
        np.array([0.7, -0.2, 1.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        duration=20.0,
        dt=0.01,
    )
    assert np.all(np.isfinite(result["omega"]))
    assert abs(np.linalg.norm(result["quaternion"][-1]) - 1.0) < 1e-12
    print("energy drift:", result["relative_energy_drift"])
    print("momentum drift:", result["momentum_drift"])


if __name__ == "__main__":
    demo()
