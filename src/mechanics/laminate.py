"""Laminate mechanics: Material, Profile, ABD matrices."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from numpy.polynomial.legendre import leggauss


@dataclass
class Material:
    """Orthotropic material with 5 elastic moduli."""
    E1: float
    E2: float
    G23: float
    G13: float
    G12: float
    nu12: float
    rho: float = 0.0

    def __post_init__(self):
        self.nu21 = self.nu12 * self.E2 / self.E1

    def Q(self) -> np.ndarray:
        """6x6 plane stress-reduced stiffness matrix."""
        denom = 1.0 - self.nu12 * self.nu21
        return np.array([
            [self.E1 / denom, self.nu21 * self.E1 / denom, 0, 0, 0, 0],
            [self.nu12 * self.E2 / denom, self.E2 / denom, 0, 0, 0, 0],
            [0, 0, self.G12, 0, 0, 0],
            [0, 0, 0, self.G23, 0, 0],
            [0, 0, 0, 0, self.G13, 0],
            [0, 0, 0, 0, 0, self.G12],
        ])

    def Qb(self, angle: float) -> np.ndarray:
        """Transformed stiffness matrix at fiber angle (radians)."""
        Q = self.Q()
        Q11, Q12, Q22 = Q[0, 0], Q[0, 1], Q[1, 1]
        Q44, Q55, Q66 = Q[3, 3], Q[4, 4], Q[2, 2]
        c = np.cos(angle)
        s = np.sin(angle)

        Qb = np.zeros((6, 6))
        Qb[0, 0] = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
        Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
        Qb[1, 0] = Qb[0, 1]
        Qb[1, 1] = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
        Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
        Qb[2, 0] = Qb[0, 2]
        Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
        Qb[2, 1] = Qb[1, 2]
        Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
        Qb[3, 3] = Q44 * c**2 + Q55 * s**2
        Qb[3, 4] = (Q55 - Q44) * c * s
        Qb[4, 3] = Qb[3, 4]
        Qb[4, 4] = Q44 * s**2 + Q55 * c**2
        Qb[0, 5] = Qb[0, 2]
        Qb[5, 0] = Qb[0, 2]
        Qb[1, 5] = Qb[1, 2]
        Qb[5, 1] = Qb[1, 2]
        Qb[5, 5] = Qb[2, 2]
        return Qb


@dataclass
class Laminate:
    """Laminate layup with ABD matrices."""
    materials: list[Material]
    angles: list[float]  # radians
    z: list[float]       # ply interface z-coordinates (length = nplies + 1)

    def __post_init__(self):
        assert len(self.materials) == len(self.angles) == len(self.z) - 1

    def ABD(self) -> np.ndarray:
        """6x6 ABD stiffness matrix [A B; B D]."""
        nplies = len(self.materials)
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        As = np.zeros((2, 2))

        for k in range(nplies):
            Qb = self.materials[k].Qb(self.angles[k])
            zk1, zk2 = self.z[k], self.z[k + 1]
            dz = zk2 - zk1
            zmid = (zk1 + zk2) / 2.0

            A += Qb[:3, :3] * dz
            B += Qb[:3, :3] * dz * zmid
            D += Qb[:3, :3] * (dz**3 / 12.0 + dz * zmid**2)
            As += Qb[3:5, 3:5] * dz

        ABBD = np.zeros((6, 6))
        ABBD[:3, :3] = A
        ABBD[:3, 3:6] = B
        ABBD[3:6, :3] = B
        ABBD[3:6, 3:6] = D
        return ABBD, As

    def I(self) -> np.ndarray:
        """Mass moments of inertia [I0, I1, I2]."""
        I0, I1, I2 = 0.0, 0.0, 0.0
        for k, mat in enumerate(self.materials):
            zk1, zk2 = self.z[k], self.z[k + 1]
            dz = zk2 - zk1
            zmid = (zk1 + zk2) / 2.0
            rho = mat.rho
            I0 += rho * dz
            I1 += rho * dz * zmid
            I2 += rho * (dz**3 / 12.0 + dz * zmid**2)
        return np.array([I0, I1, I2])

    def kappa(self) -> np.ndarray:
        """Shear correction factor matrix (2x2 diagonal).

        Vlachoutsis (1992) method matching original plate-main implementation.
        Q_inplane = Qbi[[0,1],[0,1]] = [Q11, Q22] (shape 2,)
        Q_shear   = Qbi[[4,3],[4,3]] = [Q55, Q44] (shape 2,)
        kappa = R^2 / (d * I_).
        """
        nplies = len(self.materials)
        z_arr = np.array(self.z)
        NQUAD = 6

        xi, w = leggauss(NQUAD)

        Qbs = [self.materials[k].Qb(self.angles[k]) for k in range(nplies)]

        # Q_inplane[k] = Qbi[[0,1],[0,1]] = [Q11, Q22] (2,)
        # Q_shear[k]   = Qbi[[4,3],[4,3]] = [Q55, Q44] (2,)
        Q_ins = [Qbs[k][[0, 1], [0, 1]] for k in range(nplies)]
        Q_shs = [Qbs[k][[[4, 3]], [[4, 3]]].ravel() for k in range(nplies)]

        moment = np.zeros((3, 2))
        d = np.zeros(2)
        for k in range(nplies):
            zk1, zk2 = z_arr[k], z_arr[k + 1]
            dz = (zk2 - zk1) / 2.0
            zmid = (zk1 + zk2) / 2.0
            Q_in = Q_ins[k]
            Q_sh = Q_shs[k]
            for i in range(NQUAD):
                zq = zmid + dz * xi[i]
                moment[0] += Q_in * w[i] * dz
                moment[1] += Q_in * zq * w[i] * dz
                moment[2] += Q_in * zq**2 * w[i] * dz
                d += Q_sh * w[i] * dz

        zn = np.where(moment[0] > 0, moment[1] / moment[0], 0.0)

        R = moment[2] - 2 * zn * moment[1] + zn**2 * moment[0]

        # g(zx) = -integral_{z_min}^{zx} Q_inplane * (z' - zn) dz'
        # I_ = integral of g^2 / Q_shear
        I_ = np.zeros(2)
        for k in range(nplies):
            zk1, zk2 = z_arr[k], z_arr[k + 1]
            Q_in = Q_ins[k]
            Q_sh = Q_shs[k]
            dz = (zk2 - zk1) / 2.0
            zmid = (zk1 + zk2) / 2.0

            for i in range(NQUAD):
                zx = zmid + dz * xi[i]
                g = np.zeros(2)
                # Sum over plies fully below zx
                for kk in range(nplies):
                    zk1_k, zk2_k = z_arr[kk], z_arr[kk + 1]
                    Q_in_k = Q_ins[kk]
                    if zk2_k < zx:
                        dz_k = (zk2_k - zk1_k) / 2.0
                        zmid_k = (zk1_k + zk2_k) / 2.0
                        for j in range(NQUAD):
                            zq_j = zmid_k + dz_k * xi[j]
                            g -= Q_in_k * (zq_j - zn) * w[j] * dz_k
                    else:
                        if zx > zk1_k:
                            a_k, b_k = zk1_k, zx
                            mid_k = (a_k + b_k) / 2.0
                            hs_k = (b_k - a_k) / 2.0
                            for j in range(NQUAD):
                                zq_j = mid_k + hs_k * xi[j]
                                g -= Q_in_k * (zq_j - zn) * w[j] * hs_k
                        break

                I_ += g**2 / Q_sh * w[i] * dz

        kappa = np.where(d > 0, R**2 / (d * I_), 5.0 / 6.0)

        return np.diag(kappa)
