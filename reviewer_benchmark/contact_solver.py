from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _vec2(value: Iterable[float]) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (2,):
        raise ValueError(f"expected shape (2,), got {out.shape}")
    return out


def _cross_vec_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _cross_scalar_vec(scalar: float, vec: np.ndarray) -> np.ndarray:
    return np.array([-scalar * vec[1], scalar * vec[0]], dtype=np.float64)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("contact normal must be non-zero")
    return vec / norm


@dataclass(slots=True)
class RigidBody2D:
    position: np.ndarray
    velocity: np.ndarray
    angle: float = 0.0
    angular_velocity: float = 0.0
    mass: float = np.inf
    inertia: float = np.inf
    restitution: float = 0.0
    friction: float = 0.5
    dynamic: bool = True

    def __post_init__(self) -> None:
        self.position = _vec2(self.position)
        self.velocity = _vec2(self.velocity)
        self.angle = float(self.angle)
        self.angular_velocity = float(self.angular_velocity)
        self.mass = float(self.mass)
        self.inertia = float(self.inertia)
        self.restitution = float(self.restitution)
        self.friction = float(self.friction)
        self.dynamic = bool(self.dynamic)

    @property
    def inv_mass(self) -> float:
        if not self.dynamic or not np.isfinite(self.mass) or self.mass <= 0.0:
            return 0.0
        return 1.0 / self.mass

    @property
    def inv_inertia(self) -> float:
        if not self.dynamic or not np.isfinite(self.inertia) or self.inertia <= 0.0:
            return 0.0
        return 1.0 / self.inertia

    @classmethod
    def static(
        cls,
        position: Iterable[float],
        *,
        angle: float = 0.0,
        restitution: float = 0.0,
        friction: float = 0.5,
    ) -> "RigidBody2D":
        return cls(
            position=position,
            velocity=(0.0, 0.0),
            angle=angle,
            angular_velocity=0.0,
            mass=np.inf,
            inertia=np.inf,
            restitution=restitution,
            friction=friction,
            dynamic=False,
        )


@dataclass(slots=True)
class ContactPoint:
    body_a: int
    body_b: int
    point: np.ndarray
    normal: np.ndarray
    separation: float = 0.0
    friction: float | None = None
    restitution: float | None = None
    key: object | None = None

    def __post_init__(self) -> None:
        self.body_a = int(self.body_a)
        self.body_b = int(self.body_b)
        self.point = _vec2(self.point)
        self.normal = _normalize(_vec2(self.normal))
        self.separation = float(self.separation)
        self.friction = None if self.friction is None else float(self.friction)
        self.restitution = None if self.restitution is None else float(self.restitution)


@dataclass(slots=True)
class ContactImpulse:
    key: tuple[object, ...]
    normal_impulse: float
    tangent_impulse: float


@dataclass(slots=True)
class _Constraint:
    contact: ContactPoint
    key: tuple[object, ...]
    r_a: np.ndarray
    r_b: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    normal_mass: float
    tangent_mass: float
    friction: float
    bias: float
    normal_impulse: float = 0.0
    tangent_impulse: float = 0.0


class ContactSolver:
    def __init__(
        self,
        *,
        iterations: int = 12,
        baumgarte: float = 0.2,
        penetration_slop: float = 1e-3,
        restitution_speed_threshold: float = 0.5,
    ) -> None:
        self.iterations = int(iterations)
        self.baumgarte = float(baumgarte)
        self.penetration_slop = float(penetration_slop)
        self.restitution_speed_threshold = float(restitution_speed_threshold)
        self._cache: dict[tuple[object, ...], tuple[float, float]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def solve(
        self,
        bodies: list[RigidBody2D],
        contacts: Iterable[ContactPoint],
        dt: float,
        *,
        warm_start: bool = True,
    ) -> list[ContactImpulse]:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        constraints = self._build_constraints(bodies, contacts, dt)
        if not constraints:
            self._cache.clear()
            return []

        if warm_start:
            for constraint in constraints:
                cached = self._cache.get(constraint.key)
                if cached is None:
                    continue
                constraint.normal_impulse, constraint.tangent_impulse = cached
                self._apply_impulse(
                    bodies,
                    constraint.contact.body_a,
                    constraint.contact.body_b,
                    constraint.r_a,
                    constraint.r_b,
                    constraint.normal * constraint.normal_impulse
                    + constraint.tangent * constraint.tangent_impulse,
                )

        for _ in range(self.iterations):
            for constraint in constraints:
                self._solve_normal(bodies, constraint)
                self._solve_tangent(bodies, constraint)

        self._cache = {
            constraint.key: (constraint.normal_impulse, constraint.tangent_impulse)
            for constraint in constraints
        }
        return [
            ContactImpulse(
                key=constraint.key,
                normal_impulse=constraint.normal_impulse,
                tangent_impulse=constraint.tangent_impulse,
            )
            for constraint in constraints
        ]

    def _build_constraints(
        self,
        bodies: list[RigidBody2D],
        contacts: Iterable[ContactPoint],
        dt: float,
    ) -> list[_Constraint]:
        items = []
        for index, contact in enumerate(contacts):
            if contact.body_a == contact.body_b:
                raise ValueError("contact must reference two different bodies")
            body_a = bodies[contact.body_a]
            body_b = bodies[contact.body_b]
            key = self._contact_key(contact)
            items.append((self._sort_key(contact, key, index), key, body_a, body_b, contact))
        items.sort(key=lambda item: item[0])

        constraints: list[_Constraint] = []
        for _, key, body_a, body_b, contact in items:
            r_a = contact.point - body_a.position
            r_b = contact.point - body_b.position
            normal = contact.normal
            tangent = np.array([-normal[1], normal[0]], dtype=np.float64)

            rn_a = _cross_vec_vec(r_a, normal)
            rn_b = _cross_vec_vec(r_b, normal)
            rt_a = _cross_vec_vec(r_a, tangent)
            rt_b = _cross_vec_vec(r_b, tangent)

            normal_denom = (
                body_a.inv_mass
                + body_b.inv_mass
                + body_a.inv_inertia * rn_a * rn_a
                + body_b.inv_inertia * rn_b * rn_b
            )
            tangent_denom = (
                body_a.inv_mass
                + body_b.inv_mass
                + body_a.inv_inertia * rt_a * rt_a
                + body_b.inv_inertia * rt_b * rt_b
            )
            if normal_denom <= 0.0:
                continue

            friction = (
                float(np.sqrt(body_a.friction * body_b.friction))
                if contact.friction is None
                else contact.friction
            )
            restitution = (
                max(body_a.restitution, body_b.restitution)
                if contact.restitution is None
                else contact.restitution
            )

            relative_velocity = self._relative_velocity(body_a, body_b, r_a, r_b)
            normal_speed = float(np.dot(relative_velocity, normal))
            restitution_bias = 0.0
            if normal_speed < -self.restitution_speed_threshold:
                restitution_bias = -restitution * normal_speed

            penetration = min(0.0, contact.separation + self.penetration_slop)
            stabilization_bias = -self.baumgarte * penetration / dt

            constraints.append(
                _Constraint(
                    contact=contact,
                    key=key,
                    r_a=r_a,
                    r_b=r_b,
                    normal=normal,
                    tangent=tangent,
                    normal_mass=1.0 / normal_denom,
                    tangent_mass=0.0 if tangent_denom <= 0.0 else 1.0 / tangent_denom,
                    friction=max(0.0, friction),
                    bias=restitution_bias + stabilization_bias,
                )
            )
        return constraints

    def _solve_normal(self, bodies: list[RigidBody2D], constraint: _Constraint) -> None:
        body_a = bodies[constraint.contact.body_a]
        body_b = bodies[constraint.contact.body_b]
        relative_velocity = self._relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)
        normal_speed = float(np.dot(relative_velocity, constraint.normal))
        delta = constraint.normal_mass * (-normal_speed + constraint.bias)
        next_impulse = max(0.0, constraint.normal_impulse + delta)
        delta = next_impulse - constraint.normal_impulse
        if delta == 0.0:
            return
        constraint.normal_impulse = next_impulse
        self._apply_impulse(
            bodies,
            constraint.contact.body_a,
            constraint.contact.body_b,
            constraint.r_a,
            constraint.r_b,
            constraint.normal * delta,
        )

    def _solve_tangent(self, bodies: list[RigidBody2D], constraint: _Constraint) -> None:
        if constraint.tangent_mass <= 0.0 or constraint.friction <= 0.0:
            return
        body_a = bodies[constraint.contact.body_a]
        body_b = bodies[constraint.contact.body_b]
        relative_velocity = self._relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)
        tangent_speed = float(np.dot(relative_velocity, constraint.tangent))
        delta = constraint.tangent_mass * (-tangent_speed)
        limit = constraint.friction * constraint.normal_impulse
        next_impulse = float(np.clip(constraint.tangent_impulse + delta, -limit, limit))
        delta = next_impulse - constraint.tangent_impulse
        if delta == 0.0:
            return
        constraint.tangent_impulse = next_impulse
        self._apply_impulse(
            bodies,
            constraint.contact.body_a,
            constraint.contact.body_b,
            constraint.r_a,
            constraint.r_b,
            constraint.tangent * delta,
        )

    @staticmethod
    def _relative_velocity(
        body_a: RigidBody2D,
        body_b: RigidBody2D,
        r_a: np.ndarray,
        r_b: np.ndarray,
    ) -> np.ndarray:
        vel_a = body_a.velocity + _cross_scalar_vec(body_a.angular_velocity, r_a)
        vel_b = body_b.velocity + _cross_scalar_vec(body_b.angular_velocity, r_b)
        return vel_b - vel_a

    @staticmethod
    def _apply_impulse(
        bodies: list[RigidBody2D],
        body_a_index: int,
        body_b_index: int,
        r_a: np.ndarray,
        r_b: np.ndarray,
        impulse: np.ndarray,
    ) -> None:
        body_a = bodies[body_a_index]
        body_b = bodies[body_b_index]
        if body_a.inv_mass > 0.0:
            body_a.velocity = body_a.velocity - impulse * body_a.inv_mass
            body_a.angular_velocity -= body_a.inv_inertia * _cross_vec_vec(r_a, impulse)
        if body_b.inv_mass > 0.0:
            body_b.velocity = body_b.velocity + impulse * body_b.inv_mass
            body_b.angular_velocity += body_b.inv_inertia * _cross_vec_vec(r_b, impulse)

    @staticmethod
    def _contact_key(contact: ContactPoint) -> tuple[object, ...]:
        if contact.key is not None:
            return ("user", contact.body_a, contact.body_b, repr(contact.key))
        point = tuple(np.round(contact.point, 9))
        normal = tuple(np.round(contact.normal, 9))
        return ("auto", contact.body_a, contact.body_b, point, normal)

    @staticmethod
    def _sort_key(
        contact: ContactPoint,
        key: tuple[object, ...],
        index: int,
    ) -> tuple[object, ...]:
        return (contact.body_a, contact.body_b, key, index)


def _central_elastic_collision_check() -> None:
    solver = ContactSolver(iterations=6, restitution_speed_threshold=0.0)
    bodies = [
        RigidBody2D(position=(-0.5, 0.0), velocity=(1.0, 0.0), mass=1.0, inertia=1.0, restitution=1.0),
        RigidBody2D(position=(0.5, 0.0), velocity=(-1.0, 0.0), mass=1.0, inertia=1.0, restitution=1.0),
    ]
    contacts = [
        ContactPoint(
            body_a=0,
            body_b=1,
            point=(0.0, 0.0),
            normal=(1.0, 0.0),
            separation=0.0,
            key="head_on",
        )
    ]
    impulses = solver.solve(bodies, contacts, dt=1.0 / 60.0)
    assert len(impulses) == 1
    np.testing.assert_allclose(bodies[0].velocity, (-1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(bodies[1].velocity, (1.0, 0.0), atol=1e-9)


def _resting_contact_check() -> None:
    solver = ContactSolver(iterations=20, restitution_speed_threshold=1.0)
    dt = 1.0 / 120.0
    gravity = np.array([0.0, -9.81], dtype=np.float64)
    radius = 0.5
    bodies = [
        RigidBody2D(position=(0.0, radius), velocity=(0.0, 0.0), mass=1.0, inertia=0.25, friction=0.8),
        RigidBody2D.static(position=(0.0, 0.0), friction=0.8),
    ]

    for _ in range(240):
        bodies[0].velocity = bodies[0].velocity + gravity * dt
        bodies[0].position = bodies[0].position + bodies[0].velocity * dt
        separation = bodies[0].position[1] - radius
        contacts = [
            ContactPoint(
                body_a=1,
                body_b=0,
                point=(bodies[0].position[0], 0.0),
                normal=(0.0, 1.0),
                separation=separation,
                key="ground",
            )
        ]
        solver.solve(bodies, contacts, dt=dt)
        bodies[0].position[1] = max(bodies[0].position[1], radius)

    assert abs(bodies[0].velocity[1]) < 1e-3
    assert bodies[0].position[1] >= radius - 1e-9


def _self_check() -> None:
    _central_elastic_collision_check()
    _resting_contact_check()
    print("contact_solver self-check ok")


if __name__ == "__main__":
    _self_check()
