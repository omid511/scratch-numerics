"""Boundary condition helpers."""
from __future__ import annotations

import numpy as np


# DOF indices for 5-DOF FSDT plate
DOF_U, DOF_V, DOF_W, DOF_THETA_X, DOF_THETA_Y = 0, 1, 2, 3, 4


# Deprecated: use constrained_dofs_for_edge() instead
EDGE_BC_MAP = {
    "free": (),
    "simply_supported": (DOF_U, DOF_V, DOF_W),
    "clamped": (DOF_U, DOF_V, DOF_W, DOF_THETA_X, DOF_THETA_Y),
}


def constrained_dofs_for_edge(boundary_type: str, edge: str) -> tuple[int, ...]:
    """Return DOF indices constrained for the given boundary type and edge.

    Args:
        boundary_type: one of "free", "clamped", "simply_supported_movable",
                       "simply_supported_immovable"
        edge: one of "left", "right", "top", "bottom"
    """
    if edge not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"Unknown edge: {edge}")
    if boundary_type == "free":
        return ()
    if boundary_type == "clamped":
        return (DOF_U, DOF_V, DOF_W, DOF_THETA_X, DOF_THETA_Y)
    if boundary_type == "simply_supported_movable":
        return (DOF_W,)
    if boundary_type == "simply_supported_immovable":
        if edge in {"left", "right"}:
            return (DOF_U, DOF_W)
        return (DOF_V, DOF_W)
    raise ValueError(f"Unknown boundary type: {boundary_type}")


def add_penalty_constraints(
    K_struct: np.ndarray,
    B: np.ndarray,
    penalty_factor: float = 1e6,
) -> tuple[np.ndarray, float]:
    """Apply scaled penalty method to enforce DOF constraints.

    Returns (K_constrained, k_penalty).
    """
    diag_scale = float(np.max(np.abs(np.diag(K_struct))))
    if not np.isfinite(diag_scale) or diag_scale <= 0.0:
        raise ValueError("Cannot derive a positive stiffness scale")
    k_penalty = penalty_factor * diag_scale
    K_constrained = K_struct + k_penalty * (B.T @ B)
    condition_number = np.linalg.cond(K_constrained)
    if condition_number > 1e12:
        raise np.linalg.LinAlgError(
            f"Penalty-constrained stiffness is ill-conditioned: cond={condition_number:.3e}"
        )
    return K_constrained, k_penalty


def build_boundary_springs(
    L1: float, L2: float,
    left: dict, right: dict, top: dict, bottom: dict,
    k_stiffness: float = 1e14,
) -> list[tuple[float, int, float | None, float | None]]:
    """Convert edge config to penalty spring list.

    Returns list of (value, dof, x_or_None, y_or_None).
    """
    springs = []

    def _add_edge(edge_cfg: dict, edge_axis: str, edge_pos: float):
        bc_type = edge_cfg.get("type", "clamped")
        if bc_type == "free":
            return
        if bc_type == "elastic":
            k_values = edge_cfg.get("k", [k_stiffness] * 5)
        else:
            dofs = EDGE_BC_MAP[bc_type]
            k_values = [k_stiffness if d in dofs else 0.0 for d in range(5)]

        for dof, k_val in enumerate(k_values):
            if k_val == 0.0:
                continue
            if edge_axis == "x":
                springs.append((k_val, dof, edge_pos, None))
            else:
                springs.append((k_val, dof, None, edge_pos))

    _add_edge(left, "x", 0.0)
    _add_edge(right, "x", L1)
    _add_edge(bottom, "y", 0.0)
    _add_edge(top, "y", L2)

    return springs


def scale_springs_to_penalty(
    springs: list[tuple[float, int, float | None, float | None]],
    K_struct: np.ndarray,
    penalty_factor: float,
) -> list[tuple[float, int, float | None, float | None]]:
    """Rescale boundary springs so penalty = penalty_factor * max(|diag(K_struct)|).

    This replaces the fixed absolute stiffness with a value proportional to the
    structural matrix, preventing ill-conditioning across different panel geometries.
    """
    diag_scale = float(np.max(np.abs(np.diag(K_struct))))
    if not np.isfinite(diag_scale) or diag_scale <= 0.0:
        raise ValueError("Cannot derive a positive stiffness scale from structural matrix")

    k_target = penalty_factor * diag_scale

    scaled = []
    for value, dof, x, y in springs:
        scaled.append((k_target, dof, x, y))
    return scaled
