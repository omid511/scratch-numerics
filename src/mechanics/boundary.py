"""Boundary condition helpers."""
from __future__ import annotations


EDGE_BC_MAP = {
    "free": (),
    "simply_supported": (0, 1, 2),
    "clamped": (0, 1, 2, 3, 4),
}


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
