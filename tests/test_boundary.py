"""Tests for boundary condition helpers."""
import numpy as np
from mechanics.boundary import EDGE_BC_MAP, build_boundary_springs


def test_edge_bc_map_keys():
    assert "free" in EDGE_BC_MAP
    assert "simply_supported" in EDGE_BC_MAP
    assert "clamped" in EDGE_BC_MAP


def test_edge_bc_map_dofs():
    assert EDGE_BC_MAP["free"] == ()
    assert EDGE_BC_MAP["simply_supported"] == (0, 1, 2)
    assert EDGE_BC_MAP["clamped"] == (0, 1, 2, 3, 4)


def test_free_edge_no_springs():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "free"}, right={"type": "free"},
        top={"type": "free"}, bottom={"type": "free"},
    )
    assert len(springs) == 0


def test_clamped_all_edges():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    # 4 edges x 5 dofs = 20 springs
    assert len(springs) == 20


def test_simply_supported():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "simply_supported"}, right={"type": "simply_supported"},
        top={"type": "simply_supported"}, bottom={"type": "simply_supported"},
    )
    # 4 edges x 3 dofs (0,1,2) = 12
    assert len(springs) == 12


def test_spring_positions_x():
    springs = build_boundary_springs(
        0.5, 0.3,
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "free"}, bottom={"type": "free"},
    )
    for val, dof, x, y in springs:
        if x is not None:
            assert x in (0.0, 0.5)
        assert y is None


def test_spring_positions_y():
    springs = build_boundary_springs(
        0.5, 0.3,
        left={"type": "free"}, right={"type": "free"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    for val, dof, x, y in springs:
        assert x is None
        if y is not None:
            assert y in (0.0, 0.3)


def test_elastic_edge():
    k_vals = [1e6, 2e6, 3e6, 4e6, 5e6]
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "elastic", "k": k_vals},
        right={"type": "free"},
        top={"type": "free"},
        bottom={"type": "free"},
    )
    assert len(springs) == 5
    spring_k = [s[0] for s in springs]
    np.testing.assert_allclose(spring_k, k_vals)


def test_custom_stiffness():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
        k_stiffness=1e10,
    )
    for val, dof, x, y in springs:
        assert val == 1e10


def test_mixed_boundary():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "clamped"}, right={"type": "simply_supported"},
        top={"type": "free"}, bottom={"type": "elastic", "k": [1e5, 0, 1e5, 0, 0]},
    )
    # left: 5, right: 3, top: 0, bottom: 2
    assert len(springs) == 10


def test_spring_dof_indices():
    springs = build_boundary_springs(
        0.3, 0.3,
        left={"type": "clamped"},
        right={"type": "free"},
        top={"type": "free"},
        bottom={"type": "free"},
    )
    dofs = sorted(set(s[1] for s in springs))
    assert dofs == [0, 1, 2, 3, 4]
