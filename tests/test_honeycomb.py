"""Tests for honeycomb core equivalent properties (Gibson formula)."""
import math
import pytest
import numpy as np
from mechanics.honeycomb import honeycomb_properties


def test_gibson_matches_original():
    """Validate against plate-main/honeycomb.py reference."""
    E = 70e9
    G = E / (2 * (1 + 0.33))
    rho = 2710
    t = 0.2e-3
    l1 = 3e-3
    l2 = 3e-3
    theta = 30 / 180 * math.pi

    props = honeycomb_properties(E, G, rho, t, l1, l2, theta)

    # Reference from original honeycomb.py
    assert abs(props["E1"] - 4.726844e7) / 4.726847e7 < 1e-6
    assert abs(props["E2"] - 4.754649e7) / 4.754649e7 < 1e-6
    assert abs(props["G23"] - 1.012895e9) / 1.012895e9 < 1e-6
    assert abs(props["G13"] - 1.012895e9) / 1.012895e9 < 1e-6
    assert abs(props["G12"] - 1.197467e7) / 1.197467e7 < 1e-6
    assert abs(props["nu12"] - 0.9824561) / 0.9824561 < 1e-6
    assert abs(props["rho"] - 278.1545) / 278.1545 < 1e-6


def test_symmetry_zero_angle():
    """At theta=0, E1 and E2 should be well-defined (limit case)."""
    E = 70e9
    G = E / (2.6)
    rho = 2710
    props = honeycomb_properties(E, G, rho, 0.2e-3, 3e-3, 3e-3, 0.01)
    assert props["E1"] > 0
    assert props["E2"] > 0
    assert props["rho"] > 0


def test_positive_definite():
    """All moduli should be positive for physical parameters."""
    E = 70e9
    G = E / (2.6)
    rho = 2710
    props = honeycomb_properties(E, G, rho, 0.2e-3, 3e-3, 3e-3, 30 / 180 * math.pi)
    for key in ["E1", "E2", "G23", "G13", "G12", "rho"]:
        assert props[key] > 0, f"{key} should be positive"
    assert 0 < props["nu12"] < 1


def test_dimensions():
    """Output should have exactly 7 keys."""
    props = honeycomb_properties(70e9, 26e9, 2710, 0.2e-3, 3e-3, 3e-3, math.pi / 6)
    assert set(props.keys()) == {"E1", "E2", "G23", "G13", "G12", "nu12", "rho"}


def test_large_theta():
    """At theta near 90 degrees, properties should still be finite."""
    props = honeycomb_properties(70e9, 26e9, 2710, 0.2e-3, 3e-3, 3e-3, 85 * math.pi / 180)
    for v in props.values():
        assert math.isfinite(v)


def test_l1c_zero_raises():
    with pytest.raises(ValueError, match="l1c must be positive"):
        honeycomb_properties(70e9, 26e9, 2710, 0.2e-3, 0.0, 3e-3, math.pi / 6)


def test_negative_tc_raises():
    with pytest.raises(ValueError, match="tc must be non-negative"):
        honeycomb_properties(70e9, 26e9, 2710, -0.1e-3, 3e-3, 3e-3, math.pi / 6)


def test_theta_too_large_raises():
    with pytest.raises(ValueError, match="theta_c must be in"):
        honeycomb_properties(70e9, 26e9, 2710, 0.2e-3, 3e-3, 3e-3, math.pi / 2)
