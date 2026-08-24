"""Tests for the identifiability formalization module."""
import numpy as np
import pytest

from mechanics.p2_inverse_damage.identifiability import (
    effective_rank,
    frequency_jacobian,
    nullspace_dimension,
    cramer_rao_floor,
)


class TestEffectiveRank:
    def test_full_rank_identity(self):
        j = np.eye(6, 64)
        assert effective_rank(j) == 6

    def test_low_rank(self):
        j = np.outer(np.ones(3), np.ones(64))  # rank 1
        assert effective_rank(j) == 1

    def test_zero_matrix(self):
        assert effective_rank(np.zeros((3, 64))) == 0


class TestNullspaceDimension:
    def test_full_rank_nullspace(self):
        j = np.eye(6, 64)
        assert nullspace_dimension(j) == 64 - 6

    def test_zero_matrix_nullspace(self):
        assert nullspace_dimension(np.zeros((3, 64))) == 64


class TestCramerRaoFloor:
    def test_positive_floor(self):
        j = np.random.default_rng(42).normal(0, 1, (6, 64))
        sigma = np.full(6, 0.01)
        floor = cramer_rao_floor(j, sigma)
        assert floor > 0

    def test_higher_noise_higher_floor(self):
        j = np.random.default_rng(42).normal(0, 1, (6, 64))
        lo = cramer_rao_floor(j, np.full(6, 0.005))
        hi = cramer_rao_floor(j, np.full(6, 0.02))
        assert hi > lo
