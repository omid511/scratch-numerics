"""Spatial patterns must retain channel phase without excitation gauge artifacts."""
import numpy as np
import pytest

from mechanics.p4_margin_estimation.modal_features import (
    estimate_modes, modal_spatial_row, modal_spatial_addition_row,
)

PAIRS = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])


def expected_row(shape):
    power = abs(shape)**2
    cross = shape[PAIRS[:, 0]]*shape[PAIRS[:, 1]].conj()/power.sum()
    return np.r_[power/power.sum(), cross.real, cross.imag, 1.]


def test_recovers_least_stable_sensor_pattern_not_highest_energy_mode():
    t = np.arange(512)*.001
    lead = np.array([1., 2j, -1., .5-.2j])
    other = np.array([3., -3., 2., -2.])
    x = np.real(lead[:, None]*np.exp((-1+2j*np.pi*73)*t)
                + other[:, None]*np.exp((-4+2j*np.pi*31)*t))
    x += np.array([.1, -.2, .3, -.4])[:, None]
    modes = estimate_modes(x, .001, include_shapes=True)
    assert modes['frequency'][0] < 40  # stronger mode is NOT the selected lead
    row = modal_spatial_row(modes, PAIRS)
    np.testing.assert_allclose(row, expected_row(lead), atol=1e-6)
    scaled_time = estimate_modes(x, .002, include_shapes=True)
    np.testing.assert_allclose(modal_spatial_row(scaled_time, PAIRS), row, atol=1e-6)


@pytest.mark.parametrize('scale', [3., -2., 2*np.exp(1.7j)])
def test_global_complex_excitation_scale_and_phase_do_not_change_pattern(scale):
    shape = np.array([1., 1j, -2., .5+.4j])
    modes = {'alpha': np.array([-2.]), 'frequency': np.array([40.]),
             'energy_fraction': np.array([1.]), 'mode_shapes': shape[None, :]}
    original = modal_spatial_row(modes, PAIRS)
    changed = {**modes, 'mode_shapes': scale*shape[None, :]}
    np.testing.assert_allclose(modal_spatial_row(changed, PAIRS), original, atol=1e-14)


def test_opposite_sensor_phase_is_not_erased_by_equal_channel_powers():
    t = np.arange(512)*.001
    wave = np.exp(-2*t)*np.cos(2*np.pi*40*t)
    aligned = estimate_modes(np.tile(wave, (4,1)), .001, include_shapes=True)
    opposite = estimate_modes(np.array([wave, -wave, wave, -wave]), .001, include_shapes=True)
    a, b = modal_spatial_row(aligned, PAIRS), modal_spatial_row(opposite, PAIRS)
    np.testing.assert_allclose(a[:4], b[:4], atol=1e-8)
    np.testing.assert_allclose(a[4:8], .25, atol=1e-8)
    np.testing.assert_allclose(b[4:8], -.25, atol=1e-8)


def test_mode_below_visibility_threshold_cannot_replace_visible_lead():
    modes = {'alpha': np.array([5., -2.]), 'frequency': np.array([40., 70.]),
             'energy_fraction': np.array([.005, .995]),
             'mode_shapes': np.array([[1.,1.,1.,1.], [1.,-1.,1.,-1.]])}
    np.testing.assert_allclose(modal_spatial_row(modes, PAIRS), expected_row(modes['mode_shapes'][1]))


def test_unobservable_shape_has_explicit_missing_flag():
    modes = estimate_modes(np.ones((4,512)), .001, include_shapes=True)
    np.testing.assert_array_equal(modal_spatial_row(modes, PAIRS), np.zeros(13))


@pytest.mark.parametrize('pairs', [np.array([[0,4]]), np.array([[0,-1]]), np.array([[0.,1.]])])
def test_invalid_sensor_pair_is_rejected(pairs):
    modes = estimate_modes(np.ones((4,512)), .001, include_shapes=True)
    with pytest.raises(ValueError, match='sensor pairs'):
        modal_spatial_row(modes, pairs)


def test_recovers_second_least_stable_shape_and_spatial_similarity():
    t = np.arange(512)*.001
    lead = np.array([1., 2j, -1., .5-.2j])
    second = np.array([3., -3., 2., -2.])
    x = np.real(lead[:, None]*np.exp((-1+2j*np.pi*73)*t)
                + second[:, None]*np.exp((-4+2j*np.pi*31)*t))
    modes = estimate_modes(x, .001, include_shapes=True)
    row = modal_spatial_addition_row(modes, PAIRS)
    similarity = abs(np.vdot(lead, second))**2/(np.vdot(lead, lead).real*np.vdot(second, second).real)
    np.testing.assert_allclose(row, np.r_[expected_row(second), similarity, 1.], atol=1e-6)
    changed_dt = estimate_modes(x, .002, include_shapes=True)
    np.testing.assert_allclose(modal_spatial_addition_row(changed_dt, PAIRS), row, atol=1e-6)


@pytest.mark.parametrize('scales', [(3j, -2.), (1e150, 1e-150)])
def test_similarity_preserves_independent_mode_gauges_without_overflow(scales):
    shapes = np.array([[1., 1j, -2., .5+.4j], [2., -1., .3j, 1.]])
    modes = {'alpha': np.array([-1., -3.]), 'frequency': np.array([40., 70.]),
             'energy_fraction': np.array([.4, .6]), 'mode_shapes': shapes}
    expected = modal_spatial_addition_row(modes, PAIRS)
    changed = {**modes, 'mode_shapes': shapes*np.asarray(scales)[:, None]}
    np.testing.assert_allclose(modal_spatial_addition_row(changed, PAIRS), expected, atol=1e-14)


def test_second_mode_selection_excludes_invisible_modes_and_breaks_ties_by_frequency():
    modes = {'alpha': np.array([5., -1., -2., -2.]), 'frequency': np.array([90., 80., 70., 40.]),
             'energy_fraction': np.array([.005, .495, .25, .25]),
             'mode_shapes': np.array([[1.,1.,1.,1.], [1.,-1.,1.,-1.],
                                     [1.,1j,1.,1j], [2.,1j,-1.,.5]])}
    row = modal_spatial_addition_row(modes, PAIRS)
    np.testing.assert_allclose(row[:-2], expected_row(modes['mode_shapes'][3]), atol=1e-14)


def test_missing_second_mode_and_missing_lead_do_not_imply_valid_similarity():
    modes = {'alpha': np.array([-1.]), 'frequency': np.array([40.]),
             'energy_fraction': np.array([1.]), 'mode_shapes': np.ones((1,4), complex)}
    np.testing.assert_array_equal(modal_spatial_addition_row(modes, PAIRS), np.zeros(15))
    two = {'alpha': np.array([-1., -2.]), 'frequency': np.array([40., 70.]),
           'energy_fraction': np.array([.5, .5]),
           'mode_shapes': np.array([[0.,0.,0.,0.], [1.,-1.,1.,-1.]])}
    row = modal_spatial_addition_row(two, PAIRS)
    np.testing.assert_allclose(row[:-2], expected_row(two['mode_shapes'][1]))
    np.testing.assert_array_equal(row[-2:], [0., 0.])


@pytest.mark.parametrize('second,similarity', [([1.,1.,1.,1.], 1.), ([1.,-1.,1.,-1.], 0.)])
def test_similarity_distinguishes_collinear_and_orthogonal_patterns(second, similarity):
    modes = {'alpha': np.array([-1., -2.]), 'frequency': np.array([40., 70.]),
             'energy_fraction': np.array([.5, .5]),
             'mode_shapes': np.array([[1.,1.,1.,1.], second], complex)}
    np.testing.assert_allclose(modal_spatial_addition_row(modes, PAIRS)[-2:], [similarity, 1.])
