import unittest
from types import SimpleNamespace

import numpy as np

from p1_improvement_study import conformal_state, interval_score, raw_scale
from p1_prospective import mode1_calibration, screening_action
from p1_reference_diagnosis import mass_vectors, span_diagnostics


class ShellMassTests(unittest.TestCase):
    def test_offset_coupling_and_director_inertia_include_complex_state(self):
        values = np.zeros((6, 1, 2), dtype=complex)
        displacement = np.array([[1+2j, -.5j, 3], [2j, 1, -.7]])
        director = np.array([[.2j, 1, -.3], [1, -.4j, .1]])
        values[:3, 0] = displacement.T
        values[3:, 0] = director.T
        areas = np.array([.4, .7])
        thickness, offset, density = .3, -.2, 5.
        weighted = mass_vectors(values, areas, thickness, offset, density)
        energy = np.sum(np.abs(displacement)**2, axis=1)
        coupling = 2*offset*np.real(np.sum(displacement.conj()*director, axis=1))
        inertia = (offset**2+thickness**2/12)*np.sum(np.abs(director)**2, axis=1)
        expected = density*thickness*np.dot(areas, energy+coupling+inertia)
        self.assertAlmostEqual(float(np.vdot(weighted, weighted).real), expected, places=12)
        centered = mass_vectors(values, areas, thickness, 0., density)
        self.assertGreater(abs(float(np.vdot(centered, centered).real)-expected), .1)

    def test_basis_mixing_changes_individual_mac_but_not_span(self):
        reference = np.eye(4)[:2]
        angle = .3
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        mixed = rotation@reference
        result = span_diagnostics(reference, mixed)
        np.testing.assert_allclose(result['individual_mac'], np.cos(angle)**2, atol=1e-14)
        np.testing.assert_allclose(result['principal_angles_degrees'], 0., atol=2e-6)
        mixed[0, 2] = .1
        changed = span_diagnostics(reference, mixed)
        self.assertGreater(max(changed['principal_angles_degrees']), 5.)


def calibration_fixture(run_count):
    runs = np.repeat(np.arange(1, run_count+1), 2)
    modes = np.tile([1, 2], run_count)
    interior = np.zeros((3, 3), dtype=bool)
    interior[1, 1] = True
    dataset = SimpleNamespace(run_ids=runs, mode_ids=modes, interior_mask=interior,
                              correction=np.zeros((len(runs), 3, 3)),
                              f_hf=np.full(len(runs), 100.))
    prediction = {'correction': np.zeros_like(dataset.correction),
                  'correction_std': np.ones_like(dataset.correction),
                  'frequency': dataset.f_hf.copy(), 'frequency_std': np.ones(len(runs))}
    prediction['correction'][::2, 1, 1] = np.arange(1, run_count+1)
    prediction['correction'][1::2, 1, 1] = .5
    prediction['frequency'][1::2] += 2*np.arange(1, run_count+1)
    return dataset, prediction


class CalibrationTests(unittest.TestCase):
    def test_quantile_counts_geometries_not_modal_rows(self):
        dataset, prediction = calibration_fixture(9)
        state = conformal_state(dataset, np.arange(len(dataset.run_ids)), prediction)
        self.assertEqual(state['multipliers']['90'],
                         {'field': 9., 'frequency': 18., 'joint': 18., 'field_rms': 9.})
        self.assertNotIn('95', state['multipliers'])
        dataset, prediction = calibration_fixture(19)
        state = conformal_state(dataset, np.arange(len(dataset.run_ids)), prediction)
        self.assertEqual(state['multipliers']['95']['joint'], 38.)

    def test_joint_score_uses_worst_quantity_on_each_run(self):
        dataset, prediction = calibration_fixture(9)
        prediction['correction'][0, 1, 1] = 30.
        state = conformal_state(dataset, np.arange(len(dataset.run_ids)), prediction)
        self.assertEqual(state['scores'][0]['joint'], 30.)
        self.assertEqual(state['multipliers']['90']['joint'], 30.)

    def test_nonfinite_and_nonpositive_scales_are_errors_not_unsupported_ranks(self):
        for key, value in [('correction', np.nan), ('frequency', np.inf),
                           ('correction_std', np.nan), ('frequency_std', 0.)]:
            with self.subTest(quantity=key):
                dataset, prediction = calibration_fixture(19)
                prediction[key].flat[0] = value
                with self.assertRaises(ValueError):
                    conformal_state(dataset, np.arange(len(dataset.run_ids)), prediction)

    def test_frequency_screening_calibration_does_not_inherit_other_mode_error(self):
        dataset, prediction = calibration_fixture(19)
        prediction['frequency'][::2] += np.arange(1, 20)
        prediction['frequency'][1::2] = 1.e8
        state = mode1_calibration(dataset, np.arange(len(dataset.run_ids)), prediction)
        self.assertEqual(state['multipliers']['95'], 19.)
        self.assertEqual(state['runs'], list(range(1, 20)))

    def test_raw_scale_floors_only_positive_domain_and_rejects_invalid_values(self):
        dataset, prediction = calibration_fixture(9)
        prediction['correction_std'].fill(0.)
        prediction['frequency_std'].fill(0.)
        scaled = raw_scale(prediction, dataset.interior_mask)
        np.testing.assert_array_equal(scaled['correction_std'][:, ~dataset.interior_mask], 0.)
        self.assertTrue(np.all(scaled['correction_std'][:, dataset.interior_mask] > 0))
        self.assertTrue(np.all(scaled['frequency_std'] > 0))
        prediction['frequency_std'][0] = -1.
        with self.assertRaises(ValueError):
            raw_scale(prediction, dataset.interior_mask)

    def test_proper_interval_score_penalizes_miss_and_is_sign_symmetric(self):
        errors = np.array([0., 2., 4., -4.])
        np.testing.assert_allclose(interval_score(errors, np.full(4, 2.), .1), [4., 4., 44., 44.])
        with self.assertRaises(ValueError):
            interval_score(errors, 2., 0.)

    def test_screening_threshold_equalities_do_not_false_reject(self):
        self.assertEqual(screening_action(105., 5., 100.), 'accept')
        self.assertEqual(screening_action(95., 5., 100.), 'HF')
        self.assertEqual(screening_action(94., 5., 100.), 'reject')
        self.assertEqual(screening_action(100., 0., 100.), 'accept')


class DirectHFConsumerTests(unittest.TestCase):
    def test_predicted_hf_is_independent_of_lf_and_clamped_at_boundary(self):
        import test_p1_surrogate
        from p1_improved_models import DirectHFModel
        from p1_prospective import LFInputs
        dataset = test_p1_surrogate.SurrogateTests.dataset()
        model = DirectHFModel('matern32', max_iterations=3, seed=4).fit(
            dataset, np.flatnonzero(dataset.run_ids <= 4))
        rows = np.flatnonzero(dataset.run_ids == 8)
        inputs = LFInputs(dataset.x, dataset.y, dataset.boundary_mask,
            dataset.interior_mask, dataset.lf.copy(), dataset.run_ids,
            dataset.mode_ids, dataset.parameters, dataset.parameter_names, dataset.f_lf)
        original = model.predict_rows(inputs, rows)
        original_hf = inputs.lf[rows]+original['correction']
        inputs.lf += .2
        inputs.f_lf = inputs.f_lf*7.
        changed = model.predict_rows(inputs, rows)
        changed_hf = inputs.lf[rows]+changed['correction']
        np.testing.assert_allclose(changed_hf, original_hf, rtol=1.e-12, atol=1.e-12)
        np.testing.assert_array_equal(changed['frequency'], original['frequency'])
        np.testing.assert_array_equal(changed_hf[:, ~dataset.interior_mask], 0.)


if __name__ == '__main__':
    unittest.main()
