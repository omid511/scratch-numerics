"""Reload affine heads, reproduce constraints/calibration and raw-waveform inference."""
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_correction as E
from corrective_head import functions, transform, fit_ridge_scaling
from threadpoolctl import threadpool_limits
C, np, OUT = E.C, E.np, E.OUT
spec = importlib.util.spec_from_file_location('correction_acquisition', C.OUT / 'prepare_features.py')
acquisition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquisition)


def points(q, one, full, heads):
    control = E.control_features(q, np.column_stack((one(q[:, :107]), full(q))))
    base = control[:, 0]
    result = [base, base + E.old_head_predict(heads['blend_control'], control)]
    for arm in E.ARMS[2:]:
        kind = arm.split('_')[0]
        x = np.column_stack((control, E.H.additions(q, kind)))
        result.append(base + E.correction(heads[arm], x))
    return np.array(result)


def verify():
    d = E.setup()
    fits = json.loads((OUT / 'fits.json').read_text())
    assert len(fits['fits']) == 20 and fits['all_converged']
    with np.load(OUT / 'predictions.npz') as z:
        assert json.loads(str(z['fingerprint'])) == d['fingerprint'] == fits['fingerprint']
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        for key, value in [('y', d['y']), ('ids', d['ids']), ('source_indices', d['rows']), ('row_fold', d['folds'])]:
            np.testing.assert_array_equal(z[key], value)
        assert z['arms'].tolist() == E.ARMS and z['seeds'].tolist() == [42]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    np.testing.assert_allclose(p[0], d['baseline'], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(t[0], d['baseline_thresholds'])
    models, radii, partitions, constraint_replays = {}, {}, [], []
    max_calibration_error = 0.
    for fold in range(5):
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        with np.load(OUT / f'fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
            part = json.loads(str(z['partitions']))
            for key, rows in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
                np.testing.assert_array_equal(part[key], rows)
            heads = C.cloudpickle.loads(z['heads'].tobytes())
            radii[fold] = json.loads(str(z['calibration']))
            recorded = z['calibration_points'].reshape(len(E.ARMS), -1)
            fold_records = json.loads(str(z['records']))
        objective_rows, constraint_rows = np.array(part['objective_rows']), np.array(part['constraint_rows'])
        sets = [set(d['ids'][rows]) for rows in (objective_rows, constraint_rows, cal, held)]
        assert sets[0] | sets[1] == set(d['ids'][proper])
        assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
        # Verify all four copies stay together and scaling uses objective rows only.
        with np.load(E.OLD / f'fusion_seed42_fold{fold}.npz') as z:
            E._verify_archived_contract(json.loads(str(z['fingerprint']))['frozen'], d['fingerprint']['coordinates']['contract'])
            tx = z['train_control']
        update, validation = np.searchsorted(proper, objective_rows), np.searchsorted(proper, constraint_rows)
        for arm in E.ARMS[2:]:
            kind = arm.split('_')[0]
            trainx = np.concatenate((tx, E.H.additions(d['train'][:, proper], kind)), axis=-1)
            x, vx = trainx[:, update].reshape(-1, 28), trainx[:, validation].reshape(-1, 28)
            head = heads[arm]
            mean, scale, active = fit_ridge_scaling(x)
            for key, value in [('mean', mean), ('scale', scale), ('active', active)]:
                np.testing.assert_array_equal(head[key], value)
            objective, constraints, reference = functions(transform(head, x), tx[:, update, 0].reshape(-1),
                np.tile(d['y'][objective_rows], 4), transform(head, vx), tx[:, validation, 0].reshape(-1),
                np.tile(d['y'][constraint_rows], 4))
            slacks = constraints(head['coef'])[0]
            record = next(r for r in fold_records if r['arm'] == arm)
            assert record == next(r for r in fits['fits'] if r['arm'] == arm and r['fold'] == fold)
            np.testing.assert_allclose(slacks, record['constraint_slacks'], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(reference, record['constraint_reference_normalized_metrics'], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(objective(head['coef'])[0], record['loss'], atol=1e-12, rtol=1e-12)
            if arm.endswith('_constrained'):
                assert slacks.min() >= -1e-7
            constraint_replays.append(dict(fold=fold, arm=arm, slacks=slacks.tolist()))
        one, full = C.reference_predictors(42, fold, proper, cal, held)
        models[fold] = (one, full, heads)
        cp = points(d['train'][:, cal].reshape(-1, 174), one, full, heads)
        np.testing.assert_allclose(cp, recorded, atol=1e-12, rtol=1e-12)
        max_calibration_error = max(max_calibration_error, float(abs(cp-recorded).max()))
        target, ids = np.tile(d['y'][cal], 4), np.tile(d['ids'][cal], 4)
        rates = []
        for ai, arm in enumerate(E.ARMS):
            radius = C.calibration(cp[ai], target, ids)
            assert radius == radii[fold][arm]
            score = C.distribution(cp[ai], radius)[:, 3]
            levels = [C.threshold(score, target, b) for b in C.BUDGETS]
            np.testing.assert_array_equal(levels, t[ai, fold])
            observed = [float(np.mean(-score[target>0] <= level)) for level in levels]
            assert np.all(np.array(observed) <= C.BUDGETS)
            rates.append(observed)
        partitions.append(dict(fold=fold, objective_designs=len(sets[0]), constraint_designs=len(sets[1]),
                               calibration_designs=len(sets[2]), held_designs=len(sets[3]), calibration_false_warnings=rates))
    wanted = [3696, 5198, int(d['rows'][np.flatnonzero((d['y']<0) & (d['y']>-.01))[0]]),
              int(d['rows'][np.flatnonzero(d['y']>=.15)[0]]), int(d['rows'][len(d['rows'])//2])]
    gains = np.array(json.loads((E.ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    with np.load(C.OUT / 'features.npz') as z:
        raw_hashes = json.loads(str(z['raw_hashes']))
    B = C.helpers.base
    ridge = B.PhysicsFeatureRidge()
    replays, max_features, max_prediction_error = [], 0., 0.
    for source in dict.fromkeys(wanted):
        matched = np.flatnonzero(d['rows'] == source)
        assert len(matched) == 1
        k = int(matched[0]); fold = int(d['folds'][k]); did = str(d['ids'][k])
        one, full, heads = models[fold]
        path = E.ROOT / 'p4_sensor_candidates' / f'{did}.npz'
        assert C.sha256(path) == raw_hashes[did]
        with np.load(path) as z:
            j = int(np.flatnonzero(z['source_indices'] == source)[0])
            raw, sigma = z['raw'][j], z['noise_scale'][j]
        for name in ('clean', 'iid5_new949', 'correlated5_new949_950'):
            ci = d['names'].index(name)
            signal, dt = acquisition.measured_signal(raw, sigma, source, name, d['layout'], gains, float(d['dt'][k]), {}, {})
            modes = B.estimate_modes(signal, dt, include_shapes=True)
            q = np.r_[B.base_feature_row(signal, dt, modes, ridge), B.modal_spatial_row(modes, d['pairs']),
                      B.modal_spatial_addition_row(modes, d['pairs'])][None]
            feature_error = float(abs(q-d['evaluation'][ci, k]).max())
            np.testing.assert_allclose(q[0], d['evaluation'][ci, k], atol=1e-9, rtol=1e-9)
            start = time.perf_counter()
            cp = points(q, one, full, heads)
            actual = np.array([C.distribution(cp[ai], radii[fold][arm])[0] for ai, arm in enumerate(E.ARMS)])
            seconds = time.perf_counter()-start
            error = float(abs(actual-p[:, ci, k]).max())
            np.testing.assert_allclose(actual, p[:, ci, k], atol=1e-8, rtol=1e-8)
            max_features = max(max_features, feature_error); max_prediction_error = max(max_prediction_error, error)
            replays.append(dict(source_index=source, design=did, fold=fold, condition=name,
                max_feature_error=feature_error, max_prediction_error=error, cached_feature_all_arms_inference_seconds=seconds))
            print('CORRECTION_WAVEFORM_REPLAY', source, name, feature_error, error, flush=True)
    assert len(constraint_replays) == 20
    proof = dict(all_predictions_finite=True, four_design_partitions_disjoint=True,
        frozen_baseline_replay=True, archived_control_replay=True, all_calibration_budgets_respected=True,
        fitted_affine_heads=20, constraints_replayed=constraint_replays, partitions=partitions,
        real_waveform_replays=len(replays), replays=replays, max_feature_error=max_features,
        max_prediction_error=max_prediction_error, max_calibration_error=max_calibration_error,
        numeric=E.H.numerical_check(), predictions_sha256=C.sha256(OUT / 'predictions.npz'),
        baseline_compatibility=fits['baseline_compatibility'],
        limitation='One initialization; repeated TRAIN exploration; OOF members share training designs across readout partitions. Timings exclude modal extraction.')
    C.save_json(OUT / 'verification.json', proof)
    print('CORRECTION_VERIFICATION_COMPLETE', len(replays), 'raw waveform replays; twenty reloaded constraint/objective checks', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
