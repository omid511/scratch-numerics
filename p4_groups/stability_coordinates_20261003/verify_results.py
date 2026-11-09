"""Replay new coordinate predictions from physical P4 waveforms."""
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as E
from threadpoolctl import threadpool_limits
C, np, OUT = E.C, E.np, E.OUT
spec = importlib.util.spec_from_file_location('coordinate_acquisition', C.OUT / 'prepare_features.py')
acquisition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquisition)


def load_predictor(kind, fold, proper, cal, held):
    path = OUT / f'{kind}_seed42_fold{fold}.npz'
    with np.load(path) as z:
        assert json.loads(str(z['fingerprint'])) == E.DATA['fingerprint']
        for key, value in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', held)]:
            np.testing.assert_array_equal(z[key], value)
        meta = json.loads(str(z['metadata']))
        assert meta['success']
        predict = C.cloudpickle.loads(z['predictor'].tobytes())
    return predict


def verify():
    d = E.setup()
    with np.load(OUT / 'predictions.npz') as z:
        assert json.loads(str(z['fingerprint'])) == d['fingerprint']
        p, t = z['predictions'][:, 0], z['thresholds'][:, 0]
        for key, value in [('y', d['y']), ('ids', d['ids']), ('source_indices', d['rows']), ('row_fold', d['folds'])]:
            np.testing.assert_array_equal(z[key], value)
        assert z['arms'].tolist() == E.ARMS and z['seeds'].tolist() == [42]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    np.testing.assert_array_equal(p[0], d['baseline'])
    np.testing.assert_array_equal(t[0], d['baseline_thresholds'])
    partition_proof, models, calibration = [], {}, {}
    max_calibration_replay = 0.
    for fold in range(5):
        proper, cal, held = C.split(d['ids'], d['folds'], fold)
        sets = [set(d['ids'][v]) for v in (proper, cal, held)]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        one, full = C.reference_predictors(42, fold, proper, cal, held)
        products, hurwitz = [load_predictor(k, fold, proper, cal, held) for k in ('products', 'hurwitz')]
        models[fold] = (one, full, products, hurwitz)
        q = d['train'][:, cal].reshape(-1, 174)
        op, fp = one(q[:, :107]), full(q)
        gp, hp = products(E.augmented(q, 'products')), hurwitz(E.augmented(q, 'hurwitz'))
        cps = np.array([.5*(op+fp), fp, gp, .5*(op+gp), hp, .5*(op+hp)])
        target, cids = np.tile(d['y'][cal], 4), np.tile(d['ids'][cal], 4)
        with np.load(OUT / f'fold{fold}.npz') as z:
            np.testing.assert_array_equal(z['held_rows'], held)
            np.testing.assert_array_equal(z['proper_rows'], proper)
            np.testing.assert_array_equal(z['calibration_rows'], cal)
            recorded = z['calibration_points'].reshape(len(E.ARMS), -1)
            np.testing.assert_allclose(cps, recorded, atol=1e-12, rtol=1e-12)
            max_calibration_replay = max(max_calibration_replay, float(abs(cps-recorded).max()))
            calibration[fold] = json.loads(str(z['calibration']))
        observed = []
        for ai, cp in enumerate(cps):
            radius = C.calibration(cp, target, cids)
            assert radius == calibration[fold][ai]
            scores = C.distribution(cp, radius)[:, 3]
            levels = [C.threshold(scores, target, b) for b in C.BUDGETS]
            np.testing.assert_array_equal(levels, t[ai, fold])
            rates = [float(np.mean(-scores[target > 0] <= level)) for level in levels]
            assert np.all(np.array(rates) <= C.BUDGETS)
            observed.append(rates)
        active_counts = {}
        from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
        for kind in ('products', 'hurwitz'):
            _, _, active = fit_ridge_scaling(E.augmented(d['train'][:, proper], kind).reshape(-1, 195))
            active_counts[kind] = int(active.sum())
        partition_proof.append(dict(fold=fold, proper_designs=len(sets[0]), calibration_designs=len(sets[1]),
                                    held_designs=len(sets[2]), calibration_false_warnings=observed,
                                    training_supported_columns=active_counts))
    wanted = [3696, 5198, int(d['rows'][np.flatnonzero((d['y'] < 0) & (d['y'] > -.01))[0]]),
              int(d['rows'][np.flatnonzero(d['y'] >= .15)[0]]), int(d['rows'][len(d['rows'])//2])]
    gains = np.array(json.loads((E.ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    with np.load(C.OUT / 'features.npz') as z:
        raw_hashes = json.loads(str(z['raw_hashes']))
    replay, max_features, max_predictions = [], 0., 0.
    B = C.helpers.base
    ridge = B.PhysicsFeatureRidge()
    for source in dict.fromkeys(wanted):
        match = np.flatnonzero(d['rows'] == source)
        assert len(match) == 1
        k = int(match[0])
        fold = int(d['folds'][k])
        one, full, products, hurwitz = models[fold]
        did = str(d['ids'][k])
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
            for kind in ('products', 'hurwitz'):
                np.testing.assert_allclose(E.additions(q, kind)[0], E.additions(d['evaluation'][ci, k:k+1], kind)[0], atol=1e-9, rtol=1e-9)
            start = time.perf_counter()
            op, fp = one(q[:, :107]), full(q)
            gp, hp = products(E.augmented(q, 'products')), hurwitz(E.augmented(q, 'hurwitz'))
            points = [.5*(op+fp), fp, gp, .5*(op+gp), hp, .5*(op+hp)]
            actual = np.array([C.distribution(v, calibration[fold][ai])[0] for ai, v in enumerate(points)])
            seconds = time.perf_counter() - start
            error = float(abs(actual-p[:, ci, k]).max())
            np.testing.assert_allclose(actual, p[:, ci, k], atol=1e-8, rtol=1e-8)
            max_features, max_predictions = max(max_features, feature_error), max(max_predictions, error)
            replay.append(dict(source_index=source, design=did, fold=fold, condition=name,
                               max_feature_error=feature_error, max_prediction_error=error,
                               cached_feature_all_arms_inference_seconds=seconds))
            print('COORDINATE_WAVEFORM_REPLAY', source, name, feature_error, error, flush=True)
    proof = dict(all_predictions_finite=True, all_design_partitions_disjoint=True,
                 frozen_baseline_replay=True, all_calibration_budgets_respected=True,
                 new_models=10, numeric=E.numerical_check(), partitions=partition_proof,
                 real_waveform_replays=len(replay), replays=replay, max_feature_error=max_features,
                 max_prediction_error=max_predictions, max_calibration_replay=max_calibration_replay,
                 hash=C.sha256(OUT / 'predictions.npz'),
                 limitation='One initialization; repeated TRAIN exploration; fixed-prediction bootstrap. Timings exclude modal extraction.')
    C.save_json(OUT / 'verification.json', proof)
    print('COORDINATE_VERIFICATION_COMPLETE', len(replay), 'actual waveform replays', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
