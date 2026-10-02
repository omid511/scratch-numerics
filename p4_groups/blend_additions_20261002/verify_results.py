"""Verify assembly, isolation and real waveform-to-integrated-output inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as E
C, np, OUT = E.C, E.np, E.OUT


def load_model(path, expected_rows):
    with np.load(path) as z:
        for key, expected in expected_rows.items():
            np.testing.assert_array_equal(z[key], expected)
        meta = json.loads(str(z['metadata']))
        assert meta['success']
        return E.cloudpickle.loads(z['predictor'].tobytes())


def query_models(seed, fold, d):
    proper, cal, held = C.split(d['ids'], d['folds'], fold)
    one, full = C.reference_predictors(seed, fold, proper, cal, held)
    models = {}
    for addon in E.ADDONS[1:4]:
        models[addon] = load_model(OUT / f'outer_{addon}_seed{seed}_fold{fold}.npz',
                                  {'fit_rows': proper, 'query_rows': np.concatenate([cal, held])})
    for addon in E.ADDONS[4:]:
        old_arm = 'blend_global' if addon == 'global_observability' else 'blend_local'
        models[addon] = [load_model(E.LOCAL / f'{old_arm}_{width}_seed{seed}_fold{fold}.npz',
                         {'proper_rows': proper, 'calibration_rows': cal, 'held_rows': held}) for width in (107, 174)]
    models['neighborhood'] = [E.N.DesignNeighborhood(d['train'][:, proper, :width], d['y'][proper], d['ids'][proper], d['rows'][proper])
                              for width in (107, 174)]
    with np.load(OUT / f'fusion_seed{seed}_fold{fold}.npz') as z:
        assert json.loads(str(z['fingerprint'])) == d['fingerprint']
        heads = E.cloudpickle.loads(z['heads'].tobytes())
        radii = json.loads(str(z['calibration']))
    return one, full, models, heads, radii


def predict_one(arm, q, global_features, local_features, fitted):
    one, full, models, heads, radii = fitted
    members = np.column_stack([one(q[:, :107]), full(q)])
    control = E.control_features(q, members)
    base = control[:, 0]
    if arm == 'blend_frozen':
        point = base
    elif arm in ('blend_affine', 'blend_control'):
        width = 1 if arm == 'blend_affine' else 7
        point = base + E.head_predict(heads[arm], control[:, :width])
    else:
        route, addon = arm.split('_', 1)
        if addon == 'neighborhood':
            added = .5 * sum(model.predict(q[:, :width])[0] for width, model in zip((107, 174), models[addon]))
        elif addon in E.ADDONS[1:4]:
            added = models[addon](q)
        else:
            parts = [global_features] + ([local_features] if addon == 'local_observability' else [])
            added = .5 * sum(model(np.column_stack([q[:, :width], *parts]))
                             for width, model in zip((107, 174), models[addon]))
        point = .5 * (base + added) if route == 'fixed' else base + E.head_predict(heads[arm], np.column_stack([control, added - base]))
    return C.distribution(point, radii[arm])


def verify(smoke=False):
    d = E.setup()
    prefix = 'smoke_' if smoke else ''
    path = OUT / f'{prefix}predictions.npz'
    with np.load(path) as z:
        p, t, y, ids, rows, folds = [z[k] for k in ('predictions', 'thresholds', 'y', 'ids', 'source_indices', 'row_fold')]
        names, seeds, arms = z['conditions'].tolist(), z['seeds'].tolist(), z['arms'].tolist()
        assert json.loads(str(z['fingerprint'])) == d['fingerprint']
    selected = np.flatnonzero(d['folds'] == 0) if smoke else np.arange(len(d['y']))
    for key, actual in [('y', y), ('ids', ids), ('rows', rows), ('folds', folds)]:
        np.testing.assert_array_equal(actual, d[key][selected])
    assert names == d['names'] and arms == E.ARMS
    assert set(ids).issubset(set(d['ids']))
    if not smoke:
        assert len(np.unique(ids)) == 77 and len(y) == 4454 and seeds == C.SEEDS
    np.testing.assert_allclose(p[0], d['baseline'][:len(seeds)][:, :, selected], atol=1e-12, rtol=1e-12)
    budgets = []
    for seed in seeds:
        si = seeds.index(seed)
        for fold in (range(1) if smoke else range(5)):
            proper, cal, held = C.split(d['ids'], d['folds'], fold)
            groups = [set(d['ids'][idx]) for idx in (proper, cal, held)]
            assert all(not groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
            with np.load(OUT / f'fusion_seed{seed}_fold{fold}.npz') as z:
                for key, expected in zip(('proper_rows', 'calibration_rows', 'held_rows'), (proper, cal, held)):
                    np.testing.assert_array_equal(z[key], expected)
                np.testing.assert_array_equal(p[:, si][:, :, np.searchsorted(selected, held)], z['predictions'])
                np.testing.assert_array_equal(t[:, si, fold], z['thresholds'])
                rates = json.loads(str(z['calibration_false_warning']))
                assert all(np.all(np.array(v) <= C.BUDGETS) for v in rates.values())
                budgets.append(dict(seed=seed, fold=fold, rates=rates))
    assert np.isfinite(p).all() and np.all((p[..., 3] >= 0) & (p[..., 3] <= 1))
    assert np.all(p[..., 1] <= p[..., 0]) and np.all(p[..., 0] <= p[..., 2])
    assert np.all(p[..., 4] <= p[..., 0]) and np.all(p[..., 0] <= p[..., 5])
    spec = importlib.util.spec_from_file_location('integration_waveform_replay', E.LOCAL / 'prepare_features.py')
    waveform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(waveform)
    # Reuse metadata only for acquisition dt; no solver modes/speeds/margins enter features.
    with np.load(E.ROOT / 'p4_dataset_saturation/metadata_arrays.npz') as z:
        dt = z['dts'][d['rows']]
    with np.load(C.helpers.base.PRIOR / 'features.npz') as z:
        layout, pairs = z['sensor_indices'], z['sensor_pairs']
    gains = np.array(json.loads((E.ROOT / 'p4_groups/physical_mismatch_20260929/protocol.json').read_text())['gains'])
    candidate_rows = [int(rows[0])] if smoke else list(dict.fromkeys([3696, 5198, int(rows[0])]))
    cases, timings, max_feature, max_output = [], {}, 0., 0.
    fit_cache = {}
    for source_index in candidate_rows:
        k = int(np.flatnonzero(d['rows'] == source_index)[0])
        did, fold = str(d['ids'][k]), int(d['folds'][k])
        with np.load(E.ROOT / 'p4_sensor_candidates' / f'{did}.npz') as z:
            source_rows = z['source_indices']
            j = int(np.flatnonzero(source_rows == source_index)[0])
            raw, sigma = z['raw'][j], z['noise_scale'][j]
        for name in ('iid1_new949', 'correlated5_new949_950'):
            ci = names.index(name)
            signal, reported_dt = waveform.measured_signal(raw, sigma, source_index, name, layout, gains, float(dt[k]), {}, {})
            start = time.perf_counter()
            modes = C.helpers.base.estimate_modes(signal, reported_dt, include_shapes=True)
            q = np.r_[C.helpers.base.base_feature_row(signal, reported_dt, modes, C.helpers.PhysicsFeatureRidge()),
                      C.helpers.base.modal_spatial_row(modes, pairs), C.helpers.base.modal_spatial_addition_row(modes, pairs)][None]
            gc, lc = C.feature_row(signal, reported_dt, q[0, [28, 31]])
            extraction_ms = 1000 * (time.perf_counter() - start)
            np.testing.assert_allclose(q[0], d['evaluation'][ci, k], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(gc, d['ec'][ci, k], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(lc, d['el'][ci, k], atol=1e-10, rtol=1e-10)
            max_feature = max(max_feature, float(abs(q[0] - d['evaluation'][ci, k]).max()),
                              float(abs(gc - d['ec'][ci, k]).max()), float(abs(lc - d['el'][ci, k]).max()))
            output = {}
            for si, seed in enumerate(seeds):
                key = (seed, fold)
                if key not in fit_cache:
                    fit_cache[key] = query_models(seed, fold, d)
                for ai, arm in enumerate(arms):
                    actual = predict_one(arm, q, gc[None], lc[None], fit_cache[key])[0]
                    expected = p[ai, si, ci, int(np.flatnonzero(rows == source_index)[0])]
                    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
                    max_output = max(max_output, float(abs(actual - expected).max()))
                    output[f'{seed}:{arm}'] = actual.tolist()
                    if seed == seeds[0] and source_index == candidate_rows[0] and name == 'iid1_new949':
                        elapsed = []
                        for _ in range(30):
                            start = time.perf_counter_ns()
                            predict_one(arm, q, gc[None], lc[None], fit_cache[key])
                            elapsed.append((time.perf_counter_ns() - start) / 1e6)
                        timings[arm] = dict(median_ms=float(np.median(elapsed)), p95_ms=float(np.quantile(elapsed, .95)), samples=len(elapsed))
            cases.append(dict(source_index=source_index, design=did, fold=fold, condition=name,
                              waveform_feature_extraction_ms=extraction_ms, outputs=output))
    result = dict(complete_coverage=not smoke, physical_clips=len(y), designs=len(np.unique(ids)), conditions=len(names),
                  seeds=seeds, arms=arms, frozen_distributions_and_thresholds_replayed=True,
                  full_fold_assembly_replayed=True, proper_calibration_held_designs_disjoint=True,
                  all_calibration_budgets_respected=True, waveform_cases=cases,
                  waveform_feature_max_delta=max_feature, single_query_distribution_max_delta=max_output,
                  cached_feature_inference=timings,
                  timing_boundary='Actual174-feature query and (for observability)24 descriptor values available. Includes both frozen members, added predictor(s), combiner and calibrated distribution; excludes normalization/modal/spatial/local feature extraction. Not full sensor-to-output latency.',
                  hashes={f: E.sha256(OUT / f) for f in ('protocol.json', 'run_experiment.py', 'verify_results.py', prefix + 'predictions.npz')})
    E.save_json(OUT / f'{prefix}verification.json', result)
    print('WAVEFORM_FUSION_VERIFICATION_PASSED', len(y), len(names), len(arms), 'arms;', len(cases),
          'actual waveform cases; feature delta', max_feature, 'distribution delta', max_output, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    with E.threadpool_limits(limits=1):
        verify(args.smoke)
