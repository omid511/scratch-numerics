"""TRAIN-only cross-fitted residual integration of physical readouts into the blend."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'p4_groups/physical_joint_inference_20261001'
sys.path.insert(0, str(PRIOR))
import experiment as physical
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
fit_network = physical.fit_network
save_json, save_npz, sha256, cloudpickle = physical.save_json, physical.save_npz, physical.sha256, physical.cloudpickle
SEEDS, BUDGETS = physical.SEEDS, physical.BUDGETS
ARMS = ['blend_frozen', 'blend_adaptive', 'blend_affine', 'blend_control',
        'blend_joint', 'blend_soft', 'blend_both', 'blend_sham']
FIELDS = ['point', 'raw_lower', 'raw_upper', 'p_unsafe', 'group_lower', 'group_upper']
LAMBDA = .1


def signature(raw_hashes):
    sources = [Path(__file__), OUT / 'protocol.json', PRIOR / 'experiment.py', PRIOR / 'predictions.npz',
               PRIOR / 'fits.json', ROOT / 'experiment_p4_tabular.py',
               ROOT / 'src/mechanics/p4_margin_estimation/baselines.py']
    return {'physical_sources': physical.source_signature(raw_hashes),
            'integration_sources': {str(p.relative_to(ROOT)): sha256(p) for p in sources}}


def head_fit(x, y):
    mean, scale, active = fit_ridge_scaling(x)
    z = (x - mean) / scale
    z[:, ~active] = 0
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z + LAMBDA * len(y) * np.eye(x.shape[1]), z.T @ (y - intercept))
    return {'mean': mean, 'scale': scale, 'active': active, 'coef': coefficients, 'intercept': intercept}


def head_predict(model, x):
    z = (x - model['mean']) / model['scale']
    z[:, ~model['active']] = 0
    return z @ model['coef'] + model['intercept']


def feature_blocks(q, components, views, permutation):
    one, full, blend = components[:, 0], components[:, 1], components[:, 2]
    post = components[:, 3:]
    control = np.column_stack([blend, one - full, q[:, 27], q[:, 28], q[:, 20], q[:, 19], q[:, 29]])
    joint = np.column_stack([post[:, 0] - blend, post[:, 1] - post[:, 0], post[:, 2] - post[:, 0], post[:, 3]])
    soft = np.column_stack([post[:, 4] - q[:, 27], post[:, 5]])
    both = np.column_stack([joint, soft])
    shuffled = both.reshape(views, len(permutation), 6)[:, permutation].reshape(-1, 6)
    return {'blend_frozen': control, 'blend_adaptive': control, 'blend_affine': blend[:, None],
            'blend_control': control, 'blend_joint': np.column_stack([control, joint]),
            'blend_soft': np.column_stack([control, soft]), 'blend_both': np.column_stack([control, both]),
            'blend_sham': np.column_stack([control, shuffled])}


def fitted_components(path, train, truth, ids, fit, query, seed, fingerprint):
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale inner component cache'
            np.testing.assert_array_equal(z['fit_rows'], fit)
            np.testing.assert_array_equal(z['query_rows'], query)
            return z['predictions'], json.loads(str(z['metadata']))
    assert not set(ids[fit]) & set(ids[query]), 'inner design leakage'
    x, target = train[:, fit].reshape(-1, 174), np.tile(truth[fit, 2], 4)
    q = train[:, query].reshape(-1, 174)
    one, meta_one = fit_network(x[:, :107], target, seed=seed)
    full, meta_full = fit_network(x, target, seed=seed)
    if not meta_one['success'] or not meta_full['success']:
        raise RuntimeError(f'nonconverged inner base fit: {path}: {meta_one}, {meta_full}')
    prior = physical.PhysicalPrior()
    meta_physical = prior.fit(truth[fit], train[:, fit][:, :, [27, 28]], seed)
    a, b = one(q[:, :107]), full(q)
    predictions = np.column_stack([a, b, .5 * (a + b), prior.predict(q[:, [27, 28]], 'joint_soft')])
    metadata = {'one': meta_one, 'full': meta_full, 'physical': meta_physical,
                'fit_designs': np.unique(ids[fit]).tolist(), 'query_designs': np.unique(ids[query]).tolist()}
    save_npz(path, predictions=predictions.reshape(4, len(query), 9),
             components=np.frombuffer(cloudpickle.dumps({'one': one, 'full': full, 'physical': prior}), dtype=np.uint8),
             metadata=json.dumps(metadata), fingerprint=json.dumps(fingerprint), fit_rows=fit, query_rows=query)
    print('Completed crossfit components', path.name, 'designs', len(metadata['fit_designs']), len(metadata['query_designs']), flush=True)
    return predictions.reshape(4, len(query), 9), metadata


def crossfit(train, truth, ids, proper, fold, seed, fingerprint):
    designs = np.unique(ids[proper]).copy()
    np.random.default_rng(1047 + fold).shuffle(designs)
    result = np.full((4, len(proper), 9), np.nan)
    records = []
    for inner, group in enumerate(np.array_split(designs, 3)):
        local_query = np.flatnonzero(np.isin(ids[proper], group))
        local_fit = np.flatnonzero(~np.isin(ids[proper], group))
        fit, query = proper[local_fit], proper[local_query]
        path = OUT / f'inner_seed{seed}_fold{fold}_part{inner}.npz'
        predictions, metadata = fitted_components(path, train, truth, ids, fit, query, seed, fingerprint)
        result[:, local_query] = predictions
        records.append({'fold': fold, 'seed': seed, 'inner': inner, 'hash': sha256(path), **metadata})
    assert np.isfinite(result).all(), 'incomplete proper-design crossfit table'
    return result, records


def outer_components(train, evaluation, proper, cal, test, seed, fold, reference_post):
    one, full = physical.reference_predictors(seed, fold, proper, cal, test)
    path = PRIOR / f'physical_seed{seed}_fold{fold}.npz'
    previous = json.loads((PRIOR / 'fits.json').read_text())
    recorded = next(f['model_hash'] for f in previous['fits'] if f.get('seed') == seed and f.get('fold') == fold and 'kind' not in f)
    assert sha256(path) == recorded, 'frozen physical model changed'
    with np.load(path) as z:
        for key, rows in [('proper_rows', proper), ('calibration_rows', cal), ('held_rows', test)]:
            np.testing.assert_array_equal(z[key], rows)
        prior = cloudpickle.loads(z['model'].tobytes())
    qcal = train[:, cal].reshape(-1, 174)
    qtest = evaluation[:, test].reshape(-1, 174)
    acal, bcal = one(qcal[:, :107]), full(qcal)
    atest, btest = one(qtest[:, :107]), full(qtest)
    cp = np.column_stack([acal, bcal, .5 * (acal + bcal), prior.predict(qcal[:, [27, 28]], 'joint_soft')])
    hp = np.column_stack([atest, btest, .5 * (atest + btest), reference_post.reshape(-1, 6)])
    return cp, hp


def train_heads(x, y, components, arm):
    baseline = components[:, 2]
    residual = None if arm in ('blend_frozen', 'blend_adaptive') else head_fit(x, y - baseline)
    point = baseline if residual is None else baseline + head_predict(residual, x)
    scale = None if arm == 'blend_frozen' else head_fit(x, np.log(np.maximum(abs(y - point), 1e-8)))
    return {'residual': residual, 'scale': scale}


def point_scale(heads, x, components):
    point = components[:, 2].copy()
    if heads['residual'] is not None:
        point += head_predict(heads['residual'], x)
    scale = np.ones(len(point)) if heads['scale'] is None else np.exp(head_predict(heads['scale'], x))
    if not np.isfinite(point).all() or not np.isfinite(scale).all() or not np.all(scale > 0):
        raise FloatingPointError('nonfinite integrated point or uncertainty scale')
    return point, scale


def calibration(point, scale, y, ids):
    residual = abs(y - point) / scale
    raw = float(np.quantile(residual, .9))
    maxima = np.sort([residual[ids == did].max() for did in np.unique(ids)])
    rank = int(np.ceil((len(maxima) + 1) * .9))
    if rank > len(maxima):
        raise ValueError('too few calibration designs for finite90% group calibration')
    group = float(maxima[rank - 1])
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms <= 0:
        raise ValueError('zero calibration RMS cannot define Gaussian probability comparator')
    return {'raw': raw, 'group': group, 'rms': rms}


def distribution(point, scale, levels):
    raw, group = levels['raw'] * scale, levels['group'] * scale
    return np.column_stack([point, point - raw, point + raw, ndtr(-point / (levels['rms'] * scale)),
                            point - group, point + group])


def run(smoke=False):
    train, evaluation, truth, ids, rows, names, folds, dt, clamps, raw_hashes = physical.load_data()
    with np.load(PRIOR / 'predictions.npz') as z:
        for key, value in [('y', truth[:, 2]), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names
        old = z['predictions']
    fingerprint = signature(raw_hashes)
    seeds = SEEDS[:1] if smoke else SEEDS
    output = np.full((len(ARMS), len(seeds), len(names), len(rows), len(FIELDS)), np.nan)
    thresholds = np.full((len(ARMS), len(seeds), 5, len(BUDGETS)), np.nan)
    all_records, all_splits = [], []
    for fold in (range(1) if smoke else range(5)):
        proper, cal, test = physical.split(ids, folds, fold)
        all_splits.append({'fold': fold, 'proper_rows': proper.tolist(), 'calibration_rows': cal.tolist(), 'held_rows': test.tolist(),
                           'proper_designs': np.unique(ids[proper]).tolist(), 'calibration_designs': np.unique(ids[cal]).tolist(),
                           'held_designs': np.unique(ids[test]).tolist()})
        permutations = [np.random.default_rng(2047 + 3 * fold + i).permutation(len(part)) for i, part in enumerate((proper, cal, test))]
        for si, seed in enumerate(seeds):
            source_si = SEEDS.index(seed)
            proper_components, records = crossfit(train, truth, ids, proper, fold, seed, fingerprint)
            all_records.extend(records)
            cp, hp = outer_components(train, evaluation, proper, cal, test, seed, fold, old[2, source_si][:, test])
            xproper = train[:, proper].reshape(-1, 174)
            xcal, xtest = train[:, cal].reshape(-1, 174), evaluation[:, test].reshape(-1, 174)
            component_table = proper_components.reshape(-1, 9)
            features = feature_blocks(xproper, component_table, 4, permutations[0])
            calib_features = feature_blocks(xcal, cp, 4, permutations[1])
            held_features = feature_blocks(xtest, hp, len(names), permutations[2])
            target, ycal = np.tile(truth[proper, 2], 4), np.tile(truth[cal, 2], 4)
            calids = np.tile(ids[cal], 4)
            for ai, arm in enumerate(ARMS):
                heads = train_heads(features[arm], target, component_table, arm)
                pc, sc = point_scale(heads, calib_features[arm], cp)
                ph, sh = point_scale(heads, held_features[arm], hp)
                levels = calibration(pc, sc, ycal, calids)
                cdist, hdist = distribution(pc, sc, levels), distribution(ph, sh, levels)
                output[ai, si][:, test] = hdist.reshape(len(names), len(test), len(FIELDS))
                thresholds[ai, si, fold] = [physical.warning_threshold(cdist[:, 3], ycal, budget) for budget in BUDGETS]
                if arm in ('blend_frozen', 'blend_adaptive'):
                    np.testing.assert_array_equal(hdist[:, 0], old[5, source_si][:, test, 0].reshape(-1))
                if arm == 'blend_frozen':
                    np.testing.assert_allclose(hdist[:, :4], old[5, source_si][:, test, :4].reshape(-1, 4), rtol=1e-12, atol=1e-12)
                path = OUT / f'heads_{arm}_seed{seed}_fold{fold}.npz'
                save_npz(path, heads=np.frombuffer(cloudpickle.dumps(heads), dtype=np.uint8),
                         calibration=json.dumps(levels), thresholds=thresholds[ai, si, fold],
                         fingerprint=json.dumps(fingerprint), proper_rows=proper, calibration_rows=cal, held_rows=test)
                all_records.append({'kind': 'heads', 'arm': arm, 'fold': fold, 'seed': seed,
                                    'path': path.name, 'hash': sha256(path), 'calibration': levels,
                                    'residual_coefficients': None if heads['residual'] is None else heads['residual']['coef'].tolist(),
                                    'scale_coefficients': None if heads['scale'] is None else heads['scale']['coef'].tolist()})
            print('Completed all integrated arms', 'outer', fold, 'seed', seed, flush=True)
    selected = np.flatnonzero(folds == 0) if smoke else np.arange(len(rows))
    assert np.isfinite(output[:, :, :, selected]).all()
    assert signature(raw_hashes) == fingerprint, 'source or reference changed during integration'
    target_path = OUT / ('smoke_predictions.npz' if smoke else 'predictions.npz')
    save_npz(target_path, predictions=output[:, :, :, selected], thresholds=thresholds,
             y=truth[selected, 2], ids=ids[selected], source_indices=rows[selected], row_fold=folds[selected],
             conditions=np.array(names), arms=np.array(ARMS), seeds=np.array(seeds), fields=np.array(FIELDS),
             budgets=np.array(BUDGETS), fingerprint=json.dumps(fingerprint))
    save_json(OUT / ('smoke_fits.json' if smoke else 'fits.json'),
              {'splits': all_splits, 'fits': all_records, 'fingerprint': fingerprint, 'partial_smoke': smoke})
    print('SMOKE_PASSED' if smoke else 'INTEGRATION_COMPLETE', output[:, :, :, selected].shape, 'frozen blend exact replay passed', flush=True)


def numerical_smoke():
    x = np.column_stack([np.arange(12.), np.ones(12)])
    y = 2 * x[:, 0] - 3
    head = head_fit(x, y)
    np.testing.assert_allclose(head_predict(head, x).mean(), y.mean(), atol=1e-12)
    assert not head['active'][1]
    scaled = point_scale({'residual': None, 'scale': None}, x, np.column_stack([y, y, y]))
    np.testing.assert_array_equal(scaled[0], y)
    ids = np.repeat(np.arange(13), 2)
    target = np.linspace(-.1, .1, 26)
    levels = calibration(target + .01, np.ones(26), target, ids)
    dist = distribution(target, np.ones(26), levels)
    np.testing.assert_allclose(dist[:, 2] - dist[:, 1], .02, atol=1e-12)
    assert np.all(dist[:, 4] <= dist[:, 0]) and np.all(dist[:, 0] <= dist[:, 5])
    print('Analytical ridge, no-correction identity and calibrated interval smoke passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.smoke:
            numerical_smoke()
        run(args.smoke)
