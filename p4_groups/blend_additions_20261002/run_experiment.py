"""Independent, calibration-isolated additions to the unchanged P4 spatial blend."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LOCAL = ROOT / 'p4_groups/local_observability_20261001'
OLD = ROOT / 'p4_groups/blend_physical_integration_20261001'
sys.path.insert(0, str(LOCAL))
import contract as C
sys.path.insert(0, str(ROOT / 'p4_groups/neighborhood_reference_20261002'))
import neighborhood_reference as N
np, cloudpickle = C.np, C.cloudpickle
save_json, save_npz, sha256 = C.save_json, C.save_npz, C.sha256
from mechanics.p4_margin_estimation.baselines import fit_ridge_scaling
from threadpoolctl import threadpool_limits
ADDONS = ['neighborhood', 'supervised', 'global_consistency', 'localized_consistency', 'global_observability', 'local_observability']
ARMS = ['blend_frozen', 'blend_affine', 'blend_control'] + [f'{route}_{a}' for a in ADDONS for route in ('fixed', 'learned')]
LAMBDA = .1
DATA = None


def head_fit(x, target):
    mean, scale, active = fit_ridge_scaling(x)
    z = (x - mean) / scale
    z[:, ~active] = 0
    intercept = float(target.mean())
    coef = np.linalg.solve(z.T @ z + LAMBDA * len(target) * np.eye(x.shape[1]), z.T @ (target - intercept))
    return dict(mean=mean, scale=scale, active=active, coef=coef, intercept=intercept)


def head_predict(head, x):
    z = (x - head['mean']) / head['scale']
    z[:, ~head['active']] = 0
    return z @ head['coef'] + head['intercept']


def control_features(q, members):
    one, full = members[:, 0], members[:, 1]
    return np.column_stack([.5 * (one + full), one - full, q[:, 27], q[:, 28], q[:, 20], q[:, 19], q[:, 29]])


def descriptors(base, global_features, local_features, rows, width, local):
    parts = [base[:, rows, :width], global_features[:, rows]]
    if local:
        parts.append(local_features[:, rows])
    x = np.concatenate(parts, axis=2)
    return x.reshape(-1, x.shape[-1])


def setup():
    global DATA
    data = C.load_data()
    train, evaluation, y, ids, rows, names, folds, *_ = data
    source = C.signature()
    with np.load(LOCAL / 'features.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names
        assert json.loads(str(z['fingerprint'])) == source
        tc, tl, ec, el = [z[k] for k in ('training_control', 'training_local', 'control', 'local')]
        raw = json.loads(str(z['raw_hashes']))
    for did, digest in raw.items():
        assert sha256(ROOT / 'p4_sensor_candidates' / f'{did}.npz') == digest
    with np.load(LOCAL / 'predictions.npz') as z:
        for key, value in [('y', y), ('ids', ids), ('source_indices', rows), ('row_fold', folds)]:
            np.testing.assert_array_equal(z[key], value)
        assert z['conditions'].tolist() == names and z['seeds'].tolist() == C.SEEDS
        baseline = z['predictions'][0].copy()
        baseline_thresholds = z['thresholds'][0].copy()
    old_local_fingerprint = {**source, 'feature_table': sha256(LOCAL / 'features.npz'),
                             'frozen_point_record': sha256(ROOT / 'p4_groups/physical_joint_inference_20261001/predictions.npz')}
    paths = [OUT / 'protocol.json', Path(__file__), LOCAL / 'features.npz', LOCAL / 'predictions.npz',
             Path(N.__file__), N.OUT / 'protocol.json']
    fingerprint = {'sources': {str(p.relative_to(ROOT)): sha256(p) for p in paths}, 'frozen': source}
    # These source checks cover all historical inner model shards, not their unused oracle readout.
    with np.load(OLD / 'inner_seed42_fold0_part0.npz') as z:
        old_fingerprint = json.loads(str(z['fingerprint']))
    for sources in (old_fingerprint['physical_sources']['sources'], old_fingerprint['integration_sources']):
        for path, digest in sources.items():
            assert sha256(ROOT / path) == digest, f'changed old crossfit source: {path}'
    assert old_fingerprint['physical_sources']['raw'] == raw
    assert old_fingerprint['physical_sources']['versions'] == source['versions']
    for name, digest in old_fingerprint['physical_sources']['references'].items():
        assert sha256(C.helpers.BASE / name) == digest
    DATA = dict(train=train, evaluation=evaluation, y=y, ids=ids, rows=rows, names=names, folds=folds,
                tc=tc, tl=tl, ec=ec, el=el, baseline=baseline, baseline_thresholds=baseline_thresholds,
                fingerprint=fingerprint, old_fingerprint=old_fingerprint, old_local_fingerprint=old_local_fingerprint,
                neighborhood_fingerprint=N.signature())
    for value in DATA.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return DATA


def new_fit(path, x, target, seed, fit, query, *, arm, pairs=None, weights=None):
    d = DATA
    fp = {'experiment': d['fingerprint'], 'arm': arm, 'seed': seed}
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == fp, f'stale new fit: {path}'
            np.testing.assert_array_equal(z['fit_rows'], fit)
            np.testing.assert_array_equal(z['query_rows'], query)
            predict = cloudpickle.loads(z['predictor'].tobytes())
            meta = json.loads(str(z['metadata']))
    else:
        custom = arm in ('supervised', 'global_consistency', 'localized_consistency')
        weight = 1. if arm in ('global_consistency', 'localized_consistency') else 0.
        predict, meta = C.fit_network(x, target, seed=seed, alpha=1., custom=custom,
            max_iter=20000, max_fun=400000, consistency_pairs=pairs, consistency_weight=weight,
            consistency_pair_weights=weights)
        save_npz(path, predictor=np.frombuffer(cloudpickle.dumps(predict), dtype=np.uint8), metadata=json.dumps(meta),
                 fingerprint=json.dumps(fp), fit_rows=fit, query_rows=query)
        print('ADDITION_FIT', path.name, json.dumps(meta), flush=True)
    if not meta['success']:
        raise RuntimeError(f'nonconverged new fit: {path}: {meta}')
    return predict, {'path': str(path.relative_to(ROOT)), 'hash': sha256(path), **meta}


def old_inner(seed, fold, part, fit, query):
    path = OLD / f'inner_seed{seed}_fold{fold}_part{part}.npz'
    with np.load(path) as z:
        assert json.loads(str(z['fingerprint'])) == DATA['old_fingerprint']
        np.testing.assert_array_equal(z['fit_rows'], fit)
        np.testing.assert_array_equal(z['query_rows'], query)
        meta = json.loads(str(z['metadata']))
        assert meta['one']['success'] and meta['full']['success']
        assert meta['fit_designs'] == np.unique(DATA['ids'][fit]).tolist()
        assert meta['query_designs'] == np.unique(DATA['ids'][query]).tolist()
        saved = z['predictions'][:, :, :2].copy()
        # Only one/full are used. No physical/oracle estimate enters an integration head.
        models = cloudpickle.loads(z['components'].tobytes())
    q = DATA['train'][:, query].reshape(-1, 174)
    actual = np.column_stack([models['one'](q[:, :107]), models['full'](q)]).reshape(4, len(query), 2)
    np.testing.assert_allclose(actual, saved, atol=1e-12, rtol=1e-12)
    return saved, dict(path=str(path.relative_to(ROOT)), hash=sha256(path), reused_base=True,
                       fit_designs=meta['fit_designs'], query_designs=meta['query_designs'])


def added_points(seed, fold, label, fit, query, base, gc, lc, *, outer=False):
    d = DATA
    prefix = OUT / f'{label}_seed{seed}_fold{fold}'
    cache = prefix.with_name(prefix.name + '_points.npz')
    if cache.exists():
        with np.load(cache) as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
            np.testing.assert_array_equal(z['fit_rows'], fit)
            np.testing.assert_array_equal(z['query_rows'], query)
            return z['points'], json.loads(str(z['records']))
    points, records = [], []
    target = np.tile(d['y'][fit], 4)
    q = base[:, query].reshape(-1, 174)
    if outer:
        with np.load(N.OUT / f'fold{fold}.npz') as z:
            assert json.loads(str(z['fingerprint'])) == d['neighborhood_fingerprint']
            for key, expected in [('proper_rows', fit), ('calibration_rows', C.split(d['ids'], d['folds'], fold)[1]),
                                  ('held_rows', C.split(d['ids'], d['folds'], fold)[2])]:
                np.testing.assert_array_equal(z[key], expected)
            hp = z['predictions'][:, :, 0].reshape(-1)
            cp = z['calibration_points']
            neighbors = z['neighbors']
        N.verify_neighbors(neighbors, fit, d['ids'], d['rows'])
        points.append(cp if label == 'cal' else hp)
    else:
        members = []
        for width in (107, 174):
            model = N.DesignNeighborhood(d['train'][:, fit, :width], d['y'][fit], d['ids'][fit], d['rows'][fit])
            p, refs, _ = model.predict(q[:, :width])
            N.verify_neighbors(refs, fit, d['ids'], d['rows'])
            members.append(p)
        points.append(.5 * (members[0] + members[1]))
    for arm in ADDONS[1:4]:
        n = len(fit)
        pairs = np.column_stack([n + np.arange(n), 3 * n + np.arange(n)]) if arm != 'supervised' else None
        weights = (abs(d['y'][fit]) < .05).astype(float) if arm == 'localized_consistency' else None
        if weights is not None:
            assert weights.sum() > 0
        model_label = 'outer' if outer else label
        path = OUT / f'{model_label}_{arm}_seed{seed}_fold{fold}.npz'
        # Outer models have a single reusable fit/query identity (cal+held), not one fit per acquisition table.
        model_query = np.concatenate(C.split(d['ids'], d['folds'], fold)[1:]) if outer else query
        predict, meta = new_fit(path, d['train'][:, fit].reshape(-1, 174), target, seed, fit, model_query,
                               arm=arm, pairs=pairs, weights=weights)
        points.append(predict(q))
        records.append(meta)
    for arm in ADDONS[4:]:
        local = arm == 'local_observability'
        old_arm = 'blend_local' if local else 'blend_global'
        members = []
        for width in (107, 174):
            xq = descriptors(base, gc, lc, query, width, local)
            if outer:
                path = LOCAL / f'{old_arm}_{width}_seed{seed}_fold{fold}.npz'
                with np.load(path) as z:
                    assert json.loads(str(z['fingerprint'])) == d['old_local_fingerprint']
                    for key, expected in zip(('proper_rows', 'calibration_rows', 'held_rows'), C.split(d['ids'], d['folds'], fold)):
                        np.testing.assert_array_equal(z[key], expected)
                    predict = cloudpickle.loads(z['predictor'].tobytes())
                    meta = json.loads(str(z['metadata']))
                assert meta['success']
                records.append(dict(path=str(path.relative_to(ROOT)), hash=sha256(path), reused_observability=True, **meta))
            else:
                path = OUT / f'{label}_{arm}_{width}_seed{seed}_fold{fold}.npz'
                predict, meta = new_fit(path, descriptors(d['train'], d['tc'], d['tl'], fit, width, local),
                                       target, seed, fit, query, arm=arm)
                records.append(meta)
            members.append(predict(xq))
        points.append(.5 * (members[0] + members[1]))
    points = np.column_stack(points).reshape(len(base), len(query), len(ADDONS))
    assert np.isfinite(points).all()
    save_npz(cache, points=points, fit_rows=fit, query_rows=query,
             records=json.dumps(records), fingerprint=json.dumps(d['fingerprint']))
    return points, records


def fold_job(seed, fold):
    d = DATA
    proper, cal, held = C.split(d['ids'], d['folds'], fold)
    result_path = OUT / f'fusion_seed{seed}_fold{fold}.npz'
    if result_path.exists():
        with np.load(result_path) as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
        return seed, fold
    ids = d['ids']
    designs = np.unique(ids[proper]).copy()
    np.random.default_rng(1047 + fold).shuffle(designs)
    base_oof = np.full((4, len(proper), 2), np.nan)
    addon_oof = np.full((4, len(proper), len(ADDONS)), np.nan)
    seen = np.zeros(len(proper), dtype=int)
    records = []
    for part, group in enumerate(np.array_split(designs, 3)):
        query = proper[np.isin(ids[proper], group)]
        fit = proper[~np.isin(ids[proper], group)]
        assert not set(ids[fit]) & set(ids[query])
        assert not set(ids[fit]) & set(ids[cal]) and not set(ids[fit]) & set(ids[held])
        slots = np.searchsorted(proper, query)
        members, meta = old_inner(seed, fold, part, fit, query)
        points, fits = added_points(seed, fold, f'inner{part}', fit, query, d['train'], d['tc'], d['tl'])
        base_oof[:, slots], addon_oof[:, slots] = members, points
        seen[slots] += 1
        records += [meta, *fits]
    assert np.all(seen == 1) and np.isfinite(base_oof).all() and np.isfinite(addon_oof).all()
    one, full = C.reference_predictors(seed, fold, proper, cal, held)
    qcal = d['train'][:, cal].reshape(-1, 174)
    qheld = d['evaluation'][:, held].reshape(-1, 174)
    cal_members = np.column_stack([one(qcal[:, :107]), full(qcal)])
    held_members = np.column_stack([one(qheld[:, :107]), full(qheld)])
    cal_addons, fits = added_points(seed, fold, 'cal', proper, cal, d['train'], d['tc'], d['tl'], outer=True)
    records += fits
    held_addons, fits = added_points(seed, fold, 'held', proper, held, d['evaluation'], d['ec'], d['el'], outer=True)
    records += fits
    tx = control_features(d['train'][:, proper].reshape(-1, 174), base_oof.reshape(-1, 2))
    cx, hx = control_features(qcal, cal_members), control_features(qheld, held_members)
    ta, ca, ha = [p.reshape(-1, len(ADDONS)) for p in (addon_oof, cal_addons, held_addons)]
    target, ctarget, cids = np.tile(d['y'][proper], 4), np.tile(d['y'][cal], 4), np.tile(ids[cal], 4)
    predictions, thresholds, heads, radii, false_warning = [], [], {}, {}, {}
    for arm in ARMS:
        head = None
        if arm == 'blend_frozen':
            cp, hp = cx[:, 0], hx[:, 0]
        elif arm in ('blend_affine', 'blend_control'):
            width = 1 if arm == 'blend_affine' else 7
            head = head_fit(tx[:, :width], target - tx[:, 0])
            cp, hp = cx[:, 0] + head_predict(head, cx[:, :width]), hx[:, 0] + head_predict(head, hx[:, :width])
        else:
            route, addon = arm.split('_', 1)
            ai = ADDONS.index(addon)
            if route == 'fixed':
                cp, hp = .5 * (cx[:, 0] + ca[:, ai]), .5 * (hx[:, 0] + ha[:, ai])
            else:
                blocks = [np.column_stack([x, a[:, ai] - x[:, 0]]) for x, a in ((tx, ta), (cx, ca), (hx, ha))]
                head = head_fit(blocks[0], target - tx[:, 0])
                cp, hp = cx[:, 0] + head_predict(head, blocks[1]), hx[:, 0] + head_predict(head, blocks[2])
        radius = C.calibration(cp, ctarget, cids)
        cdist = C.distribution(cp, radius)
        hpdist = C.distribution(hp, radius).reshape(len(d['names']), len(held), 6)
        levels = np.array([C.threshold(cdist[:, 3], ctarget, b) for b in C.BUDGETS])
        rates = [float(np.mean(-cdist[ctarget > 0, 3] <= t)) for t in levels]
        assert np.all(np.array(rates) <= C.BUDGETS)
        if arm == 'blend_frozen':
            si = C.SEEDS.index(seed)
            np.testing.assert_allclose(hpdist, d['baseline'][si][:, held], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(levels, d['baseline_thresholds'][si, fold], atol=1e-12, rtol=1e-12)
        predictions.append(hpdist)
        thresholds.append(levels)
        heads[arm], radii[arm], false_warning[arm] = head, radius, rates
    save_npz(result_path, predictions=np.stack(predictions), thresholds=np.stack(thresholds),
             heads=np.frombuffer(cloudpickle.dumps(heads), dtype=np.uint8), calibration=json.dumps(radii),
             calibration_false_warning=json.dumps(false_warning), records=json.dumps(records),
             train_control=tx.reshape(4, len(proper), 7), train_addons=addon_oof,
             calibration_control=cx.reshape(4, len(cal), 7), calibration_addons=cal_addons,
             held_control=hx.reshape(len(d['names']), len(held), 7), held_addons=held_addons,
             proper_rows=proper, calibration_rows=cal, held_rows=held,
             fingerprint=json.dumps(d['fingerprint']), arms=ARMS, addons=ADDONS)
    print('FUSION_FOLD_COMPLETE', seed, fold, len(held), 'frozen distribution/threshold replay; inner design isolation; all calibration budgets passed', flush=True)
    return seed, fold


def run(smoke=False, workers=4):
    d = setup()
    jobs = [(42, 0)] if smoke else [(s, f) for s in C.SEEDS for f in range(5)]
    # Fork shares immutable feature arrays; workers never reload or mutate large tables.
    if workers == 1 or smoke:
        for s, f in jobs:
            fold_job(s, f)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(fold_job, s, f) for s, f in jobs]):
                future.result()
    seeds = [42] if smoke else C.SEEDS
    selected = np.flatnonzero(d['folds'] == 0) if smoke else np.arange(len(d['y']))
    predictions = np.full((len(ARMS), len(seeds), len(d['names']), len(selected), 6), np.nan)
    thresholds = np.full((len(ARMS), len(seeds), 5, len(C.BUDGETS)), np.nan)
    records, partitions, calibration_budgets = [], [], []
    for seed, fold in jobs:
        path = OUT / f'fusion_seed{seed}_fold{fold}.npz'
        with np.load(path) as z:
            assert json.loads(str(z['fingerprint'])) == d['fingerprint']
            held = z['held_rows']
            si = seeds.index(seed)
            predictions[:, si, :, np.searchsorted(selected, held), :] = z['predictions'].transpose(2, 0, 1, 3)
            thresholds[:, si, fold] = z['thresholds']
            records += json.loads(str(z['records']))
            calibration_budgets.append({'seed': seed, 'fold': fold, 'rates': json.loads(str(z['calibration_false_warning']))})
            partitions.append({'seed': seed, 'fold': fold, **{key: z[key].tolist() for key in ('proper_rows', 'calibration_rows', 'held_rows')}})
    assert np.isfinite(predictions).all()
    assert np.isfinite(thresholds[:, :, :1 if smoke else 5]).all()
    suffix = 'smoke_' if smoke else ''
    save_npz(OUT / f'{suffix}predictions.npz', predictions=predictions, thresholds=thresholds,
             y=d['y'][selected], ids=d['ids'][selected], source_indices=d['rows'][selected], row_fold=d['folds'][selected],
             conditions=d['names'], seeds=seeds, arms=ARMS, fields=C.FIELDS, fingerprint=json.dumps(d['fingerprint']))
    unique_fits = {r['path']: r for r in records}
    save_json(OUT / f'{suffix}fits.json', dict(fingerprint=d['fingerprint'], fits=list(unique_fits.values()),
              partitions=partitions, calibration_budgets=calibration_budgets,
              all_converged=all(r.get('success', True) for r in unique_fits.values())))
    print('ADDITION_SMOKE_PASSED' if smoke else 'INDEPENDENT_ADDITIONS_COMPLETE', predictions.shape, len(unique_fits), 'unique fit/reuse records', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke, args.workers)
